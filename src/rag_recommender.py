"""
recommender_rag.py

RAG-style Recommender that:
- Ingests PDFs from data/conditions Generales/* (each product folder)
- Extracts text and chunks it
- Embeds chunks with sentence-transformers (all-MiniLM-L6-v2)
- Stores chunk metadata in memory (lightweight)
- Uses SQLAnalyzerAgent (assumed in same project) to query client profile
- Produces product recommendations with a scoring breakdown and outputs JSON + HTML dashboard

Usage:
    python recommender_rag.py --client 12122
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
from sqlalchemy import create_engine, text

# Import your SQL analyzer agent class (adjust the import path if needed)
# from sql_agent import SQLAnalyzerAgent   # OR adapt below if in different module
from sql_analyzer import SQLAnalyzerAgent  # assumes sql_analyzer.py present in PYTHONPATH

# ---------------------------
# Utility functions
# ---------------------------
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries where possible."""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current.strip())
            # if sentence itself very large, split raw
            if len(s) > max_chars:
                for i in range(0, len(s), max_chars - overlap):
                    chunks.append(s[i:i + max_chars])
                current = ""
            else:
                current = s
    if current:
        chunks.append(current)
    # add small overlap merging
    if overlap and len(chunks) > 1:
        merged = []
        for i, c in enumerate(chunks):
            if i == 0:
                merged.append(c)
            else:
                # keep overlap of previous chars
                prev = merged[-1]
                tail = prev[-overlap:] if len(prev) > overlap else prev
                merged.append(tail + " " + c)
        return merged
    return chunks

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------------------------
# PDF Ingestor
# ---------------------------
class PDFIngestor:
    def __init__(self, base_path: str, embedder_model: str = "all-MiniLM-L6-v2"):
        self.base_path = Path(base_path)
        self.model = SentenceTransformer(embedder_model)
        # in-memory store: list of dicts {id, product_folder, pdf_file, chunk_index, text, embedding}
        self.store: List[Dict[str, Any]] = []
        self._id_counter = 0

    def ingest_all(self, verbose: bool = True) -> None:
        """Walk product folders and ingest PDFs."""
        product_folders = [p for p in self.base_path.iterdir() if p.is_dir()]
        for pf in product_folders:
            pdf_files = sorted([f for f in pf.glob("*.pdf")])
            for pdf in pdf_files:
                if verbose:
                    print(f"Ingesting {pdf} ...")
                texts = self._extract_text(pdf)
                chunks = []
                for t in texts:
                    chunks.extend(chunk_text(t))
                # embed in batches
                for idx in range(0, len(chunks), 32):
                    batch = chunks[idx: idx + 32]
                    embeddings = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
                    for j, txt in enumerate(batch):
                        rec = {
                            "id": f"doc_{self._id_counter}",
                            "product_folder": pf.name,
                            "pdf_file": pdf.name,
                            "chunk_index": idx + j,
                            "text": txt,
                            "embedding": embeddings[j].astype(np.float32),
                        }
                        self.store.append(rec)
                        self._id_counter += 1

    def _extract_text(self, pdf_path: Path) -> List[str]:
        """Extract text from PDF; returns list of page texts."""
        texts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text()
                    if txt:
                        texts.append(txt)
        except Exception as e:
            print(f"❌ Failed to read {pdf_path}: {e}")
        return texts

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        q_emb = self.model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
        # compute similarity across store
        sims = []
        for rec in self.store:
            s = cosine_sim(q_emb, rec["embedding"])
            sims.append((rec, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

# ---------------------------
# RecommenderRAG
# ---------------------------
class RecommenderRAG:
    def __init__(self, pdf_base_path: str, db_url: str = "sqlite:///db/insurance.db"):
        self.pdf_ingestor = PDFIngestor(pdf_base_path)
        self.sql_agent = SQLAnalyzerAgent()  # uses Groq internally
        # connect to DB to fetch product mapping for scoring rules
        self.engine = create_engine(db_url)

    def build_index(self):
        print("🔍 Ingesting PDFs and building embeddings...")
        self.pdf_ingestor.ingest_all(verbose=True)
        print(f"✅ Ingested {len(self.pdf_ingestor.store)} chunks.")

    def _get_client_profile(self, client_id: int) -> Dict[str, Any]:
        # Use SQL analyzer to get client rows from view client_profiles
        # We'll call SQLAnalyzerAgent.ask to produce SQL and run it; but existing agent asks the LLM to generate SQL
        # Here we directly query the DB for deterministic fetch to avoid NL-to-SQL step.
        with self.engine.connect() as conn:
            q = text("SELECT * FROM client_profiles WHERE REF_PERSONNE = :id")
            row = conn.execute(q, {"id": client_id}).fetchone()
            if not row:
                return {}
            return dict(row._mapping)


    def _get_candidate_products(self, client_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Simple business-logic: fetch mapping_produits entries where LIB_BRANCHE matches client's sector
        sector = client_profile.get("LIB_SECTEUR_ACTIVITE")
        with self.engine.connect() as conn:
            q = text("""
                SELECT LIB_BRANCHE, LIB_PRODUIT, "Profils cibles"
                FROM mapping_produits
                WHERE LIB_BRANCHE = :sector
                LIMIT 50
            """)
            rows = conn.execute(q, {"sector": sector}).fetchall()
            cands = []
            for r in rows:
                cands.append({
                    "product": r["LIB_PRODUIT"],
                    "branch": r["LIB_BRANCHE"],
                    "profiles": r["Profils cibles"]
                })
            return cands

    def score_product(self, client_profile: Dict[str, Any], product_candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scoring combines:
         - Business-match score (1.0 if product branch matches client sector)
         - Coverage gap score (simple heuristic: 0/1 if product not present in client's known products)
         - Semantic relevance: similarity between product name and relevant PDF clauses (0..1)
        """
        # business match
        bm = 1.0 if client_profile.get("LIB_SECTEUR_ACTIVITE") == product_candidate.get("branch") else 0.5

        # naive coverage gap: check if client's current LIB_PRODUIT equals candidate
        # (client_profile may have LIB_PRODUIT column from view)
        existing = client_profile.get("LIB_PRODUIT") or ""
        gap = 1.0 if product_candidate["product"].strip().lower() not in existing.lower() else 0.0

        # semantic relevance: search in PDF docs for product name + "garantie" etc.
        query = f'{product_candidate["product"]} garanties conditions'
        sims = self.pdf_ingestor.semantic_search(query, top_k=5)
        sem_score = float(np.mean([s for (_, s) in sims])) if sims else 0.0

        # combine weights
        score = 0.45 * bm + 0.25 * gap + 0.30 * sem_score
        return {
            "product": product_candidate["product"],
            "branch": product_candidate["branch"],
            "profiles": product_candidate["profiles"],
            "business_match_score": round(bm, 3),
            "coverage_gap_score": round(gap, 3),
            "semantic_score": round(sem_score, 3),
            "final_score": round(score, 4),
            "top_clauses": [
                {"pdf_file": rec["pdf_file"], "chunk_index": rec["chunk_index"], "text_snippet": rec["text"][:400], "sim": round(sim, 4)}
                for rec, sim in sims
            ]
        }

    def recommend_for_client(self, client_id: int, top_n: int = 5) -> Dict[str, Any]:
        client_profile = self._get_client_profile(client_id)
        if not client_profile:
            return {"error": f"Client {client_id} not found."}

        candidates = self._get_candidate_products(client_profile)
        scored = []
        for c in candidates:
            s = self.score_product(client_profile, c)
            scored.append(s)
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        top = scored[:top_n]

        report = {
            "client_id": client_id,
            "client_profile": client_profile,
            "top_recommendations": top,
            "all_scores_count": len(scored)
        }
        return report

    def save_report(self, report: Dict[str, Any], out_dir: str = "output"):
        os.makedirs(out_dir, exist_ok=True)
        client_id = report.get("client_id")
        json_path = Path(out_dir) / f"recommendation_client_{client_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        # simple HTML dashboard
        html_path = Path(out_dir) / f"recommendation_client_{client_id}.html"
        html = ["<html><head><meta charset='utf-8'><title>Recommendation Report</title></head><body>"]
        html.append(f"<h2>Recommendation Report for Client {client_id}</h2>")
        html.append("<h3>Client Profile</h3><pre>")
        html.append(json.dumps(report["client_profile"], ensure_ascii=False, indent=2))
        html.append("</pre><h3>Top Recommendations</h3><ol>")
        for rec in report["top_recommendations"]:
            html.append(f"<li><b>{rec['product']}</b> — score: {rec['final_score']}")
            html.append("<ul>")
            html.append(f"<li>business_match_score: {rec['business_match_score']}</li>")
            html.append(f"<li>coverage_gap_score: {rec['coverage_gap_score']}</li>")
            html.append(f"<li>semantic_score: {rec['semantic_score']}</li>")
            html.append("</ul>")
            # show top clause snippet
            if rec.get("top_clauses"):
                html.append("<details><summary>Top Clauses</summary><pre>")
                for c in rec["top_clauses"]:
                    html.append(f"{c['pdf_file']} (sim={c['sim']}): {c['text_snippet']}</pre>")
                html.append("</pre></details>")
            html.append("</li>")
        html.append("</ol></body></html>")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        print(f"Saved JSON -> {json_path}")
        print(f"Saved HTML -> {html_path}")

# ---------------------------
# CLI / Example run
# ---------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=int, required=True, help="Client REF_PERSONNE to analyze")
    parser.add_argument("--pdfs", type=str, default="data\Conditions Générales", help="Base path to product condition PDF folders")
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    manager = RecommenderRAG(pdf_base_path=args.pdfs)
    # build embeddings once (this may take a minute)
    manager.build_index()
    # produce recommendations
    report = manager.recommend_for_client(args.client, top_n=args.top_n)
    # save outputs
    manager.save_report(report, out_dir="output")
    # also print brief summary
    print("\nTop recommendations:")
    for r in report.get("top_recommendations", []):
        print(f"- {r['product']} (score {r['final_score']})")
