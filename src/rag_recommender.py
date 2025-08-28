"""
rag_recommender.py

Optimized RAG-style Recommender that:
- Ingests PDFs from data/Conditions Générales/* (each product folder)
- Extracts text and chunks it
- Embeds chunks with sentence-transformers (all-MiniLM-L6-v2)
- Stores chunk metadata in memory (lightweight)
- Uses SQLAnalyzerAgent to query client profile
- Produces product recommendations with improved scoring:
  - Business match: Fuzzy match on sector/branch using Levenshtein or embedding sim
  - Coverage gap: Check against mocked/existing contracts (add column if needed)
  - Semantic relevance: Tailor query to client profile + product
  - New: Profile match score using embeddings on 'Profils cibles' vs client desc
  - Adjustable weights
- Outputs JSON + HTML dashboard with explanations

Usage:
    python rag_recommender.py --client 12122
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import pandas as pd
from sqlalchemy import create_engine, text
from fuzzywuzzy import fuzz  # pip install python-Levenshtein fuzzywuzzy for fuzzy matching

from sql_analyzer import SQLAnalyzerAgent  # Adjust import as needed

# ---------------------------
# Utility functions
# ---------------------------
def chunk_text(text: str, max_chars: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries where possible. Reduced max_chars for better relevance."""
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
            if len(s) > max_chars:
                for i in range(0, len(s), max_chars - overlap):
                    chunks.append(s[i:i + max_chars])
                current = ""
            else:
                current = s
    if current:
        chunks.append(current)
    # Add overlap
    if overlap and len(chunks) > 1:
        for i in range(1, len(chunks)):
            chunks[i] = chunks[i-1][-overlap:] + " " + chunks[i]
    return chunks

# ---------------------------
# PDF Ingestor
# ---------------------------
class PDFIngestor:
    def __init__(self, base_path: str, embedder_model: str = "all-MiniLM-L6-v2"):
        self.base_path = Path(base_path)
        self.model = SentenceTransformer(embedder_model)
        self.store: List[Dict[str, Any]] = []  # {id, product_folder, pdf_file, chunk_index, text, embedding}
        self._id_counter = 0

    def ingest_all(self, verbose: bool = True) -> None:
        product_folders = [p for p in self.base_path.iterdir() if p.is_dir()]
        for pf in tqdm(product_folders, desc="Ingesting product folders"):
            pdf_files = list(pf.glob("*.pdf"))
            for pdf in pdf_files:
                if verbose:
                    print(f"Ingesting {pdf}...")
                texts = self._extract_text(pdf)
                chunks = [c for t in texts for c in chunk_text(t)]
                if chunks:
                    embeddings = self.model.encode(chunks, show_progress_bar=verbose, convert_to_numpy=True)
                    for j, (txt, emb) in enumerate(zip(chunks, embeddings)):
                        self.store.append({
                            "id": f"doc_{self._id_counter}",
                            "product_folder": pf.name,
                            "pdf_file": pdf.name,
                            "chunk_index": j,
                            "text": txt,
                            "embedding": emb.astype(np.float32),
                        })
                        self._id_counter += 1

    def _extract_text(self, pdf_path: Path) -> List[str]:
        texts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text()
                    if txt:
                        # Clean noise: remove headers/footers if pattern known
                        txt = re.sub(r'^\s*\d+\s*$', '', txt, flags=re.M)  # Remove page numbers
                        texts.append(txt.strip())
        except Exception as e:
            print(f"❌ Failed to read {pdf_path}: {e}")
        return texts if texts else ["No text extracted."]  # Fallback

    def semantic_search(self, query: str, top_k: int = 10, min_sim: float = 0.3) -> List[Tuple[Dict[str, Any], float]]:
        if not self.store:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
        sims = [(rec, util.cos_sim(q_emb, rec["embedding"]).item()) for rec in self.store]
        sims = [(rec, s) for rec, s in sims if s >= min_sim]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

# ---------------------------
# RecommenderRAG
# ---------------------------
class RecommenderRAG:
    def __init__(self, pdf_base_path: str, db_url: str = "sqlite:///db/insurance.db"):
        self.pdf_ingestor = PDFIngestor(pdf_base_path)
        self.sql_agent = SQLAnalyzerAgent()
        self.engine = create_engine(db_url)
        self.embed_model = self.pdf_ingestor.model  # Reuse for profile embeddings
        # Weights for scoring (adjusted for better balance)
        self.weights = {
            'business_match': 0.3,
            'profile_match': 0.3,
            'coverage_gap': 0.2,
            'semantic': 0.2
        }

    def build_index(self):
        print("🔍 Ingesting PDFs and building embeddings...")
        self.pdf_ingestor.ingest_all(verbose=True)
        print(f"✅ Ingested {len(self.pdf_ingestor.store)} chunks.")

    def _get_client_profile(self, client_id: int) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            q = text("""
                SELECT REF_PERSONNE, RAISON_SOCIALE, LIB_SECTEUR_ACTIVITE,
                       GROUP_CONCAT(DISTINCT LIB_PRODUIT) AS existing_products,
                       GROUP_CONCAT(DISTINCT Profils_cibles) AS matched_profiles
                FROM client_profiles
                WHERE REF_PERSONNE = :id
                GROUP BY REF_PERSONNE
            """)
            row = conn.execute(q, {"id": client_id}).fetchone()
            if not row:
                return {}
            profile = dict(row._mapping)
            profile['client_desc'] = f"{profile.get('LIB_SECTEUR_ACTIVITE', '')} {profile.get('LIB_ACTIVITE', '')}"
            return profile

    def _get_candidate_products(self, client_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        sector = client_profile.get("LIB_SECTEUR_ACTIVITE", "")
        activity = client_profile.get("LIB_ACTIVITE", "")
        with self.engine.connect() as conn:
            q = text("""
                SELECT DISTINCT LIB_BRANCHE, LIB_SOUS_BRANCHE, LIB_PRODUIT, "Profils cibles" AS profiles
                FROM mapping_produits
                WHERE LIB_BRANCHE LIKE :sector OR LIB_PRODUIT LIKE :activity
            """)
            rows = conn.execute(q, {"sector": f"%{sector}%", "activity": f"%{activity}%"}).fetchall()
            return [dict(r._mapping) for r in rows]

    def score_product(self, client_profile: Dict[str, Any], product_candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Improved scoring:
        - Business match: Fuzzy ratio on branch/sector + sous_branche/activity
        - Profile match: Embedding sim between 'Profils cibles' and client_desc
        - Coverage gap: 1.0 if not in existing_products (now aggregated)
        - Semantic: Tailored query "product garanties for [client_desc]"
        """
        # Business match (fuzzy for robustness)
        bm_sector = fuzz.partial_ratio(client_profile.get("LIB_SECTEUR_ACTIVITE", "").lower(), product_candidate["LIB_BRANCHE"].lower()) / 100.0
        bm_activity = fuzz.partial_ratio(client_profile.get("LIB_ACTIVITE", "").lower(), product_candidate.get("LIB_SOUS_BRANCHE", "").lower()) / 100.0
        bm = max(bm_sector, bm_activity)

        # Profile match (embedding sim)
        client_emb = self.embed_model.encode(client_profile['client_desc'])
        profile_emb = self.embed_model.encode(product_candidate['profiles'] or "general")
        pm = util.cos_sim(client_emb, profile_emb).item()

        # Coverage gap
        existing = set(p.strip().lower() for p in (client_profile.get("existing_products") or "").split(','))
        gap = 1.0 if product_candidate["LIB_PRODUIT"].strip().lower() not in existing else 0.0

        # Semantic relevance (tailored query)
        query = f"{product_candidate['LIB_PRODUIT']} garanties conditions for {client_profile['client_desc']}"
        sims = self.pdf_ingestor.semantic_search(query, top_k=5, min_sim=0.4)
        sem_score = np.mean([s for _, s in sims]) if sims else 0.0

        # Weighted final
        final_score = (
            self.weights['business_match'] * bm +
            self.weights['profile_match'] * pm +
            self.weights['coverage_gap'] * gap +
            self.weights['semantic'] * sem_score
        )

        return {
            "product": product_candidate["LIB_PRODUIT"],
            "branch": product_candidate["LIB_BRANCHE"],
            "sub_branch": product_candidate.get("LIB_SOUS_BRANCHE", ""),
            "profiles": product_candidate["profiles"],
            "business_match_score": round(bm, 3),
            "profile_match_score": round(pm, 3),
            "coverage_gap_score": round(gap, 3),
            "semantic_score": round(sem_score, 3),
            "final_score": round(final_score, 4),
            "top_clauses": [
                {"pdf_file": rec["pdf_file"], "chunk_index": rec["chunk_index"], "text_snippet": rec["text"][:300] + "...", "sim": round(sim, 4)}
                for rec, sim in sims
            ],
            "explanation": f"Matched based on {bm*100}% sector similarity; {pm*100}% profile fit; {'gap detected' if gap else 'already covered'}."
        }

    def recommend_for_client(self, client_id: int, top_n: int = 5) -> Dict[str, Any]:
        client_profile = self._get_client_profile(client_id)
        if not client_profile:
            return {"error": f"Client {client_id} not found."}

        candidates = self._get_candidate_products(client_profile)
        if not candidates:
            return {"error": "No candidate products found."}

        scored = [self.score_product(client_profile, c) for c in candidates]
        scored = [s for s in scored if s['final_score'] > 0.2]  # Filter low scores
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

        # Enhanced HTML with explanations
        html_path = Path(out_dir) / f"recommendation_client_{client_id}.html"
        html = [
            "<html><head><meta charset='utf-8'><title>Recommendation Report</title>",
            "<style>body{font-family:Arial;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:8px;} th{background-color:#f2f2f2;}</style></head><body>"
        ]
        html.append(f"<h2>Recommendation Report for Client {client_id}</h2>")
        html.append("<h3>Client Profile</h3><pre>" + json.dumps(report["client_profile"], ensure_ascii=False, indent=2) + "</pre>")
        html.append("<h3>Top Recommendations</h3><table><tr><th>Product</th><th>Score</th><th>Explanation</th></tr>")
        for rec in report.get("top_recommendations", []):
            html.append(f"<tr><td>{rec['product']}</td><td>{rec['final_score']}</td><td>{rec['explanation']}</td></tr>")
        html.append("</table>")
        html.append("<h4>Detailed Scores</h4>")
        for rec in report.get("top_recommendations", []):
            html.append(f"<details><summary>{rec['product']} (Score: {rec['final_score']})</summary>")
            html.append("<ul>")
            for k in ['business_match_score', 'profile_match_score', 'coverage_gap_score', 'semantic_score']:
                html.append(f"<li>{k}: {rec[k]}</li>")
            html.append("</ul>")
            if rec.get("top_clauses"):
                html.append("<h5>Supporting Clauses from PDFs</h5><pre>")
                for c in rec["top_clauses"]:
                    html.append(f"{c['pdf_file']} (sim={c['sim']}): {c['text_snippet']}<br>")
                html.append("</pre>")
            html.append("</details>")
        html.append("</body></html>")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("".join(html))
        print(f"Saved JSON -> {json_path}")
        print(f"Saved HTML -> {html_path}")

# ---------------------------
# CLI / Example run
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", type=int, required=True, help="Client REF_PERSONNE to analyze")
    parser.add_argument("--pdfs", type=str, default="data/Conditions Générales", help="Base path to product condition PDF folders")
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    manager = RecommenderRAG(pdf_base_path=args.pdfs)
    manager.build_index()
    report = manager.recommend_for_client(args.client, top_n=args.top_n)
    manager.save_report(report, out_dir="output")
    print("\nTop recommendations:")
    for r in report.get("top_recommendations", []):
        print(f"- {r['product']} (score {r['final_score']}): {r['explanation']}")