# StarterPack AgentIA - AI-Powered Insurance Recommendation System

An intelligent insurance recommendation system that combines SQL analysis, RAG (Retrieval Augmented Generation), and NLP to provide personalized insurance product recommendations for clients based on their business profile and needs.

## 🎯 Features

- **SQL Analyzer Agent**: Natural language to SQL conversion using Groq's LLM (Llama-3.3-70b)
- **RAG-based Recommender**: Semantic search through insurance policy documents using sentence transformers
- **Client Profiling**: Automated analysis of client profiles from database
- **Smart Scoring System**: Multi-factor scoring based on:
  - Business sector matching (fuzzy matching)
  - Profile compatibility (embedding similarity)
  - Coverage gap analysis
  - Semantic relevance of policy terms
- **Automated Pitch Generation**: Generate personalized insurance pitches for clients
- **Interactive Dashboard**: HTML and JSON reports with detailed recommendations

## 🏗️ Architecture

The system consists of five main components:

1. **ETL Pipeline** (`etl.py`): Data ingestion from Excel files into SQLite database
2. **SQL Analyzer** (`sql_analyzer.py`): Natural language query interface powered by Groq LLM
3. **RAG Recommender** (`rag_recommender.py`): Document analysis and recommendation engine
4. **Orchestrator** (`orchestrator.py`): Pipeline coordination for end-to-end client analysis
5. **Pitching Bot** (`pitching_bot.py`): Pitch generation module (to be implemented)

## 📋 Prerequisites

- Python 3.8+
- SQLite
- Groq API Key (for LLM capabilities)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nour-gab/StarterPack_AgentIA.git
   cd StarterPack_AgentIA
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: Some additional dependencies used by the code need to be installed manually:
   ```bash
   pip install langchain langchain-groq python-dotenv
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Prepare the database**
   ```bash
   python src/etl.py
   ```
   This will:
   - Load data from Excel files in the `data/` directory
   - Create the SQLite database at `db/insurance.db`
   - Set up necessary tables and views

## 📦 Dependencies

Core libraries (see `requirements.txt`):
- `pandas` - Data manipulation
- `sqlalchemy` - Database operations
- `pdfplumber` / `PyPDF2` - PDF text extraction
- `sentence-transformers` - Text embeddings
- `scikit-learn` - Machine learning utilities
- `fastapi` / `uvicorn` - API framework
- `streamlit` - Web interface
- `requests` - HTTP client
- `python-Levenshtein` / `fuzzywuzzy` - Fuzzy string matching

Note: LangChain dependencies (langchain, langchain-groq, python-dotenv) are used in the code but need to be added to requirements.txt

## 🚀 Usage

### 1. Run ETL Pipeline
```bash
python src/etl.py
```

### 2. Query with Natural Language
```bash
python src/sql_analyzer.py
```
Example query: "Show me 5 clients with their sector and recommended products"

### 3. Generate Recommendations
```bash
python src/rag_recommender.py --client 12122 --top_n 5
```

This will:
- Analyze the client profile
- Search relevant policy documents
- Generate scored recommendations
- Create HTML and JSON reports in the `output/` directory

### 4. Run Complete Pipeline
```bash
python src/orchestrator.py
```
Executes the full workflow: client analysis → recommendations → pitch generation

## 📂 Project Structure

```
StarterPack_AgentIA/
├── src/
│   ├── etl.py                 # Data ingestion pipeline
│   ├── sql_analyzer.py        # NL to SQL converter with Groq LLM
│   ├── rag_recommender.py     # RAG-based recommendation engine
│   ├── pitching_bot.py        # Pitch generation module (to be implemented)
│   └── orchestrator.py        # Pipeline orchestration
├── data/
│   ├── Données_Assurance_S2.1.xlsx           # Client data
│   ├── Description des colonnes-thématique 2.xlsx  # Column descriptions
│   ├── Mapping produits vs profils_cibles.xlsx     # Product-profile mapping
│   ├── Description_garanties.xlsx             # Coverage descriptions
│   └── Conditions Générales/                  # Insurance policy PDFs
│       ├── 4-CG-IARD/                        # Property insurance docs
│       └── 5-CG-Engineering/                 # Engineering insurance docs
├── db/
│   └── insurance.db           # SQLite database (generated)
├── out/                       # Output directory
├── output/                    # Generated reports (created at runtime)
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Docker configuration (to be configured)
└── README.md                  # This file
```

## 📊 Data Files

The system uses several Excel files as data sources:

- **Données_Assurance_S2.1.xlsx**: Main client database with company information and insurance needs
- **Description des colonnes-thématique 2.xlsx**: Data dictionary for client columns
- **Mapping produits vs profils_cibles.xlsx**: Maps insurance products to target client profiles
- **Description_garanties.xlsx**: Detailed coverage and guarantee descriptions

## 🔍 How It Works

### Recommendation Scoring

The system uses a weighted scoring algorithm:

```
Final Score = 0.3 × Business Match 
            + 0.3 × Profile Match 
            + 0.2 × Coverage Gap 
            + 0.2 × Semantic Relevance
```

- **Business Match**: Fuzzy matching between client sector and product branch
- **Profile Match**: Embedding similarity between client description and target profiles
- **Coverage Gap**: Higher score for products not already held by the client
- **Semantic Relevance**: Cosine similarity of policy documents to client needs

### RAG Pipeline

1. **Ingestion**: Extract text from insurance policy PDFs
2. **Chunking**: Split documents into overlapping chunks (800 chars)
3. **Embedding**: Generate embeddings using `all-MiniLM-L6-v2`
4. **Retrieval**: Semantic search for relevant policy clauses
5. **Scoring**: Multi-factor evaluation of product fit
6. **Reporting**: Generate detailed recommendations with explanations

## 🔐 Configuration

Key configuration parameters in `rag_recommender.py`:

```python
# Scoring weights (adjustable)
weights = {
    'business_match': 0.3,
    'profile_match': 0.3,
    'coverage_gap': 0.2,
    'semantic': 0.2
}

# Text chunking
max_chars = 800
overlap = 150

# Semantic search
top_k = 10
min_similarity = 0.3
```

## 📝 Output

The system generates two types of reports for each client:

1. **JSON Report** (`recommendation_client_[ID].json`): 
   - Structured data with all scores and metadata
   - Complete client profile
   - Top recommendations with detailed scoring breakdown

2. **HTML Dashboard** (`recommendation_client_[ID].html`):
   - Interactive visualization of recommendations
   - Expandable sections for each product
   - Supporting clauses from policy documents
   - Explanations for scoring decisions

## 🐳 Docker Support

Docker Compose configuration file is present but needs to be configured for containerized deployment.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is provided as a starter pack for AI agent development in the insurance domain.

## 🙏 Acknowledgments

- Built with LangChain and Groq's LLM APIs
- Uses Sentence Transformers for embeddings
- Powered by RAG architecture for document understanding
