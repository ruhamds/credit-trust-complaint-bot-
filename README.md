# Credit Trust Complaint Bot

This repository analyzes over 9.6 million US consumer complaints and prepares a cleaned, structured, and vectorized dataset ready for downstream tasks like semantic search or Retrieval-Augmented Generation (RAG). It uses Python, FAISS, and SentenceTransformers.

---

## Project Structure

credit-trust-complaint-bot/
├── .github/workflows/
│ ├── ci.yml
│ └── unittests.yml
├── data/ # Raw data (ignored by Git)
├── notebooks/
│ ├── EDA.ipynb # Task 1: Data Exploration
│ └── visual_vector_store.ipynb # Task 2: Index Visualization
└── src/
├── RAG/
│ └── vector_store/
│ ├── init.py
│ ├── build_vector_store.py # Task 2: Vector Store Builder
│ └── visualize_chunks.py # Task 2: Analysis
├── README.MD

---

## Task 1: Data Cleaning & EDA

### Dataset Overview

- **Original rows:** 9,609,797
- **Columns:** 18
- **Missing Narratives:** ~69%
- **Final narrative-only subset:** 459,138 rows

### Target Products Extracted

Filtered based on narratives and sub-product matches:

- Credit card: 433,055
- Savings account: 355,149
- Money transfers: 150,212
- Personal loan: 22,700
- Buy Now, Pay Later (BNPL): 19,698

### Cleaning & Processing

- Removed entries with missing or short narratives (<10 words)
- Standardized overlapping product names such as:
  - `Credit card` + `Credit card or prepaid card`
  - `Money transfers` + `Money transfer, virtual currency, or money service`
- Saved cleaned output as:  
  `data/processed/filtered_complaints.csv`

---

## Task 2: Chunking & Vector Store Creation

### Chunking Strategy

- Split complaint narratives into **overlapping 300-word chunks**
- Overlap: **50 words**
- Each chunk retains metadata:
  - `complaint_id`
  - `product`
  - `chunk_index`

### Embedding Model

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension:** 384
- Chosen for:
  - Balance of speed + accuracy
  - Great for semantic search on short-to-medium texts

### FAISS Vector Store

- Used `IndexFlatIP` with **cosine similarity** (L2-normalized)
- Artifacts saved:
  - `faiss_index.index`
  - `metadata.pkl`
  - `config.pkl`
- Stored in: `src/task2_vectorstore/faiss_index/`

---

## How to Run

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run preprocessing (Task 1)
python src/task1_preprocessing/preprocess_data.py

# Run vector store builder (Task 2)
python src/task2_vectorstore/build_vector_store.py
