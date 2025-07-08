# CrediTrust Complaint-Answering Chatbot

## Project Overview
This project develops a Retrieval-Augmented Generation (RAG) system to enable semantic search and question-answering on consumer complaints related to financial products such as Credit Cards, Personal Loans, Buy Now Pay Later (BNPL), Savings Accounts, and Money Transfers.

---

## Project Structure

credit-trust-complaint-bot/
├── data/
│   ├── raw/                          # Raw CFPB complaint data
│   └── processed/
│       └── filtered_complaints.csv   # Filtered & cleaned complaint data
│       └── vector_store/
│           ├── faiss_index.index     # FAISS vector index file
│           ├── metadata.pkl          # Metadata linking chunks to complaints
│           └── config.pkl            # Configuration for embedding model
├── notebooks/
│   ├── Task1/
│   │   └── EDA.ipynb                 # Exploratory Data Analysis notebook
│   ├── Task2/
│   │   ├── chunking_eval.ipynb       # Chunking strategy experimentation
│   │   └── visual_vector_store.ipynb # Vector store visualization notebook
│   └── Task3/
│       └── rag_evaluation.ipynb      # RAG pipeline evaluation
├── src/
│   ├── app.py                       # Gradio web app for interactive chat (Task 4)
│   ├── rag_pipeline.py              # RAG pipeline class (Task 3)
│   ├── build_vector_store.py        # Script for building FAISS vector store (Task 2)
│   ├── chunking.py                  # Chunking utilities (Task 2)
│   ├── embedding.py                 # Embedding generation utilities (Task 2)
│   ├── indexing.py                  # FAISS indexing utilities (Task 2)
│   └── visualize_chunks.py          # Visualization utilities for chunk data (Task 2)
├── requirements.txt                 # Project dependencies
└── README.md                        # This file

---

## Summary of Tasks

### Task 1: Exploratory Data Analysis (EDA)
- Loaded and filtered 9.6 million consumer complaints down to 459,138 relevant complaints.
- Analyzed product distribution, narrative quality, and missing data patterns.
- Identified BNPL-related complaints via narrative keyword search due to absence of direct labels.

### Task 2: Vector Store Creation and Testing
- Built a FAISS vector store with 820,653 text chunks from the filtered complaints.
- Generated metadata linking chunks to complaint IDs and products.
- Tested retrieval accuracy using example queries, noting limited BNPL-specific results.

### Task 3: Retrieval-Augmented Generation (RAG) Pipeline
- Developed a RAG system combining dense retrieval (FAISS) with rule-based or LLM-based answer generation.
- Integrated the `SimpleRAGPipeline` class with methods for retrieval, answer generation, and end-to-end querying.
- Evaluated RAG pipeline with multiple domain-relevant questions; results saved in `rag_evaluation_results.csv`.


### Task 4: Interactive Chat Interface
- Built a user-friendly web interface using Gradio.
- Features text input, submit button, generated answer display, and source text excerpts for transparency.
- Supports easy question entry and response visualization.

---

## Setup & Usage

1. Clone the repository.
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
Build vector store (Task 2):

python src/build_vector_store.py
Run RAG evaluation (Task 3):
Execute notebooks/Task3/rag_evaluation.ipynb or use rag_pipeline.py directly.

Launch interactive app (Task 4):

python src/app.py



