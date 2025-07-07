# This script builds a vector store from consumer complaints, chunking narratives and generating embeddings.
# It uses NLTK for text processing, SentenceTransformers for embeddings, and FAISS for indexing.
import pandas as pd
import numpy as np
import yaml
import nltk
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define chunk size for processing
chunk_size = 10000  # Rows per chunk
batch_size = 1000   # Chunks per embedding batch

#The `TextChunker` class splits narratives into overlapping chunks using NLTK for accurate word counts.


class TextChunker:
    def __init__(self, chunk_size=200, chunk_overlap=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, complaint_id: str, product: str) -> List[Dict]:
        """Split text into overlapping chunks with metadata."""
        if not isinstance(text, str) or len(text.strip()) < 10:
            return []

        # Tokenize into words
        words = nltk.word_tokenize(text)
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunks.append({
                'text': chunk_text,
                'complaint_id': complaint_id,
                'product': product,
                'chunk_index': len(chunks)
            })

            if i + self.chunk_size >= len(words):
                break

        return chunks

#The `VectorStoreBuilder` class generates embeddings and builds the FAISS index.

class VectorStoreBuilder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.metadata = []

    def build_vector_store(self, df: pd.DataFrame, chunker: TextChunker, batch_size: int):
        """Build FAISS vector store from dataframe."""
        print("Starting text chunking...")
        all_chunks = []

        for _, row in df.iterrows():
            chunks = chunker.chunk_text(
                row['Consumer complaint narrative'],
                str(row['Complaint ID']),
                row['Product']
            )
            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks from {len(df)} complaints")

        # Extract texts for embedding
        texts = [chunk['text'] for chunk in all_chunks]
        self.metadata = [{
            'complaint_id': chunk['complaint_id'],
            'product': chunk['product'],
            'chunk_index': chunk['chunk_index'],
            'text': chunk['text']
        } for chunk in all_chunks]

        # Generate embeddings in batches
        print("Generating embeddings...")
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, batch_size=batch_size, show_progress_bar=True)
            embeddings.extend(batch_embeddings)

        # Normalize and add to FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)

        print(f"Vector store built with {self.index.ntotal} vectors")

    def save_vector_store(self, directory="../vector_store/faiss_index"):
        """Save the vector store and metadata."""
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.index, os.path.join(directory, "faiss_index.index"))

        with open(os.path.join(directory, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)

        config = {
            'model_name': 'all-MiniLM-L6-v2',
            'dimension': self.dimension,
            'total_vectors': self.index.ntotal,
            'chunk_size': 200,
            'chunk_overlap': 20
        }

        with open(os.path.join(directory, "config.pkl"), 'wb') as f:
            pickle.dump(config, f)

        print(f"Vector store saved to {directory}/")

#We load `filtered_complaints.csv` in chunks, chunk narratives, generate embeddings, and index them.


input_path = 'data/processed/filtered_complaints.csv'
vector_store_dir = '../vector_store/faiss_index'
chunker = TextChunker(chunk_size=200, chunk_overlap=20)
vector_builder = VectorStoreBuilder()

# Process in chunks
total_vectors = 0
for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size,
                                             usecols=['Complaint ID', 'Product', 'Consumer complaint narrative'])):
    print(f'Processing chunk {chunk_idx + 1}...')
    vector_builder.build_vector_store(chunk, chunker, batch_size)
    total_vectors += vector_builder.index.ntotal

vector_builder.save_vector_store(vector_store_dir)

## Step 5: Update Report with Task 2 Summary

#We append a Task 2 summary to `reports/eda_summary.md`, justifying the chunking strategy and embedding model.

import os # Import the os module

report_path = '../reports/eda_summary.md'
task2_summary = f"""
## Task 2: Text Chunking, Embedding, and Vector Store Indexing

### Chunking Strategy
Used a custom `TextChunker` with `chunk_size=200` words and `chunk_overlap=20` words (10%). Tested:
- **100-word chunks**: Missed context, splitting sentences (e.g., "high fees" separated from details).
- **500-word chunks**: Diluted specific issues (e.g., mixed "fees" and "customer service").
- **200-word chunks**: Balanced context and precision, capturing single issues effectively.
The 20-word overlap ensured continuity across chunks.

### Embedding Model Choice
Selected `sentence-transformers/all-MiniLM-L6-v2` for:
- **Efficiency**: Small size (22MB), 384-dimensional vectors, suitable for Colab’s ~12GB RAM.
- **Performance**: Optimized for short texts, ideal for complaint chunks.
- **Open-Source**: Freely available, no API costs.

### Vector Store
Used FAISS with cosine similarity for fast, scalable searches. Metadata (`Complaint ID`, `Product`, `chunk_index`, `text`) ensures traceability. Total vectors: {total_vectors}.

Processed in chunks of 10,000 rows and embedded in batches of 1,000 to manage memory.
"""

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(report_path), exist_ok=True)

with open(report_path, 'a') as f:
    f.write(task2_summary)

print(f'Appended Task 2 summary to {report_path}')