# scripts/vector_store.py

import pandas as pd
import numpy as np
import nltk
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from tqdm import tqdm

# Download NLTK tokenizer
nltk.download('punkt')

class TextChunker:
    def __init__(self, chunk_size=200, chunk_overlap=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, complaint_id: str, product: str) -> List[Dict]:
        """Split a single narrative into overlapping word chunks with metadata."""
        if not isinstance(text, str) or len(text.strip()) < 10:
            return []

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


class VectorStoreBuilder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # For cosine similarity
        self.metadata = []

    def build_vector_store(self, df: pd.DataFrame, chunker: TextChunker, batch_size: int = 1000):
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

        texts = [chunk['text'] for chunk in all_chunks]
        self.metadata = [chunk for chunk in all_chunks]

        print("Generating embeddings...")
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, batch_size=batch_size, show_progress_bar=True)
            embeddings.extend(batch_embeddings)

        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
        self.index.add(embeddings_array)

        print(f"Vector store built with {self.index.ntotal} vectors")

    def save_vector_store(self, directory="vector_store"):
        os.makedirs(directory, exist_ok=True)

        faiss.write_index(self.index, os.path.join(directory, "faiss_index.index"))

        with open(os.path.join(directory, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)

        config = {
            'model_name': 'all-MiniLM-L6-v2',
            'dimension': self.dimension,
            'total_vectors': self.index.ntotal,
            'chunk_size': self.metadata[0]['chunk_index'] + 1 if self.metadata else 0,
        }

        with open(os.path.join(directory, "config.pkl"), 'wb') as f:
            pickle.dump(config, f)

        print(f"Vector store saved to {directory}/")
if __name__ == "__main__":
    import traceback

    try:
        print("🔹 Loading data...")
        df = pd.read_csv("data/processed/filtered_complaints.csv", usecols=["Complaint ID", "Product", "Consumer complaint narrative"])
        
        # SAMPLE 2K for safe memory use
        df_sample = df.sample(n=10000, random_state=42).reset_index(drop=True)
        print(f"✅ Sampled {len(df_sample)} complaints")

        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        builder = VectorStoreBuilder()

        print("🔹 Starting chunking and embedding...")

        # Custom embedding logic with safe batch loop
        all_chunks = []
        for _, row in df_sample.iterrows():
            chunks = chunker.chunk_text(
                row['Consumer complaint narrative'],
                str(row['Complaint ID']),
                row['Product']
            )
            all_chunks.extend(chunks)

        print(f"✅ Created {len(all_chunks)} chunks")

        texts = [c['text'] for c in all_chunks]
        builder.metadata = all_chunks  # attach metadata before embedding

        print("🧠 Embedding and indexing in batches...")
        batch_size = 500
        embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            try:
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = builder.model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
                batch_embeddings = np.array(batch_embeddings).astype('float32')
                faiss.normalize_L2(batch_embeddings)
                builder.index.add(batch_embeddings)
                print(f"✅ Added batch {i // batch_size + 1}")
            except Exception as embed_error:
                print(f"❌ Failed batch {i // batch_size + 1}: {embed_error}")
                traceback.print_exc()

        print(f"✅ Total vectors indexed: {builder.index.ntotal}")

        print("💾 Saving vector store...")
        builder.save_vector_store("vector_store/")
        print("✅ Done!")

    except Exception as e:
        print("\n❌ Top-level script failure:")
        traceback.print_exc()

