import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

vector_store_dir = r'C:\Users\Antifragile\Desktop\credit-trust-complaint-bot\src\RAG\vector_store\faiss_index'
index_path = os.path.join(vector_store_dir, 'faiss_index.index')
metadata_path = os.path.join(vector_store_dir, 'metadata.pkl')

# Load FAISS index
index = faiss.read_index(index_path)
print(f'Loaded FAISS index with {index.ntotal} vectors')

# Load metadata
with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)
print(f'Loaded {len(metadata)} metadata entries')

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Refined query for BNPL
query = "high fees in Buy Now Pay Later services"
print(f'\nQuery: "{query}"')

# Embed query
query_embedding = model.encode([query])[0].astype('float32')
faiss.normalize_L2(query_embedding.reshape(1, -1))

# Search for top-20 candidates (to filter for BNPL)
k = 20
distances, indices = index.search(query_embedding.reshape(1, -1), k)

# Filter results for BNPL-related products
bnpl_results = []
for distance, idx in zip(distances[0], indices[0]):
    if idx < len(metadata):
        meta = metadata[idx]
        # Check if product contains BNPL-related terms
        if 'buy now pay later' in meta['product'].lower():
            bnpl_results.append((distance, idx))
    if len(bnpl_results) >= 5:  # Stop at 5 BNPL results
        break

# If no BNPL results, fall back to top-5
if not bnpl_results:
    print("No BNPL-specific complaints found. Showing top-5 general results.")
    bnpl_results = list(zip(distances[0], indices[0]))[:5]

# Print results
print(f'\nTop {min(5, len(bnpl_results))} results:')
for i, (distance, idx) in enumerate(bnpl_results):
    if idx < len(metadata):
        meta = metadata[idx]
        print(f'\nResult {i+1}:')
        print(f'Similarity Score: {distance:.4f}')
        print(f'Complaint ID: {meta["complaint_id"]}')
        print(f'Product: {meta["product"]}')
        print(f'Chunk Index: {meta["chunk_index"]}')
        print(f'Text: {meta["text"][:200]}...')
    else:
        print(f'\nResult {i+1}: No metadata available (index {idx} out of range)')