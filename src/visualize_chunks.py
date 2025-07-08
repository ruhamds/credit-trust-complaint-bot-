import matplotlib.pyplot as plt
import pickle
import os
import faiss
import numpy as np
from sklearn.decomposition import PCA

def load_metadata(path="vector_store/metadata.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)

def plot_chunk_lengths(metadata):
    chunk_lengths = [len(chunk['text'].split()) for chunk in metadata]
    
    plt.figure(figsize=(10, 6))
    plt.hist(chunk_lengths, bins=40, color='skyblue', edgecolor='black')
    plt.title("Distribution of Chunk Word Lengths")
    plt.xlabel("Chunk Length (words)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_embeddings_2d(index_path="vector_store/faiss_index.index", sample_size=1000):
    index = faiss.read_index(index_path)
    vectors = index.reconstruct_n(0, min(sample_size, index.ntotal))
    
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(np.array(vectors))

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
    plt.title(f"PCA of {sample_size} Embeddings")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
