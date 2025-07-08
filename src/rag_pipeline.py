import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from typing import List, Dict
from collections import Counter
from transformers import pipeline

# Dynamically define project root based on this file's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class SimpleRAGPipeline:
    def __init__(self, vector_store_dir="src/RAG/vector_store/", use_llm=False):
        """Initialize RAG pipeline with optional LLM."""
        self.load_vector_store(vector_store_dir)
        self.load_original_data()
        self.use_llm = use_llm
        if use_llm:
            self.generator = pipeline("text-generation", model="mistralai/Mixtral-8x7B-Instruct-v0.1", max_new_tokens=256)

    def load_vector_store(self, directory):
        """Load the vector store components."""
        self.index = faiss.read_index(os.path.join(directory, "faiss_index.index"))
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
        with open(os.path.join(directory, "config.pkl"), "rb") as f:
            self.config = pickle.load(f)
        self.model = SentenceTransformer(self.config["model_name"])
        print(f"✅ Loaded vector store with {len(self.metadata)} chunks")

    def load_original_data(self):
        """Load original complaint data."""
        try:
            csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "filtered_complaints.csv")
            self.df = pd.read_csv(
                csv_path,
                usecols=["Complaint ID", "Consumer complaint narrative"],
                dtype={"Complaint ID": str}
            )
            print(f"✅ Loaded {len(self.df)} original complaints")
        except FileNotFoundError:
            print("⚠️ Warning: Could not load data/processed/filtered_complaints.csv")
            self.df = None
        except ValueError as e:
            print(f"⚠️ Warning: Column mismatch - {e}. Loading all columns instead.")
            self.df = pd.read_csv(csv_path)
            print(f"✅ Loaded columns: {list(self.df.columns)}")

    def retrieve(self, question: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant chunks."""
        try:
            question_embedding = self.model.encode([question]).astype("float32")
            faiss.normalize_L2(question_embedding)
            scores, indices = self.index.search(question_embedding, k)

            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if 0 <= idx < len(self.metadata) and self.df is not None:
                    metadata = self.metadata[idx]
                    complaint_id = str(metadata["complaint_id"])
                    complaint_row = self.df[self.df["Complaint ID"] == complaint_id]

                    if not complaint_row.empty:
                        full_text = complaint_row["Consumer complaint narrative"].iloc[0]
                        results.append({
                            "rank": i + 1,
                            "score": float(score),
                            "complaint_id": complaint_id,
                            "product": metadata["product"],
                            "text": full_text,
                            "chunk_index": metadata["chunk_index"]
                        })
                    else:
                        print(f"⚠️ No match for Complaint ID: {complaint_id}")
                else:
                    print(f"⚠️ Invalid index or missing dataframe.")
            return results
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []

    def generate_answer(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """Generate answer using rule-based or LLM approach."""
        if not retrieved_chunks:
            return "I don't have enough information to answer this question."

        if self.use_llm:
            context_text = "\n".join([f"{c['text']}" for c in retrieved_chunks])
            prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer,
state that you don't have enough information.

Context:
{context_text}

Question: {question}
Answer:"""
            response = self.generator(prompt)[0]["generated_text"]
            return response.split("Answer:")[-1].strip()
        else:
            products = [c["product"] for c in retrieved_chunks]
            product_counts = Counter(products)
            all_text = " ".join([c["text"].lower() for c in retrieved_chunks])

            issue_keywords = {
                "unauthorized": ["unauthorized", "without permission"],
                "fees": ["fee", "charge", "cost"],
                "customer_service": ["customer service", "representative"],
                "billing": ["bill", "statement"],
                "account_access": ["access", "login"],
                "fraud": ["fraud", "scam"],
                "error": ["error", "mistake"]
            }
            found_issues = [
                issue for issue, keys in issue_keywords.items()
                if any(k in all_text for k in keys)
            ][:3]

            response = []
            if len(product_counts) == 1:
                product = list(product_counts.keys())[0]
                response.append(f"Based on {len(retrieved_chunks)} {product} complaints:")
            else:
                response.append(f"Based on {len(retrieved_chunks)} complaints across multiple products:")

            if found_issues:
                response.append("Common issues include:")
                descriptions = {
                    "unauthorized": "Unauthorized charges",
                    "fees": "Unexpected fees",
                    "customer_service": "Poor customer service",
                    "billing": "Billing problems",
                    "account_access": "Account access issues",
                    "fraud": "Fraud concerns",
                    "error": "System errors"
                }
                for issue in found_issues:
                    response.append(f"- {descriptions.get(issue, issue)}")

            excerpt = retrieved_chunks[0]["text"][:200]
            response.append(f"\nExample: \"{excerpt}...\"")

            return "\n".join(response)

    def query(self, question: str, k: int = 5) -> Dict:
        """Run the full RAG query pipeline."""
        retrieved_chunks = self.retrieve(question, k)
        answer = self.generate_answer(question, retrieved_chunks)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "num_chunks": len(retrieved_chunks)
        }


if __name__ == "__main__":
    rag = SimpleRAGPipeline(use_llm=False)
    questions = [
        "What are the most common credit card complaint issues?",
        "How do customers complain about savings account fees?",
        "What problems do people report with money transfers?",
        "What are typical complaints about Buy Now Pay Later services?"
    ]

    for q in questions:
        print("\n" + "="*80)
        result = rag.query(q)
        print(f"❓ Question: {result['question']}\n💬 Answer: {result['answer']}\n")
        for chunk in result["retrieved_chunks"][:2]:
            print(f"📚 Source: {chunk['product']} | Score: {chunk['score']:.3f}")
            print(f"    Preview: {chunk['text'][:150]}...\n")
