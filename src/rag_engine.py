import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

class ComplaintRAGEngine:
    def __init__(self, vector_store_dir='vector_store'):
        self.index_path = os.path.join(vector_store_dir, 'complaints_faiss.index')
        self.meta_path = os.path.join(vector_store_dir, 'chunks_metadata.parquet')
        
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("FAISS index or metadata parquet file missing. Run task 2 first.")
            
        print("--- Loading Persisted FAISS Index & Metadata Frame ---")
        self.index = faiss.read_index(self.index_path)
        self.metadata_df = pd.read_parquet(self.meta_path)
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pull the new valid HF token from local environment parameters
        hf_token = os.getenv("HF_TOKEN", None)
        self.client = InferenceClient(token=hf_token)

    def retrieve_context(self, query: str, product_filter: str = None, top_k: int = 5) -> str:
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        
        distances, indices = self.index.search(query_vector, k=top_k * 5)
        
        flat_indices = indices.flatten()
        valid_indices = [int(idx) for idx in flat_indices if idx != -1]
        
        if not valid_indices:
            return "No matching complaints found in database."
            
        matched_chunks = self.metadata_df.iloc[valid_indices].copy()
        
        if product_filter:
            matched_chunks = matched_chunks[matched_chunks['product_category'] == product_filter]
            
        final_hits = matched_chunks.head(top_k)
        if final_hits.empty:
            return "No matching complaints found matching the product filter constraints."
            
        context_blocks = []
        for _, row in final_hits.iterrows():
            context_blocks.append(
                f"[Product: {row['product_category']} | Issue: {row['issue']}]\n"
                f"Complaint: {row['chunk_text']}\n"
                f"---"
            )
        return "\n\n".join(context_blocks)

    def generate_answer(self, question: str, product_filter: str = None) -> dict:
        context = self.retrieve_context(query=question, product_filter=product_filter, top_k=5)
        
        prompt_template = f"""<s>[INST] You are an advanced AI Assistant for CrediTrust Financial. Analyze customer complaint trends based ONLY on the evidence provided.

CONTEXT COMPLAINTS:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Ground your analysis strictly in the complaints provided above.
2. If the context does not contain relevant insights, state that no evidence was found. Do not make things up.
3. Keep your response concise, using clear bullet points. [/INST]"""

        try:
            # Explicitly locked onto a highly stable open-source inference endpoint
            response = self.client.text_generation(
                model="Qwen/Qwen2.5-7B-Instruct",
                prompt=prompt_template,
                max_new_tokens=500,
                temperature=0.2,
                return_full_text=False
            )
            answer_text = response if isinstance(response, str) else response.get("generated_text", str(response))
            return {"answer": answer_text, "context": context}
        except Exception as e:
            return {"answer": f"Error running generation layer: {str(e)}", "context": context}
