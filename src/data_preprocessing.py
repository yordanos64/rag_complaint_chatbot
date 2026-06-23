import pandas as pd
import numpy as np
import re
import os
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class ComplaintPipeline:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.chunked_df = None
        
        # Initialize the challenge specified embedding model (384 dimensions)
        print("--- Initializing all-MiniLM-L6-v2 Embedding Model ---")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def load_and_filter(self):
        print("--- Task 1: Loading and filtering data in chunks ---")
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Critical Error: Source file not found at {self.input_path}")
            
        target_products = [
            'Credit card or prepaid card',
            'Checking or savings account',
            'Money transfer, virtual currency, or money service',
            'Payday loan, title loan, or personal consumer loan'
        ]
        
        chunks = []
        try:
            for chunk in pd.read_csv(self.input_path, chunksize=50000, low_memory=False):
                prod_col = 'Product' if 'Product' in chunk.columns else 'product'
                narr_col = 'Consumer complaint narrative' if 'Consumer complaint narrative' in chunk.columns else 'consumer_complaint_narrative'
                
                chunk = chunk.dropna(subset=[narr_col])
                chunk = chunk[chunk[prod_col].isin(target_products)]
                chunks.append(chunk)
        except Exception as e:
            raise RuntimeError(f"Failed parsing CSV stream fields: {str(e)}")
            
        if not chunks:
            raise ValueError("Execution halted: Filtered corpus configurations yielded 0 valid records.")
            
        self.df = pd.concat(chunks, ignore_index=True)
        
        # Normalize structural schema columns
        if 'Product' not in self.df.columns: self.df.rename(columns={'product': 'Product'}, inplace=True)
        if 'Consumer complaint narrative' not in self.df.columns: self.df.rename(columns={'consumer_complaint_narrative': 'Consumer complaint narrative'}, inplace=True)
        if 'Complaint ID' not in self.df.columns: self.df.rename(columns={'complaint_id': 'Complaint ID'}, inplace=True)
            
        print(f"Filtered data loaded: {len(self.df)} rows.")
        return self

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r"i am writing to file a complaint regarding|to whom it may concern", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def clean_narratives(self):
        print("--- Task 1: Cleaning text narratives ---")
        self.df['cleaned_narrative'] = self.df['Consumer complaint narrative'].apply(self._clean_text)
        return self

    def save_data(self):
        print(f"--- Task 1: Saving processed data to {self.output_path} ---")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        print("Task 1 completed successfully! 🎉")
        return self

    def create_stratified_sample(self, sample_size=12000):
        print(f"--- Task 2: Creating a stratified sample of {sample_size} rows ---")
        self.df = self.df.groupby('Product', group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 4), random_state=42)
        )
        print(f"Sample created with {len(self.df)} rows.")
        return self

    def chunk_narratives_task2(self, chunk_size=500, overlap=50):
        print("--- Task 2: Chunking narratives into 500 characters ---")
        chunked_records = []
        
        for _, row in self.df.iterrows():
            text = row['cleaned_narrative']
            if not text or len(text) < 10: continue
                
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                
                chunked_records.append({
                    'complaint_id': str(row.get('Complaint ID', '')),
                    'product_category': row.get('Product', ''),
                    'issue': row.get('Issue', ''),
                    'chunk_text': chunk_text,
                    'chunk_index': chunk_idx
                })
                
                start += (chunk_size - overlap)
                chunk_idx += 1
                
        self.chunked_df = pd.DataFrame(chunked_records)
        print(f"Total chunks created: {len(self.chunked_df)}")
        return self

    def generate_embeddings_and_vector_store(self, vector_store_dir='vector_store'):
        """Generates semantic embeddings and builds a persisted vector lookup store index file"""
        print("--- Task 2: Generating dense vector embeddings for text chunks ---")
        if self.chunked_df is None or self.chunked_df.empty:
            raise ValueError("Pipeline state error: Text chunk records must be populated before seeding vector metrics.")
            
        chunks_list = self.chunked_df['chunk_text'].tolist()
        
        # Compute vectors batch by batch with an animated progress meter
        embeddings = self.model.encode(chunks_list, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        embeddings = np.array(embeddings).astype('float32')
        
        print("--- Task 2: Building and persisting local FAISS index database ---")
        dimension = embeddings.shape[1] # 384 dimensions
        
        # Instantiating a flat L2 distance inner product matching index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        os.makedirs(vector_store_dir, exist_ok=True)
        index_path = os.path.join(vector_store_dir, 'complaints_faiss.index')
        faiss.write_index(index, index_path)
        
        # Persist associated textual chunk metadata separately alongside vector frames
        meta_path = os.path.join(vector_store_dir, 'chunks_metadata.parquet')
        self.chunked_df.to_parquet(meta_path, index=False)
        
        print(f"Vector Database successfully initialized and saved onto disk at: {vector_store_dir}/ 🎉")
        return self
