import pandas as pd
import re
import os

class ComplaintPipeline:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.chunked_df = None

    def load_and_filter(self):
        print("--- Task 1: Loading and filtering data in chunks ---")
        target_products = [
            'Credit card or prepaid card',
            'Checking or savings account',
            'Money transfer, virtual currency, or money service',
            'Payday loan, title loan, or personal consumer loan'
        ]
        
        chunks = []
        for chunk in pd.read_csv(self.input_path, chunksize=50000, low_memory=False):
            # Dynamic casing check for columns
            prod_col = 'Product' if 'Product' in chunk.columns else 'product'
            narr_col = 'Consumer complaint narrative' if 'Consumer complaint narrative' in chunk.columns else 'consumer_complaint_narrative'
            
            chunk = chunk.dropna(subset=[narr_col])
            chunk = chunk[chunk[prod_col].isin(target_products)]
            chunks.append(chunk)
            
        self.df = pd.concat(chunks, ignore_index=True)
        # Standardize column naming convention for downstream steps
        if 'Product' not in self.df.columns and 'product' in self.df.columns:
            self.df = self.df.rename(columns={'product': 'Product'})
        if 'Consumer complaint narrative' not in self.df.columns and 'consumer_complaint_narrative' in self.df.columns:
            self.df = self.df.rename(columns={'consumer_complaint_narrative': 'Consumer complaint narrative'})
            
        print(f"Filtered data loaded: {len(self.df)} rows.")
        return self

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
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
        prod_col = 'Product' if 'Product' in self.df.columns else 'product'
        
        self.df = self.df.groupby(prod_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 4), random_state=42)
        )
        print(f"Sample created with {len(self.df)} rows.")
        return self

    def chunk_narratives_task2(self, chunk_size=500, overlap=50):
        print("--- Task 2: Chunking narratives into 500 characters ---")
        chunked_records = []
        
        prod_col = 'Product' if 'Product' in self.df.columns else 'product'
        id_col = 'Complaint ID' if 'Complaint ID' in self.df.columns else ('complaint_id' if 'complaint_id' in self.df.columns else 'id')
        issue_col = 'Issue' if 'Issue' in self.df.columns else 'issue'
        
        for _, row in self.df.iterrows():
            text = row['cleaned_narrative']
            if not text:
                continue
                
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                
                chunked_records.append({
                    'complaint_id': row.get(id_col, _),
                    'product_category': row.get(prod_col, ''),
                    'issue': row.get(issue_col, ''),
                    'chunk_text': chunk_text,
                    'chunk_index': chunk_idx
                })
                
                start += (chunk_size - overlap)
                chunk_idx += 1
                
        self.chunked_df = pd.DataFrame(chunked_records)
        print(f"Total chunks created: {len(self.chunked_df)}")
        return self

    def save_chunks(self, chunks_output_path='data/processed/sampled_chunks.csv'):
        print(f"--- Task 2: Saving chunks to {chunks_output_path} ---")
        os.makedirs(os.path.dirname(chunks_output_path), exist_ok=True)
        self.chunked_df.to_csv(chunks_output_path, index=False)
        print("Task 2 completed successfully! 🎉")
        return self
