import pandas as pd
import re
import os

class ComplaintPipeline:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

    def load_and_filter(self):
        print("--- 1. Loading and filtering data in chunks ---")
        target_products = [
            'Credit card or prepaid card',
            'Checking or savings account',
            'Money transfer, virtual currency, or money service',
            'Payday loan, title loan, or personal consumer loan'
        ]
        
        chunks = []
        for chunk in pd.read_csv(self.input_path, chunksize=50000, low_memory=False):
            chunk = chunk.dropna(subset=['Consumer complaint narrative'])
            chunk = chunk[chunk['Product'].isin(target_products)]
            chunks.append(chunk)
            
        self.df = pd.concat(chunks, ignore_index=True)
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
        print("--- 2. Cleaning text narratives ---")
        self.df['cleaned_narrative'] = self.df['Consumer complaint narrative'].apply(self._clean_text)
        return self

    def save_data(self):
        print(f"--- 3. Saving processed data to {self.output_path} ---")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        print("Task 1 Pipeline finished successfully! 🎉")
