from src.data_preprocessing import ComplaintPipeline

if __name__ == "__main__":
    pipeline = ComplaintPipeline(
        input_path='data/raw/complaints.csv', 
        output_path='data/processed/filtered_complaints.csv'
    )
    pipeline.load_and_filter().clean_narratives().save_data()
