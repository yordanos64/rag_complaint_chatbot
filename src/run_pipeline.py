from src.data_preprocessing import ComplaintPipeline

if __name__ == "__main__":
    pipeline = ComplaintPipeline(
        input_path='data/raw/complaints.csv', 
        output_path='data/processed/filtered_complaints.csv'
    )
    
    # End-to-end unified production run processing sequence
    pipeline.load_and_filter() \
            .clean_narratives() \
            .save_data() \
            .create_stratified_sample(sample_size=12000) \
            .chunk_narratives_task2(chunk_size=500, overlap=50) \
            .generate_embeddings_and_vector_store(vector_store_dir='vector_store')
