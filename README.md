# Intelligent Complaint Analysis for Financial Services

## RAG-Powered Data Engineering Pipeline for CrediTrust Financial

### Project Overview

This project implements a production-ready data engineering pipeline designed to support a Retrieval-Augmented Generation (RAG) chatbot for CrediTrust Financial. The system processes consumer complaint narratives from the Consumer Financial Protection Bureau (CFPB) dataset, transforming raw unstructured text into clean, searchable chunks optimized for vector embeddings and semantic retrieval.

The pipeline emphasizes scalability, maintainability, and data quality by incorporating efficient data loading, text normalization, stratified sampling, and contextual chunking techniques commonly used in modern AI and data engineering workflows.

---

## Key Features

### Task 1: Data Ingestion and Preprocessing

#### High-Performance Data Loading

* Processes large CFPB complaint datasets using chunk-based loading (`chunksize=50000`).
* Minimizes memory consumption while maintaining processing efficiency.
* Suitable for multi-gigabyte datasets.

#### Domain-Specific Filtering

The pipeline focuses exclusively on CrediTrust's core financial product categories:

* Credit card or prepaid card
* Checking or savings account
* Money transfer, virtual currency, or money service
* Payday loan, title loan, or personal consumer loan

Additionally, records with missing complaint narratives are automatically removed to ensure data quality.

#### Text Normalization

Consumer complaint narratives undergo extensive cleaning:

* Conversion to lowercase
* Removal of common introductory boilerplate phrases
* Elimination of special characters and noise
* Standardization of whitespace
* Preparation for downstream embedding generation

---

### Task 2: Stratified Sampling and Text Chunking

#### Balanced Stratified Sampling

To ensure fair representation across financial products, the pipeline creates a balanced sample of approximately 12,000 complaints.

Benefits include:

* Reduced dataset size for experimentation
* Better category representation
* Improved retrieval quality in the RAG system

#### Context-Preserving Chunking

Long complaint narratives are divided into overlapping chunks suitable for embedding models.

Configuration:

* Chunk Size: 500 characters
* Chunk Overlap: 50 characters

This sliding-window approach helps preserve contextual information between adjacent chunks and improves retrieval accuracy.

#### Metadata Enrichment

Each generated chunk retains important business metadata:

* Complaint ID
* Product Category
* Issue Type
* Chunk Text
* Chunk Index

This structure enables traceability and source attribution during retrieval.

---

## Project Structure

```text
rag-complaint-chatbot/
├── data/
│   ├── raw/
│   │   └── complaints.csv
│   │
│   └── processed/
│       ├── filtered_complaints.csv
│       ├── sampled_chunks.csv
│       └── metrics.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   └── run_pipeline.py
│
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py
│
├── requirements.txt
└── README.md
```

---

## Engineering Design Principles

### Object-Oriented Architecture

The pipeline is built using Object-Oriented Programming (OOP) principles, providing:

* Clear separation of concerns
* Improved maintainability
* Easier testing and extension

### Method Chaining

Pipeline stages are executed through a fluent interface:

```python
pipeline = (
    ComplaintPipeline()
    .load_and_filter()
    .clean_text()
    .create_sample()
    .chunk_narratives()
    .save_outputs()
)
```

This design creates readable and scalable workflows.



## Data Quality Enhancements

The pipeline incorporates several production-grade validation measures:

### Duplicate Detection

```python
df.drop_duplicates(
    subset=["Consumer complaint narrative"]
)
```

Prevents duplicate complaints from impacting retrieval performance.

### Data Quality Metrics

The system generates processing statistics including:

* Total records processed
* Records removed
* Missing values detected
* Duplicate records removed
* Average narrative length
* Total chunks generated

Results are stored in:

```text
data/processed/metrics.json
```


## Running the Pipeline

Execute the complete workflow from the project root:

```bash
python -m src.run_pipeline
```

### Example Output

```text
--- Loading and filtering CFPB data ---
Filtered data loaded: 346,174 rows

--- Cleaning complaint narratives ---
Text normalization completed

--- Creating stratified sample ---
Sample generated: 12,000 complaints

--- Chunking narratives ---
Total chunks created: 26,194

--- Saving outputs ---
Pipeline completed successfully
```



## Technology Stack

### Core Libraries

* Pandas – Data manipulation and processing
* NumPy – Numerical computation
* PyArrow – High-performance data handling

### Testing

* Pytest – Unit testing framework

### Future RAG Integration

The processed chunk dataset is designed for direct integration with:

* OpenAI Embeddings
* Sentence Transformers
* FAISS
* ChromaDB
* Pinecone
* Weaviate



## Future Improvements

Planned enhancements include:

* Token-based chunking using tokenizer-aware splitting
* Embedding generation pipeline
* Vector database integration
* Retrieval evaluation metrics
* Airflow orchestration
* Docker containerization
* CI/CD automation
* Monitoring and logging infrastructure



## Outcome

The final output of this pipeline is a clean, balanced, and retrieval-ready corpus of financial complaint narratives. These structured chunks serve as the foundation for a production-grade RAG chatbot capable of answering customer-service and financial-domain questions using historical complaint data.
