import pytest
from src.data_preprocessing import ComplaintPipeline

def test_text_cleaning_logic():
    # Instantiate a clean mock pipeline structure
    pipeline = ComplaintPipeline(input_path="fake.csv", output_path="fake_out.csv")
    
    raw_text = "Dear Sirs, I am writing to file a complaint regarding my account!! BLOCK THIS CODE."
    expected_clean_output = "my account block this code"
    
    # Assert that the static text regularizer correctly isolates semantic target text strings
    assert pipeline._clean_text(raw_text) == expected_clean_output
