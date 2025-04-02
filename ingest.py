# src/ingest.py
import os
from sentence_transformers import SentenceTransformer

print("Starting document ingestion script...")
print("Current working directory:", os.getcwd())

def load_document(file_path):
    """
    Reads the file from the given path and returns cleaned text.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    # Basic preprocessing: remove extra whitespace and newlines
    cleaned_text = ' '.join(text.split())
    return cleaned_text

def generate_embedding(text):
    """
    Generates an embedding vector for the given text using a pre-trained model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding

if __name__ == "__main__":
    # Adjust the path based on your current working directory:
    sample_file_path = './data/sample.txt'  # or '../data/sample.txt' depending on where you run it
    print("Using sample file path:", sample_file_path)
    
    document_text = load_document(sample_file_path)
    if document_text:
        print("Loaded and Cleaned Document:")
        print(document_text)
        
        embedding = generate_embedding(document_text)
        print("Generated Embedding Vector:")
        print(embedding)
    else:
        print("Failed to load document.")