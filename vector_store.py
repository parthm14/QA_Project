import chromadb
from chromadb.config import Settings

class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_db"):
        # Initialize the ChromaDB client
        self.client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
        )
        # Create or get a collection to store document embeddings
        self.collection = self.client.get_or_create_collection(name="documents")

    def index_document(self, doc_id: str, embedding, document_text: str):
        """
        Adds a document and its embedding to the vector store.
        """
        self.collection.add(
            documents=[document_text],
            embeddings=[embedding],
            ids=[doc_id]
        )
        print(f"Document '{doc_id}' indexed successfully.")

    def query_documents(self, query_embedding, n_results=1):
        """
        Retrieves the most similar document(s) based on the query embedding.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results