from ingest import load_document, generate_embedding
from vector_store import VectorStoreManager
from web_search import search_web
from rag_chain import RAGChain
from utils import display_answer

def main():
    # Step 1: Ingest and embed a sample document
    sample_file_path = './data/sample.txt'
    document_text = load_document(sample_file_path)
    if not document_text:
        print("Document could not be loaded. Exiting...")
        return

    embedding = generate_embedding(document_text)
    print("Generated embedding for the sample document.")

    # Step 2: Set up the vector store and index the document
    vector_store = VectorStoreManager(persist_directory="./chroma_db")
    doc_id = "sample_doc"
    vector_store.index_document(doc_id, embedding, document_text)
    print("Indexed the sample document in ChromaDB.")

    # Step 3: Simulate a user query
    user_query = "What is this document about?"
    print(f"\nUser Query: {user_query}")

    # (Optional) Step 3a: Web search integration
    web_results = search_web(user_query)
    print("\nWeb Search Results (Dummy):")
    for res in web_results:
        print(f"- {res['title']}: {res['snippet']} (URL: {res['url']})")

    # Step 4: Retrieve relevant docs from vector store (RAG approach)
    # For RAGChain, we need a 'retriever' object that matches LangChain's interface.
    # We can create a simple wrapper around vector_store.query_documents.

    class ChromaDBRetriever:
        """A simple retriever interface for LangChain to use ChromaDB results."""
        def __init__(self, vs_manager):
            self.vs_manager = vs_manager

        def get_relevant_documents(self, query: str):
            # For demonstration, let's embed the query using the same embedding function
            query_emb = generate_embedding(query)
            results = self.vs_manager.query_documents(query_emb, n_results=2)
            
            # Convert results into the format LangChain expects: a list of Document objects
            # But we can simulate that with a simple namedtuple or a dictionary.
            from langchain.docstore.document import Document

            docs = []
            for doc_text, doc_id in zip(results["documents"][0], results["ids"][0]):
                docs.append(Document(page_content=doc_text, metadata={"id": doc_id}))
            return docs

    # Create a retriever from our vector store
    retriever = ChromaDBRetriever(vector_store)
    rag_chain = RAGChain(retriever=retriever)

    # Step 5: Run the RAG chain to get an answer
    final_answer = rag_chain.run(user_query)

    # Step 6: Display the answer
    display_answer(final_answer)

if __name__ == "__main__":
    main()