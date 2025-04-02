"""
rag_chain.py

Implements a retrieval-augmented generation (RAG) pipeline using LangChain.
"""

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class RAGChain:
    def __init__(self, retriever):
        """
        Initialize the RAG chain with a retriever (e.g., from ChromaDB).
        """
        self.retriever = retriever
        # Note: This uses OpenAI as an example. You can replace it with another LLM.
        self.llm = OpenAI(temperature=0.7)

    def run(self, query: str) -> str:
        """
        Executes a retrieval-augmented generation chain for the given query.
        """
        # Create a RetrievalQA chain, which fetches relevant docs and then generates an answer.
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # or "map_reduce", "refine", etc.
            retriever=self.retriever
        )
        return chain.run(query)