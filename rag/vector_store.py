"""
FAISS Vector Store Manager for indexing and similarity search.
"""

from typing import List, Optional, Tuple
import os
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class FAISSVectorStoreManager:
    """Manages creation, serialization, and retrieval for FAISS vector stores."""

    def __init__(self, embeddings: Embeddings):
        """
        Initialize the FAISS manager with an embedding model.

        Args:
            embeddings: LangChain-compatible embedding model.
        """
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None

    def build_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Build an in-memory FAISS index from document chunks.

        Args:
            documents: List of chunked Document objects.

        Returns:
            The populated FAISS vector store instance.

        Raises:
            ValueError: If the documents list is empty.
        """
        if not documents:
            raise ValueError("Cannot build a vector store from an empty document list.")

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        return self.vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Retrieve the top-k most relevant document chunks for a query.

        Args:
            query: User's question or search query.
            k: Number of relevant chunks to retrieve (default: 4, within 3-5 range).

        Returns:
            List of retrieved Document objects with content and metadata.
        """
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Build or load an index first.")

        if not query or not query.strip():
            return []

        return self.vector_store.similarity_search(query.strip(), k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve the top-k most relevant document chunks along with similarity scores.

        Args:
            query: User question.
            k: Top-k chunks.

        Returns:
            List of (Document, score) tuples.
        """
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Build or load an index first.")

        if not query or not query.strip():
            return []

        return self.vector_store.similarity_search_with_score(query.strip(), k=k)

    def save_local(self, directory_path: str) -> None:
        """Persist the FAISS index to disk."""
        if not self.vector_store:
            raise ValueError("No active vector store to save.")
        self.vector_store.save_local(directory_path)

    def load_local(self, directory_path: str) -> FAISS:
        """Load a persisted FAISS index from disk."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"FAISS index directory not found: '{directory_path}'")

        self.vector_store = FAISS.load_local(
            directory_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self.vector_store

    @property
    def is_ready(self) -> bool:
        """Check if vector store is initialized and ready for queries."""
        return self.vector_store is not None
