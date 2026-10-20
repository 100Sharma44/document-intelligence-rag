"""
Embeddings manager utilizing Google Gemini embedding models.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load variables from .env if present
load_dotenv()


class GeminiEmbeddingsManager:
    """Manages creation and configuration of Google Gemini vector embeddings."""

    DEFAULT_MODEL = "models/gemini-embedding-001"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the Gemini embeddings client.

        Args:
            api_key: Optional Gemini API key. If not provided, looks up GOOGLE_API_KEY from environment.
            model_name: Gemini embedding model name (defaults to 'models/gemini-embedding-001').
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or self.DEFAULT_MODEL

        if not self.api_key:
            raise ValueError(
                "Google API Key not found. Please set GOOGLE_API_KEY in your .env file, "
                "set the environment variable, or enter it in the application sidebar."
            )

        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.api_key,
        )

    def get_embeddings_model(self) -> GoogleGenerativeAIEmbeddings:
        """Return the initialized LangChain embedding object."""
        return self._embeddings
