"""
End-to-end smoke test using live Google Gemini API if GOOGLE_API_KEY is available.
Never logs or prints secret keys.
"""

import os
import sys
from dotenv import load_dotenv

# Ensure rag package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()


def run_smoke_test() -> bool:
    """Executes live smoke test if API key is configured."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[INFO] GOOGLE_API_KEY is not set. Skipping live external API test.")
        print("[INFO] Run `pytest` to execute all 16 automated mock & integration unit tests.")
        return True

    print("[INFO] GOOGLE_API_KEY detected. Running live end-to-end smoke test...")

    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    from rag.pdf_processor import PDFProcessor, chunk_documents
    from rag.embeddings import GeminiEmbeddingsManager
    from rag.vector_store import FAISSVectorStoreManager
    from rag.qa_pipeline import DocumentIntelligenceQAPipeline

    # 1. Create a non-sensitive sample PDF
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((50, 72), "Project Apollo Overview:\nApollo provides distributed data analytics and cloud storage.")
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Security Specifications:\nAll data in transit is encrypted using TLS 1.3 protocol.")
    pdf_bytes = doc.tobytes()
    doc.close()

    # 2. Extract & chunk
    documents = PDFProcessor.extract_from_bytes(pdf_bytes, filename="project_apollo_specs.pdf")
    chunks = chunk_documents(documents, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 2, "Expected at least 2 chunks"

    # 3. Embed & Vector Store
    embeddings_mgr = GeminiEmbeddingsManager(api_key=api_key)
    vector_mgr = FAISSVectorStoreManager(embeddings=embeddings_mgr.get_embeddings_model())
    vector_mgr.build_from_documents(chunks)
    assert vector_mgr.is_ready, "Vector store should be ready"

    # 4. QA Pipeline invocation
    qa_pipe = DocumentIntelligenceQAPipeline(vector_store_manager=vector_mgr, api_key=api_key)
    response = qa_pipe.ask("What protocol is used for data encryption in transit?", top_k=2)

    assert response.answer, "Response answer should not be empty"
    assert len(response.citations) > 0, "Response should have at least one citation"
    assert response.citations[0].source == "project_apollo_specs.pdf"

    print("[SUCCESS] Live Gemini smoke test completed successfully!")
    print(f"Sample Question: What protocol is used for data encryption in transit?")
    print(f"Verified Answer: {response.answer}")
    print(f"Verified Citation: {response.citations[0].source} (Page {response.citations[0].page})")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
