"""
Tests for rag.pdf_processor: PDF text extraction, validation, and chunking.
"""

import pytest
try:
    import pymupdf
except ImportError:
    import fitz as pymupdf
from langchain_core.documents import Document

from rag.pdf_processor import (
    PDFProcessor,
    DocumentProcessingError,
    NoExtractableTextError,
    chunk_documents,
)


def _create_mock_pdf_bytes(pages: list[str]) -> bytes:
    """Helper to create an in-memory PDF with specified text per page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestPDFProcessor:
    """Test suite for PDF extraction and edge cases."""

    def test_extract_valid_multi_page_pdf(self):
        """Verify extraction preserves text, page numbers, and total page metadata."""
        pages_content = [
            "This is page one discussing cloud architecture.",
            "This is page two covering FAISS vector indexing.",
        ]
        pdf_bytes = _create_mock_pdf_bytes(pages_content)

        docs = PDFProcessor.extract_from_bytes(pdf_bytes, filename="sample_report.pdf")

        assert len(docs) == 2
        assert "cloud architecture" in docs[0].page_content
        assert docs[0].metadata["source"] == "sample_report.pdf"
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["total_pages"] == 2

        assert "FAISS vector indexing" in docs[1].page_content
        assert docs[1].metadata["source"] == "sample_report.pdf"
        assert docs[1].metadata["page"] == 2
        assert docs[1].metadata["total_pages"] == 2

    def test_blank_pdf_raises_no_extractable_text_error(self):
        """Verify that a PDF with pages but zero readable text raises NoExtractableTextError."""
        # Create a PDF with 2 empty pages
        pdf_bytes = _create_mock_pdf_bytes(["", ""])

        with pytest.raises(NoExtractableTextError) as exc_info:
            PDFProcessor.extract_from_bytes(pdf_bytes, filename="blank_scanned.pdf")

        assert "no extractable text" in str(exc_info.value).lower()
        assert "blank_scanned.pdf" in str(exc_info.value)

    def test_empty_byte_stream_raises_processing_error(self):
        """Verify that 0-byte input raises DocumentProcessingError."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            PDFProcessor.extract_from_bytes(b"", filename="zero_bytes.pdf")

        assert "empty" in str(exc_info.value).lower()

    def test_invalid_header_raises_processing_error(self):
        """Verify that non-PDF content raises DocumentProcessingError."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            PDFProcessor.extract_from_bytes(b"NOT_A_PDF_CONTENT", filename="corrupt.pdf")

        assert "valid pdf" in str(exc_info.value).lower()

    def test_nonexistent_file_path(self):
        """Verify that passing an invalid file path raises DocumentProcessingError."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            PDFProcessor.extract_from_path("/tmp/nonexistent_document_12345.pdf")

        assert "file not found" in str(exc_info.value).lower()


class TestChunking:
    """Test suite for metadata-preserving text chunking."""

    def test_chunk_documents_preserves_metadata(self):
        """Verify that chunks retain original page and source metadata, plus chunk IDs."""
        sample_doc = Document(
            page_content="Sentence one. " * 80,  # ~1120 chars
            metadata={"source": "annual_report.pdf", "page": 3, "total_pages": 10},
        )

        chunks = chunk_documents([sample_doc], chunk_size=300, chunk_overlap=50)

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == "annual_report.pdf"
            assert chunk.metadata["page"] == 3
            assert chunk.metadata["total_pages"] == 10
            assert "chunk_id" in chunk.metadata
            assert chunk.metadata["chunk_index"] == i

    def test_chunk_empty_list(self):
        """Verify empty input returns an empty list without error."""
        chunks = chunk_documents([])
        assert chunks == []
