"""
PDF processing and text chunking module using PyMuPDF and LangChain.
"""

from typing import List, Union, Optional
try:
    import pymupdf
except ImportError:
    import fitz as pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessingError(Exception):
    """Raised when document extraction or validation fails."""
    pass


class NoExtractableTextError(DocumentProcessingError):
    """Raised when a PDF contains no extractable text (e.g. blank or scanned images)."""
    pass


class PDFProcessor:
    """Extracts text and page-level metadata from PDF documents using PyMuPDF."""

    @staticmethod
    def extract_from_bytes(
        file_bytes: bytes,
        filename: str = "uploaded_document.pdf"
    ) -> List[Document]:
        """
        Extract text page-by-page from raw PDF bytes without writing to disk.

        Args:
            file_bytes: Raw bytes of the PDF.
            filename: Original filename to store in metadata.

        Returns:
            List of LangChain Document objects with page content and metadata.

        Raises:
            DocumentProcessingError: If the file is invalid, encrypted, or corrupted.
            NoExtractableTextError: If no extractable text is found across all pages.
        """
        if not file_bytes:
            raise DocumentProcessingError(f"The file '{filename}' is empty (0 bytes).")

        # Basic PDF magic header check
        if not file_bytes.startswith(b"%PDF"):
            raise DocumentProcessingError(f"The file '{filename}' does not appear to be a valid PDF document.")

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise DocumentProcessingError(f"Failed to open '{filename}': {str(e)}") from e

        return PDFProcessor._process_fitz_doc(doc, filename)

    @staticmethod
    def extract_from_path(file_path: str) -> List[Document]:
        """
        Extract text page-by-page from a file path.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of LangChain Document objects.
        """
        import os
        filename = os.path.basename(file_path)
        if not os.path.exists(file_path):
            raise DocumentProcessingError(f"File not found: '{file_path}'")

        try:
            doc = pymupdf.open(file_path)
        except Exception as e:
            raise DocumentProcessingError(f"Failed to open '{file_path}': {str(e)}") from e

        return PDFProcessor._process_fitz_doc(doc, filename)

    @staticmethod
    def _process_fitz_doc(doc: pymupdf.Document, filename: str) -> List[Document]:
        """Internal helper to extract pages and metadata from a PyMuPDF Document."""
        try:
            if doc.is_encrypted:
                raise DocumentProcessingError(
                    f"The document '{filename}' is password-protected or encrypted. Please provide a decrypted PDF."
                )

            total_pages = len(doc)
            if total_pages == 0:
                raise DocumentProcessingError(f"The document '{filename}' contains no pages.")

            documents: List[Document] = []
            total_extracted_chars = 0

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_text = page.get_text("text").strip()
                page_number = page_idx + 1  # 1-indexed for human readability

                if page_text:
                    total_extracted_chars += len(page_text)
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": filename,
                                "page": page_number,
                                "total_pages": total_pages,
                            },
                        )
                    )

            if total_extracted_chars == 0 or not documents:
                raise NoExtractableTextError(
                    f"The document '{filename}' contains {total_pages} page(s), but no extractable text was found. "
                    "It may be a scanned document containing only images or an empty file."
                )

            return documents
        finally:
            doc.close()


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Split extracted documents into overlapping chunks while strictly preserving metadata.

    Args:
        documents: List of page-level Documents.
        chunk_size: Target character count per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunked Document objects with updated chunk metadata.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunked_docs: List[Document] = []
    chunk_counter = 0

    for doc in documents:
        splits = splitter.split_text(doc.page_content)
        for idx, text in enumerate(splits):
            meta = dict(doc.metadata)
            meta["chunk_id"] = f"{meta.get('source', 'doc')}_p{meta.get('page', 1)}_c{idx + 1}"
            meta["chunk_index"] = chunk_counter
            chunk_counter += 1
            chunked_docs.append(Document(page_content=text, metadata=meta))

    return chunked_docs
