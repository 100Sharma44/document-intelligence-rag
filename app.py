"""
Document Intelligence Assistant using RAG
Streamlit Web Application
"""

import os
import streamlit as st
from dotenv import load_dotenv

from rag.pdf_processor import (
    PDFProcessor,
    DocumentProcessingError,
    NoExtractableTextError,
    chunk_documents,
)
from rag.embeddings import GeminiEmbeddingsManager
from rag.vector_store import FAISSVectorStoreManager
from rag.qa_pipeline import DocumentIntelligenceQAPipeline, QAResponse

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom minimal executive CSS styling
st.markdown(
    """
    <style>
    /* Main container styling */
    .main {
        background-color: #f8fafc;
    }
    /* Header card */
    .header-box {
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .header-box h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .header-box p {
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-size: 0.95rem;
        color: #94a3b8;
    }
    /* Metric pill */
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        background-color: #e2e8f0;
        color: #334155;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    /* Citation card */
    .citation-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .citation-header {
        font-weight: 600;
        font-size: 0.88rem;
        color: #1e3a8a;
    }
    .citation-text {
        font-size: 0.83rem;
        color: #475569;
        margin-top: 0.35rem;
        white-space: pre-wrap;
    }
    /* Disclaimer alert */
    .disclaimer-box {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "vector_store_manager" not in st.session_state:
    st.session_state["vector_store_manager"] = None

if "qa_pipeline" not in st.session_state:
    st.session_state["qa_pipeline"] = None

if "indexed_files_summary" not in st.session_state:
    st.session_state["indexed_files_summary"] = []

if "total_pages_indexed" not in st.session_state:
    st.session_state["total_pages_indexed"] = 0

if "total_chunks_indexed" not in st.session_state:
    st.session_state["total_chunks_indexed"] = 0


# ==========================================
# Sidebar: Setup, Upload & Configuration
# ==========================================
with st.sidebar:
    st.subheader("⚙️ Configuration")

    env_api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_api_key if env_api_key else "",
        type="password",
        help="Obtain an API key from https://aistudio.google.com/app/apikey. If GOOGLE_API_KEY is set in .env, it loads automatically.",
    )

    active_api_key = api_key_input.strip() if api_key_input else env_api_key

    if active_api_key:
        st.success("API Key configured", icon="✅")
    else:
        st.warning("Please provide a Gemini API key to proceed.", icon="⚠️")

    st.divider()

    st.subheader("📂 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple readable PDF documents.",
    )

    with st.expander("🛠️ Advanced RAG Parameters"):
        chunk_size = st.slider("Chunk Size (characters)", min_value=300, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=50, max_value=400, value=150, step=25)
        top_k = st.slider("Retrieved Context Chunks (Top-K)", min_value=2, max_value=6, value=4, step=1)

    process_button = st.button("🚀 Process & Index Documents", use_container_width=True, type="primary")

    if process_button:
        if not active_api_key:
            st.error("Cannot process documents: Please enter a Google Gemini API Key first.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file before clicking Process.")
        else:
            with st.spinner("Extracting text and indexing into FAISS vector database..."):
                all_extracted_docs = []
                file_summaries = []
                total_pages = 0
                error_files = []

                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    file_bytes = uploaded_file.read()

                    try:
                        extracted = PDFProcessor.extract_from_bytes(file_bytes=file_bytes, filename=filename)
                        all_extracted_docs.extend(extracted)
                        num_pages = len(extracted)
                        total_pages += num_pages
                        file_summaries.append(f"{filename} ({num_pages} pages)")
                    except NoExtractableTextError as e:
                        error_files.append(f"⚠️ {filename}: Scanned or image-only PDF with no readable text.")
                    except DocumentProcessingError as e:
                        error_files.append(f"❌ {filename}: {str(e)}")
                    except Exception as e:
                        error_files.append(f"❌ {filename}: Unexpected error: {str(e)}")

                for err in error_files:
                    st.error(err)

                if all_extracted_docs:
                    chunks = chunk_documents(
                        all_extracted_docs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    try:
                        # Initialize Embeddings & FAISS
                        embeddings_mgr = GeminiEmbeddingsManager(api_key=active_api_key)
                        vector_mgr = FAISSVectorStoreManager(embeddings=embeddings_mgr.get_embeddings_model())
                        vector_mgr.build_from_documents(chunks)

                        # Initialize QA Pipeline
                        qa_pipe = DocumentIntelligenceQAPipeline(
                            vector_store_manager=vector_mgr,
                            api_key=active_api_key,
                        )

                        # Update Session State
                        st.session_state["vector_store_manager"] = vector_mgr
                        st.session_state["qa_pipeline"] = qa_pipe
                        st.session_state["indexed_files_summary"] = file_summaries
                        st.session_state["total_pages_indexed"] = total_pages
                        st.session_state["total_chunks_indexed"] = len(chunks)

                        st.success(f"Successfully indexed {len(file_summaries)} file(s) ({len(chunks)} chunks)!")
                    except Exception as e:
                        st.error(f"Failed to build vector index: {str(e)}")
                elif not error_files:
                    st.error("No valid documents could be processed.")

    st.divider()

    # Session Status Information
    st.subheader("📊 Session Status")
    if st.session_state["vector_store_manager"] is not None and st.session_state["vector_store_manager"].is_ready:
        st.write(f"**Indexed Documents:** {len(st.session_state['indexed_files_summary'])}")
        st.write(f"**Total Pages:** {st.session_state['total_pages_indexed']}")
        st.write(f"**Indexed Chunks:** {st.session_state['total_chunks_indexed']}")
        with st.expander("📄 Document List"):
            for summary in st.session_state["indexed_files_summary"]:
                st.write(f"• {summary}")
    else:
        st.info("No active vector store index. Upload and process documents to begin.")

    if st.button("🗑️ Reset Session & Index", use_container_width=True):
        st.session_state["chat_history"] = []
        st.session_state["vector_store_manager"] = None
        st.session_state["qa_pipeline"] = None
        st.session_state["indexed_files_summary"] = []
        st.session_state["total_pages_indexed"] = 0
        st.session_state["total_chunks_indexed"] = 0
        st.rerun()


# ==========================================
# Main Content Area
# ==========================================
st.markdown(
    """
    <div class="header-box">
        <h1>Document Intelligence Assistant</h1>
        <p>Grounded enterprise document question answering powered by Retrieval-Augmented Generation (RAG), FAISS, and Google Gemini.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Onboarding guide if no index is active
if st.session_state["vector_store_manager"] is None or not st.session_state["vector_store_manager"].is_ready:
    st.info(
        "👋 **Welcome! To get started:**\n\n"
        "1. Enter your **Google Gemini API Key** in the sidebar (or place it in `.env`).\n"
        "2. Upload one or more **PDF documents** in the sidebar.\n"
        "3. Click **'Process & Index Documents'** to extract text and compute vector embeddings.\n"
        "4. Ask questions below to retrieve relevant sections and generate answers with verified citations."
    )
else:
    # Display indexed summary banner
    summary_chips = " | ".join(st.session_state["indexed_files_summary"])
    st.markdown(
        f"**Active Index:** {summary_chips} "
        f"(<span class='metric-badge'>{st.session_state['total_chunks_indexed']} chunks</span>)",
        unsafe_allow_html=True,
    )

st.divider()

# Conversational Message Display
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display source citations if available
        if message.get("citations"):
            citations = message["citations"]
            with st.expander(f"🔍 Verified Document Citations ({len(citations)} source excerpt{'s' if len(citations) > 1 else ''})"):
                for idx, cit in enumerate(citations, start=1):
                    source_name = cit.get("source", "Unknown")
                    page_num = cit.get("page", "Unknown")
                    excerpt_text = cit.get("excerpt", "")

                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">Source #{idx}: 📄 {source_name} &nbsp;•&nbsp; Page {page_num}</div>
                            <div class="citation-text">{excerpt_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# User Chat Input
user_query = st.chat_input("Ask a question about your uploaded documents...")

if user_query:
    if not active_api_key:
        st.error("Please enter a valid Google Gemini API Key in the sidebar.")
    elif st.session_state["qa_pipeline"] is None or not st.session_state["vector_store_manager"].is_ready:
        st.warning("Please upload and index PDF documents before asking questions.")
    else:
        # Display user question
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response using RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching document index and synthesizing answer..."):
                try:
                    qa_response: QAResponse = st.session_state["qa_pipeline"].ask(
                        question=user_query,
                        top_k=top_k,
                    )
                    answer_text = qa_response.answer
                    citations_data = [c.to_dict() for c in qa_response.citations]

                    st.markdown(answer_text)

                    if citations_data:
                        with st.expander(f"🔍 Verified Document Citations ({len(citations_data)} source excerpt{'s' if len(citations_data) > 1 else ''})"):
                            for idx, cit in enumerate(citations_data, start=1):
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <div class="citation-header">Source #{idx}: 📄 {cit['source']} &nbsp;•&nbsp; Page {cit['page']}</div>
                                        <div class="citation-text">{cit['excerpt']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Record in session chat history
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations_data,
                    })
                except Exception as e:
                    error_msg = f"Error generating answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": error_msg,
                        "citations": [],
                    })

# Academic & Technical Disclaimer
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>Note on Factual Grounding:</strong> Retrieval-Augmented Generation (RAG) significantly constrains model output to retrieved excerpts and includes metadata-verified citations. However, RAG reduces but does not theoretically eliminate all hallucination risk. Users should inspect the expanded source excerpts for critical decision-making.
    </div>
    """,
    unsafe_allow_html=True,
)
