import streamlit as st
import time
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from rag_engine import RAGPipeline
from rag_engine.sample_data import SAMPLE_DATASETS

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ultimate RAG Engine | Data to Deployment",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING (Rich Aesthetics & Dark Mode Accent)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Gradient Banner */
    .main-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        border: 1px solid #4f46e5;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(67, 56, 202, 0.25);
    }
    
    .main-banner h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-banner p {
        color: #c7d2fe;
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 0;
    }
    
    /* Custom Metric Card */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Citation Pill */
    .citation-pill {
        display: inline-block;
        background: #1e293b;
        color: #cbd5e1;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.82rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Step Log Pill */
    .step-badge {
        display: inline-block;
        background-color: #312e81;
        color: #818cf8;
        font-weight: 600;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
    }
    
    /* Pipeline Step Box */
    .step-box {
        background: #0f172a;
        border-left: 4px solid #6366f1;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    /* Lost in the Middle visualizer */
    .reorder-container {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }
    .reorder-item {
        flex: 1;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .head-tail {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #34d399;
    }
    .middle-item {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR - API KEYS & HYPERPARAMETERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/bullseye.png", width=64)
    st.title("RAG Engine Config")
    st.markdown("---")
    
    # Section 1: API Key Configuration
    st.subheader("🔑 API Key Settings")
    st.caption("Active keys loaded from .env environment.")
    
    gemini_default = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    cohere_default = os.environ.get("COHERE_API_KEY", "")
    openai_default = os.environ.get("OPENAI_API_KEY", "")

    gemini_key_input = st.text_input("Google Gemini API Key", value=gemini_default, type="password", placeholder="AIzaSy...", key="gemini_key")
    cohere_key_input = st.text_input("Cohere Rerank API Key", value=cohere_default, type="password", placeholder="isHZ3...", key="cohere_key")
    openai_key_input = st.text_input("OpenAI API Key (Optional)", value=openai_default, type="password", placeholder="sk-...", key="openai_key")
    
    # Active status indicators
    gem_status = "🟢 Active (Gemini-1.5/2.0)" if gemini_key_input else "⚪ Offline (Local Fallback)"
    coh_status = "🟢 Active (Cohere-Rerank-v3.5)" if cohere_key_input else "⚪ Local Custom Reranker"
    op_status = "🟢 Active" if openai_key_input else "⚪ Optional"
    
    st.markdown(f"""
    <div style="background:#1e293b; padding:10px; border-radius:8px; font-size:0.8rem; margin-bottom:15px;">
        <strong>API Status:</strong><br/>
        • Gemini LLM & Embeddings: <code>{gem_status}</code><br/>
        • Cohere Reranker: <code>{coh_status}</code><br/>
        • Vector Engine: <code>ChromaDB / FAISS</code>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("⚙️ Pipeline Hyperparameters")
    
    chunk_size = st.slider("Chunk Size (Chars)", min_value=100, max_value=2000, value=500, step=50)
    chunk_overlap = st.slider("Chunk Overlap (Chars)", min_value=0, max_value=500, value=100, step=10)
    retrieval_mode = st.selectbox("Stage 10 Retrieval Mode", ["hybrid", "dense", "sparse"])
    
    st.markdown("---")
    st.subheader("🧩 Contextual Compression")
    enable_compression = st.toggle("Enable Compression (Stage 12)", value=True)
    reranker_model = st.selectbox("Stage 12.3 Reranker Engine", ["Cohere Rerank API", "CustomCrossEncoder (Local)", "FlashRank", "BGE Reranker"])
    top_k_first = st.slider("Top-K First-Stage Candidates", min_value=5, max_value=30, value=15)
    top_n_final = st.slider("Top-N Final Context Chunks", min_value=1, max_value=10, value=4)

# -----------------------------------------------------------------------------
# INITIALIZE OR UPDATE PIPELINE SESSION STATE
# -----------------------------------------------------------------------------
if "pipeline" not in st.session_state or st.sidebar.button("🔄 Apply Config & Reload Pipeline"):
    with st.spinner("Initializing RAG Engine Pipeline..."):
        st.session_state.pipeline = RAGPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_mode=retrieval_mode,
            enable_compression=enable_compression,
            reranker_model=reranker_model,
            top_k_first_stage=top_k_first,
            top_n_final=top_n_final,
            gemini_api_key=gemini_key_input if gemini_key_input else None,
            cohere_api_key=cohere_key_input if cohere_key_input else None,
            openai_api_key=openai_key_input if openai_key_input else None
        )
        st.success("Pipeline updated successfully!")



pipeline: RAGPipeline = st.session_state.pipeline

# -----------------------------------------------------------------------------
# HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-banner">
    <h1>🎯 Ultimate RAG Engine Dashboard</h1>
    <p>Complete Implementation of the <strong>16-Stage Ultimate RAG Roadmap</strong> (Data Collection to Evaluation & Feedback Loop)</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN TABBED INTERFACE
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Interactive Query Engine",
    "📥 Ingestion & Vector DB",
    "🧩 Contextual Compression Visualizer",
    "📊 Evaluation Metrics",
    "🗺️ Roadmap Step Inspector"
])

# =============================================================================
# TAB 1: INTERACTIVE QUERY ENGINE & PDF RAG
# =============================================================================
with tab1:
    # -------------------------------------------------------------------------
    # PROMINENT PDF FILE UPLOAD SECTION
    # -------------------------------------------------------------------------
    st.markdown("### 📄 Upload PDF Document (PDF RAG Mode)")
    pdf_col1, pdf_col2 = st.columns([3, 1])
    
    with pdf_col1:
        uploaded_pdf = st.file_uploader(
            "Upload your PDF document to query:",
            type=["pdf", "txt", "md"],
            key="pdf_rag_uploader",
            help="Upload any PDF file. The RAG pipeline will extract pages, generate embeddings, and answer queries strictly from this file."
        )
    
    with pdf_col2:
        st.markdown("<br/>", unsafe_allow_html=True)
        process_pdf_btn = st.button("📥 Index & Ingest PDF", type="secondary", use_container_width=True)

    if (process_pdf_btn or uploaded_pdf) and uploaded_pdf is not None:
        file_key = f"processed_{uploaded_pdf.name}_{uploaded_pdf.size}"
        if st.session_state.get("last_processed_pdf") != file_key:
            with st.spinner(f"Extracting pages & indexing PDF '{uploaded_pdf.name}' into Vector DB..."):
                pdf_bytes = uploaded_pdf.read()
                if uploaded_pdf.name.lower().endswith(".pdf"):
                    logs = pipeline.ingest_pdf_file(pdf_bytes, filename=uploaded_pdf.name)
                else:
                    text_str = pdf_bytes.decode("utf-8", errors="ignore")
                    logs = pipeline.ingest_documents([{"text": text_str, "metadata": {"source": uploaded_pdf.name}}])
                
                st.session_state["last_processed_pdf"] = file_key
                st.session_state["pdf_name"] = uploaded_pdf.name
                st.success(f"✅ Ingested PDF '{uploaded_pdf.name}'! Total **{len(pipeline.indexed_chunks)}** vector chunks indexed.")

    if "pdf_name" in st.session_state:
        st.info(f"📌 Active PDF Context: **{st.session_state['pdf_name']}** ({len(pipeline.indexed_chunks)} chunks indexed in Vector DB)")

    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # QUERY INPUT & PRESET SELECTION
    # -------------------------------------------------------------------------
    col_input, col_preset = st.columns([3, 1])
    
    with col_preset:
        st.subheader("💡 Sample Queries")
        preset_q = st.radio(
            "Select query:",
            [
                "Custom Query",
                "What is the main topic of the document?",
                "What is Reciprocal Rank Fusion?",
                "How does Contextual Compression work in RAG?",
                "Explain Hybrid Retrieval combining Dense and BM25"
            ]
        )
    
    with col_input:
        st.subheader("❓ Ask Questions on your PDF / Document")
        default_query = "What are the main key points explained in this document?"
        if preset_q != "Custom Query":
            default_query = preset_q
            
        user_query = st.text_area("Enter your question:", value=default_query, height=100)
        run_button = st.button("🚀 Run RAG Pipeline on Document", type="primary", use_container_width=True)

    if run_button and user_query.strip():
        with st.spinner("Orchestrating 16 RAG Pipeline Stages on PDF Context..."):
            response = pipeline.query(user_query)
            st.session_state["last_response"] = response


    if "last_response" in st.session_state:
        resp = st.session_state["last_response"]
        
        st.markdown("---")
        st.subheader("🤖 Generated Grounded Answer")
        
        # Answer Card Container
        with st.container():
            st.markdown("""
            <style>
            .answer-card {
                background: #1e293b;
                border: 1px solid #3b82f6;
                border-radius: 12px;
                padding: 24px;
                font-size: 1.02rem;
                line-height: 1.7;
                color: #f8fafc;
                margin-bottom: 20px;
                box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
            }
            .answer-card h3 {
                color: #60a5fa;
                font-size: 1.3rem;
                margin-top: 12px;
                margin-bottom: 8px;
            }
            .answer-card h4 {
                color: #38bdf8;
                font-size: 1.1rem;
                margin-top: 10px;
                margin-bottom: 6px;
            }
            .answer-card ul, .answer-card ol {
                padding-left: 20px;
                margin-bottom: 12px;
            }
            .answer-card li {
                margin-bottom: 6px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<div class="answer-card">', unsafe_allow_html=True)
            st.markdown(resp.answer)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Metadata / Citations Row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{len(resp.citations)}</div>
                <div class="metric-lbl">Citations Used</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{resp.metrics['end_to_end']['total_latency_ms']:.1f}ms</div>
                <div class="metric-lbl">Total Latency</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{resp.metrics['end_to_end']['overall_quality_score']*100:.0f}%</div>
                <div class="metric-lbl">Quality Score</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{resp.metrics['generation_metrics']['faithfulness']*100:.0f}%</div>
                <div class="metric-lbl">Faithfulness</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📚 Source Citations")
        for c in resp.citations:
            st.markdown(f"""
            <span class="citation-pill">
                <strong>{c['citation_marker']}</strong> {c['title']} (Relevance Score: {c['score']:.4f})
            </span>
            """, unsafe_allow_html=True)

        # Step Logs Breakdown
        st.markdown("### ⏱️ Step-by-Step Pipeline Execution Logs")
        with st.expander("Click to inspect execution trace across all 16 stages", expanded=False):
            for log in resp.step_logs:
                st.markdown(f"""
                <div class="step-box">
                    <span class="step-badge">Stage {log.step_number}</span> 
                    <strong>{log.step_name}</strong> 
                    <span style="float:right; color:#94a3b8; font-size:0.85rem;">⚡ {log.execution_time_ms} ms</span>
                    <br/>
                    <small style="color:#cbd5e1;">{log.description}</small><br/>
                    <div style="background:#1e293b; padding:6px 10px; border-radius:4px; margin-top:6px; font-size:0.8rem; font-family:monospace;">
                        Output: {log.output_summary}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: INGESTION & VECTOR STORE
# =============================================================================
with tab2:
    st.subheader("📥 Data Collection, Loading & Indexing (Steps 1 - 7)")
    
    ingest_col1, ingest_col2 = st.columns([1, 1])
    
    with ingest_col1:
        st.markdown("#### 1. Ingest Preset Sample Datasets")
        dataset_choice = st.selectbox("Select dataset:", list(SAMPLE_DATASETS.keys()))
        if st.button("Load Preset Dataset", use_container_width=True):
            logs = pipeline.load_preset_dataset(dataset_choice)
            st.success(f"Loaded '{dataset_choice}' into Vector Database!")

    with ingest_col2:
        st.markdown("#### 2. Add Custom Documents & PDFs")
        uploaded_file = st.file_uploader("Upload PDF, TXT, or Markdown file", type=["pdf", "txt", "md", "csv"], key="tab2_uploader")
        raw_text_input = st.text_area("Or paste text here:", height=100, key="tab2_text")
        
        if st.button("Ingest Custom Content", type="primary", use_container_width=True):
            if uploaded_file:
                file_bytes = uploaded_file.read()
                if uploaded_file.name.lower().endswith(".pdf"):
                    logs = pipeline.ingest_pdf_file(file_bytes, filename=uploaded_file.name)
                    st.success(f"Successfully extracted & indexed PDF '{uploaded_file.name}'!")
                else:
                    text_content = file_bytes.decode("utf-8", errors="ignore")
                    logs = pipeline.ingest_documents([{"text": text_content, "metadata": {"source": uploaded_file.name}}])
                    st.success(f"Successfully ingested '{uploaded_file.name}'!")
            elif raw_text_input.strip():
                logs = pipeline.ingest_documents([raw_text_input.strip()])
                st.success("Successfully ingested custom text content!")
            else:
                st.warning("Please upload a file or paste text first.")


    st.markdown("---")
    st.subheader("🗄️ Vector Database Index Inspection (Stage 7)")
    st.info(f"Total Chunks Currently Indexed: **{len(pipeline.indexed_chunks)}** documents.")
    
    chunk_data = []
    for idx, doc in enumerate(pipeline.indexed_chunks, start=1):
        chunk_data.append({
            "Index": idx,
            "Chunk ID": doc.doc_id,
            "Char Length": len(doc.page_content),
            "Word Count": len(doc.page_content.split()),
            "Title / Source": doc.metadata.get("title", "Custom Document"),
            "Sample Content": doc.page_content[:100] + "..."
        })
    st.dataframe(chunk_data, use_container_width=True)

# =============================================================================
# TAB 3: CONTEXTUAL COMPRESSION VISUALIZER
# =============================================================================
with tab3:
    st.subheader("🧩 Contextual Compression Retriever Pipeline (Stage 12)")
    st.caption("Visualizing the 5 sub-steps: Document Filtering ➡️ Redundancy Removal ➡️ Reranking ➡️ Content Extraction ➡️ Context Reordering")
    
    st.markdown("""
    | Stage | Sub-component | Functionality |
    | :--- | :--- | :--- |
    | **12.1** | **EmbeddingsFilter** | Removes documents below relevance threshold |
    | **12.2** | **EmbeddingsRedundantFilter** | Removes duplicate or near-identical passages |
    | **12.3** | **Reranker** | Cross-Encoder joint query-document relevance scoring |
    | **12.4** | **LLMChainExtractor** | Extracts only the exact query-relevant sentences |
    | **12.5** | **LongContextReorder** | Solves "Lost-in-the-Middle" by placing top docs at head & tail |
    """)
    
    if "last_response" in st.session_state:
        st.markdown("---")
        st.markdown("### 🔄 Context Reordering Visualization (Stage 12.5)")
        st.write("To prevent the LLM from ignoring information in the middle of long prompts, the highest relevance chunks are placed at the **Head (Beginning)** and **Tail (End)**.")
        
        ctx_docs = st.session_state["last_response"].context
        if ctx_docs:
            reorder_cols = st.columns(len(ctx_docs))
            for idx, d in enumerate(ctx_docs):
                pos_type = "Head (Top Priority)" if idx == 0 else ("Tail (Top Priority)" if idx == len(ctx_docs)-1 else f"Middle Rank #{idx+1}")
                class_name = "head-tail" if (idx == 0 or idx == len(ctx_docs)-1) else "middle-item"
                
                with reorder_cols[idx]:
                    st.markdown(f"""
                    <div class="reorder-item {class_name}">
                        <div>Position #{idx+1}</div>
                        <small>{pos_type}</small><br/><br/>
                        <em>"{d.page_content[:60]}..."</em><br/>
                        <strong>Score: {d.score:.4f}</strong>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Run a query in Tab 1 to see live Contextual Compression visualizer results.")

# =============================================================================
# TAB 4: EVALUATION METRICS
# =============================================================================
with tab4:
    st.subheader("📊 RAG Evaluation & Feedback Loop (Stage 16)")
    
    if "last_response" in st.session_state:
        metrics = st.session_state["last_response"].metrics
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.markdown("#### 🎯 Retrieval Metrics")
            st.json(metrics.get("retrieval_metrics", {}))
            
        with col_m2:
            st.markdown("#### 🧠 Generation Metrics")
            st.json(metrics.get("generation_metrics", {}))
            
        with col_m3:
            st.markdown("#### ⚡ End-to-End Metrics")
            st.json(metrics.get("end_to_end", {}))
    else:
        st.info("Run a query in Tab 1 to view offline + online evaluation metrics.")

# =============================================================================
# TAB 5: ROADMAP STEP INSPECTOR
# =============================================================================
with tab5:
    st.subheader("🗺️ Ultimate RAG Roadmap - 16 Pipeline Stages Inspector")
    
    stages_info = [
        ("1", "Data Collection", "Gather raw text, docx, html, web scraping with compliance checks."),
        ("2", "Data Loading", "Parse raw content into clean standard schema documents."),
        ("3", "Text Splitting", "Chunking documents with sliding window overlap."),
        ("4", "Data Cleaning", "Boilerplate removal, regex normalization, deduplication."),
        ("5", "Metadata Extraction", "Attach titles, dates, categories, authors to chunks."),
        ("6", "Embedding Generation", "Create dense semantic vectors + BM25 sparse vectors."),
        ("7", "Store in Vector Database", "Index dense embeddings and sparse term indices."),
        ("8", "User Query Processing", "Sanitize user prompt and log query telemetry."),
        ("9", "Query Understanding", "Multi-query expansion, HyDE synthesis, self-query filters."),
        ("10", "First-Stage Retrieval", "Perform dense vector search + sparse BM25 retrieval."),
        ("11", "Rank Fusion", "Reciprocal Rank Fusion (RRF) to merge candidate lists."),
        ("12", "Contextual Compression", "Filter, deduplicate, cross-encoder rerank, and extract context."),
        ("13", "Build Context", "Assemble final prompt template with system instructions."),
        ("14", "LLM Generation", "Synthesize grounded answer using LLM / API providers."),
        ("15", "Post-Processing", "Format markdown, inject citations, run safety guardrails."),
        ("16", "Evaluation & Metrics", "Compute Faithfulness, Relevance, Quality & Latency metrics.")
    ]
    
    grid_cols = st.columns(2)
    for idx, (num, name, desc) in enumerate(stages_info):
        col = grid_cols[idx % 2]
        with col:
            st.markdown(f"""
            <div style="background:#1e293b; border-left:4px solid #38bdf8; padding:14px; border-radius:8px; margin-bottom:12px;">
                <span class="step-badge">Stage {num}</span> 
                <strong style="font-size:1.05rem; color:#f8fafc;">{name}</strong>
                <p style="margin-top:6px; margin-bottom:0; color:#94a3b8; font-size:0.88rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
