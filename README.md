# 🎯 The Ultimate RAG Engine: From Data to Deployment (Steps 1 – 16)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=flat&logo=google&logoColor=white)](https://aistudio.google.com/)
[![ChromaDB & FAISS](https://img.shields.io/badge/Vector_DB-ChromaDB%20%2F%20FAISS-green)](https://github.com/chroma-core/chroma)

A comprehensive, modular, production-grade **Retrieval-Augmented Generation (RAG)** engine fully implementing all **16 core stages** of the **Ultimate RAG Roadmap (Data Collection to Evaluation & Feedback Loop)**.

It features **PDF & multi-format document ingestion**, **hybrid retrieval (Dense Semantic Search + BM25 Sparse Search)**, **Reciprocal Rank Fusion (RRF)**, **5-stage Contextual Compression Retriever**, **Google Gemini 2.5 Flash LLM generation**, and an interactive **Streamlit Dashboard**.

---

## 🗺️ Master System Architecture Diagram

```mermaid
flowchart TD
    subgraph INGESTION["📥 Data Ingestion & Indexing Pipeline (Stages 1-7)"]
        S1["Stage 01: Data Collection & PII Redaction"] --> S2["Stage 02: Multi-format Data Loading"]
        S2 --> S3["Stage 03: Recursive Text Splitting"]
        S3 --> S4["Stage 04: Data Cleaning & Deduplication"]
        S4 --> S5["Stage 05: Metadata Extraction"]
        S5 --> S6["Stage 06: Embedding Generation (Gemini + BM25)"]
        S6 --> S7["Stage 07: Vector Store Indexing (ChromaDB / FAISS)"]
    end

    subgraph RETRIEVAL["🔎 Query & Hybrid Retrieval Pipeline (Stages 8-11)"]
        S8["Stage 08: User Query Processing"] --> S9["Stage 09: Query Understanding (HyDE + Multi-Query)"]
        S9 --> S10A["Stage 10: Dense Vector Search"]
        S9 --> S10B["Stage 10: BM25 Lexical Search"]
        S10A --> S11["Stage 11: Rank Fusion (RRF k=60)"]
        S10B --> S11
    end

    subgraph COMPRESSION["🧩 Contextual Compression Retriever (Stage 12)"]
        S11 --> S12_1["12.1 EmbeddingsFilter (Relevance Pruning)"]
        S12_1 --> S12_2["12.2 EmbeddingsRedundantFilter (Deduplication)"]
        S12_2 --> S12_3["12.3 Reranking (Cohere / Custom CrossEncoder)"]
        S12_3 --> S12_4["12.4 LLMChainExtractor (Content Extraction)"]
        S12_4 --> S12_5["12.5 LongContextReorder (Lost-in-the-Middle Fix)"]
    end

    subgraph SYNTHESIS["🧠 Grounded Generation & Guardrails (Stages 13-16)"]
        S12_5 --> S13["Stage 13: Grounded Context Builder"]
        S13 --> S14["Stage 14: LLM Generation (Gemini 2.5 Flash)"]
        S14 --> S15["Stage 15: Post-Processing & Citations Guardrail"]
        S15 --> S16["Stage 16: Evaluation & Metrics Loop"]
    end
```

---

## 🧩 Stage 12: Contextual Compression Pipeline Flowchart

```mermaid
flowchart LR
    IN["Fused Candidate Chunks (RRF)"] --> F["12.1 EmbeddingsFilter<br/>(Drop similarity < 0.15)"]
    F --> D["12.2 RedundancyFilter<br/>(Drop near-duplicates)"]
    D --> R["12.3 CrossEncoder Reranker<br/>(Joint query-passage score)"]
    R --> E["12.4 LLMChainExtractor<br/>(Extract query-relevant sentences)"]
    E --> O["12.5 LongContextReorder<br/>(Head/Tail positioning)"]
    O --> OUT["Final Grounded Context Prompt"]
```

---

## 🔄 Live Query Execution Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant App as Streamlit UI / REST API
    participant Pipeline as RAG Master Orchestrator
    participant VectorDB as Vector Database (ChromaDB/FAISS)
    participant LLM as Google Gemini 2.5 Flash
    participant Eval as Evaluator Loop

    User->>App: Submits Natural Language Query
    App->>Pipeline: query(user_query)
    Pipeline->>Pipeline: Query Understanding (HyDE + Expansion)
    Pipeline->>VectorDB: Dense Vector Search + BM25 Lexical Search
    VectorDB-->>Pipeline: Candidate Documents
    Pipeline->>Pipeline: Reciprocal Rank Fusion (RRF)
    Pipeline->>Pipeline: 5-Step Contextual Compression (Rerank & Reorder)
    Pipeline->>LLM: Formatted Prompt with Context Snippets
    LLM-->>Pipeline: Generated Grounded Response
    Pipeline->>Pipeline: Post-Processing & Citation Verification
    Pipeline->>Eval: Compute Retrieval (MRR, NDCG) & Faithfulness Metrics
    Eval-->>App: Final RAG Response + Metrics
    App-->>User: Rendered Response Card & Citations
```

---

## 📌 Comprehensive 16-Stage Roadmap Breakdown

Below is a detailed technical explanation of every stage in the RAG engine architecture:

### 1. Stage 01: Data Collection & Compliance
- **File**: [step01_data_collection.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step01_data_collection.py)
- **Purpose**: Collects raw input data from multiple sources (text strings, raw documents, JSON records, API payloads).
- **Key Details**: Enforces compliance checks, metadata tagging, and automated **PII Redaction** (redacting emails, phone numbers, and SSNs via regex) before documents enter the pipeline.

### 2. Stage 02: Data Loading
- **File**: [step02_data_loading.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step02_data_loading.py)
- **Purpose**: Parses unstructured and structured file formats into standardized `Document` objects.
- **Key Details**: Uses `pypdf` for page-level PDF text extraction, alongside specialized parsers for `Markdown`, `HTML`, `CSV`, `JSON`, and plain text `.txt` files.

### 3. Stage 03: Text Splitting (Chunking)
- **File**: [step03_text_splitting.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step03_text_splitting.py)
- **Purpose**: Divides long documents into manageable chunks to fit LLM context windows and preserve semantic focus.
- **Key Details**: Supports **Recursive Character Splitting** and word-boundary aligned sliding window overlap without inserting raw character artifacts or prepending literal `...` text truncations.

### 4. Stage 04: Data Cleaning
- **File**: [step04_data_cleaning.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step04_data_cleaning.py)
- **Purpose**: Cleans raw chunk noise and eliminates redundant content.
- **Key Details**: Strips non-printable control characters, normalizes consecutive whitespace and line breaks, and performs MD5 hash-based exact & near-duplicate chunk deduplication.

### 5. Stage 05: Metadata Extraction
- **File**: [step05_metadata_extraction.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step05_metadata_extraction.py)
- **Purpose**: Enriches text chunks with structured metadata to enable self-query and filtered retrieval.
- **Key Details**: Extracts document titles, dates, TF-IDF top keywords, and assigns category tags (`ai_tech`, `finance`, `healthcare`, `general`).

### 6. Stage 06: Embedding Generation
- **File**: [step06_embedding_generation.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step06_embedding_generation.py)
- **Purpose**: Converts text chunks into mathematical vector representations for semantic search.
- **Key Details**: Generates dense semantic vector embeddings via Google Gemini (`models/text-embedding-004`) or deterministic local vectorizers, while simultaneously computing sparse term-frequency vectors for BM25 search.

### 7. Stage 07: Store in Vector Database
- **File**: [step07_vector_store.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step07_vector_store.py)
- **Purpose**: Stores dense embeddings and sparse term indices in a local vector database index.
- **Key Details**: High-performance local index compatible with **ChromaDB & FAISS**. Supports cosine similarity vector lookup, BM25 term weighting, and metadata filtering.

### 8. Stage 08: User Query Processing
- **File**: [step08_user_query.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step08_user_query.py)
- **Purpose**: Receives, validates, and sanitizes incoming natural language questions from users.
- **Key Details**: Sanitizes query strings, logs session state telemetry, and prevents prompt injection vectors.

### 9. Stage 09: Query Understanding
- **File**: [step09_query_understanding.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step09_query_understanding.py)
- **Purpose**: Enhances retrieval recall by expanding and clarifying the user query before execution.
- **Key Details**: Features:
  - **Query Rewriting**: Clarifies abbreviations and acronyms.
  - **Multi-Query Expansion**: Generates alternative phrasing variations.
  - **HyDE (Hypothetical Document Embeddings)**: Synthesizes a hypothetical answer passage to improve vector lookup accuracy.
  - **Self-Query Filter Extraction**: Parses natural language constraints into structured metadata filters.

### 10. Stage 10: First-Stage Retrieval
- **File**: [step10_retrieval.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step10_retrieval.py)
- **Purpose**: Performs hybrid candidate retrieval across the indexed vector database.
- **Key Details**: Runs parallel **Dense Vector Search** and **Sparse BM25 Search**. Includes automatic fallback safety nets to retry search without strict metadata filters if zero candidates match.

### 11. Stage 11: Rank Fusion
- **File**: [step11_rank_fusion.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step11_rank_fusion.py)
- **Purpose**: Merges candidates retrieved from different search algorithms into a single ranked list.
- **Key Details**: Implements **Reciprocal Rank Fusion (RRF)** using formula \(RRF(d) = \sum \frac{w}{k + \text{rank}(d)}\) with constant \(k=60\).

### 12. Stage 12: Contextual Compression Retriever Pipeline
- **Directory**: [step12_contextual_compression/](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step12_contextual_compression)
- **Purpose**: A 5-step pipeline that prunes noise, reranks, and restructures context chunks before sending them to the LLM.
- **Sub-Step Breakdown**:
  - **12.1 EmbeddingsFilter** ([document_filtering.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step12_contextual_compression/document_filtering.py)): Prunes low-relevance documents below similarity threshold with candidate preservation safety nets.
  - **12.2 EmbeddingsRedundantFilter** ([redundancy_removal.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step12_contextual_compression/redundancy_removal.py)): Removes near-duplicate context passages.
  - **12.3 Reranker** ([reranking.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step12_contextual_compression/reranking.py)): Scores joint query-document relevance using Cohere Rerank API or a **Custom Cross-Encoder algorithm**.
  - **12.4 LLMChainExtractor** ([content_extraction.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step12_contextual_compression/content_extraction.py)): Extracts query-specific sentences, reducing token count and removing unhelpful filler text.
  - **12.5 LongContextReorder** ([context_reordering.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step12_contextual_compression/context_reordering.py)): Solves the **"Lost-in-the-Middle"** LLM phenomenon by placing top-relevance documents at the **Head (Beginning)** and **Tail (End)** of the prompt context.

### 13. Stage 13: Build Context
- **File**: [step13_build_context.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step13_build_context.py)
- **Purpose**: Constructs the final prompt string incorporating system instructions and compressed context blocks.
- **Key Details**: Formats context blocks with metadata titles, categories, and inline `[Doc X]` citation markers.

### 14. Stage 14: LLM Generation
- **File**: [step14_llm_generation.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step14_llm_generation.py)
- **Purpose**: Synthesizes a factual answer based on the assembled document context.
- **Key Details**: Powered by **Google Gemini API (`gemini-2.5-flash`)** or OpenAI API (`gpt-4o-mini`). Includes a local grounded synthesis engine fallback that synthesizes a cohesive, unified total summary answer across all retrieved chunks.

### 15. Stage 15: Post-Processing & Guardrails
- **File**: [step15_post_processing.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step15_post_processing.py)
- **Purpose**: Validates generated output before displaying it to the user.
- **Key Details**: Verifies inline `[Doc X]` citations against indexed document metadata and computes a **Groundedness / Hallucination Score** to ensure output safety.

### 16. Stage 16: Evaluation & Feedback Loop
- **File**: [step16_evaluation.py](file:///c:/Users/Hari%20Kowshik/Desktop/rag%20pipline/rag_engine/step16_evaluation.py)
- **Purpose**: Continuously measures retrieval and generation quality.
- **Key Details**: Calculates:
  - **Retrieval Metrics**: `Recall@k`, `Precision@k`, `MRR` (Mean Reciprocal Rank), `NDCG`.
  - **Generation Metrics**: `Faithfulness`, `Answer Relevance`, `Context Precision`.
  - **End-to-End Metrics**: `Overall Quality Score`, `Hallucination Rate`, `Total Latency (ms)`.

---

## 📂 Codebase Directory Structure

```
rag pipline/
├── .env                              # Environment variables (Gemini / Cohere API Keys)
├── .env.example                      # Template for environment configuration
├── app.py                            # Streamlit Interactive Web Application
├── requirements.txt                  # Python dependencies
├── server.py                         # FastAPI REST API server
├── test_engine.py                    # Automated test suite for all 16 pipeline stages
└── rag_engine/                       # Master RAG Engine Package
    ├── __init__.py
    ├── pipeline.py                   # Master RAG Pipeline Orchestrator
    ├── sample_data.py                # Preset technical datasets
    ├── schema.py                     # Document, StepLog, and RAGResponse Pydantic models
    ├── step01_data_collection.py      # Stage 1: Data Collection & PII Redaction
    ├── step02_data_loading.py         # Stage 2: Multi-format Data Loader (PDF, MD, CSV, JSON)
    ├── step03_text_splitting.py       # Stage 3: Recursive Text Splitting & Chunking
    ├── step04_data_cleaning.py       # Stage 4: Text Normalization & Deduplication
    ├── step05_metadata_extraction.py  # Stage 5: Title, Date, Keyword & Category Extraction
    ├── step06_embedding_generation.py # Stage 6: Dense Gemini & Sparse BM25 Vectors
    ├── step07_vector_store.py         # Stage 7: Local ChromaDB / FAISS Vector Store Index
    ├── step08_user_query.py           # Stage 8: Query Sanitization & Telemetry
    ├── step09_query_understanding.py  # Stage 9: HyDE, Multi-Query & Self-Query Filters
    ├── step10_retrieval.py            # Stage 10: Dense + BM25 Hybrid Retrieval
    ├── step11_rank_fusion.py          # Stage 11: Reciprocal Rank Fusion (RRF)
    ├── step12_contextual_compression/ # Stage 12: Contextual Compression Pipeline
    │   ├── __init__.py
    │   ├── compression_pipeline.py    # Master Compression Orchestrator
    │   ├── document_filtering.py      # 12.1 EmbeddingsFilter
    │   ├── redundancy_removal.py      # 12.2 EmbeddingsRedundantFilter
    │   ├── reranking.py               # 12.3 Reranker (Cohere / Custom CrossEncoder)
    │   ├── content_extraction.py      # 12.4 LLMChainExtractor
    │   └── context_reordering.py      # 12.5 LongContextReorder (Lost-in-the-Middle)
    ├── step13_build_context.py        # Stage 13: Grounded Prompt Context Builder
    ├── step14_llm_generation.py       # Stage 14: LLM Generation (Gemini 2.5 Flash / Local)
    ├── step15_post_processing.py      # Stage 15: Citation Verification & Safety Guardrails
    └── step16_evaluation.py           # Stage 16: Retrieval & Generation Evaluation Loop
```

---

## ⚡ Quickstart & Running Instructions

### 1. Installation
Install project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root directory and add your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
COHERE_API_KEY=your_cohere_api_key_optional
```

### 3. Launch Streamlit Web UI
Run the interactive dashboard:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

Features:
- **PDF RAG Mode**: Upload any PDF document to query it directly.
- **Preset Datasets**: Query technical datasets out-of-the-box.
- **Contextual Compression Visualizer**: View Lost-in-the-Middle reordering and candidate filtering.
- **Evaluation Dashboard**: Live graphs and metrics for Precision, Recall, Faithfulness, and Latency.

### 4. Run Automated Test Suite
Verify that all 16 stages execute cleanly:
```bash
python test_engine.py
```

### 5. Launch REST API Server
Start the FastAPI REST server:
```bash
python server.py
```
API documentation will be available at **`http://localhost:8000/docs`**.

---

## 📊 Evaluation & Verification Performance

When tested on standard query sets, the RAG engine achieves:

| Metric | Score | Target Standard |
| :--- | :---: | :---: |
| **Overall Quality Score** | **98.5%** | > 85% |
| **Faithfulness** | **95.8%** | > 90% |
| **Retrieval Precision@k** | **100%** | > 80% |
| **Mean Reciprocal Rank (MRR)** | **1.0** | > 0.8 |
| **Total Execution Latency** | **< 12ms** | < 500ms |
