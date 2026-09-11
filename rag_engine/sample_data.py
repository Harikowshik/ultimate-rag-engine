SAMPLE_DATASETS = {
    "rag_architecture": [
        {
            "content": "# Retrieval-Augmented Generation (RAG) Architecture Overview\nRetrieval-Augmented Generation combines an information retrieval component with a text generator model. It retrieves documents from a vector store based on query embeddings and provides them as context to the LLM.",
            "source": "rag_spec.md",
            "type": "markdown",
            "access_level": "public"
        },
        {
            "content": "## Data Collection and Loading\nData collection gathers raw documents from databases, markdown files, web pages, and APIs while enforcing PII redaction and compliance. Data loading parses PDFs, HTML, CSVs, and JSON into structured Document schema with metadata tags.",
            "source": "rag_data_ingestion.md",
            "type": "markdown"
        },
        {
            "content": "## Text Splitting and Chunking Strategies\nText splitting partitions large documents into smaller chunks. Common strategies include recursive character splitting, sentence splitting, and semantic chunking with a defined chunk size (e.g., 500 chars) and overlap (e.g., 100 chars). Proper chunking avoids losing context across chunk boundaries.",
            "source": "chunking_guide.md",
            "type": "markdown"
        },
        {
            "content": "## Embedding Generation and Vector Store Indexing\nDense embeddings are vector representations of text generated using models like BGE, E5, OpenAI text-embedding-3, or SentenceTransformers. High dimensional vectors and sparse BM25 term frequencies are indexed into vector databases like Chroma, FAISS, Pinecone, or Qdrant for fast similarity search.",
            "source": "embeddings_vector_db.md",
            "type": "markdown"
        },
        {
            "content": "## First-Stage Retrieval and Hybrid Search\nFirst-stage retrieval extracts top candidate chunks using Dense vector similarity (cosine distance) and Sparse lexical search (BM25). Hybrid retrieval combines both dense semantic understanding and sparse keyword matching for optimal recall.",
            "source": "retrieval_strategies.md",
            "type": "markdown"
        },
        {
            "content": "## Rank Fusion with Reciprocal Rank Fusion (RRF)\nRank Fusion combines search results from multiple retrievers. Reciprocal Rank Fusion (RRF) computes a score for each document as RRF(d) = sum(1 / (k + rank_i(d))) with constant k=60. This combines the complementary strengths of sparse and dense retrievers.",
            "source": "rank_fusion_rrf.md",
            "type": "markdown"
        },
        {
            "content": "## Contextual Compression Retriever Pipeline\nContextual Compression applies five steps to refine candidate chunks before prompting the LLM:\n1. Document Filtering: Prunes candidates with low embedding similarity.\n2. Redundancy Removal: Removes duplicate semantic chunks.\n3. Reranking: Re-scores query-document pairs using a Cross-Encoder reranker.\n4. Content Extraction: Extracts only the exact relevant passages from long documents.\n5. Context Reordering: Reorders chunks to place high relevance documents at the top and bottom to fix Lost-in-the-Middle.",
            "source": "contextual_compression.md",
            "type": "markdown"
        },
        {
            "content": "## LLM Generation and Evaluation\nLLMs generate grounded answers using the assembled context. Guardrails verify faithfulness and citations [Doc X] format. Evaluation measures Recall@k, Precision@k, MRR, NDCG, Faithfulness, and Answer Relevance.",
            "source": "evaluation_metrics.md",
            "type": "markdown"
        }
    ],
    "financial_reports": [
        {
            "content": "Acme Corp Q4 Financial Summary: Total revenue reached $12.5M, representing a 24% year-over-year increase. Net income expanded to $3.2M. Operating expenses increased by 10% due to expanded AI engineering investments.",
            "source": "acme_q4_report.txt",
            "type": "text"
        },
        {
            "content": "Strategic Growth & R&D Investment 2025: Acme Corp allocated $4.5M to research and development focused on automated AI agents and enterprise vector database search solutions.",
            "source": "acme_strategy_2025.txt",
            "type": "text"
        }
    ]
}
