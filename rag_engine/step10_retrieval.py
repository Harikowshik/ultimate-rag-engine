from typing import List, Dict, Any, Tuple
from .schema import Document, StepLog
from .step07_vector_store import VectorDatabase
from .step06_embedding_generation import EmbeddingGenerator

class FirstStageRetriever:
    """
    Stage 10: RETRIEVAL (FIRST-STAGE)
    - Dense Retrieval (Vector Search)
    - Sparse Retrieval (BM25 Lexical Keyword Search)
    - Hybrid Retrieval (Combining Dense + BM25 scores)
    """

    def __init__(self, vector_db: VectorDatabase, embedder: EmbeddingGenerator, mode: str = "hybrid", top_k: int = 20):
        self.vector_db = vector_db
        self.embedder = embedder
        self.mode = mode.lower() # 'dense', 'sparse', or 'hybrid'
        self.top_k = top_k

    def retrieve(self, query: str, query_data: Dict[str, Any]) -> Tuple[List[Document], List[Document], StepLog]:
        dense_results: List[Document] = []
        sparse_results: List[Document] = []

        # Extract search parameters
        filters = query_data.get("metadata_filters", {})
        hyde_doc = query_data.get("hyde_document", query)

        # Dense retrieval embedding
        q_vec, _ = self.embedder._generate_dense_vector(hyde_doc)
        dense_results = self.vector_db.search_dense(q_vec, top_k=self.top_k, metadata_filter=filters if filters else None)

        # Fallback 1: If metadata filter produced 0 results, search without metadata filter
        if not dense_results and filters:
            dense_results = self.vector_db.search_dense(q_vec, top_k=self.top_k, metadata_filter=None)

        # Fallback 2: If HyDE produced 0 results, retry with original query embedding
        if not dense_results and hyde_doc != query:
            orig_q_vec, _ = self.embedder._generate_dense_vector(query)
            dense_results = self.vector_db.search_dense(orig_q_vec, top_k=self.top_k, metadata_filter=None)

        # Sparse BM25 retrieval
        terms = [t for t in query.lower().split() if len(t) > 2]
        sparse_results = self.vector_db.search_bm25(terms, top_k=self.top_k)
        if not sparse_results:
            sparse_results = self.vector_db.search_bm25(query.lower().split(), top_k=self.top_k)

        if self.mode == "dense":
            combined = dense_results
        elif self.mode == "sparse":
            combined = sparse_results
        else: # Hybrid mode (return both lists for Stage 11 Rank Fusion)
            combined = dense_results

        log = StepLog(
            step_number="10",
            step_name="First-Stage Retrieval",
            description=f"Retrieved candidate documents using mode='{self.mode}' (top_k={self.top_k}).",
            input_summary=f"Query search across indexed vector database.",
            output_summary=f"Retrieved {len(dense_results)} dense candidates and {len(sparse_results)} BM25 sparse candidates.",
            details={
                "mode": self.mode,
                "top_k": self.top_k,
                "num_dense": len(dense_results),
                "num_sparse": len(sparse_results)
            }
        )
        return dense_results, sparse_results, log
