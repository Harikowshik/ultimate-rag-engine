from typing import List, Tuple
from ..schema import Document, StepLog
from ..step06_embedding_generation import EmbeddingGenerator
from ..step07_vector_store import VectorDatabase

class DocumentFilter:
    """
    Step 12.1: DOCUMENT FILTERING
    - EmbeddingsFilter: Uses cosine similarity with query embedding to prune low-relevance documents below threshold.
    - LLMChainFilter: Fast heuristic relevance verification.
    """

    def __init__(self, similarity_threshold: float = 0.15, embedder: EmbeddingGenerator = None):
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder or EmbeddingGenerator()
        self.vector_db_util = VectorDatabase()

    def filter_documents(self, query: str, docs: List[Document]) -> Tuple[List[Document], StepLog]:
        query_vec, _ = self.embedder._generate_dense_vector(query)
        passed_docs = []
        doc_scores = []
        filtered_count = 0

        for doc in docs:
            if doc.embedding is None:
                vec, _ = self.embedder._generate_dense_vector(doc.page_content)
                doc.embedding = vec

            sim = self.vector_db_util.cosine_similarity(query_vec, doc.embedding)
            doc_scores.append((sim, doc))

            if sim >= self.similarity_threshold:
                # Update score with similarity
                doc_copy = Document(
                    doc_id=doc.doc_id,
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "filter_similarity": round(sim, 4)},
                    score=sim,
                    embedding=doc.embedding,
                    sparse_vector=doc.sparse_vector
                )
                passed_docs.append(doc_copy)
            else:
                filtered_count += 1

        # Fallback Safety Net: If threshold filtering pruned ALL documents, keep top candidates
        if not passed_docs and docs:
            doc_scores.sort(key=lambda x: x[0], reverse=True)
            fallback_count = min(6, len(docs))
            for sim, doc in doc_scores[:fallback_count]:
                doc_copy = Document(
                    doc_id=doc.doc_id,
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "filter_similarity": round(sim, 4), "filter_fallback": True},
                    score=sim if sim > 0 else 0.1,
                    embedding=doc.embedding,
                    sparse_vector=doc.sparse_vector
                )
                passed_docs.append(doc_copy)
            filtered_count = len(docs) - len(passed_docs)

        log = StepLog(
            step_number="12.1",
            step_name="Document Filtering (EmbeddingsFilter)",
            description=f"Filtered documents with cosine similarity < {self.similarity_threshold}.",
            input_summary=f"Evaluated {len(docs)} documents.",
            output_summary=f"Kept {len(passed_docs)} documents ({filtered_count} pruned).",
            details={
                "threshold": self.similarity_threshold,
                "input_count": len(docs),
                "output_count": len(passed_docs),
                "pruned_count": filtered_count
            }
        )
        return passed_docs, log
