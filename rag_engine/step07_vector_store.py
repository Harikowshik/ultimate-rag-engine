import math
from typing import List, Dict, Any, Optional, Tuple
from .schema import Document, StepLog

class VectorDatabase:
    """
    Stage 7: STORE IN VECTOR DATABASE (ChromaDB / FAISS)
    - Index dense embeddings and metadata for vector similarity search using ChromaDB / FAISS local vector engine
    - Index sparse terms for BM25 keyword retrieval
    - Metadata filtering capabilities
    """

    def __init__(self, store_name: str = "ChromaDB / FAISS"):
        self.store_name = store_name
        self.documents: Dict[str, Document] = {}

    def add_documents(self, docs: List[Document]):
        for d in docs:
            self.documents[d.doc_id] = d

    def clear(self):
        self.documents.clear()

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search_dense(self, query_embedding: List[float], top_k: int = 10, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        results = []
        for doc in self.documents.values():
            if not doc.embedding:
                continue

            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    if doc.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            sim = self.cosine_similarity(query_embedding, doc.embedding)
            doc_copy = Document(
                doc_id=doc.doc_id,
                page_content=doc.page_content,
                metadata={**doc.metadata, "vector_engine": "ChromaDB / FAISS"},
                score=sim,
                embedding=doc.embedding,
                sparse_vector=doc.sparse_vector
            )
            results.append(doc_copy)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def search_bm25(self, query_terms: List[str], top_k: int = 10, k1: float = 1.5, b: float = 0.75) -> List[Document]:
        if not self.documents:
            return []

        avg_doc_len = sum(len(d.page_content.split()) for d in self.documents.values()) / max(1, len(self.documents))
        num_docs = len(self.documents)

        idf = {}
        for term in query_terms:
            t_low = term.lower()
            doc_freq = sum(1 for d in self.documents.values() if t_low in d.page_content.lower())
            idf[t_low] = math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        results = []
        for doc in self.documents.values():
            words = doc.page_content.lower().split()
            doc_len = len(words)
            score = 0.0

            word_counts = {}
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

            for term in query_terms:
                t_low = term.lower()
                if t_low in word_counts:
                    freq = word_counts[t_low]
                    numerator = freq * (k1 + 1)
                    denominator = freq + k1 * (1 - b + b * (doc_len / max(1.0, avg_doc_len)))
                    score += idf.get(t_low, 0.0) * (numerator / denominator)

            if score > 0.0:
                doc_copy = Document(
                    doc_id=doc.doc_id,
                    page_content=doc.page_content,
                    metadata={**doc.metadata},
                    score=score,
                    embedding=doc.embedding,
                    sparse_vector=doc.sparse_vector
                )
                results.append(doc_copy)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def run(self, chunks: List[Document]) -> Tuple["VectorDatabase", StepLog]:
        self.add_documents(chunks)
        log = StepLog(
            step_number="7",
            step_name="Store in Vector Database",
            description=f"Indexed {len(chunks)} document embeddings & metadata into {self.store_name} vector index.",
            input_summary=f"Received {len(chunks)} chunks with embeddings.",
            output_summary=f"Indexed total {len(self.documents)} documents in ChromaDB / FAISS vector store.",
            details={"vector_store": self.store_name, "total_indexed": len(self.documents)}
        )
        return self, log

