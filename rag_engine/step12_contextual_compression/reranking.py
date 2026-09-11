import math
import re
import os
from typing import List, Tuple
from ..schema import Document, StepLog

class Reranker:
    """
    Step 12.3: RERANKING
    - Supports Cohere Rerank API (when COHERE_API_KEY is provided)
    - Custom Code Multi-Feature Cross-Encoder Reranker (Pure Python local fallback)
    """

    def __init__(self, model_type: str = "CustomCrossEncoder", top_n: int = 10, api_key: str = None):
        self.model_type = model_type
        self.top_n = top_n
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")

    def cohere_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Uses Cohere API to rerank documents if API key is supplied.
        """
        try:
            import cohere
            co = cohere.Client(api_key=self.api_key)
            doc_texts = [d.page_content for d in docs]
            res = co.v2.rerank(
                model="rerank-v3.5",
                query=query,
                documents=doc_texts,
                top_n=min(self.top_n, len(docs))
            )
            reranked = []
            for item in res.results:
                orig_doc = docs[item.index]
                doc_copy = Document(
                    doc_id=orig_doc.doc_id,
                    page_content=orig_doc.page_content,
                    metadata={**orig_doc.metadata, "rerank_score": round(item.relevance_score, 4), "reranker_model": "Cohere-Rerank-v3.5"},
                    score=item.relevance_score,
                    embedding=orig_doc.embedding,
                    sparse_vector=orig_doc.sparse_vector
                )
                reranked.append(doc_copy)
            return reranked
        except Exception:
            # Try legacy client v1 method if v2 fails
            try:
                import cohere
                co = cohere.Client(api_key=self.api_key)
                doc_texts = [d.page_content for d in docs]
                res = co.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=doc_texts,
                    top_n=min(self.top_n, len(docs))
                )
                reranked = []
                for item in res.results:
                    orig_doc = docs[item.index]
                    doc_copy = Document(
                        doc_id=orig_doc.doc_id,
                        page_content=orig_doc.page_content,
                        metadata={**orig_doc.metadata, "rerank_score": round(item.relevance_score, 4), "reranker_model": "Cohere-Rerank-v3.0"},
                        score=item.relevance_score,
                        embedding=orig_doc.embedding,
                        sparse_vector=orig_doc.sparse_vector
                    )
                    reranked.append(doc_copy)
                return reranked
            except Exception:
                return None

    def custom_cross_encoder_score(self, query: str, doc_text: str, metadata: dict = None) -> float:
        """
        Custom Cross-Encoder joint query-document interaction scoring algorithm.
        Calculates:
        1. Term Coverage Ratio
        2. Exact Phrase Proximity & Bigram Alignment
        3. Title / Heading Relevance Bonus
        4. Information Density Normalization
        """
        query_low = query.lower()
        doc_low = doc_text.lower()

        q_terms = [w for w in re.findall(r'\b\w+\b', query_low) if len(w) > 1]
        d_terms = re.findall(r'\b\w+\b', doc_low)
        d_term_set = set(d_terms)

        if not q_terms or not d_terms:
            return 0.0

        unique_q_terms = set(q_terms)
        matched_terms = unique_q_terms.intersection(d_term_set)
        term_coverage = len(matched_terms) / len(unique_q_terms)

        q_bigrams = [f"{q_terms[i]} {q_terms[i+1]}" for i in range(len(q_terms)-1)]
        bigram_bonus = 0.0
        if q_bigrams:
            bigram_matches = sum(1 for bg in q_bigrams if bg in doc_low)
            bigram_bonus = (bigram_matches / len(q_bigrams)) * 0.25

        phrase_bonus = 0.35 if query_low in doc_low else 0.0

        title_bonus = 0.0
        if metadata and "title" in metadata:
            title_low = str(metadata["title"]).lower()
            if any(t in title_low for t in q_terms):
                title_bonus = 0.15

        len_penalty = min(1.0, len(d_terms) / 15.0)
        raw_score = (term_coverage * 0.45 + phrase_bonus + bigram_bonus + title_bonus) * len_penalty
        return min(0.99, max(0.05, round(raw_score, 4)))

    def rerank(self, query: str, docs: List[Document]) -> Tuple[List[Document], StepLog]:
        top_reranked = None
        model_used = self.model_type

        # Attempt Cohere Rerank if requested or API key present
        if ("cohere" in self.model_type.lower() or self.api_key) and self.api_key:
            cohere_results = self.cohere_rerank(query, docs)
            if cohere_results:
                top_reranked = cohere_results
                model_used = "Cohere Rerank API"

        # Local Custom Code Fallback
        if top_reranked is None:
            reranked_docs = []
            for doc in docs:
                score = self.custom_cross_encoder_score(query, doc.page_content, doc.metadata)
                doc_copy = Document(
                    doc_id=doc.doc_id,
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "rerank_score": score,
                        "reranker_model": "Custom-CrossEncoder (Local)"
                    },
                    score=score,
                    embedding=doc.embedding,
                    sparse_vector=doc.sparse_vector
                )
                reranked_docs.append(doc_copy)

            reranked_docs.sort(key=lambda x: x.score, reverse=True)
            top_reranked = reranked_docs[:self.top_n]
            model_used = "Custom CrossEncoder (Local)"

        for idx, doc in enumerate(top_reranked, start=1):
            doc.rank = idx

        log = StepLog(
            step_number="12.3",
            step_name=f"Reranking ({model_used})",
            description=f"Reranked candidate documents using {model_used} scoring.",
            input_summary=f"Scored {len(docs)} input documents.",
            output_summary=f"Selected top {len(top_reranked)} highest-scoring documents.",
            details={
                "model_type": model_used,
                "top_n": self.top_n,
                "input_count": len(docs),
                "output_count": len(top_reranked),
                "top_score": top_reranked[0].score if top_reranked else 0.0
            }
        )
        return top_reranked, log



