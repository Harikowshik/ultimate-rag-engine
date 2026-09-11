from typing import List, Dict, Any, Tuple
from .schema import Document, StepLog

class RankFusion:
    """
    Stage 11: RANK FUSION
    - Merge results from multiple retrieval strategies (Dense + Sparse) using Reciprocal Rank Fusion (RRF)
    - RRF Score: Sum(1 / (k + rank_i)) across rankers where k=60
    """

    def __init__(self, k_constant: int = 60, dense_weight: float = 1.0, sparse_weight: float = 1.0):
        self.k_constant = k_constant
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def reciprocal_rank_fusion(self, list_of_ranked_lists: List[List[Document]], weights: List[float] = None) -> List[Document]:
        if not weights:
            weights = [1.0] * len(list_of_ranked_lists)

        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for list_idx, doc_list in enumerate(list_of_ranked_lists):
            w = weights[list_idx]
            for rank, doc in enumerate(doc_list, start=1):
                doc_map[doc.doc_id] = doc
                rrf_val = w * (1.0 / (self.k_constant + rank))
                scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + rrf_val

        fused_docs = []
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            orig_doc = doc_map[doc_id]
            fused_doc = Document(
                doc_id=orig_doc.doc_id,
                page_content=orig_doc.page_content,
                metadata={**orig_doc.metadata, "rrf_score": score},
                score=score,
                embedding=orig_doc.embedding,
                sparse_vector=orig_doc.sparse_vector
            )
            fused_docs.append(fused_doc)

        # Assign final rank indices
        for i, d in enumerate(fused_docs, start=1):
            d.rank = i

        return fused_docs

    def run(self, dense_list: List[Document], sparse_list: List[Document], top_k: int = 15) -> Tuple[List[Document], StepLog]:
        fused = self.reciprocal_rank_fusion([dense_list, sparse_list], weights=[self.dense_weight, self.sparse_weight])
        top_fused = fused[:top_k]

        log = StepLog(
            step_number="11",
            step_name="Rank Fusion (RRF)",
            description=f"Merged dense and sparse ranked lists using Reciprocal Rank Fusion (k={self.k_constant}).",
            input_summary=f"Dense candidates: {len(dense_list)}, Sparse candidates: {len(sparse_list)}.",
            output_summary=f"Produced {len(top_fused)} fused ranked documents.",
            details={
                "k_constant": self.k_constant,
                "top_fused_ids": [d.doc_id for d in top_fused[:5]],
                "top_rrf_score": top_fused[0].score if top_fused else 0.0
            }
        )
        return top_fused, log
