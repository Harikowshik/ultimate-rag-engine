from typing import List, Tuple
from ..schema import Document, StepLog
from ..step06_embedding_generation import EmbeddingGenerator
from ..step07_vector_store import VectorDatabase

class RedundancyRemover:
    """
    Step 12.2: REDUNDANCY REMOVAL
    - EmbeddingsRedundantFilter: Removes candidate documents that have very high similarity (> similarity_threshold) to an already accepted higher-ranked document.
    """

    def __init__(self, redundancy_threshold: float = 0.88, embedder: EmbeddingGenerator = None):
        self.redundancy_threshold = redundancy_threshold
        self.embedder = embedder or EmbeddingGenerator()
        self.vector_db_util = VectorDatabase()

    def remove_redundant(self, docs: List[Document]) -> Tuple[List[Document], StepLog]:
        if not docs:
            return [], StepLog("12.2", "Redundancy Removal", "No docs to deduplicate", "0 docs", "0 docs", {})

        non_redundant = []
        removed_count = 0
        for doc in docs:
            if doc.embedding is None:
                vec, _ = self.embedder._generate_dense_vector(doc.page_content)
                doc.embedding = vec




            is_redundant = False
            for accepted_doc in non_redundant:
                sim = self.vector_db_util.cosine_similarity(doc.embedding, accepted_doc.embedding)
                if sim >= self.redundancy_threshold:
                    is_redundant = True
                    removed_count += 1
                    break

            if not is_redundant:
                non_redundant.append(doc)

        log = StepLog(
            step_number="12.2",
            step_name="Redundancy Removal (EmbeddingsRedundantFilter)",
            description=f"Removed redundant documents with pairwise semantic similarity >= {self.redundancy_threshold}.",
            input_summary=f"Evaluated {len(docs)} documents.",
            output_summary=f"Retained {len(non_redundant)} distinct documents ({removed_count} redundant removed).",
            details={
                "redundancy_threshold": self.redundancy_threshold,
                "input_count": len(docs),
                "retained_count": len(non_redundant),
                "removed_count": removed_count
            }
        )
        return non_redundant, log
