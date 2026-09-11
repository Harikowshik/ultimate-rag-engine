from typing import List, Tuple, Dict, Any
from ..schema import Document, StepLog
from .document_filtering import DocumentFilter
from .redundancy_removal import RedundancyRemover
from .reranking import Reranker
from .content_extraction import ContentExtractor
from .context_reordering import ContextReorderer

class ContextualCompressionPipeline:
    """
    Stage 12 Master Pipeline Orchestrator:
    Runs all 5 contextual compression components sequentially:
    12.1 Document Filtering
    12.2 Redundancy Removal
    12.3 Reranking
    12.4 Content Extraction
    12.5 Context Reordering
    """

    def __init__(
        self,
        similarity_threshold: float = 0.30,
        redundancy_threshold: float = 0.88,
        reranker_model: str = "CrossEncoder",
        top_n_rerank: int = 10,
        enable_extraction: bool = True,
        enable_reordering: bool = True,
        cohere_api_key: str = None
    ):
        self.doc_filter = DocumentFilter(similarity_threshold=similarity_threshold)
        self.redundancy_remover = RedundancyRemover(redundancy_threshold=redundancy_threshold)
        self.reranker = Reranker(model_type=reranker_model, top_n=top_n_rerank, api_key=cohere_api_key)
        self.extractor = ContentExtractor()
        self.reorderer = ContextReorderer()

        self.enable_extraction = enable_extraction
        self.enable_reordering = enable_reordering


    def run(self, query: str, candidate_docs: List[Document]) -> Tuple[List[Document], List[StepLog]]:
        logs = []

        # 12.1 Filtering
        filtered_docs, log_12_1 = self.doc_filter.filter_documents(query, candidate_docs)
        logs.append(log_12_1)

        # 12.2 Redundancy Removal
        deduped_docs, log_12_2 = self.redundancy_remover.remove_redundant(filtered_docs)
        logs.append(log_12_2)

        # 12.3 Reranking
        reranked_docs, log_12_3 = self.reranker.rerank(query, deduped_docs)
        logs.append(log_12_3)

        # 12.4 Content Extraction
        if self.enable_extraction:
            extracted_docs, log_12_4 = self.extractor.extract(query, reranked_docs)
            logs.append(log_12_4)
        else:
            extracted_docs = reranked_docs

        # 12.5 Context Reordering
        if self.enable_reordering:
            final_compressed_docs, log_12_5 = self.reorderer.reorder(extracted_docs)
            logs.append(log_12_5)
        else:
            final_compressed_docs = extracted_docs

        return final_compressed_docs, logs
