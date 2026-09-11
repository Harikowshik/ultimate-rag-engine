import time
from typing import List, Dict, Any, Optional, Union
from .schema import Document, StepLog, RAGResponse
from .step01_data_collection import DataCollector
from .step02_data_loading import DataLoader
from .step03_text_splitting import TextSplitter
from .step04_data_cleaning import DataCleaner
from .step05_metadata_extraction import MetadataExtractor
from .step06_embedding_generation import EmbeddingGenerator
from .step07_vector_store import VectorDatabase
from .step08_user_query import UserQueryHandler
from .step09_query_understanding import QueryUnderstanding
from .step10_retrieval import FirstStageRetriever
from .step11_rank_fusion import RankFusion
from .step12_contextual_compression import ContextualCompressionPipeline
from .step13_build_context import ContextBuilder
from .step14_llm_generation import LLMGenerator
from .step15_post_processing import PostProcessor
from .step16_evaluation import Evaluator
from .sample_data import SAMPLE_DATASETS

class RAGPipeline:
    """
    Master RAG Engine Orchestrator
    Implements all 16 pipeline stages from the Ultimate RAG Roadmap:
    - Stage 1: Data Collection & Compliance
    - Stage 2: Data Loading
    - Stage 3: Text Splitting (Chunking)
    - Stage 4: Data Cleaning & Deduplication
    - Stage 5: Metadata Extraction
    - Stage 6: Embedding Generation (Dense + Sparse)
    - Stage 7: Store in Vector Database
    - Stage 8: User Query Processing
    - Stage 9: Query Understanding (HyDE, Multi-Query, Filters)
    - Stage 10: First-Stage Retrieval (Dense, Sparse, Hybrid)
    - Stage 11: Rank Fusion (Reciprocal Rank Fusion - RRF)
    - Stage 12: Contextual Compression Retriever (12.1-12.5)
    - Stage 13: Build Context & Grounding Prompt
    - Stage 14: LLM Generation
    - Stage 15: Post-Processing & Guardrails
    - Stage 16: Evaluation & Metrics Loop
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        retrieval_mode: str = "hybrid",
        enable_compression: bool = True,
        similarity_threshold: float = 0.25,
        redundancy_threshold: float = 0.88,
        reranker_model: str = "CrossEncoder",
        top_k_first_stage: int = 15,
        top_n_final: int = 5,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None
    ):
        self.config = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_mode": retrieval_mode,
            "enable_compression": enable_compression,
            "similarity_threshold": similarity_threshold,
            "redundancy_threshold": redundancy_threshold,
            "reranker_model": reranker_model,
            "top_k_first_stage": top_k_first_stage,
            "top_n_final": top_n_final,
            "has_openai_key": bool(openai_api_key),
            "has_gemini_key": bool(gemini_api_key),
            "has_cohere_key": bool(cohere_api_key)
        }

        # Initialize Pipeline Components
        self.collector = DataCollector()
        self.loader = DataLoader()
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.cleaner = DataCleaner()
        self.metadata_extractor = MetadataExtractor()
        self.embedder = EmbeddingGenerator(gemini_api_key=gemini_api_key)
        self.vector_db = VectorDatabase(store_name="ChromaDB / FAISS")



        self.query_handler = UserQueryHandler()
        self.query_understanding = QueryUnderstanding()
        self.first_stage_retriever = FirstStageRetriever(
            vector_db=self.vector_db,
            embedder=self.embedder,
            mode=retrieval_mode,
            top_k=top_k_first_stage
        )
        self.rank_fusion = RankFusion()
        self.compression_pipeline = ContextualCompressionPipeline(
            similarity_threshold=similarity_threshold,
            redundancy_threshold=redundancy_threshold,
            reranker_model=reranker_model,
            top_n_rerank=top_n_final,
            cohere_api_key=cohere_api_key
        )
        self.context_builder = ContextBuilder()
        self.llm_generator = LLMGenerator(
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key
        )
        self.post_processor = PostProcessor()
        self.evaluator = Evaluator()

        self.indexed_chunks: List[Document] = []
        self.indexing_logs: List[StepLog] = []

        # Auto-load preset dataset on initialization

        self.load_preset_dataset("rag_architecture")

    def ingest_pdf_file(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> List[StepLog]:
        """
        Extracts pages from PDF bytes using pypdf, chunks them, and indexes into Vector Database.
        """
        parsed_docs = self.loader.parse_pdf_bytes(pdf_bytes, filename=filename)
        if not parsed_docs:
            return []

        self.vector_db.clear()
        logs = []

        # Step 1: Collection
        t0 = time.time()
        log1 = StepLog("1", "Data Collection & Compliance", "Collected PDF binary file.", f"File: {filename}", f"Extracted {len(parsed_docs)} pages.")
        log1.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log1)

        # Step 2: Loading
        t0 = time.time()
        loaded_docs, log2 = self.loader.run(parsed_docs)
        log2.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log2)

        # Step 3: Text Splitting
        t0 = time.time()
        chunks, log3 = self.splitter.run(loaded_docs)
        log3.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log3)

        # Step 4: Data Cleaning
        t0 = time.time()
        cleaned_chunks, log4 = self.cleaner.run(chunks)
        log4.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log4)

        # Step 5: Metadata Extraction
        t0 = time.time()
        enriched_chunks, log5 = self.metadata_extractor.run(cleaned_chunks)
        log5.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log5)

        # Step 6: Embedding Generation
        t0 = time.time()
        embedded_chunks, log6 = self.embedder.run(enriched_chunks)
        log6.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log6)

        # Step 7: Store in Vector DB
        t0 = time.time()
        _, log7 = self.vector_db.run(embedded_chunks)
        log7.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log7)

        self.indexed_chunks = embedded_chunks
        self.indexing_logs = logs
        return logs

    def load_preset_dataset(self, dataset_key: str = "rag_architecture") -> List[StepLog]:
        raw_items = SAMPLE_DATASETS.get(dataset_key, SAMPLE_DATASETS["rag_architecture"])
        return self.ingest_documents(raw_items)

    def ingest_documents(self, raw_items: List[Union[str, Dict[str, Any]]]) -> List[StepLog]:
        self.vector_db.clear()
        logs = []


        # Step 1: Data Collection
        t0 = time.time()
        collected_docs, log1 = self.collector.run(raw_items)
        log1.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log1)

        # Step 2: Data Loading
        t0 = time.time()
        loaded_docs, log2 = self.loader.run(collected_docs)
        log2.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log2)

        # Step 3: Text Splitting
        t0 = time.time()
        chunks, log3 = self.splitter.run(loaded_docs)
        log3.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log3)

        # Step 4: Data Cleaning
        t0 = time.time()
        cleaned_chunks, log4 = self.cleaner.run(chunks)
        log4.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log4)

        # Step 5: Metadata Extraction
        t0 = time.time()
        enriched_chunks, log5 = self.metadata_extractor.run(cleaned_chunks)
        log5.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log5)

        # Step 6: Embedding Generation
        t0 = time.time()
        embedded_chunks, log6 = self.embedder.run(enriched_chunks)
        log6.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log6)

        # Step 7: Store in Vector Database
        t0 = time.time()
        _, log7 = self.vector_db.run(embedded_chunks)
        log7.execution_time_ms = round((time.time() - t0) * 1000, 2)
        logs.append(log7)

        self.indexed_chunks = embedded_chunks
        self.indexing_logs = logs
        return logs

    def query(self, raw_user_query: str) -> RAGResponse:
        start_time = time.time()
        query_step_logs: List[StepLog] = []

        # Step 8: User Query
        t0 = time.time()
        query_str, log8 = self.query_handler.process_query(raw_user_query)
        log8.execution_time_ms = round((time.time() - t0) * 1000, 2)
        query_step_logs.append(log8)

        # Step 9: Query Understanding
        t0 = time.time()
        query_data, log9 = self.query_understanding.run(query_str)
        log9.execution_time_ms = round((time.time() - t0) * 1000, 2)
        query_step_logs.append(log9)

        # Step 10: First-Stage Retrieval
        t0 = time.time()
        dense_candidates, sparse_candidates, log10 = self.first_stage_retriever.retrieve(query_str, query_data)
        log10.execution_time_ms = round((time.time() - t0) * 1000, 2)
        query_step_logs.append(log10)

        # Step 11: Rank Fusion (RRF)
        t0 = time.time()
        fused_candidates, log11 = self.rank_fusion.run(dense_candidates, sparse_candidates, top_k=self.config["top_k_first_stage"])
        log11.execution_time_ms = round((time.time() - t0) * 1000, 2)
        query_step_logs.append(log11)

        # Step 12: Contextual Compression Retriever Pipeline (12.1 - 12.5)
        if self.config["enable_compression"]:
            t0 = time.time()
            compressed_context, compression_logs = self.compression_pipeline.run(query_str, fused_candidates)
            for clog in compression_logs:
                clog.execution_time_ms = round((time.time() - t0) * 1000 / len(compression_logs), 2)
                query_step_logs.append(clog)
        else:
            compressed_context = fused_candidates[:self.config["top_n_final"]]

        # Step 13: Build Context
        t0 = time.time()
        final_prompt, context_str, log13 = self.context_builder.build_prompt(query_str, compressed_context)
        log13.execution_time_ms = round((time.time() - t0) * 1000, 2)
        query_step_logs.append(log13)

        # Step 14: LLM Generation
        t0 = time.time()
        raw_answer, log14 = self.llm_generator.generate_response(query_str, context_str, compressed_context)
        log14.execution_time_ms = round((time.time() - t0) * 1000, 2)
        query_step_logs.append(log14)

        # Step 15: Post-Processing & Guardrails
        t0 = time.time()
        formatted_answer, citations, safety_report, log15 = self.post_processor.run(raw_answer, compressed_context)
        log15.execution_time_ms = round((time.time() - t0) * 1000, 2)
        query_step_logs.append(log15)

        total_time_ms = (time.time() - start_time) * 1000

        # Step 16: Evaluation & Metrics Loop
        t0 = time.time()
        eval_metrics, log16 = self.evaluator.run(query_str, formatted_answer, compressed_context, total_execution_time_ms=total_time_ms)
        log16.execution_time_ms = round((time.time() - t0) * 1000, 2)
        query_step_logs.append(log16)

        # Combine all indexing + query step logs
        all_step_logs = self.indexing_logs + query_step_logs

        return RAGResponse(
            query=query_str,
            answer=formatted_answer,
            context=compressed_context,
            citations=citations,
            step_logs=all_step_logs,
            metrics=eval_metrics,
            config_used=self.config
        )
