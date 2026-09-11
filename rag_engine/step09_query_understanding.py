import re
from typing import List, Dict, Any, Tuple, Optional
from .schema import StepLog

class QueryUnderstanding:
    """
    Stage 9: QUERY UNDERSTANDING (Optional)
    - Query Rewriting: Clarify and expand prompt
    - Multi-Query Expansion: Generate semantic variations
    - HyDE (Hypothetical Document Embeddings): Synthesize hypothetical answer for embedding lookup
    - Self-Query Filter Extraction: Parse metadata constraints from natural language
    """

    def __init__(self, enable_hyde: bool = True, enable_multi_query: bool = True, enable_self_query: bool = True):
        self.enable_hyde = enable_hyde
        self.enable_multi_query = enable_multi_query
        self.enable_self_query = enable_self_query

    def rewrite_query(self, query: str) -> str:
        # Basic acronym expansion and clarification
        rewritten = query
        expansions = {
            "rag": "retrieval augmented generation",
            "llm": "large language model",
            "vec db": "vector database",
            "rrf": "reciprocal rank fusion",
        }
        for abbr, full in expansions.items():
            pattern = re.compile(rf'\b{abbr}\b', re.IGNORECASE)
            if pattern.search(rewritten):
                rewritten = pattern.sub(f"{abbr} ({full})", rewritten)
        return rewritten

    def generate_multi_queries(self, query: str) -> List[str]:
        queries = [query]
        if not self.enable_multi_query:
            return queries

        # Synthetic multi-query expansion rule
        q_low = query.lower()
        if "how" in q_low or "what" in q_low or "why" in q_low:
            queries.append(f"Key concepts and definitions related to: {query}")
            queries.append(f"Best practices and components for: {query}")
        else:
            queries.append(f"Detailed overview of {query}")
            queries.append(f"Architecture and pipeline steps of {query}")

        return list(dict.fromkeys(queries))

    def generate_hyde_document(self, query: str) -> str:
        if not self.enable_hyde:
            return query
        # Generate a hypothetical answer passage tailored for embedding match
        return f"Hypothetical Document answering '{query}': Retrieval Augmented Generation involves document collection, chunking, embedding generation, vector database search, hybrid retrieval, contextual compression reranking, context assembly, and LLM answer generation."

    def extract_self_query_filters(self, query: str) -> Tuple[str, Dict[str, Any]]:
        filters = {}
        cleaned_q = query

        # Parse category intent heuristics with word boundary checking
        q_lower = query.lower()
        if not self.enable_self_query:
            return cleaned_q, filters

        if re.search(r'\b(finance|revenue|financial|accounting|budget)\b', q_lower):
            filters["category"] = "finance"
        elif re.search(r'\b(tech|rag|ai|machine learning|deep learning)\b', q_lower):
            filters["category"] = "ai_tech"
        elif re.search(r'\b(medical|healthcare|clinical|patient)\b', q_lower):
            filters["category"] = "healthcare"

        # Parse year filter e.g. "after 2022" or "in 2024"
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match and self.enable_self_query:
            filters["date"] = year_match.group(1)

        return cleaned_q, filters

    def run(self, query: str) -> Tuple[Dict[str, Any], StepLog]:
        rewritten_q = self.rewrite_query(query)
        expanded_queries = self.generate_multi_queries(query)
        hyde_doc = self.generate_hyde_document(query)
        cleaned_query, metadata_filters = self.extract_self_query_filters(query)

        output_data = {
            "original_query": query,
            "rewritten_query": rewritten_q,
            "expanded_queries": expanded_queries,
            "hyde_document": hyde_doc,
            "metadata_filters": metadata_filters
        }

        log = StepLog(
            step_number="9",
            step_name="Query Understanding",
            description="Expanded query variants, synthesized HyDE document, and extracted metadata filters.",
            input_summary=f"Query: '{query}'",
            output_summary=f"Generated {len(expanded_queries)} query variations & HyDE document.",
            details=output_data
        )
        return output_data, log
