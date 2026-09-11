from typing import List, Tuple, Dict, Any
from .schema import Document, StepLog

class ContextBuilder:
    """
    Stage 13: BUILD CONTEXT
    - Concatenate final compressed chunks with document identifiers and metadata
    - Add system instructions and prompt template for grounding
    """

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant. Answer the user's question accurately using ONLY the provided context snippets below.\n"
            "If the answer cannot be found in the context, explicitly state that the context does not contain enough information.\n"
            "Cite your sources using [Doc X] references."
        )

    def build_prompt(self, query: str, context_docs: List[Document]) -> Tuple[str, str, StepLog]:
        formatted_context_blocks = []
        citations_index = []

        for idx, doc in enumerate(context_docs, start=1):
            source_name = doc.metadata.get("title", doc.metadata.get("source", f"Document_{doc.doc_id}"))
            category = doc.metadata.get("category", "General")
            formatted_context_blocks.append(
                f"[Doc {idx}] Source: {source_name} (Category: {category})\n{doc.page_content}"
            )
            citations_index.append({
                "doc_num": idx,
                "doc_id": doc.doc_id,
                "title": source_name,
                "score": doc.score,
                "metadata": doc.metadata
            })

        context_str = "\n\n".join(formatted_context_blocks)

        final_prompt = (
            f"=== SYSTEM INSTRUCTIONS ===\n{self.system_prompt}\n\n"
            f"=== RETRIEVED CONTEXT ===\n{context_str}\n\n"
            f"=== USER QUERY ===\n{query}\n\n"
            f"=== GROUNDED ANSWER ==="
        )

        log = StepLog(
            step_number="13",
            step_name="Build Context",
            description="Assembled final prompt template with concatenated context and citation markers.",
            input_summary=f"Formed context from {len(context_docs)} compressed chunks.",
            output_summary=f"Built prompt string ({len(final_prompt.split())} words).",
            details={
                "num_context_chunks": len(context_docs),
                "total_prompt_words": len(final_prompt.split()),
                "citations_map": citations_index
            }
        )
        return final_prompt, context_str, log
