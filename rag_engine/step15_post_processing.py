import re
from typing import List, Tuple, Dict, Any
from .schema import Document, StepLog

class PostProcessor:
    """
    Stage 15: POST-PROCESSING
    - Format output (markdown, tables, bullet formatting)
    - Verify and map inline citations to original source metadata
    - Guardrails & Safety checks: Groundedness & Hallucination verification
    """

    def __init__(self, check_hallucinations: bool = True):
        self.check_hallucinations = check_hallucinations

    def format_citations(self, raw_answer: str, context_docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
        citations = []
        doc_map = {idx: doc for idx, doc in enumerate(context_docs, start=1)}

        # Find doc citations like [Doc 1], [Doc 2]
        matches = re.findall(r'\[Doc\s+(\d+)\]', raw_answer)
        seen_docs = set()

        for doc_num_str in matches:
            doc_num = int(doc_num_str)
            if doc_num in doc_map and doc_num not in seen_docs:
                seen_docs.add(doc_num)
                doc = doc_map[doc_num]
                citations.append({
                    "citation_marker": f"[Doc {doc_num}]",
                    "doc_id": doc.doc_id,
                    "title": doc.metadata.get("title", f"Document {doc_num}"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "score": round(doc.score, 4),
                    "snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                })

        return raw_answer, citations

    def verify_groundedness(self, answer: str, context_docs: List[Document]) -> Tuple[bool, float]:
        if not context_docs:
            return False, 0.0

        ans_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer.lower()))
        ctx_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', " ".join(d.page_content for d in context_docs).lower()))

        if not ans_words:
            return True, 1.0

        overlap = ans_words.intersection(ctx_words)
        groundedness_score = round(len(overlap) / len(ans_words), 2)
        is_safe = groundedness_score >= 0.40
        return is_safe, groundedness_score

    def run(self, raw_answer: str, context_docs: List[Document]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], StepLog]:
        formatted_answer, citations = self.format_citations(raw_answer, context_docs)
        is_grounded, grounded_score = self.verify_groundedness(formatted_answer, context_docs)

        safety_report = {
            "is_grounded": is_grounded,
            "groundedness_score": grounded_score,
            "pass_guardrails": is_grounded,
            "total_citations": len(citations)
        }

        log = StepLog(
            step_number="15",
            step_name="Post-Processing & Guardrails",
            description="Formatted citations, checked guardrails, and computed hallucination score.",
            input_summary=f"Processed raw LLM output with {len(citations)} citations.",
            output_summary=f"Groundedness score: {grounded_score} ({'Passed' if is_grounded else 'Warning'}).",
            details=safety_report
        )
        return formatted_answer, citations, safety_report, log
