from typing import List, Tuple
from ..schema import Document, StepLog

class ContextReorderer:
    """
    Step 12.5: CONTEXT REORDERING
    - LongContextReorder: Solves the 'Lost-in-the-Middle' phenomenon where LLMs pay most attention to tokens at the very beginning and very end of the prompt window.
    - Reorders documents so the highest-scoring documents are placed at the Start and End, and lower-scoring items in the middle.
    """

    def reorder(self, docs: List[Document]) -> Tuple[List[Document], StepLog]:
        if len(docs) <= 2:
            return docs, StepLog("12.5", "Context Reordering (LongContextReorder)", "Fewer than 3 docs; order preserved.", f"{len(docs)} docs", f"{len(docs)} docs", {})

        # Ensure sorted by score descending first
        sorted_docs = sorted(docs, key=lambda x: x.score, reverse=True)

        reordered = []
        # Distribute items: highest score at index 0, 2nd highest at last index, 3rd highest at index 1, 4th highest at 2nd to last, etc.
        for idx, doc in enumerate(sorted_docs):
            if idx % 2 == 0:
                reordered.append(doc)
            else:
                reordered.insert(0, doc)

        log = StepLog(
            step_number="12.5",
            step_name="Context Reordering (LongContextReorder)",
            description="Reordered documents placing top relevant candidates at the beginning and end to solve lost-in-the-middle.",
            input_summary=f"Input {len(docs)} documents.",
            output_summary=f"Reordered context sequence: highest score items placed at head and tail.",
            details={
                "input_order_scores": [round(d.score, 3) for d in sorted_docs],
                "reordered_scores": [round(d.score, 3) for d in reordered]
            }
        )
        return reordered, log
