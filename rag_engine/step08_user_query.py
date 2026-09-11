from typing import Dict, Any, Optional, Tuple
from .schema import StepLog

class UserQueryHandler:
    """
    Stage 8: USER QUERY
    - Process incoming user question
    - Sanitize query string, extract query intent metadata, and maintain conversation state
    """

    def process_query(self, raw_query: str, session_id: Optional[str] = None) -> Tuple[str, StepLog]:
        query = raw_query.strip()
        if not query:
            query = "What is Retrieval-Augmented Generation?"

        log = StepLog(
            step_number="8",
            step_name="User Query Processing",
            description="Sanitized user query and prepared for query understanding pipeline.",
            input_summary=f"Raw query input: '{raw_query}'",
            output_summary=f"Processed query: '{query}'",
            details={"session_id": session_id or "default_session", "query_length": len(query)}
        )
        return query, log
