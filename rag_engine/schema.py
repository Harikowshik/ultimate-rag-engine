from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union

@dataclass
class Document:
    doc_id: str
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0
    embedding: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "page_content": self.page_content,
            "metadata": self.metadata,
            "score": round(self.score, 4),
            "rank": self.rank,
        }

@dataclass
class StepLog:
    step_number: str
    step_name: str
    description: str
    input_summary: str
    output_summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

@dataclass
class RAGResponse:
    query: str
    answer: str
    context: List[Document]
    citations: List[Dict[str, Any]]
    step_logs: List[StepLog]
    metrics: Dict[str, Any]
    config_used: Dict[str, Any]
