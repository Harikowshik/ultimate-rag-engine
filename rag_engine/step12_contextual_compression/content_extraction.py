import re
from typing import List, Tuple
from ..schema import Document, StepLog

class ContentExtractor:
    """
    Step 12.4: CONTENT EXTRACTION
    - LLMChainExtractor: Extracts only the specific sentences or passages within a chunk that directly answer/address the user's query.
    - Saves tokens and removes unhelpful background sentences.
    """

    def __init__(self, min_passage_len: int = 20):
        self.min_passage_len = min_passage_len

    def extract_relevant_passages(self, query: str, doc_text: str) -> str:
        stopwords = {"what", "are", "the", "main", "key", "points", "explained", "in", "this", "document", "is", "a", "an", "of", "and", "or", "to", "for", "with", "on", "at", "by", "from", "how", "does", "do"}
        q_terms = set(w for w in re.findall(r'\b\w+\b', query.lower()) if w not in stopwords and len(w) > 2)
        sentences = re.split(r'(?<=[.!?])\s+', doc_text)

        if not q_terms or len(sentences) <= 2:
            return doc_text

        extracted_sentences = []
        for sent in sentences:
            s_terms = set(re.findall(r'\b\w+\b', sent.lower()))
            overlap = q_terms.intersection(s_terms)
            if len(overlap) >= 1:
                extracted_sentences.append(sent)

        if extracted_sentences:
            return " ".join(extracted_sentences)
        return doc_text # Fallback to original text if extraction is too aggressive

    def extract(self, query: str, docs: List[Document]) -> Tuple[List[Document], StepLog]:
        compressed_docs = []
        original_tokens = sum(len(d.page_content.split()) for d in docs)

        for doc in docs:
            extracted_text = self.extract_relevant_passages(query, doc.page_content)
            doc_copy = Document(
                doc_id=doc.doc_id,
                page_content=extracted_text,
                metadata={
                    **doc.metadata,
                    "extracted": True,
                    "orig_char_count": len(doc.page_content),
                    "extracted_char_count": len(extracted_text)
                },
                score=doc.score,
                rank=doc.rank,
                embedding=doc.embedding,
                sparse_vector=doc.sparse_vector
            )
            compressed_docs.append(doc_copy)

        compressed_tokens = sum(len(d.page_content.split()) for d in compressed_docs)
        token_savings_pct = round((1.0 - (compressed_tokens / max(1, original_tokens))) * 100.0, 1)

        log = StepLog(
            step_number="12.4",
            step_name="Content Extraction (LLMChainExtractor)",
            description=f"Extracted query-relevant passages, saving {token_savings_pct}% tokens.",
            input_summary=f"Processed {len(docs)} documents ({original_tokens} words).",
            output_summary=f"Compressed to {compressed_tokens} words ({token_savings_pct}% reduction).",
            details={
                "original_words": original_tokens,
                "compressed_words": compressed_tokens,
                "token_savings_pct": token_savings_pct
            }
        )
        return compressed_docs, log
