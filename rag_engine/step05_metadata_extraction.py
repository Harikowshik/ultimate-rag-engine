import re
from typing import List, Dict, Any
from .schema import Document, StepLog

class MetadataExtractor:
    """
    Stage 5: METADATA EXTRACTION (Optional)
    - Extract title, section, author, dates, categories, keywords from content
    - Enables filtered and self-query retrieval downstream
    """

    def extract_metadata(self, doc: Document) -> Document:
        text = doc.page_content
        meta = doc.metadata

        # Extract Heading/Title candidate
        heading_match = re.search(r'^(?:#+|\b(?:Title|Header|Section|Chapter)\b:?)\s*(.+)$', text, re.MULTILINE | re.IGNORECASE)
        if heading_match and "title" not in meta:
            meta["title"] = heading_match.group(1).strip()
        elif "title" not in meta:
            # First non-empty line as title fallback
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            meta["title"] = lines[0][:60] if lines else "Untitled Chunk"

        # Extract dates
        date_match = re.search(r'\b(20\d{2}|19\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b|\b(20\d{2})\b', text)
        if date_match and "date" not in meta:
            meta["date"] = date_match.group(0)

        # Keyword Extraction (TF-IDF / Frequency based rule)
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', text)]
        stopwords = {"with", "that", "this", "from", "have", "they", "were", "which", "their", "there", "about", "would", "these", "other", "into", "more", "some", "such", "than", "them", "been"}
        freq = {}
        for w in words:
            if w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        meta["keywords"] = [k for k, _ in sorted_keywords[:5]]

        # Category tagging heuristic
        cat = "general"
        t_low = text.lower()
        if any(k in t_low for k in ["rag", "embedding", "vector", "retrieval", "llm", "ai", "model"]):
            cat = "ai_tech"
        elif any(k in t_low for k in ["revenue", "profit", "finance", "quarter", "growth"]):
            cat = "finance"
        elif any(k in t_low for k in ["patient", "medical", "clinical", "health"]):
            cat = "healthcare"
        meta["category"] = cat

        doc.metadata = meta
        return doc

    def run(self, chunks: List[Document]) -> tuple[List[Document], StepLog]:
        extracted_chunks = [self.extract_metadata(c) for c in chunks]

        categories_found = list({c.metadata.get("category", "general") for c in extracted_chunks})
        log = StepLog(
            step_number="5",
            step_name="Metadata Extraction",
            description="Extracted titles, dates, categories, and top keywords for structured filtering.",
            input_summary=f"Processed {len(chunks)} text chunks.",
            output_summary=f"Enriched metadata across {len(extracted_chunks)} chunks.",
            details={
                "categories_detected": categories_found,
                "sample_keywords": extracted_chunks[0].metadata.get("keywords", []) if extracted_chunks else []
            }
        )
        return extracted_chunks, log
