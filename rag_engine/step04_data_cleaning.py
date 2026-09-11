import re
import hashlib
from typing import List, Dict, Any, Set
from .schema import Document, StepLog

class DataCleaner:
    """
    Stage 4: DATA CLEANING
    - Remove noise, boilerplate, excessive spaces, weird control characters
    - Deduplicate identical or near-identical text chunks
    - Text normalization (case handling, unicode cleanup)
    """

    def __init__(self, remove_duplicates: bool = True, normalize_whitespace: bool = True):
        self.remove_duplicates = remove_duplicates
        self.normalize_whitespace = normalize_whitespace

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        # Remove control characters except newlines/tabs
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # Normalize whitespace if requested
        if self.normalize_whitespace:
            # Replace multiple non-newline spaces with single space
            text = re.sub(r'[ \t]+', ' ', text)
            # Replace 3 or more consecutive newlines with max 2 newlines
            text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _compute_hash(self, text: str) -> str:
        # Normalized hash for deduplication ignoring case and punctuation
        norm = re.sub(r'\W+', '', text.lower())
        return hashlib.md5(norm.encode('utf-8')).hexdigest()

    def run(self, chunks: List[Document]) -> tuple[List[Document], StepLog]:
        cleaned_chunks = []
        seen_hashes: Set[str] = set()
        removed_duplicates = 0
        original_count = len(chunks)

        for chunk in chunks:
            cleaned_content = self.clean_text(chunk.page_content)
            if not cleaned_content or len(cleaned_content) < 10:
                continue # filter out empty or trivial noise chunks

            if self.remove_duplicates:
                h = self._compute_hash(cleaned_content)
                if h in seen_hashes:
                    removed_duplicates += 1
                    continue
                seen_hashes.add(h)

            chunk.page_content = cleaned_content
            chunk.metadata["cleaned"] = True
            cleaned_chunks.append(chunk)

        log = StepLog(
            step_number="4",
            step_name="Data Cleaning",
            description="Cleaned noise, normalized whitespace, and deduplicated text chunks.",
            input_summary=f"Input {original_count} chunks.",
            output_summary=f"Output {len(cleaned_chunks)} clean chunks ({removed_duplicates} duplicates removed).",
            details={
                "original_count": original_count,
                "cleaned_count": len(cleaned_chunks),
                "duplicates_removed": removed_duplicates
            }
        )
        return cleaned_chunks, log
