import re
from typing import List, Dict, Any, Optional
from .schema import Document, StepLog

class TextSplitter:
    """
    Stage 3: TEXT SPLITTING (CHUNKING)
    - Split documents into manageable chunks
    - Configurable chunk size and chunk overlap
    - Strategies: Recursive character splitting, fixed size splitting, sentence/semantic splitting
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, strategy: str = "recursive"):
        self.chunk_size = max(50, chunk_size)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size - 10)
        self.strategy = strategy.lower()

    def split_document(self, doc: Document) -> List[Document]:
        text = doc.page_content
        if not text or len(text) <= self.chunk_size:
            chunk_doc = Document(
                doc_id=f"{doc.doc_id}_c0",
                page_content=text,
                metadata={**doc.metadata, "parent_id": doc.doc_id, "chunk_index": 0, "total_chunks": 1}
            )
            return [chunk_doc]

        chunks_text = []

        if self.strategy == "fixed":
            chunks_text = self._fixed_split(text)
        elif self.strategy == "sentence" or self.strategy == "semantic":
            chunks_text = self._sentence_split(text)
        else: # Default: recursive character splitting
            chunks_text = self._recursive_split(text)

        chunks = []
        total = len(chunks_text)
        for idx, chunk_str in enumerate(chunks_text):
            c_meta = {
                **doc.metadata,
                "parent_id": doc.doc_id,
                "chunk_index": idx,
                "total_chunks": total,
                "chunk_size": len(chunk_str),
                "strategy": self.strategy
            }
            c_doc = Document(
                doc_id=f"{doc.doc_id}_c{idx}",
                page_content=chunk_str.strip(),
                metadata=c_meta
            )
            chunks.append(c_doc)

        return chunks

    def _fixed_split(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)
        step = self.chunk_size - self.chunk_overlap

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start += step
        return chunks

    def _recursive_split(self, text: str, separators: List[str] = None) -> List[str]:
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        def _split_text(txt: str, seps: List[str]) -> List[str]:
            if len(txt) <= self.chunk_size:
                return [txt]
            if not seps:
                return self._fixed_split(txt)

            sep = seps[0]
            new_seps = seps[1:]

            if sep == "":
                splits = list(txt)
            else:
                splits = txt.split(sep)

            final_chunks = []
            current_chunk = ""

            for s in splits:
                piece = s + (sep if sep != "" else "")
                if len(current_chunk) + len(piece) <= self.chunk_size:
                    current_chunk += piece
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                    if len(piece) > self.chunk_size:
                        # Sub-split long piece recursively with finer separator
                        sub_chunks = _split_text(piece, new_seps)
                        final_chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = piece

            if current_chunk:
                final_chunks.append(current_chunk.strip())

            # Apply overlap across chunks
            return self._apply_overlap(final_chunks)

        return _split_text(text, separators)

    def _sentence_split(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) + 1 <= self.chunk_size:
                current = f"{current} {sent}".strip()
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)

        return self._apply_overlap(chunks)

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped = []
        for i in range(len(chunks)):
            current = chunks[i].strip()
            if i > 0 and self.chunk_overlap > 0:
                prev_text = chunks[i-1].strip()
                # Extract word-aligned overlap from previous chunk
                overlap_text = prev_text[-self.chunk_overlap:] if len(prev_text) > self.chunk_overlap else prev_text
                # Align to full word boundary
                space_idx = overlap_text.find(' ')
                if space_idx != -1 and space_idx < len(overlap_text) - 1:
                    overlap_text = overlap_text[space_idx+1:]
                if overlap_text and not current.startswith(overlap_text):
                    current = f"{overlap_text} {current}"
            overlapped.append(current)
        return overlapped

    def run(self, docs: List[Document]) -> tuple[List[Document], StepLog]:
        all_chunks = []
        for doc in docs:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)

        log = StepLog(
            step_number="3",
            step_name="Text Splitting (Chunking)",
            description=f"Split input documents into manageable chunks using strategy='{self.strategy}'.",
            input_summary=f"{len(docs)} full documents.",
            output_summary=f"Generated {len(all_chunks)} chunks (avg size ~{sum(len(c.page_content) for c in all_chunks)//max(1, len(all_chunks))} chars).",
            details={
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "strategy": self.strategy,
                "num_input_docs": len(docs),
                "num_output_chunks": len(all_chunks)
            }
        )
        return all_chunks, log
