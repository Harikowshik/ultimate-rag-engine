import json
import csv
import io
import os
from typing import List, Dict, Any, Union
from .schema import Document, StepLog

class DataLoader:
    """
    Stage 2: DATA LOADING
    - Load raw data from files/strings
    - Handle different file formats (PDF, HTML, Markdown, CSV, JSON, TXT, DB records)
    """

    def parse_pdf_bytes(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> List[Document]:
        """
        Parses PDF file bytes using pypdf to extract page text and metadata.
        """
        documents = []
        try:
            import pypdf
            pdf_file = io.BytesIO(pdf_bytes)
            reader = pypdf.PdfReader(pdf_file)
            total_pages = len(reader.pages)

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(Document(
                        doc_id=f"{filename}_page_{page_num}",
                        page_content=text.strip(),
                        metadata={
                            "source": filename,
                            "title": filename,
                            "page_number": page_num,
                            "total_pages": total_pages,
                            "format": "pdf"
                        }
                    ))
        except Exception as e:
            # Fallback text decoding if pypdf fails
            raw_str = pdf_bytes.decode("utf-8", errors="ignore")
            documents.append(Document(
                doc_id=f"{filename}_raw",
                page_content=raw_str,
                metadata={"source": filename, "format": "pdf", "error": str(e)}
            ))

        return documents

    def parse_file_content(self, content: str, file_type: str, filename: str = "uploaded_file") -> List[Document]:
        file_type = file_type.lower().strip(".")
        documents = []

        if file_type in ["json"]:
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        if isinstance(item, dict):
                            text = item.get("text", item.get("content", json.dumps(item)))
                            meta = item.get("metadata", {})
                            meta["source"] = filename
                            meta["format"] = "json"
                            documents.append(Document(doc_id=f"{filename}_{idx}", page_content=text, metadata=meta))
                        else:
                            documents.append(Document(doc_id=f"{filename}_{idx}", page_content=str(item), metadata={"source": filename, "format": "json"}))
                elif isinstance(data, dict):
                    text = data.get("text", data.get("content", json.dumps(data)))
                    documents.append(Document(doc_id=f"{filename}_0", page_content=text, metadata={"source": filename, "format": "json"}))
            except Exception as e:
                documents.append(Document(doc_id=f"{filename}_0", page_content=content, metadata={"source": filename, "format": "raw", "error": str(e)}))

        elif file_type in ["csv"]:
            try:
                reader = csv.DictReader(io.StringIO(content))
                for idx, row in enumerate(reader):
                    row_text = " | ".join([f"{k}: {v}" for k, v in row.items() if v])
                    documents.append(Document(doc_id=f"{filename}_row_{idx}", page_content=row_text, metadata={"source": filename, "format": "csv", "row_index": idx}))
            except Exception:
                documents.append(Document(doc_id=f"{filename}_0", page_content=content, metadata={"source": filename, "format": "csv"}))

        elif file_type in ["html", "htm"]:
            import re
            clean_text = re.sub(r'<style.*?>.*?</style>', '', content, flags=re.DOTALL)
            clean_text = re.sub(r'<script.*?>.*?</script>', '', clean_text, flags=re.DOTALL)
            clean_text = re.sub(r'<[^>]+>', ' ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            documents.append(Document(doc_id=f"{filename}_html", page_content=clean_text, metadata={"source": filename, "format": "html"}))

        elif file_type in ["md", "markdown"]:
            documents.append(Document(doc_id=f"{filename}_md", page_content=content, metadata={"source": filename, "format": "markdown"}))

        else:
            documents.append(Document(doc_id=f"{filename}_txt", page_content=content, metadata={"source": filename, "format": file_type or "text"}))

        return documents


    def run(self, input_docs: List[Document]) -> tuple[List[Document], StepLog]:
        # Standardize metadata format tags
        loaded_docs = []
        formats_detected = set()

        for doc in input_docs:
            fmt = doc.metadata.get("format", doc.metadata.get("source_type", "text"))
            formats_detected.add(fmt)
            doc.metadata["loaded"] = True
            doc.metadata["char_count"] = len(doc.page_content)
            loaded_docs.append(doc)

        log = StepLog(
            step_number="2",
            step_name="Data Loading",
            description="Loaded and parsed raw content into structured Document objects.",
            input_summary=f"Processed {len(input_docs)} documents.",
            output_summary=f"Successfully loaded {len(loaded_docs)} parsed documents.",
            details={"formats_detected": list(formats_detected), "total_chars": sum(d.metadata.get("char_count", 0) for d in loaded_docs)}
        )
        return loaded_docs, log
