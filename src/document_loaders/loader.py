"""
Document loaders for Sanskrit PDF and TXT files.
"""

import os
from typing import List, Dict, Any
import pypdf
from src.nlp.sanskrit_processor import SanskritProcessor


class SanskritDocumentLoader:
    """Loads and extracts text from Sanskrit documents in PDF and TXT formats."""

    def __init__(self):
        self.processor = SanskritProcessor()

    def load_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads a plain text Sanskrit file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        normalized_content = self.processor.normalize_devanagari(content)

        return [{
            "text": normalized_content,
            "metadata": {
                "source": filename,
                "file_path": file_path,
                "format": "txt",
                "page": 1,
                "total_pages": 1
            }
        }]

    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads a PDF document and extracts Sanskrit text page by page."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        documents = []

        reader = pypdf.PdfReader(file_path)
        total_pages = len(reader.pages)

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            normalized_text = self.processor.normalize_devanagari(raw_text)

            if normalized_text:
                documents.append({
                    "text": normalized_text,
                    "metadata": {
                        "source": filename,
                        "file_path": file_path,
                        "format": "pdf",
                        "page": page_num,
                        "total_pages": total_pages
                    }
                })

        return documents

    def load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads a single file (PDF or TXT)."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.load_pdf(file_path)
        elif ext in [".txt", ".text"]:
            return self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Only .pdf and .txt are supported.")

    def load_directory(self, dir_path: str, extensions: List[str] = [".pdf", ".txt"]) -> List[Dict[str, Any]]:
        """Loads all supported documents from a directory."""
        if not os.path.exists(dir_path):
            return []

        all_docs = []
        for root, _, files in os.walk(dir_path):
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    file_path = os.path.join(root, file)
                    try:
                        docs = self.load_file(file_path)
                        all_docs.extend(docs)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        return all_docs

