"""
Sanskrit-aware text splitter that preserves sentence and verse boundaries.
"""

from typing import List, Dict, Any
from src.nlp.sanskrit_processor import SanskritProcessor


class SanskritTextSplitter:
    """
    Intelligent chunking for Sanskrit literature that strictly preserves
    sentence (।) and verse (॥) integrity.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processor = SanskritProcessor()

    def split_text(self, text: str) -> List[str]:
        """Splits a single raw Sanskrit text string into chunks."""
        text = self.processor.normalize_devanagari(text)
        if not text:
            return []

        sentences = self.processor.split_sentences(text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If a single sentence exceeds chunk_size, split by whitespace
            if sentence_len > self.chunk_size:
                words = sentence.split(" ")
                sub_chunk = []
                sub_len = 0
                for word in words:
                    if sub_len + len(word) + 1 > self.chunk_size and sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = [word]
                        sub_len = len(word)
                    else:
                        sub_chunk.append(word)
                        sub_len += len(word) + 1
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                continue

            if current_len + sentence_len > self.chunk_size and current_chunk:
                chunk_str = " ".join(current_chunk)
                chunks.append(chunk_str)
                
                # Apply overlap by keeping recent sentences
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                        
                current_chunk = overlap_chunk + [sentence]
                current_len = sum(len(s) for s in current_chunk) + len(current_chunk)
            else:
                current_chunk.append(sentence)
                current_len += sentence_len + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits a list of document objects containing 'text' and 'metadata'.
        Returns chunked documents with updated metadata (chunk_id, total_chunks, start_idx).
        """
        chunked_docs = []
        
        for doc in documents:
            raw_text = doc.get("text", "")
            base_metadata = doc.get("metadata", {}).copy()
            
            chunks = self.split_text(raw_text)
            for idx, chunk in enumerate(chunks):
                meta = base_metadata.copy()
                meta["chunk_id"] = idx
                meta["total_chunks"] = len(chunks)
                meta["char_length"] = len(chunk)
                
                chunked_docs.append({
                    "text": chunk,
                    "metadata": meta
                })
                
        return chunked_docs

