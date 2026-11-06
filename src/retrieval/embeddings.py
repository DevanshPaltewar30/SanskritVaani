"""
CPU-optimized Sanskrit and Multilingual Embeddings wrapper.
"""

import os
import time
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.nlp.sanskrit_processor import SanskritProcessor
from src.config import EMBEDDING_DIR


class SanskritEmbeddings:
    """
    Embedding generator optimized for Sanskrit and multilingual queries.
    Strictly configured for CPU execution.
    """

    def __init__(self, model_path_or_name: str = EMBEDDING_DIR, device: str = "cpu"):
        self.device = device
        self.processor = SanskritProcessor()
        
        print(f"[Embeddings] Loading model from: {model_path_or_name} (device={self.device})...")
        t0 = time.time()
        self.model = SentenceTransformer(model_path_or_name, device=self.device)
        self.load_time = time.time() - t0
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embeddings] Loaded in {self.load_time:.2f}s (dim={self.embedding_dim})")

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Embeds a list of Sanskrit document chunks.
        Normalizes embeddings to unit length for exact Cosine Similarity.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        clean_texts = [self.processor.normalize_devanagari(t) for t in texts]
        embeddings = self.model.encode(
            clean_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a user query. If the query is in Roman-transliterated Sanskrit,
        it converts it to Devanagari to match Sanskrit passages semantically.
        """
        script = self.processor.detect_script(query)
        clean_query = query.strip()

        # If Roman transliterated Sanskrit, create enhanced Devanagari query
        if script in ["roman_sanskrit", "iast_transliterated"]:
            deva_query = self.processor.roman_to_devanagari(clean_query)
            # Create a combined hybrid query for highest recall
            hybrid_query = f"{clean_query} {deva_query}"
            emb = self.model.encode([hybrid_query], normalize_embeddings=True, convert_to_numpy=True)
            return emb[0].astype(np.float32)

        emb = self.model.encode([clean_query], normalize_embeddings=True, convert_to_numpy=True)
        return emb[0].astype(np.float32)

