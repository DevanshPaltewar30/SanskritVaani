"""
JnanaSetu – End-to-End Sanskrit RAG Engine.
Coordinates document ingestion, chunking, vector indexing, retrieval,
and local CPU inference with conversational memory.
"""

import os
import time
import psutil
from typing import List, Dict, Any, Optional, Generator
from src.config import (
    CORPUS_DIR,
    INDEX_DIR,
    DEFAULT_TOP_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    SCORE_THRESHOLD
)
from src.nlp.sanskrit_processor import SanskritProcessor
from src.nlp.text_splitter import SanskritTextSplitter
from src.document_loaders.loader import SanskritDocumentLoader
from src.retrieval.embeddings import SanskritEmbeddings
from src.retrieval.vector_store import SanskritFAISSStore
from src.llm.cpu_inference import CPUInferenceEngine
from src.llm.prompts import build_rag_prompt


class JnanaSetuEngine:
    """
    Main orchestration engine for JnanaSetu Sanskrit Knowledge Assistant.
    """

    def __init__(
        self,
        embeddings: Optional[SanskritEmbeddings] = None,
        vector_store: Optional[SanskritFAISSStore] = None,
        llm_engine: Optional[CPUInferenceEngine] = None
    ):
        self.processor = SanskritProcessor()
        self.loader = SanskritDocumentLoader()
        self.splitter = SanskritTextSplitter()
        
        self.embeddings = embeddings or SanskritEmbeddings()
        self.vector_store = vector_store or SanskritFAISSStore(embeddings=self.embeddings)
        self.llm = llm_engine or CPUInferenceEngine()
        
        self.chat_history: List[Dict[str, str]] = []
        
        # Load persisted index or build automatically
        self._initialize_index()

    def _initialize_index(self):
        """Loads index from disk or builds from initial corpus if not found."""
        if not self.vector_store.load():
            print("[Engine] Index not found on disk. Ingesting initial Sanskrit corpus...")
            self.index_corpus(CORPUS_DIR)

    def index_corpus(self, corpus_dir: str = CORPUS_DIR, force: bool = False):
        """Ingests and indexes all documents from a directory."""
        if force:
            self.vector_store.clear()

        raw_docs = self.loader.load_directory(corpus_dir, extensions=[".pdf", ".txt"])
        if not raw_docs:
            print(f"[Engine] No documents found in {corpus_dir}")
            return

        chunked_docs = self.splitter.split_documents(raw_docs)
        self.vector_store.add_documents(chunked_docs)
        self.vector_store.save()
        print(f"[Engine] Indexed {len(chunked_docs)} chunks from {len(raw_docs)} files.")

    def add_document(self, file_path: str):
        """Ingests a newly uploaded file and updates the FAISS index."""
        raw_docs = self.loader.load_file(file_path)
        chunked_docs = self.splitter.split_documents(raw_docs)
        self.vector_store.add_documents(chunked_docs)
        self.vector_store.save()
        return len(chunked_docs)

    def format_history_context(self) -> str:
        """Formats recent chat history turns for conversational context."""
        if not self.chat_history:
            return ""
        
        # Use last 2-3 turns
        recent_turns = self.chat_history[-3:]
        history_lines = []
        for turn in recent_turns:
            history_lines.append(f"User: {turn['question']}")
            history_lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(history_lines)

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> Dict[str, Any]:
        """
        Executes complete RAG pipeline on CPU:
        1. Query Preprocessing & Transliteration
        2. FAISS Similarity Retrieval
        3. Contextual Prompt Assembly
        4. CPU-Only LLM Generation
        5. Performance Telemetry Tracking
        """
        process = psutil.Process()
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024 * 1024)

        # Step 1: Retrieval
        t_ret_start = time.time()
        retrieved_results = self.vector_store.similarity_search(
            query=question,
            top_k=top_k,
            score_threshold=SCORE_THRESHOLD
        )
        retrieval_time = time.time() - t_ret_start

        # Check if relevant passages exist
        has_confident_match = any(r["is_confident"] for r in retrieved_results)
        
        if not retrieved_results or not has_confident_match:
            no_info_answer = (
                "The provided Sanskrit documents do not contain information to answer this question. "
                "(उपलब्धग्रन्थेषु एतस्य प्रश्नस्य उत्तरं न लभ्यते।)"
            )
            return {
                "answer": no_info_answer,
                "sources": retrieved_results,
                "retrieval_time": retrieval_time,
                "generation_time": 0.01,
                "total_time": retrieval_time + 0.01,
                "tokens_per_second": 0.0,
                "tokens_generated": 0,
                "cpu_utilization": psutil.cpu_percent(interval=None),
                "ram_usage_mb": process.memory_info().rss / (1024 * 1024),
                "query_script": self.processor.detect_script(question)
            }

        # Step 2: Build Context & Prompt
        context_parts = []
        for i, item in enumerate(retrieved_results, start=1):
            src = item["metadata"].get("source", "Unknown")
            pg = item["metadata"].get("page", 1)
            chunk = item["chunk"]
            context_parts.append(f"[Passage {i} - {src} (Page {pg})]:\n{chunk}")
        
        context_str = "\n\n".join(context_parts)
        history_context = self.format_history_context()
        prompt = build_rag_prompt(context=context_str, question=question, history_context=history_context)

        # Step 3: LLM Generation on CPU
        gen_result = self.llm.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = process.memory_info().rss / (1024 * 1024)

        answer_text = gen_result["text"]

        # Append to conversational memory
        self.chat_history.append({
            "question": question,
            "answer": answer_text
        })

        return {
            "answer": answer_text,
            "sources": retrieved_results,
            "retrieval_time": retrieval_time,
            "generation_time": gen_result["latency"],
            "total_time": retrieval_time + gen_result["latency"],
            "tokens_per_second": gen_result["tokens_per_second"],
            "tokens_generated": gen_result["tokens"],
            "cpu_utilization": round(cpu_after, 1),
            "ram_usage_mb": round(mem_after, 1),
            "query_script": self.processor.detect_script(question)
        }

    def clear_history(self):
        """Clears the multi-turn conversational history."""
        self.chat_history = []

