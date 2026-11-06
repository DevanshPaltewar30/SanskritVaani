"""
Comprehensive unit and integration test suite for JnanaSetu.
Uses Python's standard unittest framework (zero external test runner dependencies).
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp.sanskrit_processor import SanskritProcessor
from src.nlp.text_splitter import SanskritTextSplitter
from src.document_loaders.loader import SanskritDocumentLoader
from src.retrieval.embeddings import SanskritEmbeddings
from src.retrieval.vector_store import SanskritFAISSStore
from src.llm.prompts import build_rag_prompt, SYSTEM_PROMPT
from src.config import CORPUS_DIR, SCORE_THRESHOLD


class TestSanskritProcessor(unittest.TestCase):
    """Tests for Sanskrit Unicode normalization, transliteration, and segmentation."""

    def setUp(self):
        self.processor = SanskritProcessor()

    def test_unicode_normalization(self):
        # ASCII pipe conversion to Devanagari danda
        text = "सत्यं वद | धर्मं चर ||"
        normalized = self.processor.normalize_devanagari(text)
        self.assertIn("।", normalized)
        self.assertIn("॥", normalized)
        self.assertNotIn("|", normalized)

    def test_script_detection(self):
        deva = "मूर्खभृत्यस्य कथा"
        eng = "What was the servant doing?"
        translit = "chaturasya kalidasasya katha"
        
        self.assertEqual(self.processor.detect_script(deva), "devanagari")
        self.assertEqual(self.processor.detect_script(eng), "english_or_latin")
        self.assertIn(self.processor.detect_script(translit), ["roman_sanskrit", "iast_transliterated"])

    def test_roman_to_devanagari_transliteration(self):
        roman_query = "kalidasa katha"
        deva = self.processor.roman_to_devanagari(roman_query)
        # Should contain Devanagari characters
        has_deva = any(0x0900 <= ord(c) <= 0x097F for c in deva)
        self.assertTrue(has_deva)
        self.assertIn("कालिदास", deva)

    def test_sentence_segmentation(self):
        passage = "एकदा राजा सभायाम् आसीत्। तत्र कालिदासः आगतवान्॥ ततः किं जातम्?"
        sentences = self.processor.split_sentences(passage)
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[0].endswith("।"))
        self.assertTrue(sentences[1].endswith("॥"))
        self.assertTrue(sentences[2].endswith("?"))


class TestSanskritTextSplitter(unittest.TestCase):
    """Tests for Sanskrit text chunking."""

    def setUp(self):
        self.splitter = SanskritTextSplitter(chunk_size=150, chunk_overlap=30)

    def test_split_preserves_sentences(self):
        sample = (
            "एकस्मिन् नगरे धनदत्तः नाम कश्चन धनिकः वणिक् प्रतिवसति स्म। "
            "तस्य गृहे रामुः नाम एकः आज्ञापालकः भृत्यः आसीत्। "
            "यद्यपि सः स्वामिनः सेवायां सदा तत्परः आसीत्, तथापि तस्य विवेकज्ञानं शून्यम् आसीत्॥"
        )
        chunks = self.splitter.split_text(sample)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)


class TestDocumentLoader(unittest.TestCase):
    """Tests for Sanskrit document ingestion (TXT & PDF)."""

    def setUp(self):
        self.loader = SanskritDocumentLoader()

    def test_load_txt(self):
        txt_path = os.path.join(CORPUS_DIR, "01_murkhabhritya.txt")
        self.assertTrue(os.path.exists(txt_path), f"Corpus file missing: {txt_path}")
        docs = self.loader.load_txt(txt_path)
        self.assertEqual(len(docs), 1)
        self.assertIn("मूर्खभृत्यस्य", docs[0]["text"])
        self.assertEqual(docs[0]["metadata"]["format"], "txt")

    def test_load_pdf(self):
        pdf_path = os.path.join(CORPUS_DIR, "01_murkhabhritya.pdf")
        self.assertTrue(os.path.exists(pdf_path), f"Corpus PDF missing: {pdf_path}")
        docs = self.loader.load_pdf(pdf_path)
        self.assertGreaterEqual(len(docs), 1)
        self.assertIn("मूर्खभृत्यस्य", docs[0]["text"])
        self.assertEqual(docs[0]["metadata"]["format"], "pdf")
        self.assertIn("page", docs[0]["metadata"])


class TestPromptTemplates(unittest.TestCase):
    """Tests for grounded RAG prompt assembly."""

    def test_build_rag_prompt(self):
        prompt = build_rag_prompt(
            context="कालिदासः महाकविः आसीत्।",
            question="कः महाकविः आसीत्?",
            history_context=""
        )
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("कालिदासः महाकविः आसीत्।", prompt)
        self.assertIn("कः महाकविः आसीत्?", prompt)
        self.assertIn("Answer based strictly", prompt)


class TestVectorStoreAndRetrieval(unittest.TestCase):
    """Tests for FAISS vector search and score thresholding."""

    @classmethod
    def setUpClass(cls):
        cls.embeddings = SanskritEmbeddings()
        cls.store = SanskritFAISSStore(embeddings=cls.embeddings)
        cls.store.load()

    def test_index_is_loaded(self):
        self.assertGreater(self.store.index.ntotal, 0)
        self.assertEqual(len(self.store.chunks), self.store.index.ntotal)

    def test_semantic_retrieval(self):
        query = "मूर्खभृत्यस्य कथायां भृत्यः किं कृतवान्?"
        results = self.store.similarity_search(query, top_k=3)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertIn("01_murkhabhritya", top["metadata"]["source"])
        self.assertGreater(top["score"], 0.4)
        self.assertTrue(top["is_confident"])

    def test_out_of_domain_confidence(self):
        query = "What is the boiling point of liquid helium in outer space?"
        results = self.store.similarity_search(query, top_k=3)
        top_score = results[0]["score"] if results else 0.0
        self.assertLess(top_score, SCORE_THRESHOLD)


if __name__ == "__main__":
    unittest.main(verbosity=2)

