"""
Sanskrit NLP Processor: Unicode normalization, script detection,
transliteration, and sentence segmentation.
"""

import re
import unicodedata
from typing import List, Tuple, Optional

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
    INDIC_TRANSLIT_AVAILABLE = True
except ImportError:
    INDIC_TRANSLIT_AVAILABLE = False


class SanskritProcessor:
    """Specialized NLP processor for classical and contemporary Sanskrit text."""

    # Unicode ranges for Devanagari: U+0900 to U+097F
    DEVANAGARI_RANGE = (0x0900, 0x097F)
    
    # Common IAST diacritic characters
    IAST_DIACRITICS = set("āīūṛṝḷḹēōṃḥṅñṇṭḍśṣ")

    @classmethod
    def normalize_devanagari(cls, text: str) -> str:
        """
        Normalizes Devanagari text:
        - Unicode NFC normalization
        - Replaces ASCII pipes with authentic Devanagari dandas
        - Cleans abnormal whitespaces while preserving verse structure
        """
        if not text:
            return ""
            
        # Standardize Unicode canonical composition
        text = unicodedata.normalize("NFC", text)

        # Standardize Dandās: Double danda before single danda replacement
        text = re.sub(r'\|\|', '॥', text)
        text = re.sub(r'\|', '।', text)
        
        # Replace multiple consecutive spaces with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize redundant newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    @classmethod
    def detect_script(cls, text: str) -> str:
        """
        Detects primary script and language of query/document.
        Returns: 'devanagari', 'iast_transliterated', 'english_or_latin'
        """
        if not text:
            return "unknown"

        devanagari_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        total_alpha = sum(1 for c in text if c.isalpha())

        if total_alpha == 0:
            return "unknown"

        deva_ratio = devanagari_chars / total_alpha
        if deva_ratio > 0.4:
            return "devanagari"

        # Check for IAST diacritics
        if any(c in cls.IAST_DIACRITICS for c in text.lower()):
            return "iast_transliterated"

        # Check if Roman text represents transliterated Sanskrit (e.g. Harvard-Kyoto or ITRANS)
        sanskrit_phonetic_markers = [
            "sh", "bh", "dh", "th", "kh", "gh", "ch", "jh", "ph",
            "katha", "raja", "bhritya", "kalidasa", "vriddha", "sheetam",
            "bahu", "badhati", "purushartha", "daivam", "katham", "kim",
            "kuto", "sarva", "iti", "asti", "sm", "tatah", "eva"
        ]
        lower_text = text.lower()
        marker_hits = sum(1 for marker in sanskrit_phonetic_markers if marker in lower_text)
        if marker_hits >= 2:
            return "roman_sanskrit"

        return "english_or_latin"

    @classmethod
    def roman_to_devanagari(cls, text: str, scheme: Optional[str] = None) -> str:
        """
        Transliterates Roman text (IAST, ITRANS, Harvard-Kyoto, or colloquial Roman Sanskrit)
        to authentic Devanagari script.
        """
        if not INDIC_TRANSLIT_AVAILABLE or not text:
            return text

        # If already Devanagari, return as is
        script = cls.detect_script(text)
        if script == "devanagari":
            return text

        # Colloquial phonetic normalization before transliteration
        cleaned = text.strip()
        
        # Normalize common colloquial Sanskrit spelling patterns to ITRANS
        phonetic_replacements = [
            (r'\bvri', 'vR'),
            (r'\bbhri', 'bhR'),
            (r'\bkri', 'kR'),
            (r'\bdhri', 'dhR'),
            (r'\bpri', 'pR'),
            (r'\bshri', 'shrI'),
            (r'chaturya', 'chAturya'),
            (r'vriddha', 'vRddhA'),
            (r'bhritya', 'bhRtya'),
            (r'kalidasa', 'kAlidAsa'),
            (r'sheetam', 'shItaM'),
            (r'badhati', 'bAdhati'),
            (r'purushartha', 'puruShArtha'),
            (r'daivam', 'daivaM'),
            (r'katha\b', 'kathA'),
            (r'yah\b', 'yAH'),
            (r'aa', 'A'),
            (r'ee', 'I'),
            (r'oo', 'U'),
        ]
        
        normalized_roman = cleaned
        for pattern, repl in phonetic_replacements:
            normalized_roman = re.sub(pattern, repl, normalized_roman, flags=re.IGNORECASE)

        try:
            if scheme:
                return transliterate(normalized_roman, scheme, sanscript.DEVANAGARI)
            
            if script == "iast_transliterated":
                return transliterate(cleaned, sanscript.IAST, sanscript.DEVANAGARI)
            
            # Default to ITRANS for Roman transliterated Sanskrit
            devanagari_text = transliterate(normalized_roman, sanscript.ITRANS, sanscript.DEVANAGARI)
            return devanagari_text
        except Exception:
            return text

    @classmethod
    def devanagari_to_iast(cls, text: str) -> str:
        """Transliterates Devanagari text to IAST."""
        if not INDIC_TRANSLIT_AVAILABLE or not text:
            return text
        try:
            return transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
        except Exception:
            return text

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """
        Splits Sanskrit text into sentences while respecting:
        - Purna Virama (।)
        - Deergha Purna Virama (॥)
        - Question marks, exclamation marks, and standard periods
        """
        text = cls.normalize_devanagari(text)
        if not text:
            return []

        # Split on sentence terminals while keeping the delimiter
        # Pattern matches ।, ॥, ?, !, ., or double newline
        pattern = r'([^।॥\n.!?]+[।॥\n.!?]+)'
        raw_sentences = re.findall(pattern, text)
        
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        # If no punctuation matched, fallback to newline split
        if not sentences:
            sentences = [line.strip() for line in text.splitlines() if line.strip()]
            
        return sentences
