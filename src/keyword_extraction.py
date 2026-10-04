"""
Keyword Extraction Module (Forwarding interface)
-----------------------------------------------
Maps to src/keyword_extractor.py to support both naming conventions.
"""
from src.keyword_extractor import (
    extract_keywords_single,
    extract_keywords_corpus,
)

__all__ = ["extract_keywords_single", "extract_keywords_corpus"]
