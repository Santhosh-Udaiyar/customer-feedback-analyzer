"""
Text Preprocessing Module
-------------------------
Handles text cleaning, tokenization, stopword removal, and normalization.
"""

import re
import string
from typing import List, Set, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Domain-specific generic filler words that don't add sentiment or semantic value
DOMAIN_NOISE_WORDS = {
    "product", "item", "bought", "ordered", "order", "get", "got", "one",
    "also", "would", "could", "really", "even", "much", "still", "well",
    "back", "time", "day", "days", "use", "used", "using", "buy", "purchase"
}

# Words expressing negation that must NOT be removed as stopwords,
# because doing so would invert sentiment meaning (e.g. "not good" -> "good")
NEGATION_WORDS = {
    "not", "no", "never", "nor", "neither", "barely", "hardly", "scarcely",
    "cannot", "cant", "dont", "didnt", "doesnt", "wont", "wouldnt", "shouldnt",
    "couldnt", "wasnt", "werent", "isnt", "arent"
}

_STOPWORDS_CACHE: Set[str] = set()
_STOPWORDS_WITH_NEGATIONS_CACHE: Set[str] = set()


def get_stopwords(preserve_negations: bool = True) -> Set[str]:
    """
    Retrieve English stopwords extended with domain noise words.
    When preserve_negations=True (default), negation words ('not', 'no', 'never', etc.)
    are explicitly kept so that sentiment polarity is not inverted during text filtering.
    """
    global _STOPWORDS_CACHE, _STOPWORDS_WITH_NEGATIONS_CACHE
    
    if not _STOPWORDS_CACHE:
        try:
            nltk.download("stopwords", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            base_stops = set(stopwords.words("english"))
        except Exception:
            base_stops = {
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
                "he", "she", "it", "they", "them", "what", "which", "who", "whom", "this",
                "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
                "the", "and", "or", "because", "as", "until", "while", "of", "at", "by",
                "for", "with", "about", "against", "between", "into", "through", "during",
                "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
                "on", "off", "over", "under", "again", "further", "then", "once", "here",
                "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
                "more", "most", "other", "some", "such", "only", "own",
                "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
                "should", "now"
            }
        full_stops = base_stops.union(DOMAIN_NOISE_WORDS)
        _STOPWORDS_CACHE = set(full_stops)
        # Exclude negation words from the stopword set
        _STOPWORDS_WITH_NEGATIONS_CACHE = set(full_stops - NEGATION_WORDS)

    return _STOPWORDS_WITH_NEGATIONS_CACHE if preserve_negations else _STOPWORDS_CACHE


def clean_text(text: str) -> str:
    """
    Cleans raw customer feedback:
    1. Removes URLs and hyperlinks
    2. Strips HTML tags
    3. Normalizes whitespace and unprintable characters
    4. Preserves casing, punctuation, and negations for sentiment analysis (VADER).
    """
    if text is None:
        return ""
    text_str = str(text).strip()
    if not text_str or text_str.lower() in ["nan", "none", "null"]:
        return ""

    # Remove URLs
    text_str = re.sub(r"https?://\S+|www\.\S+", "", text_str)
    # Remove HTML tags
    text_str = re.sub(r"<.*?>", "", text_str)
    # Normalize multiple whitespace/newlines
    text_str = re.sub(r"\s+", " ", text_str).strip()
    return text_str


def normalize_for_nlp(text: str) -> str:
    """
    Normalizes text for keyword & topic processing:
    Lowercases, removes punctuation, and standardizes spacing.
    """
    cleaned = clean_text(text).lower()
    # Replace punctuation with single space
    cleaned = re.sub(f"[{re.escape(string.punctuation)}]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_and_remove_stopwords(text: str, preserve_negations: bool = True) -> List[str]:
    """
    Splits normalized text into tokens and filters out stopwords,
    while optionally preserving negation words (e.g. 'not', 'no', 'never').
    """
    normalized = normalize_for_nlp(text)
    stops = get_stopwords(preserve_negations=preserve_negations)
    
    # Tokenize words containing alphanumeric characters
    tokens = re.findall(r"\b[a-z]{2,}\b", normalized)
    filtered = [token for token in tokens if token not in stops]
    return filtered


def preprocess_feedback(text: str, preserve_negations: bool = True) -> Dict[str, Any]:
    """
    Complete NLP preprocessing pipeline following the project specification:
    Raw Feedback -> Whitespace & HTML Cleaning -> Lowercasing -> Punctuation Handling ->
    Tokenization -> Stopword Removal (preserving negations) -> Clean Tokens & Text.
    """
    cleaned_surface = clean_text(text)
    lowercased = cleaned_surface.lower()
    # Punctuation handling
    no_punct = re.sub(f"[{re.escape(string.punctuation)}]", " ", lowercased)
    no_punct = re.sub(r"\s+", " ", no_punct).strip()
    
    # Tokenization
    tokens = re.findall(r"\b[a-z]{2,}\b", no_punct)
    
    # Stopword removal
    stops = get_stopwords(preserve_negations=preserve_negations)
    filtered_tokens = [t for t in tokens if t not in stops]
    cleaned_text_reconstructed = " ".join(filtered_tokens)
    
    return {
        "raw_text": text,
        "cleaned_surface": cleaned_surface,
        "normalized_text": no_punct,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "clean_text": cleaned_text_reconstructed
    }
