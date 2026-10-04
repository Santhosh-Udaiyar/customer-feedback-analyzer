"""
Keyword Extraction Module
-------------------------
Extracts meaningful keywords and keyphrases from customer feedback using
TF-IDF (Term Frequency-Inverse Document Frequency) and N-gram frequency analysis.
"""

import re
from collections import Counter
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import normalize_for_nlp, get_stopwords


def extract_keywords_single(text: str, top_n: int = 5) -> List[str]:
    """
    Extracts key terms and meaningful phrases from a single customer feedback string.
    Combines bigram/unigram analysis with stopword elimination.

    Parameters:
        text (str): Feedback string.
        top_n (int): Number of top keywords to return.

    Returns:
        List[str]: Ranked salient keywords/keyphrases.
    """
    normalized = normalize_for_nlp(text)
    if not normalized:
        return []

    stops = get_stopwords()
    tokens = [t for t in re.findall(r"\b[a-z]{3,}\b", normalized) if t not in stops]
    if not tokens:
        return []

    # If the feedback has enough words, extract candidate n-grams (unigrams + bigrams)
    candidates = []
    # Add unigrams
    candidates.extend(tokens)

    # Add meaningful adjacent bigrams
    all_words = normalized.split()
    for i in range(len(all_words) - 1):
        w1, w2 = all_words[i], all_words[i + 1]
        if len(w1) >= 3 and len(w2) >= 3:
            # Exclude if both words are stopwords
            if (w1 not in stops) and (w2 not in stops):
                candidates.append(f"{w1} {w2}")
            # Meaningful compound phrases like "customer support", "product quality"
            elif (w1 not in stops and w2 in {"support", "quality", "service", "care", "speed"}):
                candidates.append(f"{w1} {w2}")
            elif (w1 in {"customer", "product", "build", "battery"} and w2 not in stops):
                candidates.append(f"{w1} {w2}")

    # Use TF-IDF scoring if we have a moderate length sentence
    if len(tokens) >= 4:
        try:
            tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words=list(stops),
                max_features=top_n * 2
            )
            matrix = tfidf.fit_transform([normalized])
            feature_names = tfidf.get_feature_names_out()
            scores = matrix.toarray()[0]
            ranked_indices = scores.argsort()[::-1]
            extracted = [feature_names[i] for i in ranked_indices if scores[i] > 0]
            if extracted:
                return extracted[:top_n]
        except Exception:
            pass

    # Frequency-based fallback ranking
    counts = Counter(candidates)
    return [term for term, _ in counts.most_common(top_n)]


def extract_keywords_corpus(texts: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
    """
    Extracts the most significant keywords across an entire corpus of feedback using TF-IDF.

    Parameters:
        texts (List[str]): List of customer feedback texts.
        top_n (int): Number of top keywords to return.

    Returns:
        List[Tuple[str, float]]: List of (keyword, tfidf_weight) tuples.
    """
    clean_texts = [normalize_for_nlp(t) for t in texts if t and str(t).strip()]
    if not clean_texts:
        return []

    stops = list(get_stopwords())
    try:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=stops,
            max_df=0.85,
            min_df=1,
            max_features=100,
            sublinear_tf=True
        )
        tfidf_matrix = tfidf.fit_transform(clean_texts)
        feature_names = tfidf.get_feature_names_out()
        mean_weights = tfidf_matrix.mean(axis=0).A1
        scored = list(zip(feature_names, mean_weights))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
    except Exception:
        # Fallback to token counts
        all_tokens = []
        for t in clean_texts:
            tokens = [tok for tok in re.findall(r"\b[a-z]{3,}\b", t) if tok not in stops]
            all_tokens.extend(tokens)
        counts = Counter(all_tokens)
        return [(word, float(freq)) for word, freq in counts.most_common(top_n)]
