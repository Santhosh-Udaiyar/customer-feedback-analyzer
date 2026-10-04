"""
Sentiment Analysis Module
-------------------------
Implements rule-based lexical sentiment analysis using NLTK VADER
(Valence Aware Dictionary and sEntiment Reasoner).
"""

from typing import Dict, Any
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

_ANALYZER_INSTANCE = None


def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    """Singleton accessor for SentimentIntensityAnalyzer."""
    global _ANALYZER_INSTANCE
    if _ANALYZER_INSTANCE is None:
        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception:
            pass
        _ANALYZER_INSTANCE = SentimentIntensityAnalyzer()
    return _ANALYZER_INSTANCE


def analyze_sentiment(
    text: str,
    pos_threshold: float = 0.05,
    neg_threshold: float = -0.05
) -> Dict[str, Any]:
    """
    Performs genuine sentiment analysis on input text using VADER.

    Parameters:
        text (str): Customer feedback text.
        pos_threshold (float): Threshold above which compound is classified as Positive (default 0.05).
        neg_threshold (float): Threshold below which compound is classified as Negative (default -0.05).

    Returns:
        dict: {
            'sentiment': 'Positive' | 'Negative' | 'Neutral',
            'compound': float,
            'positive': float,
            'neutral': float,
            'negative': float,
            'score': float, # Primary display score (compound)
            'explanation': str
        }
    """
    if not text or not str(text).strip():
        return {
            "sentiment": "Neutral",
            "compound": 0.0,
            "positive": 0.0,
            "neutral": 1.0,
            "negative": 0.0,
            "score": 0.0,
            "explanation": "No text provided to evaluate."
        }

    analyzer = get_sentiment_analyzer()
    scores = analyzer.polarity_scores(str(text))
    compound = round(scores["compound"], 4)
    pos = round(scores["pos"], 4)
    neu = round(scores["neu"], 4)
    neg = round(scores["neg"], 4)

    # Classification logic according to standard VADER conventions
    if compound >= pos_threshold:
        sentiment = "Positive"
        explanation = f"Compound score of {compound:+.2f} meets positive threshold (>= {pos_threshold})."
    elif compound <= neg_threshold:
        sentiment = "Negative"
        explanation = f"Compound score of {compound:+.2f} falls into negative threshold (<= {neg_threshold})."
    else:
        sentiment = "Neutral"
        explanation = f"Compound score of {compound:+.2f} lies in neutral band ({neg_threshold} to {pos_threshold})."

    return {
        "sentiment": sentiment,
        "compound": compound,
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "score": compound,
        "explanation": explanation
    }
