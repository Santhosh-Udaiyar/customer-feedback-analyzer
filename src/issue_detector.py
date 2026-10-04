"""
Customer Issue & Topic Detection Module
----------------------------------------
Identifies key operational topics and customer concerns (Product Quality, Delivery,
Pricing, Customer Support, User Experience, Performance, Reliability) using
transparent lexical aspect matching and sentiment cross-tabulation.
"""

import re
from typing import Dict, List, Any, Optional
import pandas as pd

# Transparent, explainable issue taxonomy covering standard customer touchpoints
ISSUE_CATEGORIES = {
    "Product Quality": {
        "keywords": [
            "quality", "material", "materials", "build", "broken", "defective",
            "defect", "cracked", "damage", "damaged", "cheap", "durability",
            "durable", "sturdy", "flimsy", "switch", "parts", "casing"
        ],
        "icon": "📦",
        "description": "Physical build, craftsmanship, materials, and product defects."
    },
    "Delivery": {
        "keywords": [
            "delivery", "shipping", "shipment", "package", "packaging", "courier",
            "tracking", "arrived", "delay", "delayed", "late", "transit",
            "lost package", "fast shipping", "on time", "quick delivery"
        ],
        "icon": "🚚",
        "description": "Shipping timelines, parcel condition, tracking, and carrier service."
    },
    "Customer Support": {
        "keywords": [
            "support", "customer service", "agent", "representative", "rep",
            "call", "email", "phone", "chat", "contact", "rude", "polite",
            "unhelpful", "helpful", "hung up", "hold", "response", "assistance"
        ],
        "icon": "🎧",
        "description": "Staff responsiveness, helpfulness, wait times, and resolution speed."
    },
    "Pricing": {
        "keywords": [
            "price", "pricing", "cost", "expensive", "overpriced", "cheap",
            "money", "charge", "charged", "billing", "refund", "subscription",
            "fee", "fees", "value", "worth", "discount", "bill"
        ],
        "icon": "💳",
        "description": "Cost-value perception, billing accuracy, subscription charges, and refund processing."
    },
    "User Experience": {
        "keywords": [
            "ui", "ux", "interface", "design", "look", "aesthetic", "easy to use",
            "intuitive", "confusing", "manual", "setup", "learning curve",
            "onboarding", "instructions", "navigation", "sleek"
        ],
        "icon": "✨",
        "description": "Interface elegance, ease of use, documentation, and user onboarding."
    },
    "Performance": {
        "keywords": [
            "performance", "speed", "fast", "slow", "lag", "laggy", "smooth",
            "crash", "crashing", "freeze", "freezing", "bug", "buggy", "glitch",
            "glitches", "battery", "drain", "sync", "syncing"
        ],
        "icon": "⚡",
        "description": "Software execution speed, system responsiveness, lag, and battery consumption."
    },
    "Reliability": {
        "keywords": [
            "reliable", "reliability", "stable", "stability", "dependable",
            "inconsistent", "fails", "failed", "shut down", "disconnect",
            "lost data", "error", "errors", "uptime"
        ],
        "icon": "🛡️",
        "description": "System stability, uptime, data integrity, and failure rates over time."
    }
}


def detect_issues_single(text: str) -> List[Dict[str, Any]]:
    """
    Detects potential customer issues / topic mentions in a single piece of feedback.
    Returns matched categories along with matched trigger keywords.
    """
    if not text or not str(text).strip():
        return []

    text_lower = str(text).lower()
    matched = []

    for category, meta in ISSUE_CATEGORIES.items():
        found_keywords = []
        for kw in meta["keywords"]:
            # Match whole words or multi-word phrases
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower):
                found_keywords.append(kw)

        if found_keywords:
            matched.append({
                "category": category,
                "icon": meta["icon"],
                "description": meta["description"],
                "matched_keywords": found_keywords,
                "confidence": min(1.0, 0.4 + 0.2 * len(found_keywords))
            })

    return matched


def analyze_issues_dataset(df: pd.DataFrame, text_column: str, sentiment_column: str = "Sentiment") -> List[Dict[str, Any]]:
    """
    Computes genuine issue distribution across a dataset.
    Calculates total mentions, positive count, negative count, and neutral count per issue category.

    Parameters:
        df (pd.DataFrame): Enriched dataset containing feedback and sentiment predictions.
        text_column (str): Name of column containing feedback text.
        sentiment_column (str): Name of column containing 'Positive', 'Negative', 'Neutral'.

    Returns:
        List[Dict[str, Any]]: Aggregated metrics per issue category, sorted by total mentions descending.
    """
    if df.empty or text_column not in df.columns or sentiment_column not in df.columns:
        return []

    results = []
    total_reviews = len(df)

    for category, meta in ISSUE_CATEGORIES.items():
        pattern = r"\b(?:" + "|".join(re.escape(k) for k in meta["keywords"]) + r")\b"
        mask = df[text_column].astype(str).str.contains(pattern, case=False, regex=True)
        sub_df = df[mask]
        mentions = len(sub_df)

        if mentions > 0:
            pos_count = (sub_df[sentiment_column] == "Positive").sum()
            neg_count = (sub_df[sentiment_column] == "Negative").sum()
            neu_count = (sub_df[sentiment_column] == "Neutral").sum()
            mention_pct = round((mentions / total_reviews) * 100, 1)

            # Sample customer quotes for this issue
            sample_quotes = sub_df[text_column].head(2).tolist()

            results.append({
                "category": category,
                "icon": meta["icon"],
                "mentions": int(mentions),
                "mention_percentage": mention_pct,
                "positive": int(pos_count),
                "negative": int(neg_count),
                "neutral": int(neu_count),
                "description": meta["description"],
                "sample_quotes": sample_quotes
            })

    # Sort categories by total mentions descending
    results.sort(key=lambda x: x["mentions"], reverse=True)
    return results
