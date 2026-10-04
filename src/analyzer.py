"""
Analyzer Orchestration Module
------------------------------
Integrates the complete end-to-end NLP pipeline:
Preprocessing -> Sentiment Analysis -> Keyword Extraction -> Issue Detection -> Data-Driven Insights.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.preprocessing import clean_text
from src.sentiment import analyze_sentiment
from src.keyword_extractor import extract_keywords_single, extract_keywords_corpus
from src.issue_detector import detect_issues_single, analyze_issues_dataset


class FeedbackAnalyzer:
    """End-to-end Natural Language Processing orchestrator for customer feedback."""

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def analyze_single(self, text: str, top_keywords: int = 5) -> Dict[str, Any]:
        """
        Executes complete NLP pipeline on a single feedback string.
        """
        cleaned = clean_text(text)
        sentiment_res = analyze_sentiment(
            cleaned,
            pos_threshold=self.pos_threshold,
            neg_threshold=self.neg_threshold
        )
        keywords = extract_keywords_single(cleaned, top_n=top_keywords)
        issues = detect_issues_single(cleaned)

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "sentiment": sentiment_res["sentiment"],
            "compound_score": sentiment_res["compound"],
            "positive_score": sentiment_res["positive"],
            "neutral_score": sentiment_res["neutral"],
            "negative_score": sentiment_res["negative"],
            "primary_score": sentiment_res["score"],
            "explanation": sentiment_res["explanation"],
            "keywords": keywords,
            "detected_issues": issues
        }

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        text_column: str,
        top_keywords_per_row: int = 5
    ) -> pd.DataFrame:
        """
        Executes batch NLP processing across a pandas DataFrame.
        """
        results = []
        for text in df[text_column]:
            single_res = self.analyze_single(str(text), top_keywords=top_keywords_per_row)
            issue_names = [iss["category"] for iss in single_res["detected_issues"]]
            
            results.append({
                "Cleaned_Feedback": single_res["cleaned_text"],
                "Sentiment": single_res["sentiment"],
                "Sentiment_Score": single_res["compound_score"],
                "Positive_Score": single_res["positive_score"],
                "Neutral_Score": single_res["neutral_score"],
                "Negative_Score": single_res["negative_score"],
                "Extracted_Keywords": ", ".join(single_res["keywords"]),
                "Detected_Issues": ", ".join(issue_names) if issue_names else "None Identified"
            })

        df_results = pd.DataFrame(results)
        enriched_df = pd.concat([df.reset_index(drop=True), df_results], axis=1)
        return enriched_df

    @staticmethod
    def generate_insights(df: pd.DataFrame, issues_data: List[Dict[str, Any]]) -> List[str]:
        """
        Generates genuine, business-oriented insights dynamically derived from the analyzed dataset.
        No static or fake statements.
        """
        if df.empty:
            return ["No data available to generate insights."]

        total = len(df)
        pos_count = (df["Sentiment"] == "Positive").sum()
        neg_count = (df["Sentiment"] == "Negative").sum()
        neu_count = (df["Sentiment"] == "Neutral").sum()

        pos_pct = round((pos_count / total) * 100, 1)
        neg_pct = round((neg_count / total) * 100, 1)
        neu_pct = round((neu_count / total) * 100, 1)
        net_sentiment = round(pos_pct - neg_pct, 1)

        insights = []

        # 1. Overall sentiment health
        if net_sentiment > 25:
            insights.append(
                f"🌟 **Strong Customer Satisfaction:** Net Sentiment Score is **{net_sentiment:+.1f}%** "
                f"with **{pos_pct}%** positive feedback outpacing negative remarks ({neg_pct}%)."
            )
        elif net_sentiment < -10:
            insights.append(
                f"🚨 **Critical Attention Needed:** Negative feedback stands at **{neg_pct}%**, yielding a negative "
                f"Net Sentiment Score of **{net_sentiment:+.1f}%**."
            )
        else:
            insights.append(
                f"⚖️ **Balanced Customer Sentiment:** Net Sentiment Score is **{net_sentiment:+.1f}%**, "
                f"with **{pos_pct}%** positive, **{neg_pct}%** negative, and **{neu_pct}%** neutral feedback."
            )

        # 2. Key problem driver from negative feedback
        neg_issues = sorted(
            [iss for iss in issues_data if iss["negative"] > 0],
            key=lambda x: x["negative"],
            reverse=True
        )
        if neg_issues:
            top_neg_issue = neg_issues[0]
            neg_share = round((top_neg_issue["negative"] / max(1, neg_count)) * 100, 1)
            insights.append(
                f"• **Negative feedback is concentrated around {top_neg_issue['category']}:** "
                f"Appears in **{top_neg_issue['negative']} negative reviews** ({neg_share}% of all negative feedback)."
            )
            if len(neg_issues) > 1:
                second_neg = neg_issues[1]
                sec_share = round((second_neg["negative"] / max(1, neg_count)) * 100, 1)
                insights.append(
                    f"• **{second_neg['category']} appears in a significant portion of negative feedback:** "
                    f"Mentioned in **{second_neg['negative']} negative reviews** ({sec_share}% of negative feedback)."
                )

        # 3. Bright spots / positive drivers
        pos_issues = sorted(
            [iss for iss in issues_data if iss["positive"] > 0],
            key=lambda x: x["positive"],
            reverse=True
        )
        if pos_issues:
            top_pos_issue = pos_issues[0]
            insights.append(
                f"• **{top_pos_issue['category']} is frequently mentioned in positive reviews:** "
                f"Received the highest praise with **{top_pos_issue['positive']} positive mentions**."
            )

        # 4. Neutral / Passive customer segment
        if neu_pct >= 20:
            insights.append(
                f"• **Passive customer opportunity:** **{neu_pct}%** of feedback is neutral, "
                f"representing an opportunity for proactive engagement before potential customer churn."
            )

        return insights

