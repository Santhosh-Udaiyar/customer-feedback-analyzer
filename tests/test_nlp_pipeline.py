"""
Unit & Integration Tests for NLP Pipeline
"""

import os
import unittest
import pandas as pd

from src.preprocessing import clean_text, tokenize_and_remove_stopwords, normalize_for_nlp
from src.sentiment import analyze_sentiment
from src.keyword_extractor import extract_keywords_single, extract_keywords_corpus
from src.issue_detector import detect_issues_single, analyze_issues_dataset, ISSUE_CATEGORIES
from src.analyzer import FeedbackAnalyzer


class TestNLPPipeline(unittest.TestCase):

    def test_text_cleaning(self):
        raw = "Check out https://example.com for <b>details</b>!   Extra  spaces.  "
        cleaned = clean_text(raw)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertEqual(cleaned, "Check out for details! Extra spaces.")

    def test_tokenization_and_stopwords(self):
        text = "The product quality was amazing and delivery was fast"
        tokens = tokenize_and_remove_stopwords(text)
        self.assertIn("quality", tokens)
        self.assertIn("amazing", tokens)
        self.assertIn("delivery", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("was", tokens)

    def test_sentiment_positive(self):
        res = analyze_sentiment("The product quality is excellent, durable, and fantastic!")
        self.assertEqual(res["sentiment"], "Positive")
        self.assertGreater(res["compound"], 0.05)
        self.assertGreater(res["positive"], 0.0)

    def test_sentiment_negative(self):
        res = analyze_sentiment("Terrible experience. The package arrived broken and support was rude!")
        self.assertEqual(res["sentiment"], "Negative")
        self.assertLess(res["compound"], -0.05)
        self.assertGreater(res["negative"], 0.0)

    def test_sentiment_neutral(self):
        res = analyze_sentiment("The package arrived on Tuesday.")
        self.assertEqual(res["sentiment"], "Neutral")
        self.assertTrue(-0.05 <= res["compound"] <= 0.05)

    def test_keyword_extraction_single(self):
        text = "The delivery was late and customer support was slow, but the product quality was excellent."
        kws = extract_keywords_single(text, top_n=5)
        self.assertTrue(len(kws) > 0)
        # Check that meaningful tokens or phrases like 'delivery', 'customer support', or 'quality' are identified
        joined = " ".join(kws).lower()
        self.assertTrue(any(term in joined for term in ["delivery", "support", "quality", "customer support", "product quality"]))

    def test_issue_detection_single(self):
        text = "The delivery was late and customer support was slow, but the product quality was excellent."
        issues = detect_issues_single(text)
        issue_names = [i["category"] for i in issues]
        self.assertIn("Delivery", issue_names)
        self.assertIn("Customer Support", issue_names)
        self.assertIn("Product Quality", issue_names)

    def test_dataset_analysis(self):
        csv_path = os.path.join("data", "sample_feedback.csv")
        self.assertTrue(os.path.exists(csv_path), "Sample CSV file must exist")
        df = pd.read_csv(csv_path)
        self.assertIn("feedback", df.columns)

        analyzer = FeedbackAnalyzer()
        enriched = analyzer.analyze_dataset(df, text_column="feedback")
        self.assertEqual(len(enriched), len(df))
        self.assertIn("Sentiment", enriched.columns)
        self.assertIn("Sentiment_Score", enriched.columns)
        self.assertIn("Extracted_Keywords", enriched.columns)
        self.assertIn("Detected_Issues", enriched.columns)

        issues_data = analyze_issues_dataset(enriched, text_column="feedback")
        self.assertTrue(len(issues_data) > 0)

        insights = analyzer.generate_insights(enriched, issues_data)
        self.assertTrue(len(insights) > 0)


if __name__ == "__main__":
    unittest.main()
