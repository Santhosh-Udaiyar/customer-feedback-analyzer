"""
Unit & Integration Tests for Customer Feedback Analyzer
-------------------------------------------------------
Covers:
Individual Feedback:
1. Positive feedback
2. Negative feedback
3. Neutral feedback
4. Mixed feedback
5. Very short feedback
6. Long feedback
7. Empty input
8. Feedback containing punctuation
9. Feedback containing "not"
10. Feedback with unusual wording

CSV Processing:
1. Valid CSV
2. Missing text column
3. Empty CSV
4. Missing values
5. Duplicate rows
6. Different possible feedback column names
7. Large enough sample dataset
8. Download analyzed CSV
"""

import os
import io
import unittest
import pandas as pd

from src.preprocessing import (
    clean_text,
    tokenize_and_remove_stopwords,
    normalize_for_nlp,
    preprocess_feedback,
    NEGATION_WORDS,
)
from src.sentiment import analyze_sentiment
from src.keyword_extractor import extract_keywords_single, extract_keywords_corpus
from src.issue_detector import detect_issues_single, analyze_issues_dataset, ISSUE_CATEGORIES
from src.analyzer import FeedbackAnalyzer
from src.data_processing import (
    detect_feedback_column,
    detect_date_column,
    validate_and_clean_dataset,
    prepare_download_dataframe,
)


class TestCustomerFeedbackAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = FeedbackAnalyzer(pos_threshold=0.05, neg_threshold=-0.05)

    # =========================================================================
    # INDIVIDUAL FEEDBACK TESTS (10 Specific Scenarios)
    # =========================================================================

    def test_01_positive_feedback(self):
        """1. Test positive feedback evaluation."""
        text = "The product quality is absolutely outstanding and customer service was amazing!"
        res = self.analyzer.analyze_single(text)
        self.assertEqual(res["sentiment"], "Positive")
        self.assertGreater(res["compound_score"], 0.05)
        self.assertGreater(res["positive_score"], 0.0)

    def test_02_negative_feedback(self):
        """2. Test negative feedback evaluation."""
        text = "Terrible experience. The package arrived broken, cracked, and support was rude!"
        res = self.analyzer.analyze_single(text)
        self.assertEqual(res["sentiment"], "Negative")
        self.assertLess(res["compound_score"], -0.05)
        self.assertGreater(res["negative_score"], 0.0)

    def test_03_neutral_feedback(self):
        """3. Test neutral feedback evaluation."""
        text = "The package arrived on Tuesday afternoon."
        res = self.analyzer.analyze_single(text)
        self.assertEqual(res["sentiment"], "Neutral")
        self.assertTrue(-0.05 <= res["compound_score"] <= 0.05)

    def test_04_mixed_feedback(self):
        """4. Test mixed feedback containing contrasting aspects."""
        text = "The product quality is good but delivery took too long."
        res = self.analyzer.analyze_single(text)
        self.assertIn(res["sentiment"], ["Positive", "Neutral", "Negative"])
        # Both product quality and delivery issues must be identified
        issues = [iss["category"] for iss in res["detected_issues"]]
        self.assertIn("Product Quality", issues)
        self.assertIn("Delivery", issues)

    def test_05_very_short_feedback(self):
        """5. Test very short feedback."""
        text = "Good"
        res = self.analyzer.analyze_single(text)
        self.assertEqual(res["sentiment"], "Positive")
        self.assertGreater(res["compound_score"], 0.0)

    def test_06_long_feedback(self):
        """6. Test lengthy customer feedback."""
        text = (
            "We have been using this platform across our entire engineering organization for over six months now. "
            "The initial onboarding experience was relatively seamless, and the intuitive user interface made it easy "
            "for non-technical team members to adopt. However, over the past few weeks, we noticed considerable performance "
            "degradation and recurrent system lag when generating quarterly reports. Furthermore, when our team reached out to "
            "customer support, the wait times exceeded forty-five minutes and the representative was unable to resolve the ticket. "
            "Despite these performance and support hurdles, the underlying product quality and core reliability remain decent."
        )
        res = self.analyzer.analyze_single(text)
        self.assertTrue(len(res["keywords"]) >= 3)
        issues = [iss["category"] for iss in res["detected_issues"]]
        self.assertTrue(any(iss in issues for iss in ["Customer Support", "Performance", "Product Quality", "User Experience"]))

    def test_07_empty_input(self):
        """7. Test empty feedback string and whitespace handling."""
        res_empty = self.analyzer.analyze_single("")
        self.assertEqual(res_empty["sentiment"], "Neutral")
        self.assertEqual(res_empty["compound_score"], 0.0)
        self.assertEqual(res_empty["keywords"], [])
        self.assertEqual(res_empty["detected_issues"], [])

        res_spaces = self.analyzer.analyze_single("    \n\t   ")
        self.assertEqual(res_spaces["sentiment"], "Neutral")
        self.assertEqual(res_spaces["compound_score"], 0.0)

    def test_08_feedback_containing_punctuation(self):
        """8. Test feedback with heavy punctuation and HTML artifacts."""
        raw = "!!! Absolutely LOVED it :) :) Best purchase EVER... Highly, highly recommend!!! <br>5/5 Stars!"
        res = self.analyzer.analyze_single(raw)
        self.assertEqual(res["sentiment"], "Positive")
        self.assertGreater(res["compound_score"], 0.5)

    def test_09_feedback_containing_not(self):
        """9. Test negation handling ('not good' must be Negative, NOT positive)."""
        text = "The product is not good."
        res = self.analyzer.analyze_single(text)
        self.assertEqual(res["sentiment"], "Negative", "Negation 'not good' should produce negative sentiment")
        self.assertLess(res["compound_score"], 0.0)

        # Verify preprocessing explicitly preserves negation words
        tokens = tokenize_and_remove_stopwords("The product is not good")
        self.assertIn("not", tokens)
        self.assertIn("good", tokens)

    def test_10_feedback_with_unusual_wording(self):
        """10. Test feedback with slang, informal phrasing, or mixed symbols."""
        text = "meh... works sorta okayish, kinda meh battery drain tho"
        res = self.analyzer.analyze_single(text)
        self.assertIsNotNone(res["sentiment"])
        self.assertIsInstance(res["keywords"], list)

    # =========================================================================
    # CSV PROCESSING TESTS (8 Specific Scenarios)
    # =========================================================================

    def test_csv_01_valid_csv(self):
        """1. Test analysis on valid CSV format."""
        data = {
            "id": [1, 2, 3],
            "feedback": [
                "Fast shipping and great packaging!",
                "Terrible customer service, very rude agent.",
                "The product works as described, standard quality."
            ]
        }
        df = pd.DataFrame(data)
        enriched = self.analyzer.analyze_dataset(df, text_column="feedback")
        self.assertEqual(len(enriched), 3)
        self.assertIn("Sentiment", enriched.columns)
        self.assertIn("Sentiment_Score", enriched.columns)
        self.assertIn("Extracted_Keywords", enriched.columns)
        self.assertIn("Detected_Issues", enriched.columns)

    def test_csv_02_missing_text_column(self):
        """2. Test missing text column detection and error reporting."""
        df = pd.DataFrame({"random_col1": [1, 2], "random_col2": ["foo", "bar"]})
        cleaned, val_summary = validate_and_clean_dataset(df, text_column="non_existent_column")
        self.assertIsNotNone(val_summary["error"])

    def test_csv_03_empty_csv(self):
        """3. Test empty CSV handling."""
        empty_df = pd.DataFrame({"feedback": []})
        cleaned, val_summary = validate_and_clean_dataset(empty_df, text_column="feedback")
        self.assertEqual(val_summary["rows_analyzed"], 0)
        self.assertEqual(len(cleaned), 0)

    def test_csv_04_missing_values(self):
        """4. Test missing and null values safely filtered."""
        data = {
            "feedback": [
                "Good product",
                None,
                "",
                "   ",
                "nan",
                "Delayed shipment was terrible."
            ]
        }
        df = pd.DataFrame(data)
        cleaned, val_summary = validate_and_clean_dataset(df, text_column="feedback")
        self.assertEqual(val_summary["total_rows"], 6)
        self.assertEqual(val_summary["empty_rows_removed"], 4)
        self.assertEqual(val_summary["rows_analyzed"], 2)
        self.assertEqual(len(cleaned), 2)

    def test_csv_05_duplicate_rows(self):
        """5. Test duplicate feedback detection."""
        data = {
            "feedback": [
                "Identical review text",
                "Identical review text",
                "Another unique review"
            ]
        }
        df = pd.DataFrame(data)
        cleaned, val_summary = validate_and_clean_dataset(df, text_column="feedback")
        self.assertEqual(val_summary["duplicates_detected"], 1)
        self.assertEqual(val_summary["total_rows"], 3)

    def test_csv_06_different_possible_feedback_column_names(self):
        """6. Test auto-detection across various feedback column names."""
        test_headers = [
            ["id", "customer_feedback", "score"],
            ["id", "review_text", "timestamp"],
            ["id", "comments", "user"],
            ["user_id", "opinion", "created_at"],
            ["entry", "review", "city"]
        ]
        for headers in test_headers:
            detected = detect_feedback_column(headers)
            self.assertIsNotNone(detected, f"Failed to detect feedback column in {headers}")

    def test_csv_07_large_enough_sample_dataset(self):
        """7. Test processing realistic pre-built sample dataset."""
        csv_path = os.path.join("data", "sample_feedback.csv")
        self.assertTrue(os.path.exists(csv_path), "sample_feedback.csv must exist")
        df = pd.read_csv(csv_path)
        self.assertGreaterEqual(len(df), 20, "Sample dataset should contain at least 20 records")

        col = detect_feedback_column(df.columns.tolist())
        self.assertEqual(col, "feedback")

        enriched = self.analyzer.analyze_dataset(df, text_column=col)
        self.assertEqual(len(enriched), len(df))

        # Check sentiment aggregation
        pos_count = (enriched["Sentiment"] == "Positive").sum()
        neg_count = (enriched["Sentiment"] == "Negative").sum()
        self.assertGreater(pos_count, 0)
        self.assertGreater(neg_count, 0)

        # Check issue analysis and data-driven insights
        issues = analyze_issues_dataset(enriched, text_column=col)
        self.assertGreater(len(issues), 0)
        insights = self.analyzer.generate_insights(enriched, issues)
        self.assertGreater(len(insights), 0)

    def test_csv_08_download_analyzed_csv(self):
        """8. Test preparation of formatted dataset for download."""
        df = pd.DataFrame({
            "id": [101, 102],
            "feedback": ["Great customer service", "Defective power cord and broken switch"]
        })
        enriched = self.analyzer.analyze_dataset(df, text_column="feedback")
        download_df = prepare_download_dataframe(enriched, original_text_col="feedback")

        self.assertIn("feedback", download_df.columns)
        self.assertIn("sentiment", download_df.columns)
        self.assertIn("sentiment_score", download_df.columns)
        self.assertIn("detected_issue", download_df.columns)
        self.assertIn("important_keywords", download_df.columns)

        # Verify it serializes to valid CSV bytes without error
        csv_buffer = io.StringIO()
        download_df.to_csv(csv_buffer, index=False)
        output_str = csv_buffer.getvalue()
        self.assertIn("sentiment_score", output_str)
        self.assertIn("Great customer service", output_str)


if __name__ == "__main__":
    unittest.main()
