"""
Data Processing & Validation Module
-----------------------------------
Handles CSV dataset ingestion, column auto-detection, data validation,
cleaning, and preparation of results for download.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd


# Likely candidates for customer feedback text columns
FEEDBACK_COLUMN_CANDIDATES = [
    "feedback",
    "review",
    "comment",
    "customer_feedback",
    "review_text",
    "text",
    "opinion",
    "message",
    "comments",
    "reviews",
    "user_feedback",
    "feedback_text"
]

# Likely candidates for timestamp or date columns
DATE_COLUMN_CANDIDATES = [
    "date",
    "review_date",
    "timestamp",
    "created_at",
    "posted_date",
    "time",
    "datetime"
]


def detect_feedback_column(columns: List[str]) -> Optional[str]:
    """
    Intelligently identifies a likely customer feedback text column from a list of column names.
    Checks exact matches first, then substring containment.
    """
    col_map = {col.strip().lower(): col for col in columns}

    # 1. Exact match pass
    for cand in FEEDBACK_COLUMN_CANDIDATES:
        if cand in col_map:
            return col_map[cand]

    # 2. Substring match pass
    for cand in FEEDBACK_COLUMN_CANDIDATES:
        for lower_col, orig_col in col_map.items():
            if cand in lower_col:
                return orig_col

    return None


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Intelligently detects if the dataset contains a usable date/time column.
    Validates by attempting date conversion on sample values.
    """
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if any(d_cand in col_clean for d_cand in DATE_COLUMN_CANDIDATES):
            # Verify if values can actually be parsed as dates
            try:
                sample = df[col].dropna().head(10)
                if not sample.empty:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() >= len(sample) * 0.7:
                        return str(col)
            except Exception:
                continue

    return None


def validate_and_clean_dataset(
    df: pd.DataFrame,
    text_column: str,
    remove_duplicates: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs data validation and cleaning on uploaded feedback data:
    1. Checks total rows
    2. Identifies and removes null / empty feedback entries
    3. Detects duplicate feedback texts
    4. Gathers column overview

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned dataframe and validation metrics dictionary.
    """
    total_raw_rows = len(df)
    available_cols = list(df.columns)

    if text_column not in df.columns:
        return df, {
            "total_rows": total_raw_rows,
            "empty_rows_removed": 0,
            "duplicates_detected": 0,
            "rows_analyzed": 0,
            "columns": available_cols,
            "error": f"Column '{text_column}' not found in dataset."
        }

    # Identify empty or null text rows
    is_null_or_empty = df[text_column].isna() | (df[text_column].astype(str).str.strip() == "") | (df[text_column].astype(str).str.lower().isin(["nan", "none", "null"]))
    empty_count = int(is_null_or_empty.sum())

    cleaned_df = df[~is_null_or_empty].copy()

    # Detect duplicates on the text column
    duplicate_count = int(cleaned_df.duplicated(subset=[text_column]).sum())

    if remove_duplicates and duplicate_count > 0:
        cleaned_df = cleaned_df.drop_duplicates(subset=[text_column]).copy()

    rows_analyzed = len(cleaned_df)

    validation_summary = {
        "total_rows": total_raw_rows,
        "empty_rows_removed": empty_count,
        "duplicates_detected": duplicate_count,
        "rows_analyzed": rows_analyzed,
        "columns": available_cols,
        "error": None
    }

    return cleaned_df, validation_summary


def prepare_download_dataframe(df_analyzed: pd.DataFrame, original_text_col: str) -> pd.DataFrame:
    """
    Formats the enriched dataset for CSV export with clear, standardized headers:
    - feedback (original text)
    - sentiment (Positive, Negative, Neutral)
    - sentiment_score (Compound score)
    - detected_issue (Detected operational issues)
    - important_keywords (Extracted keyphrases)
    along with any additional original columns.
    """
    export_df = df_analyzed.copy()

    # Standardize column names if needed
    column_renames = {}
    if "Sentiment" in export_df.columns:
        column_renames["Sentiment"] = "sentiment"
    if "Sentiment_Score" in export_df.columns:
        column_renames["Sentiment_Score"] = "sentiment_score"
    if "Detected_Issues" in export_df.columns:
        column_renames["Detected_Issues"] = "detected_issue"
    if "Extracted_Keywords" in export_df.columns:
        column_renames["Extracted_Keywords"] = "important_keywords"
    if original_text_col in export_df.columns and original_text_col.lower() != "feedback":
        column_renames[original_text_col] = f"{original_text_col} (feedback)"

    export_df = export_df.rename(columns=column_renames)
    return export_df
