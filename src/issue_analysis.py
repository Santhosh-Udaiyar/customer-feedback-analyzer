"""
Issue & Topic Analysis Module (Forwarding interface)
-----------------------------------------------------
Maps to src/issue_detector.py to support both naming conventions.
"""
from src.issue_detector import (
    detect_issues_single,
    analyze_issues_dataset,
    ISSUE_CATEGORIES,
)

__all__ = ["detect_issues_single", "analyze_issues_dataset", "ISSUE_CATEGORIES"]
