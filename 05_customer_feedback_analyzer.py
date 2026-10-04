"""
Customer Feedback Analyzer - Application Entry Point
---------------------------------------------------
Runs the complete customer feedback analyzer application.

Can be run via:
    streamlit run 05_customer_feedback_analyzer.py
    or
    streamlit run app.py
"""

import os
import sys
import runpy

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    runpy.run_path(app_path, run_name="__main__")
