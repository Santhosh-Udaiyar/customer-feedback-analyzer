"""
Customer Feedback Analyzer
--------------------------
Streamlit Application for Natural Language Processing of Customer Feedback.
Supports Individual Feedback Analysis and Batch CSV Dataset Processing.

Run using:
    streamlit run app.py
"""

import os
import io
import re
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Modular NLP package imports
from src.preprocessing import clean_text, preprocess_feedback, tokenize_and_remove_stopwords, NEGATION_WORDS
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


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Feedback Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Professional Design & CSS Tokens
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Container */
.header-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 24px 30px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}
.header-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.header-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
    font-weight: 400;
}

/* Metric KPI Cards */
.kpi-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
}
.kpi-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.1;
    color: #f8fafc;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 4px;
}

/* Sentiment Badges */
.badge-pos {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.82rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.badge-neg {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.82rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.badge-neu {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.82rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Keyword & Issue Chips */
.keyword-tag {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #c7d2fe;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.82rem;
    margin: 2px 4px 2px 0;
    font-weight: 500;
}
.issue-tag {
    display: inline-block;
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #fca5a5;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.82rem;
    margin: 2px 4px 2px 0;
    font-weight: 600;
}

/* Insight Box */
.insight-box {
    background: rgba(30, 41, 59, 0.5);
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #e2e8f0;
    font-size: 0.95rem;
}

/* Validation Banner */
.val-banner {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 18px;
    display: flex;
    gap: 28px;
    flex-wrap: wrap;
    font-size: 0.9rem;
    color: #94a3b8;
}
.val-item strong {
    color: #f8fafc;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 10px 0 14px 0;">
    <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; color: #818cf8;">💬 Feedback Analyzer</h2>
    <p style="margin: 2px 0 0 0; font-size: 0.8rem; color: #94a3b8;">Text & Speech Analysis Mini-Project</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Choose Input / Mode:",
    ["Individual Feedback", "Upload CSV / Batch Analysis", "📖 Project Documentation & Viva Guide"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ NLP Calibration")
pos_thresh = st.sidebar.slider(
    "Positive Threshold (Compound >=)",
    min_value=0.01,
    max_value=0.30,
    value=0.05,
    step=0.01,
    help="Reviews with a VADER compound score at or above this threshold are classified as Positive."
)
neg_thresh = st.sidebar.slider(
    "Negative Threshold (Compound <=)",
    min_value=-0.30,
    max_value=-0.01,
    value=-0.05,
    step=0.01,
    help="Reviews with a VADER compound score at or below this threshold are classified as Negative."
)
num_keywords = st.sidebar.slider("Keywords to Extract per Feedback", min_value=3, max_value=8, value=5)

analyzer = FeedbackAnalyzer(pos_threshold=pos_thresh, neg_threshold=neg_thresh)


# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown("""
<div class="header-box">
    <h1 class="header-title">Customer Feedback Analyzer</h1>
    <p class="header-subtitle">Analyze customer feedback using NLP and sentiment analysis.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODE 1 — INDIVIDUAL FEEDBACK
# =========================================================
if app_mode == "Individual Feedback":
    st.markdown("### 📝 Individual Feedback Analysis")
    st.markdown("Enter customer feedback to evaluate sentiment, extract salient keywords, and identify operational concerns in real time.")

    # Quick Presets
    st.markdown("##### 💡 Try an Example:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    preset_choice = None
    
    with col_p1:
        if st.button("⚖️ Quality & Slow Delivery", use_container_width=True):
            preset_choice = "The product quality is good but delivery took too long."
    with col_p2:
        if st.button("🌟 Positive Customer Service", use_container_width=True):
            preset_choice = "Outstanding customer service! The representative resolved my billing question within minutes. Highly recommended!"
    with col_p3:
        if st.button("🚨 Broken Part & Defect", use_container_width=True):
            preset_choice = "The product is not good. The plastic casing arrived broken and cracked, and the app keeps crashing on checkout!"
    with col_p4:
        if st.button("💳 Billing & Cost Complaint", use_container_width=True):
            preset_choice = "Way overpriced for what you get. Got charged twice on my credit card and nobody in billing answered my emails."

    if preset_choice:
        st.session_state["single_input"] = preset_choice

    default_text = st.session_state.get(
        "single_input",
        "The product quality is good but delivery took too long."
    )

    feedback_text = st.text_area(
        "Customer Feedback:",
        value=default_text,
        height=120,
        placeholder="Enter or paste customer feedback here..."
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Feedback", type="primary", use_container_width=True)

    if analyze_clicked or "single_result" in st.session_state:
        # Error handling for empty or whitespace-only input
        if not feedback_text or not feedback_text.strip():
            st.error("Please enter customer feedback before analyzing.")
        elif len(feedback_text.strip()) < 3:
            st.warning("⚠️ The entered feedback is too short to produce meaningful NLP analysis. Please enter at least 3 characters.")
        else:
            with st.spinner("Analyzing feedback through NLP pipeline..."):
                res = analyzer.analyze_single(feedback_text, top_keywords=num_keywords)
                # Compute preprocessing pipeline step-by-step
                prep_stages = preprocess_feedback(feedback_text)
                st.session_state["single_result"] = res
                st.session_state["single_prep"] = prep_stages

            res = st.session_state["single_result"]
            prep_stages = st.session_state.get("single_prep", {})
            st.markdown("<br>", unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([1, 1])

            # LEFT COLUMN: Sentiment & Scores
            with res_col1:
                st.markdown("#### 🎯 Sentiment Analysis")
                sent = res["sentiment"]
                badge_class = "badge-pos" if sent == "Positive" else ("badge-neg" if sent == "Negative" else "badge-neu")
                icon = "🟢" if sent == "Positive" else ("🔴" if sent == "Negative" else "🟡")

                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-size: 1rem; color: #94a3b8; font-weight: 600;">Overall Classification:</span>
                        <span class="{badge_class}">{icon} {sent.upper()}</span>
                    </div>
                    <div style="margin-bottom: 10px;">
                        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Compound Sentiment Score:</div>
                        <div style="font-size: 1.85rem; font-weight: 800; color: #f8fafc;">{res['compound_score']:+.4f}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">Scale: -1.00 (most negative) to +1.00 (most positive)</div>
                    </div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; background: rgba(15, 23, 42, 0.5); padding: 8px 12px; border-radius: 6px;">
                        💡 <strong>Score Interpretation:</strong> {res['explanation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # VADER Score Components Chart
                scores_df = pd.DataFrame({
                    "Component": ["Positive", "Neutral", "Negative"],
                    "Proportion": [res["positive_score"], res["neutral_score"], res["negative_score"]]
                })
                fig_scores = px.bar(
                    scores_df,
                    x="Proportion",
                    y="Component",
                    orientation="h",
                    title="<b>Lexicon Score Breakdown</b>",
                    color="Component",
                    color_discrete_map={"Positive": "#10b981", "Neutral": "#f59e0b", "Negative": "#ef4444"},
                    text_auto='.3f'
                )
                fig_scores.update_layout(
                    template="plotly_dark",
                    height=200,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", size=12),
                    xaxis=dict(range=[0, 1.05], title="Score"),
                    yaxis=dict(title="", autorange="reversed"),
                    margin=dict(t=35, b=20, l=10, r=10),
                    showlegend=False
                )
                st.plotly_chart(fig_scores, use_container_width=True)

            # RIGHT COLUMN: Keywords & Detected Issues
            with res_col2:
                st.markdown("#### 🔑 Important Keywords")
                if res["keywords"]:
                    chips_html = "".join([f"<span class='keyword-tag'>{kw}</span>" for kw in res["keywords"]])
                    st.markdown(f"<div style='margin-bottom: 20px;'>{chips_html}</div>", unsafe_allow_html=True)
                else:
                    st.info("No distinctive keywords extracted from this feedback.")

                st.markdown("#### 🚨 Detected Issues / Topics")
                if res["detected_issues"]:
                    for iss in res["detected_issues"]:
                        kws_str = ", ".join(iss["matched_keywords"])
                        st.markdown(f"""
                        <div style="background: rgba(30, 41, 59, 0.5); border-left: 4px solid #ef4444; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                            <div style="font-weight: 700; color: #fca5a5; font-size: 0.95rem;">
                                {iss['icon']} {iss['category']}
                            </div>
                            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">
                                {iss['description']}
                            </div>
                            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                                Matched Triggers: <code>{kws_str}</code>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: rgba(30, 41, 59, 0.4); border-radius: 8px; padding: 12px; color: #94a3b8; font-size: 0.88rem;">
                        ℹ️ <em>No predefined operational concerns detected in this feedback text.</em>
                    </div>
                    """, unsafe_allow_html=True)

                st.caption(
                    "⚠️ *Note: Issue categories are identified using transparent lexical aspect patterns based on actual feedback words.*"
                )

            # Basic Interpretation Summary
            st.markdown("---")
            st.markdown("#### 📋 Basic Feedback Interpretation")
            interp_col1, interp_col2 = st.columns(2)
            with interp_col1:
                status_text = "Satisfied" if sent == "Positive" else ("Dissatisfied" if sent == "Negative" else "Neutral / Mixed")
                st.markdown(f"**Customer Stance:** `{status_text}`")
                if res["detected_issues"]:
                    detected_str = ", ".join([f"{i['icon']} {i['category']}" for i in res["detected_issues"]])
                    st.markdown(f"**Topics Discussed:** {detected_str}")
                else:
                    st.markdown("**Topics Discussed:** General customer feedback")
            with interp_col2:
                if res["keywords"]:
                    st.markdown(f"**Core Subject Matter:** `{', '.join(res['keywords'][:4])}`")
                if sent == "Positive":
                    st.markdown("**Key Takeaway:** Customer expresses positive evaluation and satisfaction.")
                elif sent == "Negative":
                    st.markdown("**Key Takeaway:** Customer raises dissatisfaction or friction points requiring operational attention.")
                else:
                    st.markdown("**Key Takeaway:** Feedback contains balanced or neutral remarks without extreme polarity.")

            # Preprocessing Pipeline Breakdown (Viva / Educational inspection)
            with st.expander("🔬 View NLP Preprocessing Pipeline Breakdown"):
                st.markdown("""
                Demonstrates how raw text is systematically cleaned, normalized, tokenized, and filtered while preserving crucial negation words:
                """)
                st.json({
                    "1. Raw Feedback": prep_stages.get("raw_text", ""),
                    "2. Cleaned Surface (URLs & HTML stripped)": prep_stages.get("cleaned_surface", ""),
                    "3. Normalized Text (Lowercased & Punctuation stripped)": prep_stages.get("normalized_text", ""),
                    "4. All Tokens": prep_stages.get("tokens", []),
                    "5. Filtered Tokens (Stopwords removed, negations preserved)": prep_stages.get("filtered_tokens", []),
                    "6. Clean Text Output": prep_stages.get("clean_text", "")
                })


# =========================================================
# MODE 2 — CSV BATCH ANALYSIS
# =========================================================
elif app_mode == "Upload CSV / Batch Analysis":
    st.markdown("### 📁 Batch Feedback Dataset Analysis")
    st.markdown("Upload a CSV containing customer reviews or load a pre-built sample dataset for instant analysis.")

    col_up, col_btn_sample = st.columns([3, 2])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload CSV feedback file:",
            type=["csv", "tsv", "txt"],
            help="Upload a CSV with customer feedback/reviews."
        )

    with col_btn_sample:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        sample_choice = st.selectbox(
            "Or load sample demonstration dataset:",
            ["None", "sample_feedback.csv (25 reviews)", "sample_customer_reviews.csv (30 reviews)"]
        )

    raw_df = None
    data_source_title = ""

    # Load from upload or sample dataset
    if uploaded_file is not None:
        try:
            content = uploaded_file.read()
            uploaded_file.seek(0)
            try:
                raw_df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
            except UnicodeDecodeError:
                raw_df = pd.read_csv(io.BytesIO(content), encoding="latin1")

            data_source_title = uploaded_file.name
            st.success(f"Loaded '{uploaded_file.name}' successfully ({len(raw_df)} rows).")
        except Exception as e:
            st.error(f"❌ Could not read the uploaded CSV file: {str(e)}")
            raw_df = None

    elif sample_choice != "None" or "active_batch_df" not in st.session_state:
        target_csv = "sample_feedback.csv" if "25" in sample_choice or sample_choice == "None" else "sample_customer_reviews.csv"
        paths_to_check = [
            os.path.join(os.getcwd(), "data", target_csv),
            os.path.join(os.getcwd(), target_csv),
            os.path.join(os.getcwd(), "data", "sample_feedback.csv")
        ]
        sample_file_path = None
        for p in paths_to_check:
            if os.path.exists(p):
                sample_file_path = p
                break

        if sample_file_path and os.path.exists(sample_file_path):
            raw_df = pd.read_csv(sample_file_path)
            data_source_title = f"{os.path.basename(sample_file_path)} (Demo Dataset)"
            st.info(f"Loaded demo dataset: `{os.path.basename(sample_file_path)}` ({len(raw_df)} reviews).")
        else:
            st.warning("Sample dataset not found. Please upload a CSV file.")

    if raw_df is not None:
        if raw_df.empty:
            st.error("❌ The selected dataset is empty (0 rows). Please upload a valid CSV with data.")
        else:
            st.session_state["active_batch_df"] = raw_df
            st.session_state["active_source_title"] = data_source_title

    if "active_batch_df" in st.session_state and not st.session_state["active_batch_df"].empty:
        batch_df = st.session_state["active_batch_df"].copy()

        # Intelligently identify feedback column
        detected_col = detect_feedback_column(batch_df.columns.tolist())
        if not detected_col:
            st.warning("No suitable feedback column was detected. Please select the correct text column.")
            default_index = 0
        else:
            default_index = batch_df.columns.get_loc(detected_col)

        col_col_select, col_run_btn = st.columns([3, 2])
        with col_col_select:
            selected_text_col = st.selectbox(
                "Select column containing feedback text:",
                options=batch_df.columns.tolist(),
                index=int(default_index),
                help="Common names: feedback, review, comment, text"
            )

        with col_run_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_batch_nlp = st.button("⚡ Process Batch Feedback", type="primary", use_container_width=True)

        # Trigger Analysis or Check Cache
        if run_batch_nlp or "analyzed_batch_df" not in st.session_state or st.session_state.get("last_processed_col") != selected_text_col:
            # Perform dataset validation and cleaning
            clean_df, val_summary = validate_and_clean_dataset(batch_df, selected_text_col)

            if clean_df.empty:
                st.error("❌ All records in the selected column are empty or null.")
            else:
                with st.spinner("Analyzing feedback, extracting keywords, and aggregating customer issues..."):
                    analyzed_res_df = analyzer.analyze_dataset(
                        clean_df,
                        text_column=selected_text_col,
                        top_keywords_per_row=num_keywords
                    )
                    st.session_state["analyzed_batch_df"] = analyzed_res_df
                    st.session_state["last_processed_col"] = selected_text_col
                    st.session_state["val_summary"] = val_summary
                    st.success(f"Analyzed {len(analyzed_res_df)} feedback records successfully!")

        if "analyzed_batch_df" in st.session_state:
            df_analyzed = st.session_state["analyzed_batch_df"]
            val_summary = st.session_state.get("val_summary", {})

            # DATA VALIDATION SUMMARY BANNER (Requirement 17)
            if val_summary:
                st.markdown(f"""
                <div class="val-banner">
                    <span class="val-item">📋 <strong>Rows analyzed:</strong> {val_summary['rows_analyzed']:,}</span>
                    <span class="val-item">🧹 <strong>Empty feedback removed:</strong> {val_summary['empty_rows_removed']}</span>
                    <span class="val-item">🔄 <strong>Duplicates detected:</strong> {val_summary['duplicates_detected']}</span>
                    <span class="val-item">📁 <strong>Available columns:</strong> {', '.join(val_summary['columns'][:6])}{'...' if len(val_summary['columns']) > 6 else ''}</span>
                </div>
                """, unsafe_allow_html=True)

            # Compute Dataset Overview Statistics
            total_records = len(df_analyzed)
            pos_records = (df_analyzed["Sentiment"] == "Positive").sum()
            neg_records = (df_analyzed["Sentiment"] == "Negative").sum()
            neu_records = (df_analyzed["Sentiment"] == "Neutral").sum()

            pos_pct = round((pos_records / total_records) * 100, 1) if total_records > 0 else 0.0
            neg_pct = round((neg_records / total_records) * 100, 1) if total_records > 0 else 0.0
            neu_pct = round((neu_records / total_records) * 100, 1) if total_records > 0 else 0.0
            net_sentiment = round(pos_pct - neg_pct, 1)

            # Issue aggregation
            issues_aggregated = analyze_issues_dataset(
                df_analyzed,
                text_column=selected_text_col,
                sentiment_column="Sentiment"
            )

            # ---------------------------------------------------------
            # Overview KPI Cards
            # ---------------------------------------------------------
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

            with kpi1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Total Feedback</div>
                    <div class="kpi-value">{total_records:,}</div>
                    <div class="kpi-sub">Analyzed entries</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi2:
                st.markdown(f"""
                <div class="kpi-card" style="border-left: 4px solid #10b981;">
                    <div class="kpi-label" style="color: #34d399;">Positive</div>
                    <div class="kpi-value" style="color: #34d399;">{pos_records} <span style="font-size: 1.1rem; font-weight: 500;">({pos_pct}%)</span></div>
                    <div class="kpi-sub">Favorable reviews</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi3:
                st.markdown(f"""
                <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
                    <div class="kpi-label" style="color: #fbbf24;">Neutral</div>
                    <div class="kpi-value" style="color: #fbbf24;">{neu_records} <span style="font-size: 1.1rem; font-weight: 500;">({neu_pct}%)</span></div>
                    <div class="kpi-sub">Balanced feedback</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi4:
                st.markdown(f"""
                <div class="kpi-card" style="border-left: 4px solid #ef4444;">
                    <div class="kpi-label" style="color: #f87171;">Negative</div>
                    <div class="kpi-value" style="color: #f87171;">{neg_records} <span style="font-size: 1.1rem; font-weight: 500;">({neg_pct}%)</span></div>
                    <div class="kpi-sub">Critical feedback</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi5:
                nss_color = "#34d399" if net_sentiment >= 0 else "#f87171"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Net Sentiment</div>
                    <div class="kpi-value" style="color: {nss_color};">{net_sentiment:+0.1f}%</div>
                    <div class="kpi-sub">% Pos − % Neg</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------------------------------------------------
            # Dynamic Feedback Insights (Requirement 12)
            # ---------------------------------------------------------
            st.markdown("#### 💡 Customer Feedback Insights (Calculated from Data)")
            insights_list = analyzer.generate_insights(df_analyzed, issues_aggregated)
            for ins in insights_list:
                st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------------------------------------------------
            # Dashboard Tabs: Visualizations, Issues, Keywords, Explorer
            # ---------------------------------------------------------
            d_tab1, d_tab2, d_tab3, d_tab4 = st.tabs([
                "📊 Sentiment Charts & Trends",
                "🚨 Customer Issue Analysis",
                "🔑 Top Keywords & Word Cloud",
                "📋 Data Explorer & Export"
            ])

            # TAB 1: SENTIMENT CHARTS & TRENDS
            with d_tab1:
                chart_col1, chart_col2 = st.columns(2)
                color_mapping = {
                    "Positive": "#10b981",
                    "Neutral": "#f59e0b",
                    "Negative": "#ef4444"
                }

                with chart_col1:
                    # Sentiment Distribution Donut Chart
                    sent_df = pd.DataFrame({
                        "Sentiment": ["Positive", "Neutral", "Negative"],
                        "Count": [pos_records, neu_records, neg_records]
                    })

                    fig_pie = px.pie(
                        sent_df,
                        names="Sentiment",
                        values="Count",
                        hole=0.55,
                        title="<b>Sentiment Share Distribution</b>",
                        color="Sentiment",
                        color_discrete_map=color_mapping
                    )
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#0f172a', width=2))
                    )
                    fig_pie.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", size=13),
                        margin=dict(t=50, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with chart_col2:
                    # Polarity Score Distribution
                    fig_dist = px.histogram(
                        df_analyzed,
                        x="Sentiment_Score",
                        color="Sentiment",
                        nbins=20,
                        title="<b>Sentiment Score Distribution (-1.0 to +1.0)</b>",
                        color_discrete_map=color_mapping,
                        marginal="box"
                    )
                    fig_dist.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", size=13),
                        xaxis=dict(title="VADER Compound Score", range=[-1.05, 1.05]),
                        yaxis=dict(title="Review Count"),
                        margin=dict(t=50, b=20, l=20, r=20),
                        bargap=0.1
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                # Sentiment Trend over Time (Requirement 11)
                date_column_found = detect_date_column(df_analyzed)
                if date_column_found:
                    st.markdown("---")
                    st.markdown(f"#### 📈 Sentiment Trend Over Time (Detected Date Column: `{date_column_found}`)")
                    try:
                        trend_df = df_analyzed.copy()
                        trend_df["parsed_date"] = pd.to_datetime(trend_df[date_column_found], errors="coerce")
                        trend_df = trend_df.dropna(subset=["parsed_date"])

                        if len(trend_df) >= 3:
                            trend_summary = trend_df.groupby(trend_df["parsed_date"].dt.date).agg(
                                avg_score=("Sentiment_Score", "mean"),
                                total_reviews=("Sentiment_Score", "count"),
                                positive_count=("Sentiment", lambda s: (s == "Positive").sum()),
                                neutral_count=("Sentiment", lambda s: (s == "Neutral").sum()),
                                negative_count=("Sentiment", lambda s: (s == "Negative").sum())
                            ).reset_index().sort_values("parsed_date")

                            t_col1, t_col2 = st.columns([3, 2])
                            with t_col1:
                                fig_trend = go.Figure()
                                fig_trend.add_trace(go.Scatter(
                                    x=trend_summary["parsed_date"],
                                    y=trend_summary["avg_score"],
                                    mode="lines+markers",
                                    name="Avg Sentiment Score",
                                    line=dict(color="#818cf8", width=3),
                                    marker=dict(size=8, color="#6366f1")
                                ))
                                fig_trend.add_hline(y=0.05, line_dash="dot", line_color="#10b981", annotation_text="Positive Threshold (+0.05)")
                                fig_trend.add_hline(y=-0.05, line_dash="dot", line_color="#ef4444", annotation_text="Negative Threshold (-0.05)")
                                fig_trend.update_layout(
                                    title="<b>Average Sentiment Score Over Time</b>",
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(family="Inter", size=12),
                                    xaxis=dict(title="Date"),
                                    yaxis=dict(title="Mean Compound Score", range=[-1.05, 1.05]),
                                    margin=dict(t=50, b=30, l=20, r=20)
                                )
                                st.plotly_chart(fig_trend, use_container_width=True)

                            with t_col2:
                                fig_vol = px.bar(
                                    trend_summary,
                                    x="parsed_date",
                                    y=["positive_count", "neutral_count", "negative_count"],
                                    title="<b>Daily Feedback Volume by Sentiment</b>",
                                    color_discrete_map={
                                        "positive_count": "#10b981",
                                        "neutral_count": "#f59e0b",
                                        "negative_count": "#ef4444"
                                    },
                                    labels={"value": "Count", "variable": "Sentiment", "parsed_date": "Date"}
                                )
                                fig_vol.update_layout(
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(family="Inter", size=12),
                                    barmode="stack",
                                    margin=dict(t=50, b=30, l=20, r=20)
                                )
                                st.plotly_chart(fig_vol, use_container_width=True)
                    except Exception as e:
                        st.info(f"Could not render time-trend for column `{date_column_found}`: {str(e)}")

            # TAB 2: CUSTOMER ISSUE ANALYSIS
            with d_tab2:
                st.markdown("### 🚨 Customer Issue & Topic Analysis")
                st.markdown(
                    "Each operational topic is tracked with actual mention counts and associated sentiment breakdown calculated directly from the uploaded data."
                )

                if issues_aggregated:
                    issue_df = pd.DataFrame(issues_aggregated)

                    i_col1, i_col2 = st.columns([3, 2])

                    with i_col1:
                        # Stacked Bar Chart for Issues by Sentiment
                        fig_issues = px.bar(
                            issue_df,
                            y="category",
                            x=["negative", "neutral", "positive"],
                            title="<b>Issue Mentions Breakdown by Sentiment</b>",
                            orientation='h',
                            color_discrete_map={
                                "negative": "#ef4444",
                                "neutral": "#f59e0b",
                                "positive": "#10b981"
                            },
                            labels={"value": "Total Mentions", "variable": "Sentiment", "category": "Issue Area"}
                        )
                        fig_issues.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter", size=12),
                            yaxis=dict(autorange="reversed"),
                            margin=dict(t=40, b=20, l=20, r=20),
                            barmode="stack"
                        )
                        st.plotly_chart(fig_issues, use_container_width=True)

                    with i_col2:
                        st.markdown("#### 📊 Issue Statistics Table")
                        display_issues_df = issue_df[["category", "mentions", "negative", "neutral", "positive"]].copy()
                        display_issues_df.columns = ["Issue Category", "Mentions", "Negative", "Neutral", "Positive"]
                        st.dataframe(display_issues_df, use_container_width=True, hide_index=True)

                    st.markdown("#### 🗣️ Sample Feedback per Issue Area:")
                    for item in issues_aggregated:
                        with st.expander(f"{item['icon']} {item['category']} — {item['mentions']} mentions (Negative: {item['negative']}, Positive: {item['positive']})"):
                            st.write(f"**Focus:** {item['description']}")
                            if item["sample_quotes"]:
                                for quote in item["sample_quotes"]:
                                    st.markdown(f"> *\"{quote}\"*")
                else:
                    st.info("No specific operational issues detected in the current dataset.")

            # TAB 3: KEYWORDS & WORD CLOUD
            with d_tab3:
                st.markdown("### 🔑 Important Keywords Extraction")
                st.markdown(
                    "Computed using **TF-IDF (Term Frequency-Inverse Document Frequency)** across the corpus, "
                    "identifying high-impact words and bi-grams while removing noise and stopwords."
                )

                corpus_kws = extract_keywords_corpus(df_analyzed[selected_text_col].tolist(), top_n=15)

                kw_col1, kw_col2 = st.columns(2)

                with kw_col1:
                    if corpus_kws:
                        kw_df = pd.DataFrame(corpus_kws, columns=["Keyword", "TFIDF_Score"])
                        fig_kws = px.bar(
                            kw_df,
                            x="TFIDF_Score",
                            y="Keyword",
                            orientation='h',
                            title="<b>Top 15 Keywords / Phrases by TF-IDF Weight</b>",
                            color="TFIDF_Score",
                            color_continuous_scale="Purples",
                            text_auto='.3f'
                        )
                        fig_kws.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Inter", size=12),
                            yaxis=dict(autorange="reversed"),
                            margin=dict(t=40, b=20, l=10, r=10),
                            showlegend=False
                        )
                        st.plotly_chart(fig_kws, use_container_width=True)
                    else:
                        st.info("Not enough text to extract corpus-level keywords.")

                with kw_col2:
                    st.markdown("#### ☁️ Feedback Word Cloud")
                    joined_text = " ".join(df_analyzed["Cleaned_Feedback"].astype(str))
                    if len(joined_text.strip()) > 15:
                        try:
                            wc = WordCloud(
                                width=700,
                                height=420,
                                background_color="#0f172a",
                                colormap="coolwarm",
                                max_words=80
                            ).generate(joined_text)

                            fig_wc, ax = plt.subplots(figsize=(8, 4.8), facecolor="#0f172a")
                            ax.imshow(wc, interpolation="bilinear")
                            ax.axis("off")
                            plt.tight_layout(pad=0)
                            st.pyplot(fig_wc)
                            plt.close(fig_wc)
                        except Exception as e:
                            st.warning(f"Word cloud rendering note: {str(e)}")
                    else:
                        st.info("Insufficient feedback text for word cloud.")

            # TAB 4: DATA EXPLORER & EXPORT
            with d_tab4:
                st.markdown("### 📋 Interactive Feedback Explorer & Export")

                f_c1, f_c2, f_c3 = st.columns([1, 1, 2])
                with f_c1:
                    sent_selection = st.multiselect(
                        "Filter Sentiment:",
                        options=["Positive", "Neutral", "Negative"],
                        default=["Positive", "Neutral", "Negative"]
                    )
                with f_c2:
                    issue_filter_options = ["All Issues"] + list(ISSUE_CATEGORIES.keys())
                    selected_issue_filter = st.selectbox("Filter Issue Area:", options=issue_filter_options)

                with f_c3:
                    text_search = st.text_input("🔍 Search feedback text:", "")

                # Apply Filters
                filtered = df_analyzed[df_analyzed["Sentiment"].isin(sent_selection)].copy()

                if selected_issue_filter != "All Issues":
                    filtered = filtered[filtered["Detected_Issues"].str.contains(selected_issue_filter, case=False, na=False)]

                if text_search.strip():
                    filtered = filtered[filtered[selected_text_col].astype(str).str.contains(text_search, case=False, na=False)]

                st.write(f"Displaying **{len(filtered)} of {len(df_analyzed)}** matching feedback entries:")

                # Column selection for clean display
                display_cols = [selected_text_col, "Sentiment", "Sentiment_Score", "Extracted_Keywords", "Detected_Issues"]
                for c in ["id", "Review_ID", "customer_name", "Customer_Name", "date", "Date", "category", "Rating"]:
                    if c in filtered.columns and c not in display_cols:
                        display_cols.insert(0, c)

                st.dataframe(
                    filtered[display_cols],
                    use_container_width=True,
                    height=380,
                    column_config={
                        "Sentiment_Score": st.column_config.ProgressColumn(
                            "Score",
                            help="VADER compound score from -1.00 to +1.00",
                            format="%+.2f",
                            min_value=-1.0,
                            max_value=1.0
                        )
                    }
                )

                # Export Analyzed Dataset (Requirement 14)
                st.markdown("---")
                download_df = prepare_download_dataframe(df_analyzed, selected_text_col)
                csv_buffer = io.StringIO()
                download_df.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode("utf-8")

                st.download_button(
                    label="📥 Download Analyzed Feedback",
                    data=csv_bytes,
                    file_name="analyzed_customer_feedback.csv",
                    mime="text/csv",
                    type="primary"
                )


# =========================================================
# MODE 3 — DOCUMENTATION & VIVA GUIDE
# =========================================================
elif app_mode == "📖 Project Documentation & Viva Guide":
    st.markdown("### 🎓 Academic & Viva Examination Guide")
    st.markdown("""
    This guide provides clear explanations of the NLP principles, algorithms, and architectural decisions
    used throughout the **Customer Feedback Analyzer**.
    """)

    viva_col1, viva_col2 = st.columns(2)

    with viva_col1:
        st.markdown("""
        #### 1. What is NLP?
        **Natural Language Processing (NLP)** is a subfield of Artificial Intelligence combining computer science and linguistics to enable computers to understand, interpret, and manipulate human language.

        #### 2. What is Sentiment Analysis?
        Sentiment Analysis (opinion mining) computationally determines whether textual data expresses a **Positive**, **Negative**, or **Neutral** attitude towards an entity, product, or experience.

        #### 3. Why Tokenization & Stopword Removal?
        - **Tokenization:** Breaks continuous text into individual words or meaningful tokens.
        - **Stopword Removal:** Eliminates high-frequency grammatical glue words (`and`, `the`, `is`) that carry zero discriminatory sentiment value, reducing dimensionality.
        - **Negation Handling:** Crucially, negation words such as `not`, `no`, and `never` are **preserved**, preventing semantic inversion (e.g. ensuring *"not good"* does not turn into *"good"*).

        #### 4. How does VADER Work?
        **VADER** (*Valence Aware Dictionary and sEntiment Reasoner*) is a rule-based lexicon explicitly tuned for consumer reviews and social text:
        - Uses an empirically validated sentiment lexicon.
        - Accounts for punctuation emphasis (`great!`), capitalization (`TERRIBLE`), degree adverbs (`extremely slow`), and contrastive conjunctions (`good, BUT slow`).
        """)

    with viva_col2:
        st.markdown("""
        #### 5. How are Sentiment Thresholds Defined?
        VADER outputs a normalized **compound score** ranging between **-1.0 and +1.0**:
        - **Positive:** Compound Score $\\ge +0.05$
        - **Negative:** Compound Score $\\le -0.05$
        - **Neutral:** $-0.05 <$ Compound Score $< +0.05$

        #### 6. How are Keywords Extracted?
        Uses **TF-IDF (Term Frequency - Inverse Document Frequency)**:
        $$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\log\\left(\\frac{N}{\\text{DF}(t)}\\right)$$
        - Promotes words that appear frequently within a specific review but rarely across unrelated reviews.
        - Extracts unigrams and bigrams like `customer support`, `product quality`, `fast delivery`.

        #### 7. How are Customer Issues Identified?
        Uses transparent, explainable **Aspect-Based Category Mapping** across seven key customer service touchpoints:
        1. **Product Quality** 📦
        2. **Delivery** 🚚
        3. **Customer Support** 🎧
        4. **Pricing** 💳
        5. **User Experience** ✨
        6. **Performance** ⚡
        7. **Reliability** 🛡️
        """)

    st.markdown("---")
    st.markdown("#### 🏗️ Architecture & Processing Pipeline")
    st.code("""
Customer Feedback
       ↓
Data Validation (Row counting, duplicate detection, empty feedback stripping)
       ↓
Text Preprocessing (Lowercasing, whitespace cleaning, punctuation handling, tokenization, stopword removal with negation preservation)
       ↓
Sentiment Analysis (VADER compound & component calculations)
       ↓
Keyword / Topic Analysis (TF-IDF & N-gram candidate ranking)
       ↓
Issue / Feedback Analysis (Aspect-based category mapping across 7 touchpoints)
       ↓
Visual Insights (Sentiment distribution, time trends, issue breakdowns, word clouds)
       ↓
Downloadable Results (Enriched CSV export with original and analyzed columns)
    """, language="text")
