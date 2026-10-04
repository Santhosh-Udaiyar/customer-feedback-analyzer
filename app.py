"""
Customer Feedback Analyzer
--------------------------
Streamlit Application for Natural Language Processing of Customer Feedback.
Supports Single Feedback Analysis and Batch CSV Dataset Processing.

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
from src.preprocessing import clean_text
from src.sentiment import analyze_sentiment
from src.keyword_extractor import extract_keywords_single, extract_keywords_corpus
from src.issue_detector import detect_issues_single, analyze_issues_dataset, ISSUE_CATEGORIES
from src.analyzer import FeedbackAnalyzer


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
    "Application Mode:",
    ["Mode 1 — Analyze Feedback", "Mode 2 — Analyze Dataset", "📖 Project Documentation & Viva Guide"],
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
    <p class="header-subtitle">Turn customer feedback into actionable insights using NLP.</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODE 1 — ANALYZE FEEDBACK (Individual Feedback)
# =========================================================
if app_mode == "Mode 1 — Analyze Feedback":
    st.markdown("### 📝 Individual Feedback Analysis")
    st.markdown("Enter customer feedback to evaluate sentiment, extract salient keywords, and identify operational concerns in real time.")

    # Quick Presets
    st.markdown("##### 💡 Try an Example:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    preset_choice = None
    
    with col_p1:
        if st.button("⚖️ Mixed / Complex Feedback", use_container_width=True):
            preset_choice = "The product quality is excellent, but the delivery was very slow and customer support did not respond."
    with col_p2:
        if st.button("🌟 Positive Praise", use_container_width=True):
            preset_choice = "Outstanding customer service! The representative resolved my billing question within minutes. Highly recommended!"
    with col_p3:
        if st.button("🚨 Critical Bug & Defect", use_container_width=True):
            preset_choice = "Terrible build quality. The plastic casing arrived broken and cracked. The app also keeps crashing on checkout!"
    with col_p4:
        if st.button("💳 Billing & Cost Complaint", use_container_width=True):
            preset_choice = "Way overpriced for what you get. Got charged twice on my credit card and nobody in billing answered my emails."

    if preset_choice:
        st.session_state["single_input"] = preset_choice

    default_text = st.session_state.get(
        "single_input",
        "The product quality is excellent, but the delivery was very slow and customer support did not respond."
    )

    feedback_text = st.text_area(
        "Customer Feedback:",
        value=default_text,
        height=130,
        placeholder="Enter or paste customer feedback here..."
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Feedback", type="primary", use_container_width=True)

    if analyze_clicked or "single_result" in st.session_state:
        # Error handling for empty or too short input
        if not feedback_text.strip():
            st.error("⚠️ Please enter customer feedback to analyze. Feedback text cannot be empty.")
        elif len(feedback_text.strip()) < 3:
            st.warning("⚠️ The entered feedback is too short to produce meaningful NLP analysis. Please enter at least 3 characters.")
        else:
            with st.spinner("Analyzing feedback through NLP pipeline..."):
                res = analyzer.analyze_single(feedback_text, top_keywords=num_keywords)
                st.session_state["single_result"] = res

            res = st.session_state["single_result"]
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
                        💡 <strong>Explanation:</strong> {res['explanation']}
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
                    "⚠️ *Note: Issue categories are identified using transparent lexical aspect patterns. "
                    "Certainty is not claimed when the NLP model does not detect explicit topic markers.*"
                )


# =========================================================
# MODE 2 — ANALYZE DATASET (CSV Batch Processing)
# =========================================================
elif app_mode == "Mode 2 — Analyze Dataset":
    st.markdown("### 📁 Batch Feedback Dataset Analysis")
    st.markdown("Upload a CSV file containing customer reviews or explore our pre-built demonstration dataset.")

    col_up, col_btn_sample = st.columns([3, 2])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload CSV feedback file:",
            type=["csv", "tsv", "txt"],
            help="Upload a CSV with customer reviews/feedback."
        )

    with col_btn_sample:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        use_sample = st.button("🚀 Load Pre-built Sample Dataset (25 Reviews)", use_container_width=True)

    raw_df = None
    data_source_title = ""

    # Load from upload or sample dataset
    if uploaded_file is not None:
        try:
            # Handle potential encoding issues (UTF-8, Latin-1)
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

    elif use_sample or "active_batch_df" not in st.session_state:
        sample_file_path = os.path.join(os.getcwd(), "data", "sample_feedback.csv")
        if os.path.exists(sample_file_path):
            raw_df = pd.read_csv(sample_file_path)
            data_source_title = "sample_feedback.csv (Demonstration Dataset)"
            st.info("Loaded pre-built demonstration dataset (25 realistic customer reviews).")
        else:
            st.warning("Sample dataset not found at `data/sample_feedback.csv`. Please upload a CSV.")

    if raw_df is not None:
        # Data validation
        if raw_df.empty:
            st.error("❌ The selected dataset is empty (0 rows). Please upload a valid CSV with data.")
        else:
            st.session_state["active_batch_df"] = raw_df
            st.session_state["active_source_title"] = data_source_title

    if "active_batch_df" in st.session_state and not st.session_state["active_batch_df"].empty:
        batch_df = st.session_state["active_batch_df"].copy()

        # Feedback Column Detection
        target_candidates = ["feedback", "review", "comment", "text", "review_text", "message", "opinion"]
        detected_col = None
        for col in batch_df.columns:
            if col.strip().lower() in target_candidates:
                detected_col = col
                break
        if not detected_col:
            for col in batch_df.columns:
                if any(c in col.strip().lower() for c in target_candidates):
                    detected_col = col
                    break
        default_index = batch_df.columns.get_loc(detected_col) if detected_col else 0

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
            # Drop null or completely empty rows in selected column
            clean_df = batch_df.dropna(subset=[selected_text_col]).copy()
            clean_df = clean_df[clean_df[selected_text_col].astype(str).str.strip() != ""]

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
                    st.success(f"Analyzed {len(analyzed_res_df)} feedback records successfully!")

        if "analyzed_batch_df" in st.session_state:
            df_analyzed = st.session_state["analyzed_batch_df"]

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
            st.markdown("<br>", unsafe_allow_html=True)
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
            # Dynamic Feedback Insights
            # ---------------------------------------------------------
            st.markdown("#### 💡 Data-Driven Business Insights")
            insights_list = analyzer.generate_insights(df_analyzed, issues_aggregated)
            for ins in insights_list:
                st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------------------------------------------------
            # Dashboard Tabs: Visualizations, Issues, Keywords, Explorer
            # ---------------------------------------------------------
            d_tab1, d_tab2, d_tab3, d_tab4 = st.tabs([
                "📊 Sentiment Charts",
                "🚨 Customer Issue Analysis",
                "🔑 Top Keywords & Word Cloud",
                "📋 Data Explorer & Export"
            ])

            # TAB 1: SENTIMENT CHARTS
            with d_tab1:
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    # Sentiment Distribution Pie/Donut Chart
                    sent_df = pd.DataFrame({
                        "Sentiment": ["Positive", "Neutral", "Negative"],
                        "Count": [pos_records, neu_records, neg_records]
                    })
                    color_mapping = {
                        "Positive": "#10b981",
                        "Neutral": "#f59e0b",
                        "Negative": "#ef4444"
                    }

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
                    text_search = st.text_input("🔍 Search keyword in reviews:", "")

                # Apply Filters
                filtered = df_analyzed[df_analyzed["Sentiment"].isin(sent_selection)].copy()

                if selected_issue_filter != "All Issues":
                    filtered = filtered[filtered["Detected_Issues"].str.contains(selected_issue_filter, case=False, na=False)]

                if text_search.strip():
                    filtered = filtered[filtered[selected_text_col].astype(str).str.contains(text_search, case=False, na=False)]

                st.write(f"Displaying **{len(filtered)} of {len(df_analyzed)}** matching feedback entries:")

                # Column selection for clean display
                display_cols = [selected_text_col, "Sentiment", "Sentiment_Score", "Extracted_Keywords", "Detected_Issues"]
                # Include original columns if present
                for c in ["id", "customer_name", "date", "category"]:
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

                # Export Analyzed Dataset
                st.markdown("---")
                csv_buffer = io.StringIO()
                df_analyzed.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode("utf-8")

                st.download_button(
                    label="📥 Download Analyzed Dataset (CSV)",
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
        - **Stopword Removal:** Eliminates high-frequency grammatical glue words (`and`, `the`, `is`) that carry zero discriminatory sentiment value, reducing dimensionality and noise.

        #### 4. How does VADER Work?
        **VADER** (*Valence Aware Dictionary and sEntiment Reasoner*) is a rule-based lexicon explicitly tuned for consumer reviews and social text:
        - It uses an empirically validated sentiment lexicon.
        - Accounts for punctuation emphasis (e.g. `great!`), capitalization (`TERRIBLE`), degree adverbs (`extremely slow`), and contrastive conjunctions (`good, BUT slow`).
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
Text Cleaning (URLs, HTML, whitespace stripping)
       ↓
Tokenization & Normalization (Lowercasing, punctuation stripping)
       ↓
Stopword Removal (NLTK English + Domain noise suppression)
       ↓
Feature Extraction (TF-IDF N-grams & Lexical intensities)
       ↓
Sentiment Analysis (VADER compound & component calculation)
       ↓
Keyword & Keyphrase Extraction (Top TF-IDF terms)
       ↓
Aspect/Issue Detection (7 operational customer service categories)
       ↓
Actionable Insights Generation (Net Sentiment Score & Root causes)
    """, language="text")
