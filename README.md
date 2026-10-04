# Customer Feedback Analyzer

> **Turn customer feedback into actionable insights using Natural Language Processing (NLP).**

A standalone, explainable NLP web application built with **Python**, **Streamlit**, **NLTK**, and **scikit-learn** designed to analyze customer reviews, detect sentiment polarity, extract high-impact keywords, and surface recurring operational pain points.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK%20VADER-green.svg)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20TF--IDF-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 1. Problem Statement

Modern consumer-facing businesses, e-commerce platforms, and SaaS products receive thousands of customer reviews, survey responses, and support tickets daily across multiple touchpoints.

Manually reviewing every piece of feedback is:
- **Time-Consuming & Expensive:** Requires substantial human resources to manually triage tickets and reviews.
- **Subjective & Inconsistent:** Human annotators often interpret sentiment, urgency, and criticism with bias.
- **Lacking Real-Time Agility:** Critical product bugs, delivery delays, and payment errors often go unnoticed until customer churn escalates.

Organizations need an automated, explainable, and cost-effective NLP system to continuously ingest feedback, quantify customer sentiment, pinpoint specific operational bottlenecks, and track customer satisfaction trends.

---

## 2. Project Objective

The **Customer Feedback Analyzer** automates the end-to-end feedback analysis lifecycle without relying on expensive, black-box paid APIs or heavy cloud LLMs:
1. **Analyze Individual Reviews:** Real-time sentiment evaluation, polarity breakdown, keyword mining, and issue identification.
2. **Batch CSV Processing:** Upload and analyze entire datasets of customer feedback with automated column detection and validation.
3. **Operational Issue Clustering:** Detect mentions across seven key business areas (*Product Quality, Delivery, Customer Support, Pricing, User Experience, Performance, Reliability*).
4. **Data-Driven Insights:** Dynamically compute business metrics, including the **Net Sentiment Score (NSS)**, issue frequency distributions, and friction drivers.

---

## 3. Key Features

- **Dual Operational Modes:**
  - **Mode 1 — Analyze Feedback:** Real-time analysis for single customer reviews with quick sample presets.
  - **Mode 2 — Analyze Dataset:** Scalable batch CSV analysis with interactive dashboards and visual charts.
- **Genuine NLP Sentiment Scoring:**
  - Transparent VADER lexical engine calculating **Positive**, **Neutral**, **Negative**, and **Compound** scores.
- **Salient Keyword & Phrase Extraction:**
  - Sublinear TF-IDF (Term Frequency-Inverse Document Frequency) identifying meaningful unigrams and bigrams (e.g., `customer support`, `product quality`, `fast delivery`).
- **Customer Issue & Concern Detection:**
  - Transparent aspect-based categorization tracking mention counts and sentiment breakdown per operational category.
- **Interactive Visualizations:**
  - Sentiment Share Donut Charts, Polarity Distribution Histograms, Top Keywords Bar Charts, and Feedback Word Clouds.
- **Interactive Review Explorer & Export:**
  - Multi-criteria filtering by sentiment, issue topic, search query, and polarity range, with 1-click enriched CSV export.
- **Academic & Viva Friendly:**
  - Built-in viva reference guide explaining every NLP stage, formula, and threshold.

---

## 4. NLP Pipeline Architecture

```text
                        Customer Feedback Input
                                  │
                                  ▼
                            Text Cleaning
                     (Strips URLs, HTML & Whitespace)
                                  │
                                  ▼
                      Tokenization & Normalization
                  (Lowercasing & Word Segmentation)
                                  │
                                  ▼
                          Stopword Removal
              (Filters English Grammar & Domain Noise Words)
                                  │
                                  ▼
                         Feature Extraction
              (TF-IDF N-grams & Lexicon Valence Intensities)
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
  Sentiment Analysis       Keyword Extraction       Issue & Topic Detection
  (VADER Lexicon Model)    (TF-IDF Uni/Bi-grams)   (7 Operational Aspects)
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                      Customer Insights Engine
              (Net Sentiment Score, Pain Points, KPIs)
                                  │
                                  ▼
                      Interactive Streamlit UI
```

---

## 5. Sentiment Analysis Methodology

The application utilizes **NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)**, an empirically validated rule-based sentiment model optimized for customer feedback:

### Score Breakdown
- **Positive Score (`pos`):** Proportion of words carrying positive sentiment $[0.0, 1.0]$.
- **Neutral Score (`neu`):** Proportion of words carrying neutral sentiment $[0.0, 1.0]$.
- **Negative Score (`neg`):** Proportion of words carrying negative sentiment $[0.0, 1.0]$.
- **Compound Score:** A normalized, weighted composite index calculated by summing the valence scores of each word in the lexicon and adjusting for intensity:
  $$\text{Compound} = \frac{\sum v}{\sqrt{\left(\sum v\right)^2 + \alpha}}$$
  where $v$ is the valence score of a token and $\alpha$ is a normalization parameter (typically $15$). The compound score ranges from **$-1.0$ (most negative)** to **$+1.0$ (most positive)**.

### Classification Thresholds
- **Positive:** $\text{Compound Score} \ge +0.05$
- **Negative:** $\text{Compound Score} \le -0.05$
- **Neutral:** $-0.05 < \text{Compound Score} < +0.05$

*(Thresholds are adjustable via the interactive sidebar in the application).*

---

## 6. Keyword & Keyphrase Extraction

Keyword extraction does **not** rely on static hardcoded lists. Instead, it utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)**:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$

- **Stopword Suppression:** Filters out common English functional words (`the`, `is`, `at`, `which`) and generic domain filler words (`product`, `item`, `bought`, `ordered`).
- **N-Gram Range $(1, 2)$:** Extracts both unigrams (`delivery`, `switch`, `pricing`) and compound phrases (`customer support`, `product quality`, `fast shipping`).
- **Sublinear Term Frequency Scaling:** Replaces raw term count with $1 + \log(\text{TF})$ to avoid giving excessive weight to words repeated redundantly in single reviews.

---

## 7. Customer Issue & Topic Detection

Customer feedback is categorized across seven operational touchpoints:

| Category | Focus Area | Example Trigger Keywords |
| :--- | :--- | :--- |
| **Product Quality** 📦 | Physical build & craftsmanship | *quality, broken, defective, cracked, material, durability* |
| **Delivery** 🚚 | Shipping speed & courier care | *delivery, shipping, delay, package, courier, tracking* |
| **Customer Support** 🎧 | Helpfulness & responsiveness | *support, representative, rude, unhelpful, hung up, hold* |
| **Pricing** 💳 | Value, billing accuracy, fees | *price, cost, expensive, charged twice, refund, fee* |
| **User Experience** ✨ | UI elegance, manual, onboarding | *ui, ux, easy to use, confusing, manual, setup, design* |
| **Performance** ⚡ | Speed, crashes, battery life | *crash, bug, slow, lag, freezing, battery drain* |
| **Reliability** 🛡️ | Stability, data safety, uptime | *reliable, stable, fails, error, lost data, disconnect* |

For batch datasets, the system calculates:
$$\text{Mentions}, \quad \text{Negative Mentions}, \quad \text{Neutral Mentions}, \quad \text{Positive Mentions}$$
directly from the analyzed data without synthetic numbers.

---

## 8. Dataset Ingestion & Validation

- **Format Support:** Supports standard `.csv`, `.tsv`, and `.txt` files.
- **Smart Column Detection:** Automatically detects target feedback columns (`feedback`, `review`, `comment`, `text`, `review_text`), with manual override if needed.
- **Encoding Resilience:** Gracefully falls back from UTF-8 to Latin-1 encoding for files exported from legacy systems.
- **Validation:** Identifies empty files, null rows, and empty feedback entries to prevent processing interruptions.

---

## 9. Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.10+** | Core programming language |
| **Streamlit** | Interactive modern web dashboard framework |
| **Pandas** | High-performance dataset ingestion, manipulation, and filtering |
| **NLTK** | Tokenization, English stopwords, and VADER SentimentIntensityAnalyzer |
| **Scikit-Learn** | TfidfVectorizer for unigram and bigram keyword weighting |
| **Plotly Express / Graph Objects** | Responsive, interactive charts (Donut charts, Histograms, Bar plots) |
| **WordCloud & Matplotlib** | Aesthetic visual word frequency representation |

---

## 10. Project Structure

```text
customer-feedback-analyzer/
│
├── app.py                             # Main Streamlit application
├── 05_customer_feedback_analyzer.py   # Application entry point script
│
├── src/                               # Modular NLP Package
│   ├── __init__.py                    # Module export declarations
│   ├── preprocessing.py               # Text cleaning, normalization, stopwords
│   ├── sentiment.py                   # VADER sentiment scoring and threshold mapping
│   ├── keyword_extractor.py           # TF-IDF unigram & bigram phrase extraction
│   ├── issue_detector.py              # Operational issue taxonomy & cross-tabulation
│   └── analyzer.py                    # End-to-end pipeline orchestrator & insights
│
├── data/
│   └── sample_feedback.csv            # 25 realistic demonstration reviews
│
├── tests/
│   └── test_nlp_pipeline.py           # Unit and integration test suite
│
├── requirements.txt                   # Production dependencies
├── README.md                          # Comprehensive documentation & Viva guide
└── .gitignore                         # Standard Python & Streamlit exclusions
```

---

## 11. Installation & Setup

### Prerequisites
- Python 3.10 or higher installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/Santhosh-Udaiyar/customer-feedback-analyzer.git
cd customer-feedback-analyzer
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Essential NLTK Resources (One-Time)
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 12. Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```
*Alternatively, you can also run:*
```bash
streamlit run 05_customer_feedback_analyzer.py
```

The application will open in your default browser at:
`http://localhost:8501`

---

## 13. Example Input & Expected Output

### Input Feedback:
> *"The product quality is excellent, but the delivery was very slow and customer support did not respond."*

### Analysis Output:
- **Sentiment Classification:** `Negative` (or `Neutral` depending on threshold)
- **Compound Score:** `-0.3182`
- **Component Breakdown:** Positive: `0.182`, Neutral: `0.545`, Negative: `0.273`
- **Extracted Keywords:** `product quality`, `delivery`, `customer support`, `slow`, `excellent`
- **Detected Issues:**
  - 📦 **Product Quality:** *quality, excellent*
  - 🚚 **Delivery:** *delivery, slow*
  - 🎧 **Customer Support:** *customer support, support, respond*

---

## 14. Testing

Run the test suite to verify the NLP pipeline:

```bash
python -m unittest tests/test_nlp_pipeline.py
```

### Test Coverage:
- Text cleaning (URL, HTML tag, and whitespace stripping)
- Tokenization & stopword suppression
- Positive, Negative, and Neutral sentiment evaluation
- Unigram and bigram keyword extraction
- Transparent operational issue detection
- Batch dataset analysis and insight generation

---

## 15. College Viva Questions & Answers

<details>
<summary><strong>Click to expand Viva Q&A Guide</strong></summary>

### Q1: What is NLP and why is it important in customer service?
> **Answer:** Natural Language Processing (NLP) is a branch of Artificial Intelligence that enables computers to understand and extract meaning from human language. In customer service, NLP automates the triage of high-volume unstructured feedback, identifying urgent complaints (e.g., billing errors or defective products) without manual reading.

### Q2: Why did you choose VADER instead of training a machine learning classifier?
> **Answer:** VADER is a rule-based lexicon specifically tailored for consumer reviews and short text. It understands capitalization emphasis (`TERRIBLE`), punctuation (`great!`), degree modifiers (`very slow`), and negation (`not good`). It requires zero training data, executes with low latency, and is explainable for academic inspection.

### Q3: What is the significance of the VADER compound score?
> **Answer:** The compound score is a single normalized metric ranging between -1.0 (extremely negative) and +1.0 (extremely positive). By setting empirical thresholds (e.g., $\ge +0.05$ for positive and $\le -0.05$ for negative), we map continuous sentiment valence into actionable categories.

### Q4: How does TF-IDF extract keywords?
> **Answer:** TF-IDF balances Term Frequency (how often a term appears in a document) with Inverse Document Frequency (how unique that term is across the entire corpus). Words that appear frequently in a single review but rarely in others receive the highest score.

### Q5: How are customer issues detected without an LLM?
> **Answer:** We employ an explainable aspect-based lexical pattern matching system across seven operational categories (*Product Quality, Delivery, Customer Support, Pricing, User Experience, Performance, Reliability*). Each review is evaluated against domain trigger words, and metrics (mentions, negative vs positive counts) are aggregated from the dataset.

</details>

---

## 16. Future Enhancements

- [ ] **Multilingual Sentiment Analysis:** Integration with XLM-RoBERTa for cross-lingual customer reviews.
- [ ] **Aspect-Based Sentiment Analysis (ABSA):** Assign separate sentiment polarities to different aspects within a single sentence (e.g., *Positive* for Quality, *Negative* for Delivery).
- [ ] **Transformer-Based Embeddings:** Sentence-Transformers for semantic similarity clustering of open-ended complaints.
- [ ] **RAG-Powered Feedback Querying:** Retrieval-Augmented Generation allowing managers to chat with customer review repositories.
- [ ] **Real-Time Webhook Ingestion:** Ingest feedback directly from Zendesk, Freshdesk, and Google Play Store APIs.

---

## 17. License

This project is licensed under the [MIT License](LICENSE).
