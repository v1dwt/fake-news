# TruthLens: Advanced Fraud & Fake News Detection System

**TruthLens** is an end-to-end, multi-layered Forensic Natural Language Processing (NLP) and Machine Learning platform designed to detect **Fraudulent News, Financial Scams, Clickbait Hoaxes, and Deceptive Misinformation**.

Built with academic compliance to **IEEE Std 830-1998**, TruthLens addresses key vulnerabilities in standard fake news classifiers by introducing **publisher de-biasing**, **domain-specific fraud pattern detection**, **multi-model ML consensus**, and **Explainable AI (XAI)**.

---

## 🚀 Key Improvements & Innovations

### 1. Elimination of Publisher Data Leakage (The "Reuters" Artifact)
* **The Problem:** In standard academic training using the ISOT dataset, legitimate articles almost universally start with wire datelines like `WASHINGTON (Reuters) -`. Standard TF-IDF classifiers inadvertently memorize `reuters` as the strongest feature for "Real News" (and flag any legitimate news lacking the word "reuters" as fake with 80%+ confidence).
* **The Solution in TruthLens:** We built a dedicated cleaning pipeline (`clean_wire_and_agency_artifacts`) in `model_trainer.py` that strips wire datelines, photo attributions (`image via...`), and publisher signatures. This forces the ML models to learn genuine linguistic, semantic, and syntactic deception cues.

### 2. Domain-Specific Forensic Fraud & Scam Analyzer
Fraud news differs from general political misinformation through targeted psychological manipulation:
* **Financial Scam Detection:** Identifies get-rich-quick schemes, guaranteed high-yield ROI claims, crypto doubler formulas, and bogus trading algorithms.
* **Phishing & Impersonation Detection:** Flags regulatory impersonation (IRS, tax authorities, central banks), fake account suspension warnings, and credential harvesting triggers.
* **Sensationalism & Clickbait Index (0–100):** Measures uppercase letter density, exaggerated punctuation (`!!!`, `???`), and clickbait hyperbole ("shocking revelation", "miracle cure", "you won't believe").
* **Urgency & Psychological Pressure Index (0–100):** Detects artificial deadlines ("act now", "immediate action required", "limited slots") used to bypass critical thinking.
* **Journalistic Attribution & Credibility Score:** Analyzes direct quotation marks and formal citation phrases.

### 3. Multi-Model Consensus Engine
Rather than relying on a single black-box classifier, TruthLens evaluates articles through an ensemble:
1. **Logistic Regression (Primary):** Calibrated probabilistic baseline (98.86% Test Accuracy, 98.91% F1).
2. **Multinomial Naive Bayes:** Probabilistic token distribution classifier (95.65% Test Accuracy).
3. **Passive-Aggressive Classifier:** Margin-maximizing linear classifier (99.30% Test Accuracy).

### 4. Explainable AI (XAI) Word Inspector
TruthLens provides mathematical transparency by computing active token contributions:
$$\text{Impact}(t) = \text{TF-IDF}(t) \times w_t$$
* **Suspicious Terms (Red Badges):** Words pushing the prediction toward Fraud/Deception.
* **Credibility Terms (Green Badges):** Words indicating formal journalistic or objective reporting.

### 5. Reverse News Search & Verified Media Corroboration
To cross-reference claims against the real-world journalistic record, TruthLens incorporates a real-time Reverse Search engine (`news_search.py`):
* **Intelligent Query Extraction:** Cleanses clickbait noise and derives the core entity query.
* **Live Verified Wire Search:** Queries Google News RSS and cross-references against 25+ verified tier-1 publishers (Reuters, BBC, WSJ, Bloomberg, AP, The Hindu, Financial Times).
* **Corroboration & Fact-Check Assessment:**
  - 🟢 **Verified Coverage Found:** Multiple credible outlets actively reporting on the topic.
  - 🔴 **Debunk / Scam Warning Detected:** Fact-checkers (Snopes, PolitiFact, etc.) have flagged the claim as a hoax.
  - 🟡 **Unverified Claim:** Zero credible reporting exists for an extraordinary claim.
* **1-Click External Import:** Users can click "⚡ Analyze Story" on any retrieved article to pipe it directly into TruthLens (fulfilling IEEE SRS FR-3.2).

---

## 📁 Project Architecture

```
d:/empt/
├── app.py                     # Flask Web Application & JSON REST API Server
├── fraud_detector.py          # Core Fraud Forensic Engine & XAI Attributor
├── news_search.py             # Reverse News Search & Media Corroboration Engine
├── model_trainer.py           # De-biased multi-model training pipeline
├── test_app.py                # Automated 8-point system test suite
├── test_reverse_search.py     # Automated 5-point reverse search test suite
├── requirements.txt           # Python dependency specifications
├── model.pkl                  # Fitted de-biased Logistic Regression model
├── nb_model.pkl               # Fitted Multinomial Naive Bayes model
├── pac_model.pkl              # Fitted Passive-Aggressive classifier
├── vectorizer.pkl             # Fitted 25,000-feature TF-IDF vectorizer (unigrams + bigrams)
├── model_metrics.json         # Exported model evaluation metrics & confusion matrices
├── templates/
│   └── index.html             # High-performance, dark-themed responsive dashboard
└── archive/
    ├── Fake.csv               # ISOT Fake News Dataset (23,481 articles)
    └── True.csv               # ISOT True News Dataset (21,417 articles)
```

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+ (Tested on Python 3.11 / 3.14)
* pip package manager

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Retrain Models
To retrain the de-biased models on the raw ISOT dataset:
```bash
python model_trainer.py
```
*Average training time: ~50 seconds for 44,898 articles.*

### 3. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### 4. Run Automated Tests
```bash
python test_app.py
```

---

## 🌐 REST API Usage

TruthLens provides REST endpoints conforming to IEEE SRS specifications (`FR-1.3` and `FR-3.3`).

### 1. Analyze an Article
**Endpoint:** `POST /api/analyze` (or `POST /api/predict`)

**Request Payload:**
```json
{
  "headline": "Secret Bitcoin Loophole Guarantees 500% Returns",
  "content": "Turn $100 into $10,000 overnight with this new automated trading bot! Wire money immediately before it's deleted!"
}
```

**Response (Summary):**
```json
{
  "fraud_risk_score": 99.0,
  "risk_level": "HIGH FRAUD RISK",
  "primary_verdict": "FAKE / FRAUD NEWS",
  "models": {
    "consensus": "3/3 Models Flagged as Deceptive",
    "logistic_regression": { "verdict": "Fake / Fraud", "fake_prob_pct": 89.2 },
    "naive_bayes": { "verdict": "Fake / Fraud", "fake_prob_pct": 98.4 },
    "passive_aggressive": { "verdict": "Fake / Fraud", "confidence_pct": 99.1 }
  },
  "forensics": {
    "scams": {
      "scam_score": 70,
      "detected_patterns": [
        { "category": "Financial / Investment Scam", "label": "Guaranteed Investment Return Claim" },
        { "category": "Financial / Investment Scam", "label": "Unrealistic Wealth Multiplication" }
      ]
    },
    "sensationalism": { "score": 42.5, "caps_ratio_pct": 14.2 },
    "urgency": { "score": 66.0, "urgency_hits": ["immediately", "act now"] }
  },
  "xai": {
    "suspicious_terms": [
      { "word": "wire", "weight": 1.103 },
      { "word": "secret", "weight": 0.350 }
    ],
    "credible_terms": []
  },
  "elapsed_ms": 14.8
}
```

### 2. Pre-Configured Test Samples
**Endpoint:** `GET /api/samples`

### 3. Model Training Metrics
**Endpoint:** `GET /api/metrics`

---

## 🎓 Viva & Project Presentation Guide

### Common Questions Asked by Evaluators:

1. **Why did you use TF-IDF with unigrams and bigrams instead of just word count?**
   * *Answer:* Bag-of-words simply counts frequency, which overweights common, uninformative words. TF-IDF downweights ubiquitous words while magnifying distinctive terms. Incorporating bigrams (`ngram_range=(1,2)`) allows the model to capture deceptive phrases like *"guaranteed profit"*, *"risk free"*, or *"act now"* which convey crucial semantic meaning together.

2. **What was the major flaw in the original dataset, and how did you resolve it?**
   * *Answer:* The ISOT True news dataset consists of Reuters wire articles that began with `WASHINGTON (Reuters) -`. Untreated baseline models suffer from severe dataset leakage by simply checking if "Reuters" is present. We implemented dateline and agency stripping before vectorization, ensuring the model generalizes to any news publisher (BBC, Bloomberg, etc.).

3. **How does your system distinguish between fake political news and financial fraud news?**
   * *Answer:* Political fake news often relies on ideological slant and hyperbole, whereas financial fraud relies on coercive urgency, unrealistic ROI promises, crypto doublers, and phishing formulas. TruthLens couples statistical ML classifiers with a domain-specific forensic rule engine (`detect_fraud_and_scams`, `analyze_urgency`) to flag predatory scams that standard text classifiers might overlook.

4. **What is Explainable AI (XAI), and how does TruthLens implement it?**
   * *Answer:* Neural networks and complex models often function as black boxes. In TruthLens, we compute the mathematical product of the document's TF-IDF vector and the trained linear model coefficients ($TF\text{-}IDF_i \times \beta_i$). This decomposes the prediction into transparent positive (fraud-indicative) and negative (credibility-indicative) word weights displayed in the user interface.
