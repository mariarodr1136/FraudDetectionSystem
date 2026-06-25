# FraudWatch — Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8+-3776AB) ![Flask](https://img.shields.io/badge/Flask-3.0-000000) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E) ![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75) ![Render](https://img.shields.io/badge/Deploy-Render-46E3B7)

**FraudWatch** is a full-stack machine learning application that detects fraudulent credit card transactions with **99.99% accuracy**. A scikit-learn Random Forest classifier — trained on 284,807 real-world transactions with `class_weight='balanced'` to handle extreme class imbalance — is served through a Flask REST API and paired with an interactive Plotly dashboard showing live predictions, a confusion matrix heatmap, and ranked feature importance charts. Every metric is computed on a stratified held-out test set so the numbers reflect true generalization, not training performance.

---

Live Application: https://frauddetectionsystem-qcmh.onrender.com/

Note: The live app is hosted on Render's free tier, so the backend may take 1-2 minutes to wake up after inactivity.

---

https://github.com/user-attachments/assets/f47355f8-5fc8-4263-bf0f-65a650807e83

---

## Table of Contents
- [Features](#features)
- [Model Performance](#model-performance)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Languages & Frameworks Used](#languages--frameworks-used)
- [Code Structure](#code-structure)
- [Installation](#installation)
- [Requirements](#requirements)
- [Inspiration](#inspiration)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Features

**Dashboard**
- **Five metric cards**: Accuracy, Precision, Recall, F1-Score, and False Positive Rate at the top of the page — each color-coded with an SVG icon and a gradient bottom accent bar
- **Interactive bar chart**: Per-metric bar chart using a curated indigo → violet → sky → teal → rose colorscale with percentage labels, rendered inline via Plotly
- **Confusion matrix heatmap**: TP / TN / FP / FN counts from the held-out test set visualized as a 4-stop lavender-to-indigo gradient heatmap with cell gap separation
- **Feature importance chart**: Top 15 PCA components ranked by Random Forest importance score and color-encoded by magnitude

**Model & Pipeline**
- **Stratified split**: 80/20 train/test split with `stratify=y` to preserve the 0.17% fraud class in both sets
- **Class imbalance handling**: `class_weight='balanced'` weights each class inversely proportional to its frequency — no oversampling or SMOTE required
- **Model persistence**: Classifier and scaler serialized with joblib and committed to the repo — the server loads them at startup with no retraining delay
- **Pre-computed metrics**: All dashboard numbers are derived from TP / FP / FN / TN constants at import time; no in-memory dataset required at runtime

**Live Prediction API**
- **Random transaction loader**: Generates a synthetic transaction by sampling from scaler statistics and populates the 12-feature tile grid in the dashboard
- **Real-time inference**: POSTs all 30 features to `/predict` and returns a prediction and fraud probability in under 100ms
- **REST endpoints**: Clean JSON API at `/random_transaction` and `/predict` for external integrations

**UI & Polish**
- **Plus Jakarta Sans**: Modern sans-serif font loaded via Google Fonts across all UI elements
- **Frosted glass navbar**: Sticky nav with `backdrop-filter: blur(20px)`, gradient violet brand icon, and animated live model badge
- **Gradient header title**: Page title with indigo-to-cyan CSS `background-clip: text` gradient
- **Responsive layout**: CSS Grid with breakpoints at 1100px and 768px — metric grid collapses from 5 → 3 → 2 columns
- **Smooth interactions**: Card hover lift, feature tile hover highlight, button press states, and spinner loading states on all async actions

---

## Model Performance

Evaluated on **56,962 held-out test transactions** (~98 fraud cases):

| Metric | Score | What It Means |
|---|---|---|
| **Accuracy** | **99.99%** | Correct predictions across all transactions |
| **Precision** | **100.0%** | Every flagged transaction is genuinely fraudulent |
| **Recall (TPR)** | **94.90%** | ~95 in 100 fraudulent transactions successfully caught |
| **F1-Score** | **97.38%** | Harmonic mean of precision and recall |
| **False Positive Rate** | **0.0%** | No legitimate transactions incorrectly flagged |

> Zero false positives across 56,864 legitimate test transactions — making this model practical for real-world card flagging without alert fatigue.

---

## How It Works

```
284,807 transactions (Kaggle Credit Card Fraud Dataset)
        │
        ▼
  Train / Test Split (80/20, stratified)
        │
   ┌────┴────┐
   │         │
Training   Test Set
  Set      (56,962 rows)
   │
   ▼
StandardScaler.fit_transform()
   │
   ▼
RandomForestClassifier
  • 100 estimators
  • class_weight='balanced'
  • n_jobs=-1
   │
   ▼
joblib.dump() → model.joblib + scaler.joblib
   │
   ▼
Flask Dashboard + REST API
  • /                   → interactive Plotly dashboard
  • /random_transaction → synthetic test transaction
  • /predict            → real-time inference
```

---

## Dataset

Source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)

Transactions made by European cardholders in September 2013 — 284,807 transactions over two days, 492 fraudulent (0.172%). Original features are confidential; V1–V28 are PCA-transformed principal components provided by the dataset authors.

| Feature | Description |
|---|---|
| `V1` – `V28` | PCA principal components (confidential source features) |
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `Amount` | Transaction amount in euros |
| `Class` | Target — `1` = fraudulent, `0` = legitimate |

---

## Languages & Frameworks Used

### Machine Learning
- **scikit-learn 1.6**: `RandomForestClassifier` with balanced class weights, `StandardScaler` for feature normalization, and `train_test_split` with stratification
- **NumPy 2.2**: Array operations for batch transaction generation and importance score ranking
- **Pandas 2.2**: DataFrame construction for scaler-compatible named-feature input
- **joblib 1.4**: Model and scaler serialization — persisted artifacts mean zero retraining on server start

### Backend
- **Python 3.8+**: Core language
- **Flask 3.0**: Lightweight web framework handling three routes (`/`, `/random_transaction`, `/predict`)
- **Gunicorn**: WSGI server for production deployment on Render

### Visualization
- **Plotly 5.24**: All three charts (bar, heatmap, horizontal bar) rendered as inline HTML fragments — no static files, no separate JS bundle

### Frontend
- **HTML5 / CSS3 / JavaScript (ES6+)**: Single-template dashboard with CSS custom properties, CSS Grid layouts, async `fetch` calls, and animated UI states — no frontend framework
- **Plus Jakarta Sans** (Google Fonts): Primary typeface

### Deployment
- **Render**: Free-tier web service — Gunicorn process defined in `Procfile`, pre-trained artifacts committed to the repo so the server starts without retraining

### Version Control
- **Git / GitHub**: Source hosting

---

## Code Structure

```
FraudDetectionSystem/
├── app.py              # Flask app, ML pipeline, chart generation, REST API
├── model.joblib        # Pre-trained Random Forest (100 estimators, balanced weights)
├── scaler.joblib       # Fitted StandardScaler (30 features: Time, V1–V28, Amount)
├── requirements.txt    # Python dependencies
├── Procfile            # Gunicorn process definition for Render
├── templates/
│   └── index.html      # Dashboard — styles, chart containers, live demo, JS
└── static/
    └── background.jpg  # Static asset
```

Key sections inside `app.py`:
- **Model loading** — `joblib.load()` for classifier and scaler; feature names pulled from `scaler.feature_names_in_`
- **Pre-computed metrics** — TP / FP / FN / TN constants; accuracy, precision, recall, F1, and FPR derived once at startup
- **`_gen_transaction(fraud, rng)`** — generates synthetic transactions by sampling from scaler mean and scale; fraud samples shift key PCA features (V1, V3, V14, V17) by several standard deviations to trigger model detection
- **`build_samples()`** — batch-generates 400 transactions, runs inference in one pass, and selects the first 3 correctly-predicted fraud and 3 legitimate for the dashboard table
- **Chart functions** — `metrics_chart_html()`, `conf_matrix_html()`, `feature_importance_html()` produce Plotly HTML fragments at startup; Plotly JS loaded once via CDN
- **`/predict`** — accepts a JSON feature dict, wraps in a named DataFrame, scales, infers, and returns `{prediction, fraud_probability}`

Key sections inside `templates/index.html`:
- **`<style>`** — CSS custom properties, navbar, page header, metric cards, chart cards, table, live demo card, result banner, and responsive breakpoints
- **Metric grid** — five color-coded cards (emerald, violet, amber, sky, rose) with SVG icons, bold values, and gradient bottom bars
- **Charts section** — three `chart-card` wrappers around Plotly HTML fragments, each with a title and tag header
- **Live demo** — feature tile grid populated by `/random_transaction`; SVG result banner rendered from `/predict` response
- **`<script>`** — `loadTransaction()` and `runPrediction()` async functions with button spinner states and inline SVG result icons

---

## Installation

### 1. Clone
```bash
git clone https://github.com/mariarodr1136/FraudDetectionSystem.git
cd FraudDetectionSystem
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python app.py
```

Visit [http://localhost:3000](http://localhost:3000) in your browser.

> The pre-trained `model.joblib` and `scaler.joblib` are included in the repo — no dataset download required. The app generates synthetic transactions from scaler statistics for the dashboard table and live prediction demo.

---

## Requirements
- Python 3.8+
- Dependencies in `requirements.txt`: Flask, scikit-learn, Plotly, Pandas, NumPy, joblib, Gunicorn
- No dataset download needed — pre-trained model artifacts are committed to the repo

---

## Inspiration

Credit card fraud detection is a textbook class-imbalance problem: the signal is real, the stakes are high, and the naive model — predict everything as legitimate — scores 99.8% accuracy while catching zero fraud. The interesting engineering is in the gap between that number and something actually useful: handling skewed classes, choosing the right tradeoff between precision and recall, and building a dashboard honest enough to show false negatives and false positives alongside the headline accuracy figure.

The dashboard is designed to make the model legible rather than just impressive. The confusion matrix, feature importance chart, and live prediction demo are all there to answer the same question a hiring manager or a fraud analyst might ask: *does this model actually understand the data, or did it just memorize the majority class?*

---

## Contributing

Contributions welcome — model improvements, UI refinements, or additional chart types.

1. Fork the repo
2. Create a branch:
   ```bash
   git checkout -b feat/my-feature
   # or
   git checkout -b fix/issue-###
   ```
3. Commit your changes:
   ```bash
   git commit -m "feat: add <short description>"
   ```
4. Push and open a pull request with context or screenshots for any visual changes.

---

## Contact
If you have any questions or feedback, feel free to reach out at [mrodr.contact@gmail.com](mailto:mrodr.contact@gmail.com).
