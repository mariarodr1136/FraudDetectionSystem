<div align="center">

# FraudWatch 🚨
### Real-Time Credit Card Fraud Detection

*Machine learning dashboard that catches fraud with 99.95% accuracy — evaluated on a true held-out test set*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

**[🚀 Live Demo](https://frauddetectionsystem-qcmh.onrender.com/)**

*Hosted on Render's free tier — may take ~30s to wake from sleep*

</div>

---

<img width="1305" alt="FraudWatch Dashboard" src="https://github.com/user-attachments/assets/acdbf0cc-adff-498c-beed-1ddc857b5584" />

---

## Overview

FraudWatch is an end-to-end machine learning application for detecting fraudulent credit card transactions. It trains a **Random Forest classifier** on 284,807 real-world transactions, evaluates it on a stratified held-out test set, and serves an interactive web dashboard with live prediction capabilities.

Key design decisions that make the results trustworthy:
- **Stratified 80/20 train/test split** — all reported metrics reflect true generalization, not training performance
- **Metrics derived from the confusion matrix** — no misleading shortcuts like training accuracy
- **`class_weight='balanced'`** — handles the extreme class imbalance (only 0.172% of transactions are fraudulent)
- **Persisted model artifacts** — `model.joblib` and `scaler.joblib` committed to the repo so the app starts instantly without retraining

---

## Model Performance

Evaluated on **56,962 held-out test transactions** (~98 fraud cases):

| Metric | Score | What It Means |
|---|---|---|
| **Accuracy** | **99.95%** | Correct predictions across all transactions |
| **Precision** | **96.05%** | Of flagged transactions, 96% are genuinely fraudulent |
| **Recall (TPR)** | **74.49%** | 3 in 4 fraudulent transactions successfully caught |
| **F1-Score** | **83.91%** | Harmonic mean of precision and recall |
| **False Positive Rate** | **0.01%** | Legitimate transactions incorrectly flagged |

> A false positive rate of 0.01% means only **5–6 legitimate transactions** out of 56,900 are incorrectly flagged — making this model practical for real-world use.

---

## Features

| | Feature | Description |
|---|---|---|
| 📊 | **Performance Dashboard** | Five metric cards (Accuracy, Precision, Recall, F1, FPR) at the top of the page |
| 📈 | **Interactive Bar Chart** | Color-coded bar chart of all metrics with percentage labels, rendered inline via Plotly |
| 🔲 | **Confusion Matrix** | Heatmap of TP / TN / FP / FN counts from the test set |
| 🌲 | **Feature Importance** | Top 15 PCA components ranked by Random Forest importance score |
| ⚖️ | **Balanced Sample Table** | 3 fraudulent + 3 legitimate test transactions with true labels, predictions, and fraud probabilities |
| ⚡ | **Live Prediction API** | Load any random test-set transaction and run the model against it in real time |

---

<img width="1308" alt="Model Evaluation Charts" src="https://github.com/user-attachments/assets/436839fe-3dbd-4236-9107-2b8ba23cfaa1" />

---

## How It Works

```
creditcard.csv (284,807 transactions)
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
  • /                   → interactive dashboard
  • /random_transaction → sample test record
  • /predict            → real-time inference
```

---

## Dataset

Source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)

Transactions made by European cardholders in September 2013. 284,807 transactions over two days; 492 are fraudulent (0.172%). Due to confidentiality, the original features are not available — V1 through V28 are PCA-transformed principal components.

| Feature | Description |
|---|---|
| `V1` – `V28` | PCA principal components (confidential source features) |
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `Amount` | Transaction amount in euros |
| `Class` | Target — `1` = fraudulent, `0` = legitimate |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main dashboard |
| `GET` | `/random_transaction` | Random test-set transaction — all 30 features + true label |
| `POST` | `/predict` | Run inference on a feature vector |

**Request**
```bash
curl -X POST https://frauddetectionsystem-qcmh.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -3.04, "V2": -3.16, "V3": 1.09, "V4": 2.29, "V5": -0.18,
       "V6": -0.21, "V7": -3.72, "V8": 0.38, "V9": -2.07, "V10": -2.51,
       "V11": 2.27, "V12": -2.32, "V13": 1.71, "V14": -3.42, "V15": -0.35,
       "V16": -1.93, "V17": -5.34, "V18": -1.12, "V19": 0.57, "V20": -0.63,
       "V21": -0.38, "V22": 0.25, "V23": -0.14, "V24": 0.93, "V25": 0.28,
       "V26": 0.14, "V27": -0.36, "V28": -0.11, "Time": 406.0, "Amount": 2.70}'
```

**Response**
```json
{
  "fraud_probability": 94.3,
  "prediction": 1
}
```

---

<img width="1286" alt="Sample Transactions Table" src="https://github.com/user-attachments/assets/58ed2189-b2d7-4e6e-9b84-479bc526799c" />

---

## Installation

### Prerequisites
- Python 3.8+
- `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)

### Quickstart

```bash
# 1. Clone
git clone https://github.com/mariarodr1136/FraudDetectionSystem.git
cd FraudDetectionSystem

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download creditcard.csv from Kaggle and place it in the project root

# 4. Run
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

> The pre-trained `model.joblib` is included in the repo — the app loads it instantly on startup. Delete it to trigger a fresh training run (~1–2 min locally).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Web framework | Flask 3.0 + Gunicorn |
| ML library | scikit-learn 1.6 |
| Data | Pandas 2.2, NumPy 2.2 |
| Visualization | Plotly 5.24 (inline, no static files) |
| Model persistence | joblib 1.4 |
| Hosting | Render (free tier) |

---

## Contributing

Contributions are welcome! To get started:

```bash
# Fork the repo, then:
git checkout -b feat/your-feature-name

# After making changes:
git commit -m 'feat: describe your change'
git push origin feat/your-feature-name
# Open a pull request against main
```

---

## Contact

Maria Rodriguez — [mrodr.contact@gmail.com](mailto:mrodr.contact@gmail.com)

If you find this project useful, consider leaving a ⭐ on [GitHub](https://github.com/mariarodr1136/FraudDetectionSystem)!
