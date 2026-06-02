# FraudWatch: Credit Card Fraud Detection System 🚨

![Python](https://img.shields.io/badge/Python-Programming%20Language-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgreen)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Manipulation-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-blue)
![Plotly](https://img.shields.io/badge/Plotly-Data%20Visualization-brightgreen)


FraudWatch is a credit card fraud detection system that identifies fraudulent transactions using machine learning. Built with **Python** and **Flask**, it trains a **Random Forest classifier** on the Kaggle Credit Card Fraud dataset, evaluates it on a held-out test set, and serves an interactive dashboard with real-time prediction capabilities.

The system uses a proper **train/test split** so all reported metrics reflect true generalization — not training performance. The trained model and scaler are **persisted with joblib**, so subsequent server restarts load instantly without retraining. Visualizations are rendered inline via **Plotly** without static HTML files.

---

Live Application: https://frauddetectionsystem-qcmh.onrender.com/

*Note: The live application is hosted on Render's free tier, so the backend may take up to a minute to wake up after a period of inactivity.*

---

<img width="1305" alt="Screenshot 2025-01-07 at 6 41 12 PM" src="https://github.com/user-attachments/assets/acdbf0cc-adff-498c-beed-1ddc857b5584" />

---

## Table of Contents
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [API Endpoints](#api-endpoints)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact-)

---

## Technologies Used
- **Python**: Backend logic and ML pipeline.
- **Flask**: Web framework serving the dashboard and REST API.
- **scikit-learn**: Random Forest classifier, StandardScaler, train/test split, confusion matrix.
- **Pandas / NumPy**: Data loading, manipulation, and feature extraction.
- **Plotly**: Interactive inline charts (no static HTML files).
- **joblib**: Model and scaler persistence.

---

<img width="1308" alt="Screenshot 2025-01-07 at 6 41 21 PM" src="https://github.com/user-attachments/assets/436839fe-3dbd-4236-9107-2b8ba23cfaa1" />

---

## Features

- **Proper train/test evaluation**: An 80/20 stratified split ensures reported metrics represent true out-of-sample performance.
- **Model persistence**: The Random Forest model (100 estimators) and StandardScaler are saved to `model.joblib` / `scaler.joblib` after the first training run. Subsequent starts skip retraining entirely.
- **Correct metric reporting**: Accuracy, Precision, Recall (TPR), F1-Score, and False Positive Rate are all derived from the confusion matrix on the test set — not from training accuracy.
- **Interactive bar chart**: Model performance metrics displayed as a color-coded bar chart with percentage labels.
- **Confusion matrix heatmap**: TP / TN / FP / FN counts from the test set, rendered as an interactive Plotly heatmap.
- **Feature importance chart**: Top 15 PCA components ranked by Random Forest feature importance.
- **Balanced sample table**: 3 fraudulent + 3 legitimate transactions from the test set, showing true label, prediction, and fraud probability side-by-side.
- **Live prediction demo**: Load a random test-set transaction and run the model against it in real time. The result shows the prediction, fraud probability, true label, and whether the model was correct.

---

<img width="1286" alt="Screenshot 2025-01-07 at 6 43 08 PM" src="https://github.com/user-attachments/assets/58ed2189-b2d7-4e6e-9b84-479bc526799c" />

---

## Dataset

The data used in this project is from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download).

The dataset contains transactions made by credit cards in September 2013 by European cardholders — 284,807 transactions over two days, of which 492 (0.172%) are fraudulent. It is highly imbalanced; the classifier uses `class_weight='balanced'` to compensate.

All input features are numerical. V1–V28 are principal components from a PCA transformation (original features are confidential). The two non-PCA features are:

- **`Time`**: Seconds elapsed since the first transaction in the dataset.
- **`Amount`**: Transaction amount.
- **`Class`**: Target variable — 1 for fraudulent, 0 for legitimate.

---

## Model Performance

Evaluated on the held-out 20% test set (56,962 transactions, ~98 fraud cases):

| Metric | Value |
|---|---|
| Accuracy | 99.95% |
| Precision (fraud) | 96.05% |
| Recall / TPR (fraud) | 74.49% |
| F1-Score (fraud) | 83.91% |
| False Positive Rate | 0.01% |

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Main dashboard |
| `GET` | `/random_transaction` | Returns a random test-set transaction (all 30 features + true label) |
| `POST` | `/predict` | Accepts a JSON object of feature values, returns prediction and fraud probability |

**Example `/predict` request:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": -1.359807, "V2": -0.072781, ..., "Amount": 149.62, "Time": 0}'
```

**Example response:**
```json
{
  "fraud_probability": 94.3,
  "prediction": 1
}
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mariarodr1136/FraudDetectionSystem.git
   cd FraudDetectionSystem
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download) and place it in the project root.

4. Run the app:
   ```bash
   python app.py
   ```
   On first run, the model trains and saves `model.joblib` and `scaler.joblib` (takes ~1–2 minutes). Subsequent starts load the saved files instantly.

5. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Usage

Once running, the dashboard provides:

1. **Performance cards** — Accuracy, Precision, Recall, F1-Score, and False Positive Rate at a glance.
2. **Bar chart** — All five metrics visualized with percentage labels.
3. **Confusion matrix** — TP / TN / FP / FN counts from the test set.
4. **Feature importance** — Which PCA components the model relies on most.
5. **Sample transactions** — A balanced table of 3 fraudulent and 3 legitimate test transactions with predictions and fraud probabilities.
6. **Live prediction demo** — Click "Load Random Transaction" to pull a random test-set record, then "Predict" to run the model and see the result versus ground truth.

---

## Contributing
Feel free to submit issues or pull requests for improvements or bug fixes. All contributions are welcome!

To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. Make your changes and commit with a descriptive message:
   ```bash
   git commit -m 'add your commit message'
   ```
4. Push to your fork and open a pull request against `main`.

---

## Contact 🌐
If you have any questions or feedback, feel free to reach out at [mrodr.contact@gmail.com](mailto:mrodr.contact@gmail.com).
