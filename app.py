import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

MODEL_PATH = 'model.joblib'
SCALER_PATH = 'scaler.joblib'

app = Flask(__name__)

data = pd.read_csv('creditcard.csv')
data.columns = data.columns.str.strip()
X = data.drop(columns=['Class'])
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    scaler   = joblib.load(SCALER_PATH)
    rf_model = joblib.load(MODEL_PATH)
else:
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    rf_model       = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(rf_model, MODEL_PATH)

X_test_scaled = scaler.transform(X_test)
y_pred        = rf_model.predict(X_test_scaled)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy  = (tp + tn) / (tp + tn + fp + fn) * 100
precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
fpr       = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0


def _chart(fig, first=False):
    return fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False)


def metrics_chart_html():
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'False Positive Rate']
    values  = [accuracy, precision, recall, f1, fpr]
    colors  = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#ef4444']
    fig = go.Figure([go.Bar(
        x=metrics, y=values, marker_color=colors,
        text=[f'{v:.1f}%' for v in values], textposition='outside',
        marker=dict(color=colors, line=dict(width=0)),
    )])
    fig.update_layout(
        title=dict(text='Model Performance Metrics (Test Set)', font=dict(size=14, color='#1e293b')),
        yaxis=dict(range=[0, 115], gridcolor='#f1f5f9', tickfont=dict(color='#64748b')),
        xaxis=dict(tickfont=dict(color='#64748b')),
        showlegend=False, height=360,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=50, b=10, l=10, r=10)
    )
    return _chart(fig, first=True)


def conf_matrix_html():
    labels = ['Legitimate', 'Fraudulent']
    fig = go.Figure(go.Heatmap(
        z=[[int(tn), int(fp)], [int(fn), int(tp)]],
        x=labels, y=labels,
        colorscale=[[0, '#eef2ff'], [1, '#4338ca']],
        text=[[f'TN: {tn:,}', f'FP: {fp:,}'], [f'FN: {fn:,}', f'TP: {tp:,}']],
        texttemplate='%{text}', textfont={'size': 14, 'color': '#1e293b'}, showscale=False
    ))
    fig.update_layout(
        title=dict(text='Confusion Matrix (Test Set)', font=dict(size=14, color='#1e293b')),
        xaxis=dict(title='Predicted', tickfont=dict(color='#64748b')),
        yaxis=dict(title='Actual', tickfont=dict(color='#64748b')),
        height=360, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=50, b=10, l=10, r=10)
    )
    return _chart(fig)


def feature_importance_html():
    names       = X.columns.tolist()
    importances = rf_model.feature_importances_
    top         = np.argsort(importances)[::-1][:15]
    fig = go.Figure([go.Bar(
        x=[importances[i] for i in top],
        y=[names[i] for i in top],
        orientation='h',
        marker=dict(
            color=[importances[i] for i in top],
            colorscale=[[0, '#c7d2fe'], [1, '#4338ca']],
            showscale=False,
            line=dict(width=0)
        )
    )])
    fig.update_layout(
        title=dict(text='Top 15 Feature Importances', font=dict(size=14, color='#1e293b')),
        xaxis=dict(title='Importance Score', gridcolor='#f1f5f9', tickfont=dict(color='#64748b')),
        yaxis=dict(autorange='reversed', tickfont=dict(color='#64748b')),
        height=430, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=50, b=10, l=10, r=10)
    )
    return _chart(fig)


chart_metrics    = metrics_chart_html()
chart_conf       = conf_matrix_html()
chart_importance = feature_importance_html()


def build_samples():
    rows = []
    for label in [1, 0]:
        subset = X_test[y_test == label].head(3)
        for i in range(len(subset)):
            row_s  = subset.iloc[i]                          # Series of feature values
            row_df = pd.DataFrame([row_s])                   # 1-row DataFrame for scaler
            scaled = scaler.transform(row_df)
            pred   = int(rf_model.predict(scaled)[0])
            prob   = float(rf_model.predict_proba(scaled)[0][1])
            rows.append({
                'amount':     round(float(row_s['Amount']), 2),
                'v1':         round(float(row_s['V1']), 3),
                'v2':         round(float(row_s['V2']), 3),
                'v3':         round(float(row_s['V3']), 3),
                'v4':         round(float(row_s['V4']), 3),
                'true_class': label,
                'prediction': pred,
                'fraud_prob': round(prob * 100, 1),
            })
    return rows


sample_transactions = build_samples()


@app.route('/')
def home():
    return render_template('index.html',
        chart_metrics=chart_metrics,
        chart_conf=chart_conf,
        chart_importance=chart_importance,
        sample_transactions=sample_transactions,
        accuracy=round(accuracy, 2),
        precision=round(precision, 2),
        recall=round(recall, 2),
        f1=round(f1, 2),
        fpr=round(fpr, 2),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn)
    )


@app.route('/random_transaction')
def random_transaction():
    idx      = int(np.random.randint(len(X_test)))
    row      = X_test.iloc[idx]
    features = {col: round(float(row[col]), 6) for col in X.columns}
    return jsonify({'features': features, 'true_label': int(y_test.iloc[idx])})


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    try:
        df     = pd.DataFrame([payload]).reindex(columns=X.columns, fill_value=0)
        scaled = scaler.transform(df)
        pred   = int(rf_model.predict(scaled)[0])
        prob   = float(rf_model.predict_proba(scaled)[0][1])
        return jsonify({'prediction': pred, 'fraud_probability': round(prob * 100, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
