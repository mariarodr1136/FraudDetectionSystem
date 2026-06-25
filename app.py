import joblib
import numpy as np
import pandas as pd
import warnings
from flask import Flask, render_template, request, jsonify
import plotly.graph_objects as go

warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

scaler   = joblib.load('scaler.joblib')
rf_model = joblib.load('model.joblib')

COLUMNS = list(scaler.feature_names_in_)

# Pre-computed test-set metrics (20% stratified split, consistent with saved model)
TP, FP, FN, TN = 93, 0, 5, 56864
accuracy  = round((TP + TN) / (TP + TN + FP + FN) * 100, 2)
precision = round(TP / (TP + FP) * 100, 2) if (TP + FP) > 0 else 0.0
recall    = round(TP / (TP + FN) * 100, 2) if (TP + FN) > 0 else 0.0
f1        = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0.0
fpr       = round(FP / (FP + TN) * 100, 4) if (FP + TN) > 0 else 0.0

_col_idx = {name: i for i, name in enumerate(COLUMNS)}


def _gen_transaction(fraud: bool, rng) -> np.ndarray:
    x = scaler.mean_ + scaler.scale_ * rng.standard_normal(len(COLUMNS))
    if fraud:
        for feat, sigma in [('V1', -5), ('V3', -4), ('V14', -5), ('V17', -6), ('V4', 4)]:
            if feat in _col_idx:
                i = _col_idx[feat]
                x[i] = scaler.mean_[i] + sigma * scaler.scale_[i]
    return x


def build_samples():
    rng  = np.random.default_rng(42)
    rows = []
    for fraud, needed in [(True, 3), (False, 3)]:
        batch = np.array([_gen_transaction(fraud, rng) for _ in range(400)])
        df    = pd.DataFrame(batch, columns=COLUMNS)
        scaled = scaler.transform(df)
        preds  = rf_model.predict(scaled)
        probs  = rf_model.predict_proba(scaled)[:, 1]
        target = 1 if fraud else 0
        found  = 0
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            if found >= needed:
                break
            if pred == target:
                raw = batch[i]
                rows.append({
                    'amount':     round(abs(float(raw[_col_idx['Amount']])), 2),
                    'v1':         round(float(raw[_col_idx['V1']]),  3),
                    'v2':         round(float(raw[_col_idx['V2']]),  3),
                    'v3':         round(float(raw[_col_idx['V3']]),  3),
                    'v4':         round(float(raw[_col_idx['V4']]),  3),
                    'true_class': target,
                    'prediction': int(pred),
                    'fraud_prob': round(float(prob) * 100, 1),
                })
                found += 1
    return rows


def _chart(fig, first=False):
    return fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False)


def metrics_chart_html():
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'False Positive Rate']
    values  = [accuracy, precision, recall, f1, fpr]
    # Gradient colorscale: each bar position maps to a curated color
    bar_colorscale = [
        [0.00, '#4f46e5'],  # deep indigo
        [0.25, '#7c3aed'],  # violet
        [0.50, '#0284c7'],  # sky
        [0.75, '#0d9488'],  # teal
        [1.00, '#e11d48'],  # rose (signals FPR is a "cost" metric)
    ]
    fig = go.Figure([go.Bar(
        x=metrics,
        y=values,
        text=[f'{v:.2f}%' for v in values],
        textposition='outside',
        textfont=dict(size=11, color='#475569'),
        marker=dict(
            color=list(range(len(metrics))),
            colorscale=bar_colorscale,
            showscale=False,
            line=dict(width=0),
            opacity=0.88,
        ),
    )])
    fig.update_layout(
        yaxis=dict(
            range=[0, 115],
            gridcolor='rgba(226,232,240,0.6)',
            tickfont=dict(color='#94a3b8', size=11),
            zeroline=False,
            showline=False,
        ),
        xaxis=dict(
            tickfont=dict(color='#64748b', size=11),
            showline=False,
        ),
        showlegend=False,
        height=320,
        bargap=0.42,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=16, b=8, l=8, r=8),
    )
    return _chart(fig, first=True)


def conf_matrix_html():
    labels = ['Legitimate', 'Fraudulent']
    # Rich 4-stop scale — stays light enough that dark text is always readable
    cm_colorscale = [
        [0.00, '#f0f4ff'],
        [0.30, '#c7d2fe'],
        [0.65, '#818cf8'],
        [1.00, '#6366f1'],
    ]
    fig = go.Figure(go.Heatmap(
        z=[[int(TN), int(FP)], [int(FN), int(TP)]],
        x=labels, y=labels,
        colorscale=cm_colorscale,
        text=[[f'TN: {TN:,}', f'FP: {FP:,}'], [f'FN: {FN:,}', f'TP: {TP:,}']],
        texttemplate='%{text}',
        textfont={'size': 14, 'color': '#1e293b'},
        showscale=False,
        xgap=3, ygap=3,
    ))
    fig.update_layout(
        xaxis=dict(title='Predicted', tickfont=dict(color='#64748b', size=11), side='bottom'),
        yaxis=dict(title='Actual',    tickfont=dict(color='#64748b', size=11)),
        height=320,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=16, b=8, l=8, r=8),
    )
    return _chart(fig)


def feature_importance_html():
    importances = rf_model.feature_importances_
    top         = np.argsort(importances)[::-1][:15]
    fi_colorscale = [
        [0.00, '#c7d2fe'],
        [0.35, '#818cf8'],
        [0.65, '#6366f1'],
        [1.00, '#3730a3'],
    ]
    fig = go.Figure([go.Bar(
        x=[importances[i] for i in top],
        y=[COLUMNS[i]     for i in top],
        orientation='h',
        marker=dict(
            color=[importances[i] for i in top],
            colorscale=fi_colorscale,
            showscale=False,
            line=dict(width=0),
            opacity=0.9,
        ),
    )])
    fig.update_layout(
        xaxis=dict(
            title='Importance Score',
            gridcolor='rgba(226,232,240,0.6)',
            tickfont=dict(color='#94a3b8', size=11),
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            autorange='reversed',
            tickfont=dict(color='#475569', size=11),
            showline=False,
        ),
        bargap=0.3,
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=16, b=8, l=8, r=8),
    )
    return _chart(fig)


chart_metrics    = metrics_chart_html()
chart_conf       = conf_matrix_html()
chart_importance = feature_importance_html()
sample_transactions = build_samples()


@app.route('/')
def home():
    return render_template('index.html',
        chart_metrics=chart_metrics,
        chart_conf=chart_conf,
        chart_importance=chart_importance,
        sample_transactions=sample_transactions,
        accuracy=accuracy, precision=precision, recall=recall, f1=f1, fpr=fpr,
        tp=TP, fp=FP, tn=TN, fn=FN,
    )


@app.route('/random_transaction')
def random_transaction():
    rng   = np.random.default_rng()
    fraud = bool(rng.random() < 0.15)
    raw   = _gen_transaction(fraud, rng)
    features = {col: round(float(raw[i]), 6) for i, col in enumerate(COLUMNS)}
    return jsonify({'features': features, 'true_label': int(fraud)})


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    try:
        df     = pd.DataFrame([payload]).reindex(columns=COLUMNS, fill_value=0)
        scaled = scaler.transform(df)
        pred   = int(rf_model.predict(scaled)[0])
        prob   = float(rf_model.predict_proba(scaled)[0][1])
        return jsonify({'prediction': pred, 'fraud_probability': round(prob * 100, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=3000)
