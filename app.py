from flask import Flask, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go

app = Flask(__name__)

# Load the data
data = pd.read_csv('creditcard.csv')

# Remove any extra spaces from the column names
data.columns = data.columns.str.strip()

# Features and target variable
X = data.drop(columns=['Class'])
y = data['Class']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=42)
rf_model.fit(X_scaled, y)

# Predict the target variable
y_pred = rf_model.predict(X_scaled)

# Calculate the classification report and confusion matrix
class_report = classification_report(y, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y, y_pred)

# Extracting metrics for visualization
accuracy = class_report['accuracy'] * 100
false_positive = class_report['0']['precision'] * 100  # False positives from precision of class '0' (legitimate)
false_negative = class_report['1']['recall'] * 100  # False negatives from recall of class '1' (fraudulent)
true_positive = class_report['1']['f1-score'] * 100  # True positives from f1-score of class '1' (fraudulent)
true_negative = class_report['0']['f1-score'] * 100  # True negatives from f1-score of class '0' (legitimate)

# Create a line chart of model metrics
def create_metrics_chart():
    metrics = ['Accuracy', 'False Positives', 'False Negatives', 'True Positives', 'True Negatives']
    values = [accuracy, false_positive, false_negative, true_positive, true_negative]

    fig = go.Figure(data=[go.Scatter(x=metrics, y=values, mode='lines+markers', line=dict(color='royalblue', width=3))])
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Percentage",
        showlegend=False,
        height=400
    )

    fig.write_html('static/model_metrics_line_chart.html')
    return 'static/model_metrics_line_chart.html'

# Create chart and get its path
metrics_chart_path = create_metrics_chart()

# Sample transactions for display
sample_transactions = data[['V1', 'V2', 'V3', 'V4', 'Class']].iloc[:6]  # Showing first 6 transactions for simplicity
sample_transactions['Prediction'] = y_pred[:6]

@app.route('/')
def home():
    return render_template('index.html', 
                           metrics_chart_path=metrics_chart_path,
                           sample_transactions=sample_transactions.to_dict(orient='records'),
                           accuracy=accuracy,
                           false_positive=false_positive,
                           false_negative=false_negative,
                           true_positive=true_positive,
                           true_negative=true_negative)

if __name__ == '__main__':
    app.run(debug=True)
