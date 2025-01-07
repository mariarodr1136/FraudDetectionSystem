from flask import Flask, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go

app = Flask(__name__)

data = pd.read_csv('creditcard.csv')

data.columns = data.columns.str.strip()

X = data.drop(columns=['Class']) 
y = data['Class'] 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf_model = RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=42)
rf_model.fit(X_scaled, y)

y_pred = rf_model.predict(X_scaled)

# Calculate the classification report and confusion matrix
class_report = classification_report(y, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y, y_pred)

def create_confusion_matrix_heatmap(conf_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=['Predicted Legitimate', 'Predicted Fraudulent'],
        y=['Actual Legitimate', 'Actual Fraudulent'],
        colorscale='Viridis',
        colorbar=dict(title='Count'),
        zmin=0, zmax=conf_matrix.max()
    ))
    fig.update_layout(
        title="Confusion Matrix Heatmap",
        xaxis_title="Predicted Labels",
        yaxis_title="True Labels",
        autosize=True,
    )
    # Save the figure as an HTML file in the static folder
    fig.write_html('static/confusion_matrix_heatmap.html')  
    return 'static/confusion_matrix_heatmap.html'

# Create heatmap and get its path
cm_plot_path = create_confusion_matrix_heatmap(conf_matrix)

# Debugging: Check the confusion matrix path
print("Confusion Matrix Path:", cm_plot_path)

@app.route('/')
def home():
    # Debugging: Print classification report and confusion matrix path
    print("Classification Report:", class_report)
    print("Confusion Matrix Path:", cm_plot_path) 

    return render_template('index.html', 
                           classification_report=class_report, 
                           cm_plot_path=cm_plot_path)

if __name__ == '__main__':
    app.run(debug=True)  # Run with debugging enabled
