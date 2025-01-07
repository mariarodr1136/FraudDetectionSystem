# FraudWatch: Credit Card Fraud Detection System 🚨

FraudWatch is an advanced credit card fraud detection system designed to identify fraudulent transactions in real-time using machine learning techniques. Built with **Python** and **Flask**, the system processes transaction data to predict fraud using a **Random Forest classifier**, a powerful ensemble learning algorithm known for its high accuracy in classification tasks. The backend leverages **scikit-learn** for model training and evaluation, while the web application, built with **Flask**, serves as an interactive platform for users to engage with the results.

FraudWatch incorporates robust data preprocessing steps such as feature scaling using **StandardScaler** to ensure optimal model performance. For visualization, the system employs **Plotly**, a data visualization library that creates an interactive chart for the model's performance metrics, allowing users to explore how well the model classifies legitimate and fraudulent transactions. The chart enhances user experience by offering an intuitive, interactive view of model accuracy, false positives, false negatives, true positives, and true negatives.

![Python](https://img.shields.io/badge/Python-Programming%20Language-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgreen)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Manipulation-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-blue)
![Plotly](https://img.shields.io/badge/Plotly-Data%20Visualization-brightgreen)

---
<img width="1305" alt="Screenshot 2025-01-07 at 6 41 12 PM" src="https://github.com/user-attachments/assets/acdbf0cc-adff-498c-beed-1ddc857b5584" />

---

## Table of Contents
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)
- [Contact](#contact-)

---

## Technologies Used
- **Python**: The primary programming language for the backend logic.
- **Flask**: Web framework for creating the web application and serving the results.
- **Pandas**: Data manipulation and analysis library used for loading and preparing the dataset.
- **scikit-learn**: Machine learning library used for building and evaluating the Random Forest model.
- **Plotly**: Data visualization library for generating interactive charts for model performance metrics.

---

<img width="1308" alt="Screenshot 2025-01-07 at 6 41 21 PM" src="https://github.com/user-attachments/assets/436839fe-3dbd-4236-9107-2b8ba23cfaa1" />

---

## Features
- **Fraud Detection**: The system predicts whether a credit card transaction is fraudulent or legitimate using a Random Forest classifier.
- **Feature Scaling**: Data is preprocessed with **StandardScaler** to normalize the features.
- **Classification Report**: Detailed classification report with precision, recall, and F1-score for both classes (legitimate and fraudulent transactions).
- **Model Performance Metrics Chart**: Visualizes the performance of the model through an interactive line chart for accuracy, false positives, false negatives, true positives, and true negatives using Plotly.
- **Sample Transaction Table**: Displays a table with a few sample transactions, showing their true class and predicted class, with color-coded indicators for correct or incorrect predictions.
- **User-friendly Web Application**: The system is packaged in a Flask web application to display results in an accessible and interactive format.

---

<img width="1286" alt="Screenshot 2025-01-07 at 6 43 08 PM" src="https://github.com/user-attachments/assets/58ed2189-b2d7-4e6e-9b84-479bc526799c" />

---

## Future Enhancements

- **Model Optimization**: Explore and integrate additional machine learning models such as **XGBoost** or **Gradient Boosting** to improve classification accuracy and model efficiency.
- **User Authentication**: Implement a user authentication system using **OAuth** or **JWT** for secure, personalized access to the application.
- **Real-Time Fraud Detection**: Integrate **real-time transaction data** for live fraud detection and alert generation.
- **Model Interpretability**: Incorporate **SHAP** or **LIME** for better understanding and visualization of how the model makes decisions.
- **Additional Data Sources**: Expand the dataset to include more features such as **transaction location** or **merchant information** to enhance fraud prediction.
- **Frontend Enhancements**: Improve the user interface by integrating more interactive elements and visualizations, such as **line charts** or **bar graphs**, for deeper insights into the model's performance.

---

## Installation

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Setup Instructions

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/fraudwatch.git
   cd fraudwatch
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Download the dataset creditcard.csv and place it in the project directory.
4. Run the Flask application:
   ```bash
   python app.py
5. Open your browser and go to http://127.0.0.1:5000/ to view the application.

## Usage

Once the application is running, you can access the following features:

1. **Classification Report**: View the precision, recall, and F1-score for both classes (legitimate and fraudulent).
2. **Interactive Model Metrics Line Chart**: See the model performance metrics (accuracy, false positives, false negatives, true positives, and true negatives) visualized interactively in a line chart, where you can hover over the points for detailed values.
3. **Sample Transactions Table**: View the first few sample transactions, with true and predicted classes, color-coded to indicate correct or incorrect predictions.

---

## Example

### Classification Report

| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Legitimate   | 0.99      | 0.99   | 0.99     |
| Fraudulent   | 0.98      | 0.95   | 0.97     |

---

### Model Performance Metrics Line Chart

The line chart visualizes the model's performance metrics, such as accuracy, false positives, false negatives, true positives, and true negatives.

---

### Sample Transactions Table

| Transaction (V1, V2, V3, V4)           | True Class | Prediction  |
|----------------------------------------|------------|-------------|
| -0.1348, 0.5127, -0.1256, -0.3432      | Legitimate | Fraudulent  |
| -0.0321, 0.3548, -0.2156, -0.2342      | Fraudulent | Fraudulent  |
| ...                                    | ...        | ...         |

---

## Contributing
Feel free to submit issues or pull requests for improvements or bug fixes. You can also open issues to discuss potential changes or enhancements. All contributions are welcome to enhance the app’s features or functionality!

To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feat/your-feature-name
- Alternatively, for bug fixes:
   ```bash
   git checkout -b fix/your-bug-fix-name
3. Make your changes and run all tests before committing the changes and make sure all tests are passed.
4. After all tests are passed, commit your changes with descriptive messages:
   ```bash
   git commit -m 'add your commit message'
5. Push your changes to your forked repository:
   ```bash
   git push origin feat/your-feature-name.
6. Submit a pull request to the main repository, explaining your changes and providing any necessary details.

## Contact 🌐
If you have any questions or feedback, feel free to reach out at [mrodr.contact@gmail.com](mailto:mrodr.contact@gmail.com).
