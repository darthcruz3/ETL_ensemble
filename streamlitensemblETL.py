import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# Set Streamlit page layout
st.set_page_config(page_title="Exhalation Technology Limited Ensemble Classifier", layout="wide")

# GitHub repository URL for the model and scaler
model_url = "https://github.com/darthcruz3/ETL_ensemble/blob/models/voting_ensemble_model.pkl"
scaler_url = "https://github.com/darthcruz3/ETL_ensemble/blob/models/scaler.pkl"

# Function to download and load model from URL
def load_model_from_url(url):
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

# Load trained model and scaler from GitHub
model = load_model_from_url(model_url)
scaler = load_model_from_url(scaler_url)

# Streamlit App Title
st.title("ETL Ensemble Classifier")

# Sidebar File Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file for prediction", type=["csv"])

# If file is uploaded, process it
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure correct feature columns
    feature_columns = df.columns[1:6]  # Exclude ID column
    X_new = df[feature_columns].values

    # Standardize features using the saved scaler
    X_new_scaled = scaler.transform(X_new)

    # Make predictions
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)[:, 1]

    # Add results to DataFrame
    df["Predicted_Class"] = predictions
    df["Probability"] = probabilities

    # Display results
    st.subheader("Prediction Results")
    st.write(df)

    # Save predictions as CSV
    df.to_csv("streamlit_predictions.csv", index=False)
    st.sidebar.success("Predictions saved as 'streamlit_predictions.csv'.")

    # Model Performance Metrics
    st.subheader("Ensemble_ETL Model Performance")
    y_true = df.iloc[:, -3].values  # Assuming last column before predictions is actual class
    y_pred = df["Predicted_Class"].values

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    roc_auc = roc_auc_score(y_true, probabilities)

    metrics_data = {
        "Metric": ["Accuracy", "Sensitivity", "Specificity", "ROC-AUC"],
        "Value": [accuracy, sensitivity, specificity, roc_auc]
    }
    metrics_df = pd.DataFrame(metrics_data)

    st.table(metrics_df)

    # ROC Curve
    st.subheader("Ensemble_ETL ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, probabilities, pos_label=2)
    plt.figure(figsize=(5, 3))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(plt)

# Footer
