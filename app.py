# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load CICIDS dataset (you must preprocess this in code directly)
@st.cache_data
def load_data():
    # Load your dataset here (replace with your exact path or logic)
    df = pd.read_csv("cicids_sample.csv")  # Update path if needed

    # Keep only selected features and drop missing
    selected_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
        'Flow Bytes/s', 'Flow IAT Mean', 'Label'
    ]
    df = df[selected_features].dropna()
    return df

# 2. User input function
def user_input_features():
    st.sidebar.header("🔧 Enter Network Flow Features")

    flow_duration = st.sidebar.number_input("Flow Duration", min_value=0.0, value=1000.0)
    total_fwd = st.sidebar.number_input("Total Fwd Packets", min_value=0.0, value=10.0)
    total_bwd = st.sidebar.number_input("Total Backward Packets", min_value=0.0, value=10.0)
    fwd_pkt_len = st.sidebar.number_input("Fwd Packet Length Mean", min_value=0.0, value=50.0)
    bwd_pkt_len = st.sidebar.number_input("Bwd Packet Length Mean", min_value=0.0, value=50.0)
    flow_bytes = st.sidebar.number_input("Flow Bytes/s", min_value=0.0, value=100.0)
    flow_iat = st.sidebar.number_input("Flow IAT Mean", min_value=0.0, value=200.0)

    data = {
        'Flow Duration': flow_duration,
        'Total Fwd Packets': total_fwd,
        'Total Backward Packets': total_bwd,
        'Fwd Packet Length Mean': fwd_pkt_len,
        'Bwd Packet Length Mean': bwd_pkt_len,
        'Flow Bytes/s': flow_bytes,
        'Flow IAT Mean': flow_iat
    }
    return pd.DataFrame(data, index=[0])

# 3. Main app logic
def main():
    st.set_page_config(page_title="Threat Detection", layout="centered")
    st.title("🛡️ Real-Time Threat Detection")
    st.markdown("Enter 7 key features manually to predict whether the network traffic is **Malicious or Benign** using a Voting Classifier.")

    df = load_data()

    # Split & Train
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model setup
    lr = LogisticRegression()
    mlp = MLPClassifier(max_iter=300)
    svm = SVC(probability=True)

    voting_clf = VotingClassifier(
        estimators=[('lr', lr), ('mlp', mlp), ('svm', svm)],
        voting='soft'
    )

    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    # Add white-border CSS
    st.markdown("""
        <style>
        .custom-table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }
        .custom-table th, .custom-table td {
            border: 1px solid white !important;
            padding: 8px 12px !important;
            text-align: center;
            color: white;
        }
        .custom-table th {
            background-color: #444;
        }
        .custom-table tr:nth-child(even) {
            background-color: #222;
        }
        .custom-table tr:nth-child(odd) {
            background-color: #111;
        }
        </style>
    """, unsafe_allow_html=True)

    # Metrics
    st.subheader("📈 Model Evaluation on Test Data")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    styled_report = report_df.style.set_table_attributes('class="custom-table"')
    st.write(styled_report.to_html(), unsafe_allow_html=True)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # User input and prediction
    st.subheader("🔮 Predict from Your Input")
    user_df = user_input_features()

    prediction = voting_clf.predict(user_df)[0]
    proba = voting_clf.predict_proba(user_df)[0]

    result = "Malicious" if prediction == 1 else "Benign"
    st.write(f"**Prediction**: {result}")
    st.write("**Confidence Scores**:", {"Benign": round(proba[0], 2), "Malicious": round(proba[1], 2)})

if __name__ == '__main__':
    main()
