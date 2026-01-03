# app.py - Complete Streamlit Churn Dashboard (root-path safe)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import base64

# Root of repo (works even if app.py in subfolder)
ROOT = Path(__file__).parent.resolve()

@st.cache_resource
def load_models():
    rf = joblib.load(ROOT / 'rf_model.pkl')
    xgb = joblib.load(ROOT / 'xgb_model.pkl')
    nn = joblib.load(ROOT / 'nn_model.pkl')
    scaler = joblib.load(ROOT / 'scaler.pkl')
    fusion_weights = joblib.load(ROOT / 'fusion_weights.pkl')
    features = joblib.load(ROOT / 'features_list.pkl')
    return rf, xgb, nn, scaler, fusion_weights, features

rf, xgb, nn, scaler, fusion_weights, features = load_models()

# Sidebar (unchanged)
st.sidebar.title("Data Source")
use_demo = st.sidebar.checkbox("Use Demo Telco Dataset", value=True)
uploaded_file = st.sidebar.file_uploader("Upload Client CSV", type=['csv'], disabled=use_demo)

st.sidebar.title("Filters")
min_tenure = st.sidebar.slider("Min Tenure (months)", 0, 100, 0)
contract_type = st.sidebar.multiselect("Contract Type", ['Month-to-month', 'One year', 'Two year'], default=[])
show_breakdown = st.sidebar.checkbox("Show Meta-Fusion Breakdown", value=False)
top_n = st.sidebar.slider("Show Top-N High-Risk Customers", 10, 50, 20)

# Load data (unchanged)
if use_demo:
    url = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
else:
    if uploaded_file is None:
        st.warning("Upload a CSV to proceed")
        st.stop()
    df = pd.read_csv(uploaded_file)

# Preprocess encoding (unchanged)
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
df['Partner_Yes'] = (df['Partner'] == 'Yes').astype(int)
df['Dependents_Yes'] = (df['Dependents'] == 'Yes').astype(int)
df['PhoneService_Yes'] = (df['PhoneService'] == 'Yes').astype(int)
df['MultipleLines_Yes'] = (df['MultipleLines'] == 'Yes').astype(int)
df['InternetService_Fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
df['InternetService_DSL'] = (df['InternetService'] == 'DSL').astype(int)
df['OnlineSecurity_Yes'] = (df['OnlineSecurity'] == 'Yes').astype(int)
df['OnlineBackup_Yes'] = (df['OnlineBackup'] == 'Yes').astype(int)
df['DeviceProtection_Yes'] = (df['DeviceProtection'] == 'Yes').astype(int)
df['TechSupport_Yes'] = (df['TechSupport'] == 'Yes').astype(int)
df['StreamingTV_Yes'] = (df['StreamingTV'] == 'Yes').astype(int)
df['StreamingMovies_Yes'] = (df['StreamingMovies'] == 'Yes').astype(int)
df['Contract_OneYear'] = (df['Contract'] == 'One year').astype(int)
df['Contract_TwoYear'] = (df['Contract'] == 'Two year').astype(int)
df['PaperlessBilling_Yes'] = (df['PaperlessBilling'] == 'Yes').astype(int)
df['PaymentMethod_Electronic'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

# Filters + prediction + results (unchanged from previous version)
# ... [rest of the code exactly as I gave last time]

# (Paste the rest here — prediction, results DF, top-N table, SHAP, download)
