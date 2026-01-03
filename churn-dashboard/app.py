# app.py - Full Complete Streamlit Churn Dashboard
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

st.set_page_config(page_title="ChurnGuard Pro", layout="wide")
st.title("ChurnGuard Pro – Customer Retention Dashboard")
st.markdown("Ranked high-risk customers • $ revenue impact • SHAP explainability • Interactive filters")

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

# Sidebar
st.sidebar.title("Data Source")
use_demo = st.sidebar.checkbox("Use Demo Telco Dataset", value=True)
uploaded_file = st.sidebar.file_uploader("Upload Client CSV", type=['csv'], disabled=use_demo)

st.sidebar.title("Filters")
min_tenure = st.sidebar.slider("Min Tenure (months)", 0, 100, 0)
contract_type = st.sidebar.multiselect("Contract Type", ['Month-to-month', 'One year', 'Two year'], default=[])
show_breakdown = st.sidebar.checkbox("Show Meta-Fusion Breakdown", value=False)
top_n = st.sidebar.slider("Show Top-N High-Risk Customers", 10, 50, 20)

# Load data
if use_demo:
    url = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
else:
    if uploaded_file is None:
        st.warning("Please upload a CSV to proceed")
        st.stop()
    df = pd.read_csv(uploaded_file)

# Encoding (exact match to training)
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

# Apply filters
df_filtered = df[df['tenure'] >= min_tenure].copy()

if contract_type:
    mask = pd.Series([False] * len(df_filtered))
    if 'Month-to-month' in contract_type:
        mask |= (df_filtered['Contract_OneYear'] == 0) & (df_filtered['Contract_TwoYear'] == 0)
    if 'One year' in contract_type:
        mask |= df_filtered['Contract_OneYear'] == 1
    if 'Two year' in contract_type:
        mask |= df_filtered['Contract_TwoYear'] == 1
    df_filtered = df_filtered[mask]

if len(df_filtered) == 0:
    st.error("⚠️ No customers match the current filters. Try lowering Min Tenure or clearing Contract Type.")
    st.stop()

# Prediction
X = df_filtered[features]
X_scaled = scaler.transform(X)

rf_probs = rf.predict_proba(X_scaled)[:, 1]
xgb_probs = xgb.predict_proba(X_scaled)[:, 1]
nn_probs = nn.predict_proba(X_scaled)[:, 1]

meta_scores = (fusion_weights[0] * rf_probs +
               fusion_weights[1] * xgb_probs +
               fusion_weights[2] * nn_probs)

# Results
results = df_filtered.copy()
results['Churn_Probability'] = meta_scores

if show_breakdown:
    results['RF_Score'] = rf_probs
    results['XGB_Score'] = xgb_probs
    results['NN_Score'] = nn_probs

# Top-N table + revenue
top_n_df = results.sort_values('Churn_Probability', ascending=False).head(top_n).copy()
top_n_df['Annual_Revenue_At_Risk'] = top_n_df['MonthlyCharges'] * 12

cols_to_show = ['customerID', 'Churn_Probability', 'tenure', 'MonthlyCharges',
                'Contract', 'InternetService', 'TechSupport', 'Annual_Revenue_At_Risk']
if show_breakdown:
    cols_to_show += ['RF_Score', 'XGB_Score', 'NN_Score']

st.subheader(f"Top {top_n} Highest-Risk Customers")
st.dataframe(top_n_df[cols_to_show].style.format({
    'Churn_Probability': '{:.1%}',
    'RF_Score': '{:.1%}',
    'XGB_Score': '{:.1%}',
    'NN_Score': '{:.1%}',
    'MonthlyCharges': '${:.0f}',
    'Annual_Revenue_At_Risk': '${:.0f}'
}))

total_risk = top_n_df['Annual_Revenue_At_Risk'].sum()
st.metric("Total Annual Revenue At Risk (Top-N)", f"${total_risk:,.0f}")

# SHAP plot
st.subheader("Global Feature Impact (SHAP)")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_scaled)
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, feature_names=features, show=False)
st.pyplot(fig)

# Download CSV
csv = top_n_df.to_csv(index=False).encode()
b64 = base64.b64encode(csv).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="top_{top_n}_churn_risks.csv">Download Top-{top_n} as CSV</a>'
st.markdown(href, unsafe_allow_html=True)

