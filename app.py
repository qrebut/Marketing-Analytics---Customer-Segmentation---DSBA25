import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load models
@st.cache_resource
def load_models():
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    cluster_names = joblib.load('cluster_names.pkl')
    return kmeans, scaler, cluster_names

kmeans, scaler, cluster_names = load_models()

# App Title
st.title("🏦 Credit Card Segmentation & Limit Recommender")

st.markdown("""
This dashboard allows bank managers to input customer details and receive 
AI-driven segmentation and credit limit recommendations.
""")

# Sidebar Inputs
st.sidebar.header("Customer Data Input")

def user_input_features():
    balance = st.sidebar.number_input("Balance", min_value=0.0, value=1000.0)
    purchases = st.sidebar.number_input("Purchases", min_value=0.0, value=500.0)
    cash_advance = st.sidebar.number_input("Cash Advance", min_value=0.0, value=0.0)
    credit_limit = st.sidebar.number_input("Current Credit Limit", min_value=0.0, value=3000.0)
    payments = st.sidebar.number_input("Payments Made", min_value=0.0, value=800.0)
    tenure = st.sidebar.slider("Tenure", 6, 12, 12)
    
    # We need to approximate other features with averages for the model to work 
    # if we don't ask the user for all 17 features.
    # For a robust app, we should ask for all, but for this demo, we zero-fill or average-fill others.
    data = {
        'BALANCE': balance,
        'BALANCE_FREQUENCY': 0.8, # Assumed average
        'PURCHASES': purchases,
        'ONEOFF_PURCHASES': purchases * 0.6, # Assumed ratio
        'INSTALLMENTS_PURCHASES': purchases * 0.4,
        'CASH_ADVANCE': cash_advance,
        'PURCHASES_FREQUENCY': 0.5,
        'ONEOFF_PURCHASES_FREQUENCY': 0.3,
        'PURCHASES_INSTALLMENTS_FREQUENCY': 0.3,
        'CASH_ADVANCE_FREQUENCY': 0.1,
        'CASH_ADVANCE_TRX': 1,
        'PURCHASES_TRX': 10,
        'CREDIT_LIMIT': credit_limit,
        'PAYMENTS': payments,
        'MINIMUM_PAYMENTS': payments * 0.5,
        'PRC_FULL_PAYMENT': 0.1,
        'TENURE': tenure
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display User Input
st.subheader("Customer Profile")
st.write(input_df[['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']])

if st.button('Analyze Customer'):
    # Scale
    input_scaled = scaler.transform(input_df)
    
    # Predict
    cluster = kmeans.predict(input_scaled)[0]
    segment_name = cluster_names[cluster]
    
    # Logic for Recommendation (Same as notebook)
    def get_recommendation(row, segment):
        if segment == "VIP / Big Spenders":
            return max(row['CREDIT_LIMIT'] * 1.5, row['PURCHASES'] * 3)
        elif segment == "Cash Advancers":
            return max(row['CREDIT_LIMIT'], row['BALANCE'] * 1.1)
        elif segment == "High Balance / Revolvers":
            return row['CREDIT_LIMIT']
        else:
            return max(1500, row['CREDIT_LIMIT'])

    rec_limit = get_recommendation(input_df.iloc[0], segment_name)
    
    # Display Results
    st.success(f"**Assigned Segment:** {segment_name}")
    st.metric(label="Recommended Credit Limit", value=f"${rec_limit:,.2f}", delta=rec_limit - input_df['CREDIT_LIMIT'].iloc[0])
    
    # Explanation
    st.info(f"Reasoning: This customer falls into the '{segment_name}' category based on spending and balance patterns.")