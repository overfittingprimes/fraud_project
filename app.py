import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 300;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

try:
    with open("fraud_detection_model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    st.error("Model file not found")
    st.stop()

st.markdown('<div class="main-header">Transaction Risk Assessment</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Transaction Details")
    
    distance_home = st.number_input("Distance from home address", 0.0, 500.0, 37.31)
    distance_last = st.number_input("Distance from last transaction", 0.0, 200.0, 7.12)
    price_ratio = st.number_input("Price vs typical spending", 0.1, 20.0, 2.96)
    
    repeat_retailer = st.selectbox("Previously used retailer", ["No", "Yes"])
    used_chip = st.selectbox("Chip card used", ["No", "Yes"])
    used_pin = st.selectbox("PIN entered", ["No", "Yes"])
    online_order = st.selectbox("Online transaction", ["No", "Yes"])

repeat_val = 1 if repeat_retailer == "Yes" else 0
chip_val = 1 if used_chip == "Yes" else 0
pin_val = 1 if used_pin == "Yes" else 0
online_val = 1 if online_order == "Yes" else 0

X = np.array([[distance_home, distance_last, price_ratio, repeat_val, chip_val, pin_val, online_val]])
prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0][1]

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability * 100,
    title={'text': "Fraud Risk"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#34495e"},
        'steps': [
            {'range': [0, 25], 'color': "#2ecc71"},
            {'range': [25, 50], 'color': "#f39c12"},
            {'range': [50, 75], 'color': "#e67e22"},
            {'range': [75, 100], 'color': "#e74c3c"}
        ]
    }
))
fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

if probability > 0.7:
    recommendation = "Block transaction and contact cardholder immediately"
elif probability > 0.4:
    recommendation = "Flag for manual review before processing"
else:
    recommendation = "Process transaction normally"

st.info(f"Recommendation: {recommendation}")
