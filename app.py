import streamlit as st
import pickle
import numpy as np
import pandas as pd
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
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .alert-danger {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
    }
    .alert-success {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
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

col1, col2 = st.columns([2, 1])

with col1:
    if prediction == 1:
        st.markdown('<div class="alert-danger">High fraud risk detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-success">Transaction appears legitimate</div>', unsafe_allow_html=True)
    
    st.subheader("Risk Analysis")
    
    risk_factors = []
    if distance_home > 100:
        risk_factors.append(f"Unusual location ({distance_home:.0f}km from home)")
    if distance_last > 50:
        risk_factors.append(f"Rapid location change ({distance_last:.0f}km)")
    if price_ratio > 5:
        risk_factors.append(f"Unusually large purchase ({price_ratio:.1f}x typical)")
    if not repeat_val:
        risk_factors.append("New merchant")
    if not chip_val and not pin_val:
        risk_factors.append("Low security authentication")
    
    if risk_factors:
        st.write("Key risk indicators:")
        for factor in risk_factors:
            st.write(f"• {factor}")
    else:
        st.write("No significant risk factors identified")

with col2:
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
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <h4>Risk Score</h4>
        <h2>{probability:.1%}</h2>
    </div>
    """, unsafe_allow_html=True)

st.subheader("Transaction Profile")

profile_data = {
    "Location": [f"{distance_home:.1f}km from home", f"{distance_last:.1f}km from last"],
    "Purchase": [f"{price_ratio:.1f}x typical amount", "Repeat retailer" if repeat_val else "New merchant"],
    "Security": [f"Chip: {used_chip}", f"PIN: {used_pin}"],
    "Channel": ["Online" if online_val else "In-store", ""]
}

for category, details in profile_data.items():
    if details[0]:
        st.write(f"**{category}:** {details[0]}")
        if details[1]:
            st.write(f"**{category}:** {details[1]}")

if probability > 0.7:
    recommendation = "Block transaction and contact cardholder immediately"
elif probability > 0.4:
    recommendation = "Flag for manual review before processing"
else:
    recommendation = "Process transaction normally"

st.info(f"Recommendation: {recommendation}")
