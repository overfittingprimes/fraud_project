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

MEANS = {
    'distance_from_home': 37.3137,
    'distance_from_last_transaction': 7.1226,
    'ratio_to_median_purchase_price': 2.958,
    'repeat_retailer': 0.8813,
    'used_chip': 0.3261,
    'used_pin_number': 0.07503,
    'online_order': 0.7317
}

with st.sidebar:
    st.header("Transaction Details")
    
    distance_home_input = st.selectbox("Distance from home address", ["None", "Custom"])
    if distance_home_input == "Custom":
        distance_home = st.number_input("Distance (km)", 0.0, 500.0, 37.31)
    else:
        distance_home = MEANS['distance_from_home']
    
    distance_last_input = st.selectbox("Distance from last transaction", ["None", "Custom"])
    if distance_last_input == "Custom":
        distance_last = st.number_input("Distance (km)", 0.0, 200.0, 7.12)
    else:
        distance_last = MEANS['distance_from_last_transaction']
    
    price_ratio_input = st.selectbox("Price vs typical spending", ["None", "Custom"])
    if price_ratio_input == "Custom":
        price_ratio = st.number_input("Ratio", 0.1, 20.0, 2.96)
    else:
        price_ratio = MEANS['ratio_to_median_purchase_price']
    
    repeat_retailer = st.selectbox("Previously used retailer", ["None", "No", "Yes"])
    used_chip = st.selectbox("Chip card used", ["None", "No", "Yes"])
    used_pin = st.selectbox("PIN entered", ["None", "No", "Yes"])
    online_order = st.selectbox("Online transaction", ["None", "No", "Yes"])

repeat_val = MEANS['repeat_retailer'] if repeat_retailer == "None" else (1 if repeat_retailer == "Yes" else 0)
chip_val = MEANS['used_chip'] if used_chip == "None" else (1 if used_chip == "Yes" else 0)
pin_val = MEANS['used_pin_number'] if used_pin == "None" else (1 if used_pin == "Yes" else 0)
online_val = MEANS['online_order'] if online_order == "None" else (1 if online_order == "Yes" else 0)

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
fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

if probability > 0.7:
    recommendation = "Block transaction and contact cardholder immediately"
elif probability > 0.4:
    recommendation = "Flag for manual review before processing"
else:
    recommendation = "Process transaction normally"

st.info(f"Recommendation: {recommendation}")
