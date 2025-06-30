import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .fixed-button {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 100;
    }
    .fixed-button button {
        background-color: #34495e;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-right: 0.5rem;
        border: none;
    }
    .fixed-button button:hover {
        background-color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    with open("fraud_detection_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found")
    st.stop()

st.markdown(
    '<h1 style="text-align:center;color:white;font-weight:300;">Transaction Risk Assessment</h1>',
    unsafe_allow_html=True
)

fraud_example = {
    "distance_from_home": 2.13,
    "distance_from_last_transaction": 56.37,
    "ratio_to_median_purchase_price": 6.36,
    "repeat_retailer": 1.0,
    "used_chip": 0.0,
    "used_pin_number": 0.0,
    "online_order": 1.0,
}
valid_example = {
    "distance_from_home": 8.08,
    "distance_from_last_transaction": 0.14,
    "ratio_to_median_purchase_price": 0.38,
    "repeat_retailer": 1.0,
    "used_chip": 0.0,
    "used_pin_number": 1.0,
    "online_order": 0.0,
}

with st.sidebar:
    st.header("Transaction Details")
    use_home = st.checkbox("Distance from home", key='use_home', value=False)
    distance_home = st.number_input("Distance (km)", 0.0, 500.0, 0.0, key='distance_from_home')
    use_last = st.checkbox("Distance from last transaction", key='use_last', value=False)
    distance_last = st.number_input("Distance (km)", 0.0, 200.0, 0.0, key='distance_from_last_transaction')
    use_price = st.checkbox("Price vs typical spending", key='use_price', value=False)
    price_ratio = st.number_input("Ratio", 0.0, 20.0, 0.0, key='ratio_to_median_purchase_price')
    repeat_retailer = st.checkbox("Previously used retailer", key='repeat_retailer', value=False)
    used_chip = st.checkbox("Chip card used", key='used_chip', value=False)
    used_pin = st.checkbox("PIN entered", key='used_pin', value=False)
    online_order = st.checkbox("Online transaction", key='online_order', value=False)

if st.button("Example Fraud", key='btn_fraud', help='Load example fraud transaction'):
    for k,v in fraud_example.items(): st.session_state[k] = v
    st.session_state['use_home'] = True
    st.session_state['use_last'] = True
    st.session_state['use_price'] = True
    st.session_state['repeat_retailer'] = fraud_example['repeat_retailer']==1
    st.session_state['used_chip'] = fraud_example['used_chip']==1
    st.session_state['used_pin'] = fraud_example['used_pin_number']==1
    st.session_state['online_order'] = fraud_example['online_order']==1

if st.button("Example Valid", key='btn_valid', help='Load example valid transaction'):
    for k,v in valid_example.items(): st.session_state[k] = v
    st.session_state['use_home'] = True
    st.session_state['use_last'] = True
    st.session_state['use_price'] = True
    st.session_state['repeat_retailer'] = valid_example['repeat_retailer']==1
    st.session_state['used_chip'] = valid_example['used_chip']==1
    st.session_state['used_pin'] = valid_example['used_pin_number']==1
    st.session_state['online_order'] = valid_example['online_order']==1

X = np.array([[
    st.session_state['distance_from_home'] if st.session_state['use_home'] else 0.0,
    st.session_state['distance_from_last_transaction'] if st.session_state['use_last'] else 0.0,
    st.session_state['ratio_to_median_purchase_price'] if st.session_state['use_price'] else 0.0,
    1.0 if st.session_state['repeat_retailer'] else 0.0,
    1.0 if st.session_state['used_chip'] else 0.0,
    1.0 if st.session_state['used_pin'] else 0.0,
    1.0 if st.session_state['online_order'] else 0.0
]])
prob = model.predict_proba(X)[0][1]

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob*100,
    title={'text':"Credit Card Fraud Detection"},
    gauge={
        'axis':{'range':[0,100]},
        'bar':{'color":"#34495e"},
        'steps':[
            {'range':[0,25],'color":"#2ecc71"},
            {'range':[25,50],'color":"#f39c12"},
            {'range':[50,75],'color":"#e67e22"},
            {'range':[75,100],'color":"#e74c3c"}
        ]
    }
))
fig.update_layout(height=380, margin=dict(l=20,r=20,t=60,b=20), font=dict(size=14))
st.plotly_chart(fig, use_container_width=True)

if prob>0.7: rec="Block transaction and contact cardholder immediately"
elif prob>0.4: rec="Flag for manual review before processing"
else: rec="Process transaction normally"

st.info(f"Recommendation: {rec}")
