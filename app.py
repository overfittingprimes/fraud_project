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
""", unsafe_allow_html=True)

try:
    with open("fraud_detection_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found")
    st.stop()

st.markdown('<h1 style="text-align:center;color:white;font-weight:300;">Transaction Risk Assessment</h1>', unsafe_allow_html=True)

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

if st.markdown(
    '<div class="fixed-button">'
    '<button id="fraud">Example Fraud</button>'
    '<button id="valid">Example Valid</button>'
    '</div>',
    unsafe_allow_html=True
):
    pass

st.components.v1.html("""
<script>
const send = (btn) => {
    const payload = btn === 'fraud' ? 'fraud' : 'valid';
    window.parent.postMessage({func: payload}, '*');
};
document.getElementById('fraud').onclick = () => send('fraud');
document.getElementById('valid').onclick = () => send('valid');
</script>
""", height=0)

st.experimental_data_editor({})

if st.session_state.get('message') == 'fraud':
    for k,v in fraud_example.items():
        st.session_state[k] = v
    st.session_state['repeat_retailer_chk'] = fraud_example['repeat_retailer'] == 1
    st.session_state['used_chip_chk'] = fraud_example['used_chip'] == 1
    st.session_state['used_pin_number_chk'] = fraud_example['used_pin_number'] == 1
    st.session_state['online_order_chk'] = fraud_example['online_order'] == 1
    st.session_state['message'] = None

if st.session_state.get('message') == 'valid':
    for k,v in valid_example.items():
        st.session_state[k] = v
    st.session_state['repeat_retailer_chk'] = valid_example['repeat_retailer'] == 1
    st.session_state['used_chip_chk'] = valid_example['used_chip'] == 1
    st.session_state['used_pin_number_chk'] = valid_example['used_pin_number'] == 1
    st.session_state['online_order_chk'] = valid_example['online_order'] == 1
    st.session_state['message'] = None

with st.sidebar:
    st.header("Transaction Details")

    use_distance_home = st.checkbox("Distance from home", key='use_home', value=False)
    distance_home = st.number_input("km", 0.0, 500.0, st.session_state.get('distance_from_home', 0.0), key='distance_from_home')

    use_distance_last = st.checkbox("Distance from last transaction", key='use_last', value=False)
    distance_last = st.number_input("km", 0.0, 200.0, st.session_state.get('distance_from_last_transaction', 0.0), key='distance_from_last_transaction')

    use_price_ratio = st.checkbox("Price vs typical", key='use_price', value=False)
    price_ratio = st.number_input("ratio", 0.0, 20.0, st.session_state.get('ratio_to_median_purchase_price', 0.0), key='ratio_to_median_purchase_price')

    repeat_retailer = st.checkbox("Repeat retailer", key='repeat_retailer_chk', value=False)
    used_chip = st.checkbox("Chip used", key='used_chip_chk', value=False)
    used_pin = st.checkbox("PIN entered", key='used_pin_number_chk', value=False)
    online_order = st.checkbox("Online order", key='online_order_chk', value=False)

X = np.array([[
    distance_home if use_distance_home else 0.0,
    distance_last if use_distance_last else 0.0,
    price_ratio if use_price_ratio else 0.0,
    1.0 if repeat_retailer else 0.0,
    1.0 if used_chip else 0.0,
    1.0 if used_pin else 0.0,
    1.0 if online_order else 0.0
]])
pred = model.predict(X)[0]
prob = model.predict_proba(X)[0][1]

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob * 100,
    title={'text': "Fraud Risk (%)"},
    gauge={
        'axis': {'range': [0,100]},
        'bar': {'color': "#34495e"},
        'steps': [
            {'range':[0,25],'color':"#2ecc71"},
            {'range':[25,50],'color':"#f39c12"},
            {'range':[50,75],'color':"#e67e22"},
            {'range':[75,100],'color':"#e74c3c"},
        ]
    }
))
fig.update_layout(height=380, margin=dict(l=20,r=20,t=60,b=20), font=dict(size=14))
st.plotly_chart(fig, use_container_width=True)

if prob > 0.7:
    rec = "Block & contact cardholder"
elif prob > 0.4:
    rec = "Flag for manual review"
else:
    rec = "Process normally"
st.info(f"Recommendation: {rec}")
