import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.25rem;
        color: black;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.25rem;
        color: black;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        color: black;
    }
    .info-box h4 {
        color: black;
    }
</style>
""", unsafe_allow_html=True)

try:
    with open("fraud_detection_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please make sure fraud_detection_model.pkl is in the directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.markdown('<div class="main-header">Fraud Detection System</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h4>About This System</h4>
    This machine learning model analyzes transaction patterns to detect potential credit card fraud.
    Adjust the parameters below to see how different transaction characteristics affect the fraud probability.
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Transaction Parameters")
    st.markdown("---")
    
    if st.button("Reset to Defaults", use_container_width=True):
        st.rerun()
    
    st.markdown("### Location Features")
    distance_from_home = st.number_input(
        "Distance from Home (km)",
        min_value=0.0,
        max_value=500.0,
        value=37.31,
        help="How far the transaction location is from the cardholder's home address",
        format="%.2f"
    )
    
    distance_from_last = st.number_input(
        "Distance from Last Transaction (km)",
        min_value=0.0,
        max_value=200.0,
        value=7.12,
        help="Distance between current and previous transaction locations",
        format="%.2f"
    )
    
    st.markdown("### Purchase Amount")
    price_ratio = st.number_input(
        "Ratio to Median Purchase Price",
        min_value=0.1,
        max_value=20.0,
        value=2.96,
        help="Current transaction amount compared to user's median transaction amount",
        format="%.2f"
    )
    
    st.markdown("### Transaction Details")
    repeat_retailer = st.selectbox(
        "Repeat Retailer",
        options=[0, 1],
        index=1,
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether the cardholder has used this retailer before"
    )
    
    st.markdown("### Security Features")
    used_chip = st.selectbox(
        "Used Chip",
        options=[0, 1],
        index=0,
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether the transaction used chip technology"
    )
    
    used_pin = st.selectbox(
        "Used PIN Number",
        options=[0, 1],
        index=0,
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether a PIN was required for the transaction"
    )
    
    online_order = st.selectbox(
        "Online Order",
        options=[0, 1],
        index=1,
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether this was an online transaction"
    )

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Feature Values")
    
    features = {
        'Distance from Home': distance_from_home,
        'Distance from Last': distance_from_last,
        'Price Ratio': price_ratio,
        'Repeat Retailer': repeat_retailer,
        'Used Chip': used_chip,
        'Used PIN': used_pin,
        'Online Order': online_order
    }
    
    feature_names = list(features.keys())
    values = list(features.values())
    
    fig = go.Figure(data=[
        go.Bar(x=values, y=feature_names, orientation='h',
               marker_color=['red' if v > np.mean(values) else 'blue' for v in values])
    ])
    fig.update_layout(
        title="Current Transaction Features",
        xaxis_title="Values",
        yaxis_title="Features",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Fraud Detection Results")
    
    try:
        X = np.array([[distance_from_home, distance_from_last, price_ratio, 
                      repeat_retailer, used_chip, used_pin, online_order]])
        
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        col_pred, col_prob = st.columns(2)
        
        with col_pred:
            if prediction == 1:
                st.error("Fraud Detected")
            else:
                st.success("Legitimate Transaction")
        
        with col_prob:
            st.metric(
                label="Fraud Probability",
                value=f"{probability:.1%}",
                delta=f"{probability - 0.5:.1%}" if probability != 0.5 else "0.0%"
            )
        
        if probability > 0.75:
            risk_class = "risk-high"
            risk_text = "Very high risk - immediate attention required"
        elif probability > 0.5:
            risk_class = "risk-high"
            risk_text = "High risk - manual review recommended"
        elif probability > 0.25:
            risk_class = "risk-low"
            risk_text = "Moderate risk - monitor transaction"
        else:
            risk_class = "risk-low"
            risk_text = "Low risk - transaction appears normal"
        
        st.markdown(f'<div class="{risk_class}">{risk_text}</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

with col2:
    st.subheader("Risk Meter")
    try:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fraud Risk %"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    except:
        st.info("Risk meter will appear after prediction")
    
    st.subheader("Transaction Summary")
    summary_data = {
        "Feature": ["Distance from Home", "Distance from Last", "Price Ratio", "Security Score"],
        "Value": [
            f"{distance_from_home:.1f} km",
            f"{distance_from_last:.1f} km", 
            f"{price_ratio:.2f}x",
            f"{used_chip + used_pin}/2"
        ],
        "Status": [
            "High" if distance_from_home > 100 else "Normal",
            "High" if distance_from_last > 50 else "Normal",
            "High" if price_ratio > 5 else "Normal",
            "Secure" if used_chip + used_pin >= 1 else "Insecure"
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>Fraud Detection System | Built with Streamlit & Machine Learning</p>
    <p><em>This system is for demonstration purposes. Always verify results with additional security measures.</em></p>
</div>
""", unsafe_allow_html=True)
