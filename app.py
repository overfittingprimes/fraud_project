import streamlit as st
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px

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
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fraud detection model with caching."""
    try:
        with open("fraud_detection_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'fraud_detection_model.pkl' not found. Please ensure the model file is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

FEATURE_INFO = {
    'distance_from_home': {
        'default': 37.3137,
        'description': 'Distance from cardholder\'s home address (km)',
        'min_val': 0.0,
        'max_val': 500.0,
        'help': 'How far the transaction location is from the cardholder\'s registered home address'
    },
    'distance_from_last_transaction': {
        'default': 7.1226,
        'description': 'Distance from previous transaction location (km)',
        'min_val': 0.0,
        'max_val': 200.0,
        'help': 'Distance between current and previous transaction locations'
    },
    'ratio_to_median_purchase_price': {
        'default': 2.958,
        'description': 'Ratio to median purchase price',
        'min_val': 0.1,
        'max_val': 20.0,
        'help': 'Current transaction amount compared to user\'s median transaction amount'
    },
    'repeat_retailer': {
        'default': 0.8813,
        'description': 'Repeat retailer (0=No, 1=Yes)',
        'options': [0, 1],
        'help': 'Whether the cardholder has used this retailer before'
    },
    'used_chip': {
        'default': 0.3261,
        'description': 'Used chip (0=No, 1=Yes)',
        'options': [0, 1],
        'help': 'Whether the transaction used chip technology'
    },
    'used_pin_number': {
        'default': 0.07503,
        'description': 'Used PIN number (0=No, 1=Yes)',
        'options': [0, 1],
        'help': 'Whether a PIN was required for the transaction'
    },
    'online_order': {
        'default': 0.7317,
        'description': 'Online order (0=No, 1=Yes)',
        'options': [0, 1],
        'help': 'Whether this was an online transaction'
    }
}

def create_risk_gauge(probability: float) -> go.Figure:
    """Create a risk gauge chart."""
    fig = go.Figure(go.Indicator(
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
    fig.update_layout(height=300)
    return fig

def get_feature_importance_chart(features: Dict[str, float]) -> go.Figure:
    """Create a feature importance chart."""
    feature_names = list(features.keys())
    values = list(features.values())
    
    fig = go.Figure(data=[
        go.Bar(x=values, y=feature_names, orientation='h',
               marker_color=['red' if v > np.mean(values) else 'blue' for v in values])
    ])
    fig.update_layout(
        title="Input Feature Values",
        xaxis_title="Feature Values",
        yaxis_title="Features",
        height=400
    )
    return fig

def main():
    model = load_model()
    
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
        
        st.markdown("### Adjust the parameters below:")
        
        features = {}
        
        st.markdown("#### Location Features")
        features['distance_from_home'] = st.number_input(
            "Distance from Home (km)",
            min_value=FEATURE_INFO['distance_from_home']['min_val'],
            max_value=FEATURE_INFO['distance_from_home']['max_val'],
            value=FEATURE_INFO['distance_from_home']['default'],
            help=FEATURE_INFO['distance_from_home']['help'],
            format="%.2f"
        )
        
        features['distance_from_last_transaction'] = st.number_input(
            "Distance from Last Transaction (km)",
            min_value=FEATURE_INFO['distance_from_last_transaction']['min_val'],
            max_value=FEATURE_INFO['distance_from_last_transaction']['max_val'],
            value=FEATURE_INFO['distance_from_last_transaction']['default'],
            help=FEATURE_INFO['distance_from_last_transaction']['help'],
            format="%.2f"
        )
        
        st.markdown("#### Purchase Amount")
        features['ratio_to_median_purchase_price'] = st.number_input(
            "Ratio to Median Purchase Price",
            min_value=FEATURE_INFO['ratio_to_median_purchase_price']['min_val'],
            max_value=FEATURE_INFO['ratio_to_median_purchase_price']['max_val'],
            value=FEATURE_INFO['ratio_to_median_purchase_price']['default'],
            help=FEATURE_INFO['ratio_to_median_purchase_price']['help'],
            format="%.3f"
        )
        
        st.markdown("#### Transaction Details")
        features['repeat_retailer'] = st.selectbox(
            "Repeat Retailer",
            options=[0, 1],
            index=1 if FEATURE_INFO['repeat_retailer']['default'] > 0.5 else 0,
            format_func=lambda x: "Yes" if x == 1 else "No",
            help=FEATURE_INFO['repeat_retailer']['help']
        )
        
        st.markdown("#### Security Features")
        features['used_chip'] = st.selectbox(
            "Used Chip",
            options=[0, 1],
            index=1 if FEATURE_INFO['used_chip']['default'] > 0.5 else 0,
            format_func=lambda x: "Yes" if x == 1 else "No",
            help=FEATURE_INFO['used_chip']['help']
        )
        
        features['used_pin_number'] = st.selectbox(
            "Used PIN Number",
            options=[0, 1],
            index=1 if FEATURE_INFO['used_pin_number']['default'] > 0.5 else 0,
            format_func=lambda x: "Yes" if x == 1 else "No",
            help=FEATURE_INFO['used_pin_number']['help']
        )
        
        features['online_order'] = st.selectbox(
            "Online Order",
            options=[0, 1],
            index=1 if FEATURE_INFO['online_order']['default'] > 0.5 else 0,
            format_func=lambda x: "Yes" if x == 1 else "No",
            help=FEATURE_INFO['online_order']['help']
        )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current Feature Values")
        fig_features = get_feature_importance_chart(features)
        st.plotly_chart(fig_features, use_container_width=True)
        
        st.subheader("Fraud Detection Results")
        
        try:
            X = np.array([[features[key] for key in ['distance_from_home', 'distance_from_last_transaction', 
                                                   'ratio_to_median_purchase_price', 'repeat_retailer', 
                                                   'used_chip', 'used_pin_number', 'online_order']]])
            
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0][1]
            
            col_pred, col_prob = st.columns(2)
            
            with col_pred:
                if prediction == 1:
                    st.error("**FRAUD DETECTED**")
                else:
                    st.success("**LEGITIMATE TRANSACTION**")
            
            with col_prob:
                st.metric(
                    label="Fraud Probability",
                    value=f"{probability:.1%}",
                    delta=f"{probability - 0.5:.1%}" if probability != 0.5 else "0.0%"
                )
            
            if probability > 0.75:
                risk_class = "risk-high"
                risk_text = "**VERY HIGH RISK** - Immediate attention required"
            elif probability > 0.5:
                risk_class = "risk-high"
                risk_text = "**HIGH RISK** - Manual review recommended"
            elif probability > 0.25:
                risk_class = "risk-low"
                risk_text = "**MODERATE RISK** - Monitor transaction"
            else:
                risk_class = "risk-low"
                risk_text = "**LOW RISK** - Transaction appears normal"
            
            st.markdown(f'<div class="{risk_class}">{risk_text}</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        st.subheader("Risk Meter")
        try:
            fig_gauge = create_risk_gauge(probability)
            st.plotly_chart(fig_gauge, use_container_width=True)
        except:
            st.info("Risk meter will appear after prediction")
        
        st.subheader("Transaction Summary")
        summary_data = {
            "Feature": ["Distance from Home", "Distance from Last", "Price Ratio", "Security Score"],
            "Value": [
                f"{features['distance_from_home']:.1f} km",
                f"{features['distance_from_last_transaction']:.1f} km", 
                f"{features['ratio_to_median_purchase_price']:.2f}x",
                f"{features['used_chip'] + features['used_pin_number']}/2"
            ],
            "Status": [
                "High" if features['distance_from_home'] > 100 else "Normal",
                "High" if features['distance_from_last_transaction'] > 50 else "Normal",
                "High" if features['ratio_to_median_purchase_price'] > 5 else "Normal",
                "Secure" if features['used_chip'] + features['used_pin_number'] >= 1 else "Insecure"
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

if __name__ == "__main__":
    main()
