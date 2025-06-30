import streamlit as st
import pickle
import numpy as np

# Load model
with open("fraud_detection_model.pkl", "rb") as f:
    model = pickle.load(f)

# Hard‑coded means from training set
MEANS = {'distance_from_home': 37.3137, 'distance_from_last_transaction': 7.1226, 'ratio_to_median_purchase_price': 2.958, 'repeat_retailer': 0.8813, 'used_chip': 0.3261, 'used_pin_number': 0.07503, 'online_order': 0.7317}
st.title("Fraud Detection")

dfh = st.number_input("Distance from Home", value=float(MEANS['distance_from_home']))
dflt = st.number_input("Distance from Last Transaction", value=float(MEANS['distance_from_last_transaction']))
ratio = st.number_input("Ratio to Median Purchase Price", value=float(MEANS['ratio_to_median_purchase_price']))
rpt = st.number_input("Repeat Retailer (0 or 1)", value=float(MEANS['repeat_retailer']))
chip = st.number_input("Used Chip (0 or 1)", value=float(MEANS['used_chip']))
pin = st.number_input("Used PIN Number (0 or 1)", value=float(MEANS['used_pin_number']))
online = st.number_input("Online Order (0 or 1)", value=float(MEANS['online_order']))

if st.button("Predict"):
    X = np.array([[dfh, dflt, ratio, rpt, chip, pin, online]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    st.write("**Prediction:**", "FRAUD" if pred==1 else "LEGITIMATE")
    st.write(f"**Fraud Probability:** {prob:.2%}")
    st.write("HIGH RISK" if prob>0.5 else "LOW RISK")
