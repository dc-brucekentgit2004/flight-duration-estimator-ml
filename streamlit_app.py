# app/streamlit_app.py
import streamlit as st
import pandas as pd
from src.model_predict import predict_price, load_model_and_preprocessor
from src.preprocess import load_data
from joblib import load

st.set_page_config(page_title="Advanced Flight Delay/Price Predictor", layout="centered")

st.title("🚀 Advanced Flight Delay & Price Predictor")

st.sidebar.header("Input Flight Details")
with st.sidebar.form("flight_form"):
    date = st.date_input("Journey Date")
    carrier = st.selectbox("Airline", ["IndiGo","AirIndia","SpiceJet","Vistara","GoAir","AirAsia"], index=0)
    origin = st.selectbox("Origin", ["DEL","BOM","BLR","HYD","MAA","CCU","GOI","COK","TRV","PNQ"])
    dest = st.selectbox("Destination", ["DEL","BOM","BLR","HYD","MAA","CCU","GOI","COK","TRV","PNQ"])
    dep_time = st.text_input("Scheduled Dep Time (HHMM)", "0930")
    arr_time = st.text_input("Scheduled Arr Time (HHMM)", "1130")
    dist = st.number_input("Distance (km)", value=800)
    dep_delay = st.number_input("Departure Delay (mins)", value=0, step=1)
    submitted = st.form_submit_button("Predict")

if submitted:
    # build single-row dataframe
    df = pd.DataFrame([{
        "FL_DATE": date.strftime("%Y-%m-%d"),
        "OP_CARRIER": carrier,
        "ORIGIN": origin,
        "DEST": dest,
        "CRS_DEP_TIME": dep_time,
        "CRS_ARR_TIME": arr_time,
        "DISTANCE": dist,
        "DEP_DELAY": dep_delay,
        "ARR_DELAY": 0
    }])
    try:
        pred = predict_price(df)
        st.success(f"Predicted arrival delay (minutes): {pred:.2f} (positive means late)")
    except Exception as e:
        st.error("Prediction failed. Ensure model is trained and saved in model/ directory.")
        st.exception(e)

st.markdown("---")
st.write("This app uses an advanced stacked ensemble trained on engineered aviation features.")
