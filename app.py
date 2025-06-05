
import streamlit as st
import joblib
import numpy as np

model = joblib.load('Prg1_dt_model.pkl')

st.title("Intrusion Detection System")
input_data = st.text_input("Enter 41 comma-separated values")
if st.button("Predict"):
    input_array = np.array([float(x) for x in input_data.split(',')]).reshape(1, -1)
    result = model.predict(input_array)
    st.write("🔒 Normal Traffic" if result[0] == 0 else "⚠️ Intrusion Detected!")
