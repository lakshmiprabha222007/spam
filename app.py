import streamlit as st
import joblib
import numpy as np

model = joblib.load("trained_spam_classifier_model.pkl")

st.title("Easy Prediction App")

value = st.number_input("Enter any number", value=1)

if st.button("Predict"):
    result = model.predict(np.array([[value]]))[0]
    st.write("Output:", result)
