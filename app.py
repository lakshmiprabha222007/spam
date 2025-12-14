import streamlit as st
import pickle

with open("trained_spam_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Spam Message Classifier")

message = st.text_area("Enter the message")

if st.button("Predict"):
    if message.strip():
        prediction = model.predict([message])

        if prediction[0] == 1 or prediction[0] == "spam":
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam Message")
