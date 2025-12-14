import streamlit as st
import pickle

with open("trained_spam_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📩 Spam Message Classifier")

message = st.text_area("Enter Message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        result = model.predict([message])[0]

        if result == 1:
            st.error("🚫 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")
