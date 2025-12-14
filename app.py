import streamlit as st
import joblib

# Load model
model = joblib.load("trained_spam_classifier_model.pkl")

st.title("📩 Spam Message Classifier")

# Text input with example
message = st.text_area(
    "Enter Message",
    placeholder="Example: Congratulations! You won a free prize. Click now!"
)

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        result = model.predict([message])[0]

        if result == 1:
            st.error("🚫 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")
