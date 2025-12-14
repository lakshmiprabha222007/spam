import streamlit as st
import joblib

# Load trained spam detection pipeline
model = joblib.load("trained_spam_classifier_model.pkl")

st.title("📩 Spam Detection App")

# Text input
message = st.text_area(
    "Enter Message",
    placeholder="Example: Congratulations! You won a prize. Click now!"
)

if st.button("Check Spam"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        prediction = model.predict([message])[0]

        if prediction == 1:
            st.error("🚫 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")
