import streamlit as st
import pickle

# Load model
with open("trained_spam_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Spam Message Classifier")

message = st.text_area("Enter the message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        # Convert text → numeric features
        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)

        if prediction[0] == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam Message")
