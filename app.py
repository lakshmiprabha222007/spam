import streamlit as st
import pickle

# Load trained spam classifier model
with open("trained_spam_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📩 Spam Message Classifier")

st.write("Enter a message to check whether it is Spam or Not Spam")

# Text input
message = st.text_area("Enter Message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        prediction = model.predict([message])

        if prediction[0] == 1:
            st.error("🚫 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")
