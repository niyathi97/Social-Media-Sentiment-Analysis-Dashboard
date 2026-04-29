import streamlit as st
import pickle
import sys
import os

# 🔹 Fix path for src folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import preprocess_text

# 🔹 Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# 🔹 Load vectorizer
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 🔹 Prediction function
def predict_sentiment(text):
    clean = preprocess_text(text)
    vector = vectorizer.transform([clean])
    prediction = model.predict(vector)[0]
    return prediction

# ---------------- UI ----------------

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="centered")

st.title("📊 Social Media Sentiment Analysis Dashboard")

st.write("Analyze public sentiment from text (positive / negative)")

# 🔹 Input box
user_input = st.text_area("Enter your text here:")

# 🔹 Button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict_sentiment(user_input)

        if result == "positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😡 Negative Sentiment")

# 🔹 Footer
st.write("---")
st.caption("Built using NLP + Machine Learning + Streamlit")