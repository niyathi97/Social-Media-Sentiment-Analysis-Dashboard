import pickle
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

# 🔹 Test input
if __name__ == "__main__":
    while True:
        text = input("Enter text (or 'exit'): ")
        
        if text.lower() == 'exit':
            break
        
        result = predict_sentiment(text)
        print("Sentiment:", result)