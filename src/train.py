import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocessing import preprocess_text

# 🔹 Ensure models folder exists
os.makedirs("models", exist_ok=True)

# 🔹 Load dataset (Kaggle)
columns = ['target', 'id', 'date', 'flag', 'user', 'text']
data = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    encoding='latin-1',
    names=columns
)

# 🔹 Reduce dataset size (for faster training)
data = data.sample(20000, random_state=42)

print("Dataset Loaded:", data.shape)

# 🔹 Convert labels (0 → negative, 4 → positive)
data['sentiment'] = data['target'].map({0: 'negative', 4: 'positive'})

# 🔹 Preprocess text
data['clean_text'] = data['text'].apply(preprocess_text)

print("Preprocessing Done")

# 🔹 TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text'])
y = data['sentiment']

print("TF-IDF Done:", X.shape)

# 🔹 Train Model
model = LogisticRegression()
model.fit(X, y)

print("Model Training Completed")

# 🔹 Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# 🔹 Save vectorizer
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model Saved Successfully")