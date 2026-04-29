import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (only first time)
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# 🔹 Function to clean text
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# 🔹 Function for preprocessing
def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

# 🔹 Test
if __name__ == "__main__":
    sample = "I love this product! It's amazing 😍 http://test.com"
    print("Original:", sample)
    print("Processed:", preprocess_text(sample))