import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib

# Initialize NLTK and utilities
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    else:
        return ''  # Return an empty string if the text is not a string

# Load and preprocess data
df = pd.read_csv('steam_reviews.csv')
df['cleaned_review_text'] = df['review_text'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_review_text'])

# LSA Model (TruncatedSVD)
lsa_model = TruncatedSVD(n_components=10, random_state=42)
lsa_matrix = lsa_model.fit_transform(tfidf_matrix)

# Save the model and vectorizer
joblib.dump(lsa_model, 'lsa_model.pkl')
joblib.dump(tfidf_vectorizer, 'lsa_vectorizer.pkl')

print("LSA Model and Vectorizer saved!")
