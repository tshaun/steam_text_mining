import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import os

# Load the sentiment data (already tokenized)
df = pd.read_csv('sentiment.csv')

# Split into positive and negative sentiment
df_positive = df[df['bert_sentiment'] == 1]
df_negative = df[df['bert_sentiment'] == 0]

# Define a function to run and save LSA model for each sentiment group
def run_lsa_and_save(df_subset, label):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_subset['cleaned_review_text'])

    lsa_model = TruncatedSVD(n_components=10, random_state=42)
    lsa_model.fit(tfidf_matrix)

    # Save model and vectorizer
    joblib.dump(lsa_model, f'lsa_model_{label}.pkl')
    joblib.dump(tfidf_vectorizer, f'lsa_vectorizer_{label}.pkl')

# Run LSA for both sentiments
run_lsa_and_save(df_positive, 'positive')
run_lsa_and_save(df_negative, 'negative')

print("LSA models for positive and negative sentiment saved.")
