import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import gensim
# from gensim.models import Word2Vec

import re

import matplotlib.pyplot as plt
import seaborn as sns

# from wordcloud import WordCloud
# import spacy

import random

# Download required resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Define a function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))  # Get stopwords list
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize each word
    return ' '.join(tokens)

# Define a function to preprocess reviews from a CSV file
def preprocess_reviews_from_csv(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Drop rows where 'review_text' is NaN
    df = df.dropna(subset=['review_text'])

    # Check if 'review_text' column exists
    if 'review_text' not in df.columns:
        print("Error: The input CSV does not contain a 'review_text' column.")
        return None
    
    # Apply the preprocessing function to the review_text column
    df['cleaned_review_text'] = df['review_text'].apply(lambda x: preprocess_text(x) if isinstance(x, str) else x)
    
    # Replace empty strings with NaN in 'cleaned_review_text' column
    df['cleaned_review_text'].replace('', np.nan, inplace=True)

    # Check for missing values in the cleaned_review_text column
    missing_rows = df[df['cleaned_review_text'].isnull()]

    if not missing_rows.empty:
        print(f"Missing values detected in {len(missing_rows)} rows. Replacing empty strings with NaN.")
    else:
        print("No missing values detected in cleaned_review_text column.")

    # Drop rows with missing values in the 'cleaned_review_text' column
    df = df.dropna(subset=['cleaned_review_text'])
    
    # After dropping missing rows, check if there are any remaining missing values
    missing_after_drop = df[df['cleaned_review_text'].isnull()]
    if not missing_after_drop.empty:
        print(f"After dropping, {len(missing_after_drop)} rows still have missing values.")
    else:
        print("No missing values after dropping rows.")
    
    # Sentiment Analysis: Add sentiment scores to the DataFrame
    df['sentiment'] = df['cleaned_review_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    return df

# Define a function to detect contradicting reviews
def detect_contradicting_reviews(row):
    # Check for rating-text mismatch (Contradiction rule)
    if row['review_score'] == 1 and row['sentiment'] < 0:
        return 1  # Mark as a contradiction
    elif row['review_score'] == -1 and row['sentiment'] > 0:
        return 1
    else:
        return 0  # Not a contradiction

# Example usage:
if __name__ == "__main__":
    csv_file = "filtered_dataset.csv"  # Path to your CSV file
    
    # Preprocess reviews and analyze sentiment
    df = preprocess_reviews_from_csv(csv_file)
    
    if df is not None:
        # Apply the contradiction detection function
        df['contradicting'] = df.apply(detect_contradicting_reviews, axis=1)

        # Create a new DataFrame with contradicting reviews
        contradicting_review = df[df['contradicting'] == 1]
        
        # Save contradicting reviews to a CSV file
        contradicting_review.to_csv('contradicting_reviews.csv', index=False)
        print(f"Contradicting reviews saved to 'contradicting_reviews.csv'")

        # Drop the contradicting reviews from the original DataFrame
        df = df[df['contradicting'] == 0]
        print(f"Contradicting reviews removed from the main DataFrame.")
        
        # Optional: Save the cleaned DataFrame without contradicting reviews
        df.to_csv('cleaned_reviews.csv', index=False)
        print(f"Cleaned data saved to 'cleaned_reviews.csv'")

        # TF-IDF Vectorization
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_review_text'].dropna())

        # Convert the TF-IDF matrix to a DataFrame (using sparse matrix format)
        tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, columns=tfidf_vectorizer.get_feature_names_out())

        # Define features and target
        X = tfidf_matrix          # TF-IDF features (sparse matrix)
        y = df['review_score']  # Target variable: 1 (positive/recommend) or -1 (negative/not recommend)

        # --- Train-Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- Oversampling using SMOTE ---
        # Initialize SMOTE
        smote = SMOTE(random_state=42)

        # Apply SMOTE to the training data
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        lr.fit(X_train_resampled, y_train_resampled)

        y_pred = lr.predict(X_test)

        print(y_pred)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))