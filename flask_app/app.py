from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import accelerate
import transformers

# Initialize Flask app
app = Flask(__name__)

# Initialize NLTK and other utilities
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize sentiment analyzer and lemmatizer
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))  # Get stopwords list
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize each word
    return ' '.join(tokens)

# Define a function to preprocess reviews from CSV file
def preprocess_reviews_from_csv(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Drop rows where 'review_text' is NaN
    df = df.dropna(subset=['review_text'])

    # Check if 'review_text' column exists
    if 'review_text' not in df.columns:
        return None
    
    # Apply preprocessing to 'review_text'
    df['cleaned_review_text'] = df['review_text'].apply(lambda x: preprocess_text(x) if isinstance(x, str) else x)
    df['cleaned_review_text'].replace('', np.nan, inplace=True)
    
    # Remove rows with missing values in 'cleaned_review_text'
    df = df.dropna(subset=['cleaned_review_text'])
    
    # Sentiment Analysis: Add sentiment scores to the DataFrame
    df['sentiment'] = df['cleaned_review_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    return df

# Define a function to detect contradicting reviews
def detect_contradicting_reviews(row):
    if row['review_score'] == 1 and row['sentiment'] < 0:
        return 1
    elif row['review_score'] == -1 and row['sentiment'] > 0:
        return 1
    else:
        return 0
    
    # Route to serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html')  # This will load your index.html

# Endpoint for prediction and model training
@app.route('/train', methods=['POST'])
def train_model():
    # Receive CSV file from the client
    file = request.files.get('file')

    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Save file temporarily
    file_path = 'temp_reviews.csv'
    file.save(file_path)
    
    # Preprocess reviews and analyze sentiment
    df = preprocess_reviews_from_csv(file_path)
    
    if df is None:
        return jsonify({'error': 'The file does not contain review_text column or is improperly formatted'}), 400
    
    # Apply contradiction detection
    df['contradicting'] = df.apply(detect_contradicting_reviews, axis=1)
    df = df[df['contradicting'] == 0]
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_review_text'])
    X, y = tfidf_matrix, df['review_score']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Oversampling using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Logistic Regression Model
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions
    y_pred = lr.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Save model and vectorizer for future use
    joblib.dump(lr, 'sentiment_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    return jsonify({
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'message': 'Model trained and saved successfully'
    })

# Endpoint for sentiment prediction using the trained model
@app.route('/predict', methods=['POST'])
def predict():
    # Receive review text for prediction
    data = request.get_json()
    review_text = data.get('review_text')

    if not review_text:
        return jsonify({'error': 'No review text provided'}), 400

    # Preprocess review text
    cleaned_review = preprocess_text(review_text)
    
    # Load the saved model and vectorizer
    lr = joblib.load('sentiment_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Convert the review text to TF-IDF vector
    tfidf_matrix = tfidf_vectorizer.transform([cleaned_review])

    # Predict sentiment
    prediction = lr.predict(tfidf_matrix)
    
    return jsonify({
        'prediction': prediction
    })

# Unzip the BERT model files into flask_app directory if not done already
@app.route("/predict_bert", methods=["POST"])
def predict_bert():
    # Receive review text for prediction
    data = request.get_json()
    review_text = data.get("review_text")
    if not review_text:
        return jsonify({"error": "No review text provided"}), 400

    # Preprocess review text
    cleaned_review = preprocess_text(review_text)
    print(cleaned_review)
    #Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("./saved_model")
    model = BertForSequenceClassification.from_pretrained("./saved_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Function to predict sentiment
    def predict_sentiment(text):
        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        return prediction

    prediction = predict_sentiment(cleaned_review)
    
    print(prediction);
    return jsonify({
        "prediction": prediction
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
