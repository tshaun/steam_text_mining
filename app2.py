from flask import Flask, request, jsonify
import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import numpy as np
from werkzeug.utils import secure_filename

# NLTK setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Flask app
server = Flask(__name__)

# Directory for processed data (new uploaded dataset)
PROCESSED_DATA_DIR = "processed_data2"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"csv"}

# Load BERT model and tokenizer
try:
    bert_tokenizer = BertTokenizer.from_pretrained("./saved_model")
    bert_model = BertForSequenceClassification.from_pretrained("./saved_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
except Exception as e:
    print(f"Error loading BERT model: {e}")
    raise

# Load or initialize LSA vectorizer & model
try:
    lsa_vectorizer = joblib.load("lsa_vectorizer.pkl")
    lsa_model = joblib.load("lsa_model.pkl")
except Exception as e:
    print(f"Error loading LSA model: {e}")
    raise

# Preprocessing function for text cleaning
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Function to apply BERT for sentiment analysis
def predict_sentiment(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "Positive" if prediction == 1 else "Negative"

# Function to apply LSA for topic modeling and assign topic
def predict_topic(df):
    df['cleaned_text'] = df['review_text'].apply(preprocess_text)
    
    tfidf_matrix = lsa_vectorizer.transform(df['cleaned_text'])
    topic_vectors = lsa_model.transform(tfidf_matrix)
    
    df['predicted_topic'] = np.argmax(topic_vectors, axis=1) + 1  # Topics are 1-based
    return df

# Process uploaded file and generate processed CSVs
def process_uploaded_file(csv_path):
    try:
        print(f"[PROCESS] Reading uploaded CSV: {csv_path}")
        df = pd.read_csv(csv_path)

        if "review_text" not in df.columns:
            raise ValueError("Uploaded file must contain a 'review_text' column.")
        
        print("[PROCESS] Data Loaded. First 5 rows:")
        print(df.head()) 

        df['review_text'] = df['review_text'].fillna('').astype(str)

        print("[PROCESS] Preprocessing text...")
        df["bert_sentiment"] = df["review_text"].apply(lambda x: 1 if predict_sentiment(x) == "Positive" else 0)

        print("[PROCESS] Running topic modeling...")
        df = predict_topic(df)

        print("[PROCESS] Processed DataFrame:")
        print(df[["review_text", "bert_sentiment", "predicted_topic"]].head())

        sentiment_df = df[["review_text", "bert_sentiment"]]
        sentiment_path = os.path.join(PROCESSED_DATA_DIR, "sentiment.csv")
        sentiment_df.to_csv(sentiment_path, index=False)
        print(f"[SAVED] Sentiment CSV saved to: {sentiment_path}")

        terms = lsa_vectorizer.get_feature_names_out()
        topic_terms = lsa_model.components_

        topic_data = []
        for topic_idx, weights in enumerate(topic_terms):
            sorted_terms = sorted(zip(terms, weights), key=lambda x: x[1], reverse=True)[:10]

            for term, weight in sorted_terms:
                if term not in stop_words and bool(re.match('^[a-zA-Z]+$', term)):
                    topic_data.append({
                        "topic": topic_idx + 1,
                        "term": term,
                        "weightage": np.log(weight + 1e-10)  # Log-transform the weight
                    })

        topic_df = pd.DataFrame(topic_data)
        topic_path = os.path.join(PROCESSED_DATA_DIR, "topic.csv")
        topic_df.to_csv(topic_path, index=False)
        print(f"[SAVED] Topic CSV saved to: {topic_path}")

    except Exception as e:
        print(f"[ERROR] Error in file processing: {str(e)}")
        raise

# Upload endpoint
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@server.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        try:
            process_uploaded_file(filepath)
            return jsonify({"message": "File uploaded and processed successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type"}), 400

# Dash app for visualization (integrated with Flask server)
dash_app = Dash(__name__, server=server, url_base_pathname="/dashboard/")

def load_data():
    sentiment_path = os.path.join(PROCESSED_DATA_DIR, "sentiment.csv")
    topic_path = os.path.join(PROCESSED_DATA_DIR, "topic.csv")
    sentiment_df = pd.read_csv(sentiment_path) if os.path.exists(sentiment_path) else pd.DataFrame()
    topic_df = pd.read_csv(topic_path) if os.path.exists(topic_path) else pd.DataFrame()
    return sentiment_df, topic_df

dash_app.layout = html.Div([
    html.H1("Game Reviews NLP Dashboard (Uploaded Dataset)", style={"textAlign": "center"}),
    html.Div([
        dcc.Tabs([
            dcc.Tab(label="Sentiment Analysis", value="sentiment"),
            dcc.Tab(label="Topic Modeling", value="topic")
        ], id="tabs", value="sentiment")
    ]),

    html.Div(id="sentiment-dashboard", children=[
        html.H2("Sentiment Summary"),
        dcc.Graph(id="sentiment-bar-chart-bert"),
        dcc.Interval(id="interval-component", interval=60 * 1000, n_intervals=0)
    ]),

    html.Div(id="topic-dashboard", children=[
        html.H2("Topic Analysis"),
        dcc.Dropdown(id="topic-dropdown", options=[], value=None),
        dcc.Graph(id="topic-bar-chart"),
        dcc.Interval(id="interval-topic-component", interval=60 * 1000, n_intervals=0)
    ])
])

@dash_app.callback(
    Output("sentiment-dashboard", "style"),
    Output("topic-dashboard", "style"),
    Input("tabs", "value")
)
def switch_tab(tab_value):
    if tab_value == "sentiment":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}

@dash_app.callback(
    Output("sentiment-bar-chart-bert", "figure"),
    Input("interval-component", "n_intervals")
)
def update_sentiment_dashboard(n_intervals):
    sentiment_df, _ = load_data()
    if sentiment_df.empty:
        return {}

    sentiment_bert_summary = sentiment_df.groupby("bert_sentiment").size().reset_index(name="count")
    sentiment_bert_summary["bert_sentiment"] = sentiment_bert_summary["bert_sentiment"].map({0: "Negative", 1: "Positive"})

    return px.bar(
        sentiment_bert_summary,
        x="bert_sentiment",
        y="count",
        color="bert_sentiment",
        title="BERT Sentiment Analysis"
    )

@dash_app.callback(
    [Output("topic-dropdown", "options"),
     Output("topic-bar-chart", "figure")],
    Input("interval-topic-component", "n_intervals"),
    Input("topic-dropdown", "value")
)
def update_topic_dashboard(n_intervals, selected_topic):
    _, topic_df = load_data()

    if topic_df.empty or 'term' not in topic_df or 'weightage' not in topic_df:
        return [], {}

    topic_df = topic_df.dropna(subset=['weightage'])
    topic_df = topic_df[topic_df['weightage'].apply(lambda x: isinstance(x, (int, float)))] 

    topics = topic_df['topic'].unique()
    topic_options = [{'label': f"Topic {t}", 'value': t} for t in sorted(topics)]

    if selected_topic is None:
        selected_topic = topic_options[0]['value']

    topic_filtered = topic_df[topic_df['topic'] == selected_topic]
    topic_filtered = topic_filtered.sort_values(by='weightage', ascending=False)

    topic_bar = px.bar(
        topic_filtered,
        x="term",
        y="weightage",
        color="term",
        title=f"Topic {selected_topic} Analysis",
        labels={"weightage": "Weightage (Log Transformed)"},
        hover_data={"term": True, "weightage": True}
    )

    return topic_options, topic_bar

if __name__ == "__main__":
    server.run(debug=False, port=8052, use_reloader=False)
