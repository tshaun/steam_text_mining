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
import glob
import atexit
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from werkzeug.utils import secure_filename
import logging
import nltk
import joblib

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
logging.basicConfig(level=logging.INFO)

server = Flask(__name__)
PROCESSED_DATA_DIR = "processed_data2"
UPLOAD_FOLDER = "uploads"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"csv"}

def cleanup_folders():
    for f in glob.glob(os.path.join(UPLOAD_FOLDER, '*')):
        try:
            os.remove(f)
        except Exception as e:
            logging.warning(f"Failed to delete {f}: {e}")

atexit.register(cleanup_folders)

bert_tokenizer = BertTokenizer.from_pretrained("./saved_model")
bert_model = BertForSequenceClassification.from_pretrained("./saved_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def predict_sentiment(text):
    if not text.strip():
        return "Negative"
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "Positive" if prediction == 1 else "Negative"

def predict_topic(df, sentiment="positive", n_topics=5):
    df['cleaned_text'] = df['review_text'].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
    if tfidf_matrix.shape[0] == 0:
        return df, pd.DataFrame()
    svd = TruncatedSVD(n_components=min(n_topics, tfidf_matrix.shape[1]), random_state=42)
    topic_vectors = svd.fit_transform(tfidf_matrix)
    df['predicted_topic'] = np.argmax(topic_vectors, axis=1) + 1
    terms = vectorizer.get_feature_names_out()
    topic_data = []

    for topic_idx, component in enumerate(svd.components_):
        top_terms = sorted(zip(terms, component), key=lambda x: x[1], reverse=True)[:10]
        for term, weight in top_terms:
            weightage = float(-1 * np.log(weight + 1e-10)) if weight > 0 else 0.0
            topic_data.append({"topic": topic_idx + 1, "term": term, "weightage": weightage})

    topic_df = pd.DataFrame(topic_data)

    joblib.dump(vectorizer, os.path.join(PROCESSED_DATA_DIR, f"vectorizer_{sentiment}.joblib"))
    joblib.dump(svd, os.path.join(PROCESSED_DATA_DIR, f"svd_{sentiment}.joblib"))

    return df, topic_df

def predict_topic_single(text, sentiment="positive", n_topics=5):
    try:
        cleaned_text = preprocess_text(text)
        vec_path = os.path.join(PROCESSED_DATA_DIR, f"vectorizer_{sentiment}.joblib")
        svd_path = os.path.join(PROCESSED_DATA_DIR, f"svd_{sentiment}.joblib")

        logging.info(f"Loading vectorizer from: {vec_path}")
        logging.info(f"Loading SVD model from: {svd_path}")

        if not os.path.exists(vec_path) or not os.path.exists(svd_path):
            logging.error(f"Model files not found for sentiment '{sentiment}'.")
            return "Unknown"

        vectorizer = joblib.load(vec_path)
        svd = joblib.load(svd_path)

        tfidf_matrix = vectorizer.transform([cleaned_text])
        topic_vector = svd.transform(tfidf_matrix)

        logging.info(f"Topic vector: {topic_vector}")

        if np.linalg.norm(topic_vector) < 1e-6:
            logging.warning(f"Topic vector is near-zero for text: {text}")
            return "Unknown"

        predicted_topic = np.argmax(topic_vector) + 1
        return predicted_topic
    except Exception as e:
        logging.error(f"Topic prediction error: {e}")
        return "Unknown"

def process_uploaded_file(filepath):
    df = pd.read_csv(filepath)
    if "review_text" not in df.columns:
        raise ValueError("CSV must contain 'review_text' column.")
    df['review_text'] = df['review_text'].fillna("").astype(str)
    df['bert_sentiment'] = df['review_text'].apply(lambda x: 1 if predict_sentiment(x) == "Positive" else 0)
    df[['review_text', 'bert_sentiment']].to_csv(os.path.join(PROCESSED_DATA_DIR, "sentiment.csv"), index=False)

    df_pos = df[df['bert_sentiment'] == 1].copy()
    df_neg = df[df['bert_sentiment'] == 0].copy()

    if not df_pos.empty:
        _, topic_pos_df = predict_topic(df_pos, sentiment="positive")
        topic_pos_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "topic_positive.csv"), index=False)
    if not df_neg.empty:
        _, topic_neg_df = predict_topic(df_neg, sentiment="negative")
        topic_neg_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "topic_negative.csv"), index=False)

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
            logging.error(f"Processing failed: {e}")
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file type"}), 400

app = Dash(__name__, server=server, url_base_pathname="/dashboard/")

def load_sentiment_data():
    path = os.path.join(PROCESSED_DATA_DIR, "sentiment.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def load_topic_data(sentiment="positive"):
    filename = "topic_positive.csv" if sentiment == "positive" else "topic_negative.csv"
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def empty_figure(title="No data available"):
    return {
        "data": [],
        "layout": {
            "title": {"text": title, "x": 0.5},
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{
                "text": title,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 20}
            }]
        }
    }

app.layout = html.Div([
    html.H1("Game Reviews NLP Dashboard (Uploaded Dataset)", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", value="sentiment", children=[
        dcc.Tab(label="Sentiment Analysis", value="sentiment"),
        dcc.Tab(label="Topic Modeling", value="topic")
    ]),
    html.Div(id="sentiment-dashboard", style={"display": "block"}, children=[
        html.H2("Sentiment Summary"),
        dcc.Graph(id="sentiment-bar-chart"),
        dcc.Interval(id="interval-sentiment", interval=60000, n_intervals=0)
    ]),
    html.Div(id="topic-dashboard", children=[
        html.H2("Topic Analysis"),
        html.Div([
            html.Div([
                html.Label("Select Sentiment:"),
                dcc.Dropdown(
                    id="sentiment-filter",
                    options=[
                        {"label": "Positive", "value": "positive"},
                        {"label": "Negative", "value": "negative"}
                    ],
                    value="positive"
                ),
            ], style={"width": "50%", "paddingRight": "10px"}),
            html.Div([
                html.Label("Select Topic:"),
                dcc.Dropdown(id="topic-dropdown"),
            ], style={"width": "50%", "paddingLeft": "10px"})
        ], style={"display": "flex", "gap": "10px"}),
        dcc.Graph(id="topic-bar-chart"),
        dcc.Interval(id="interval-topic-component", interval=60000, n_intervals=0),
    ])
])

@app.callback(
    Output("sentiment-dashboard", "style"),
    Output("topic-dashboard", "style"),
    Input("tabs", "value")
)
def switch_tab(tab):
    return ({"display": "block"}, {"display": "none"}) if tab == "sentiment" else ({"display": "none"}, {"display": "block"})

@app.callback(
    Output("sentiment-bar-chart", "figure"),
    Input("interval-sentiment", "n_intervals")
)
def update_sentiment(_):
    df = load_sentiment_data()
    if df.empty:
        return empty_figure("No sentiment data found")
    df['bert_sentiment'] = df['bert_sentiment'].map({0: "Negative", 1: "Positive"})
    summary = df.groupby("bert_sentiment").size().reset_index(name="count")
    return px.bar(summary, x="bert_sentiment", y="count", color="bert_sentiment", title="BERT Sentiment Summary")

@app.callback(
    Output("topic-dropdown", "options"),
    Output("topic-dropdown", "value"),
    Input("sentiment-filter", "value"),
    Input("interval-topic-component", "n_intervals")
)
def update_topic_dropdown(sentiment, _):
    df = load_topic_data(sentiment)
    if df.empty or 'topic' not in df.columns:
        return [], None
    topics = sorted(df['topic'].dropna().unique())
    return [{"label": f"Topic {t}", "value": t} for t in topics], topics[0] if topics else None

@app.callback(
    Output("topic-bar-chart", "figure"),
    Input("sentiment-filter", "value"),
    Input("topic-dropdown", "value"),
    Input("interval-topic-component", "n_intervals")
)
def update_topic_chart(sentiment, topic, _):
    df = load_topic_data(sentiment)
    if df.empty or topic is None:
        return empty_figure("No topic data available")
    df = df[df['topic'] == topic].sort_values(by="weightage", ascending=False)
    return px.bar(df, x="term", y="weightage", color="term", title=f"Topic {topic} - {sentiment.capitalize()} Reviews")

@server.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "review_text" not in data:
            logging.error("No review_text provided in the request.")
            return jsonify({"error": "No review_text provided"}), 400

        text = data["review_text"]
        if not text.strip():
            return jsonify({"error": "Empty review text provided"}), 400

        sentiment = predict_sentiment(text)
        logging.info(f"Predicted sentiment: {sentiment}")

        topic = predict_topic_single(text, sentiment.lower())
        logging.info(f"Predicted topic: {topic}")

        if topic == "Unknown":
            message = f"Sentiment is **{sentiment.upper()}**, but topic prediction failed (maybe missing uploaded CSV)."
        else:
            message = f"This review is predicted to have **{sentiment.upper()}** sentiment and belongs to **Topic {topic}**."

        return jsonify({
            "sentiment": sentiment,
            "topic": int(topic) if isinstance(topic, (np.integer, np.int64, np.int32)) else topic,
            "message": message
        })

    except Exception as e:
        logging.exception("Prediction error:")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    server.run(debug=True, port=8052)
