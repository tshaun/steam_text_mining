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

# NLTK setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Flask app
server = Flask(__name__)

PROCESSED_DATA_DIR = "processed_data2"
UPLOAD_FOLDER = "uploads"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"csv"}

# Global vectorizer and SVD models
global_vectorizer = None
global_svd = None

# Clean up files on exit
def cleanup_folders():
    for folder in [UPLOAD_FOLDER, PROCESSED_DATA_DIR]:
        files = glob.glob(os.path.join(folder, '*'))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

atexit.register(cleanup_folders)

# Load BERT model
bert_tokenizer = BertTokenizer.from_pretrained("./saved_model")
bert_model = BertForSequenceClassification.from_pretrained("./saved_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# BERT Sentiment
def predict_sentiment(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "Positive" if prediction == 1 else "Negative"

# Topic modeling using dynamic LSA
def predict_topic(df, n_topics=5):
    global global_vectorizer, global_svd

    df['cleaned_text'] = df['review_text'].apply(preprocess_text)

    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])

    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    topic_vectors = svd.fit_transform(tfidf_matrix)

    df['predicted_topic'] = np.argmax(topic_vectors, axis=1) + 1

    global_vectorizer = vectorizer
    global_svd = svd

    terms = vectorizer.get_feature_names_out()
    topic_data = []
    for topic_idx, component in enumerate(svd.components_):
        top_terms = sorted(zip(terms, component), key=lambda x: x[1], reverse=True)[:10]
        for term, weight in top_terms:
            weightage = float(-1 * np.log(weight + 1e-10)) if weight > 0 else 0.0
            topic_data.append({
                "topic": topic_idx + 1,
                "term": term,
                "weightage": weightage
            })

    topic_df = pd.DataFrame(topic_data)
    topic_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "topic.csv"), index=False)
    return df

# Process uploaded file
def process_uploaded_file(csv_path):
    df = pd.read_csv(csv_path)
    if "review_text" not in df.columns:
        raise ValueError("CSV must contain 'review_text' column.")
    df['review_text'] = df['review_text'].fillna("").astype(str)
    df["bert_sentiment"] = df["review_text"].apply(lambda x: 1 if predict_sentiment(x) == "Positive" else 0)
    df = predict_topic(df)
    sentiment_df = df[["review_text", "bert_sentiment"]]
    sentiment_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "sentiment.csv"), index=False)

# File type check
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

@server.route("/predict", methods=["POST"])
def predict():
    global global_vectorizer, global_svd
    data = request.get_json()
    if not data or "review_text" not in data:
        return jsonify({"error": "Missing review_text"}), 400

    review = data["review_text"]
    cleaned = preprocess_text(review)

    try:
        sentiment = predict_sentiment(review)
    except Exception as e:
        return jsonify({"error": f"BERT sentiment prediction failed: {str(e)}"}), 500

    if global_vectorizer is None or global_svd is None:
        return jsonify({"error": "Topic model not trained. Please upload a CSV first."}), 400

    try:
        tfidf_vector = global_vectorizer.transform([cleaned])
        topic_vector = global_svd.transform(tfidf_vector)
        predicted_topic = int(np.argmax(topic_vector) + 1)
    except Exception as e:
        return jsonify({"error": f"Topic prediction failed: {str(e)}"}), 500

    return jsonify({
        "sentiment": sentiment,
        "topic": predicted_topic
    })

# Dash app
dash_app = Dash(__name__, server=server, url_base_pathname="/dashboard/")

def load_data():
    sentiment_path = os.path.join(PROCESSED_DATA_DIR, "sentiment.csv")
    topic_path = os.path.join(PROCESSED_DATA_DIR, "topic.csv")
    sentiment_df = pd.read_csv(sentiment_path) if os.path.exists(sentiment_path) else pd.DataFrame()
    topic_df = pd.read_csv(topic_path) if os.path.exists(topic_path) else pd.DataFrame()
    return sentiment_df, topic_df

dash_app.layout = html.Div([
    html.H1("Game Reviews NLP Dashboard (Uploaded Dataset)", style={"textAlign": "center"}),
    dcc.Tabs([
        dcc.Tab(label="Sentiment Analysis", value="sentiment"),
        dcc.Tab(label="Topic Modeling", value="topic")
    ], id="tabs", value="sentiment"),

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
    return ({"display": "block"}, {"display": "none"}) if tab_value == "sentiment" else ({"display": "none"}, {"display": "block"})

@dash_app.callback(
    Output("sentiment-bar-chart-bert", "figure"),
    Input("interval-component", "n_intervals")
)
def update_sentiment_dashboard(n):
    sentiment_df, _ = load_data()
    if sentiment_df.empty:
        return {}

    sentiment_summary = sentiment_df.groupby("bert_sentiment").size().reset_index(name="count")
    sentiment_summary["bert_sentiment"] = sentiment_summary["bert_sentiment"].map({0: "Negative", 1: "Positive"})

    return px.bar(sentiment_summary, x="bert_sentiment", y="count", color="bert_sentiment", title="BERT Sentiment Analysis")

@dash_app.callback(
    [Output("topic-dropdown", "options"),
     Output("topic-bar-chart", "figure")],
    Input("interval-topic-component", "n_intervals"),
    Input("topic-dropdown", "value")
)
def update_topic_dashboard(n, selected_topic):
    _, topic_df = load_data()
    if topic_df.empty or 'term' not in topic_df or 'weightage' not in topic_df:
        return [], {}

    topic_df = topic_df.dropna(subset=["weightage"])
    topics = topic_df["topic"].unique()
    topic_options = [{'label': f"Topic {i+1}", 'value': i+1} for i in range(len(topics))]

    if not selected_topic and topic_options:
        selected_topic = topic_options[0]["value"]

    filtered = topic_df[topic_df["topic"] == selected_topic]
    filtered = filtered.sort_values(by="weightage", ascending=False)

    topic_bar = px.bar(
        filtered,
        x="term",
        y="weightage",
        color="term",
        title=f"Topic {selected_topic} Analysis",
        labels={"weightage": "Weightage (Log Transformed)"},
        hover_data={"term": True, "weightage": True}
    )

    topic_bar.update_layout(
        yaxis=dict(autorange=True),
        xaxis=dict(categoryorder="total descending"),
        bargap=0.2
    )

    return topic_options, topic_bar

if __name__ == "__main__":
    server.run(debug=False, port=8052, use_reloader=False)
