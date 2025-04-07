from flask import Flask, request, jsonify
import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import numpy as np

# NLTK setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Flask app
server = Flask(__name__)

# Directory for processed data
PROCESSED_DATA_DIR = "processed_data1"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("./saved_model")
bert_model = BertForSequenceClassification.from_pretrained("./saved_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# Load LSA vectorizer/model (for fallback in prediction logic)
lsa_vectorizer = joblib.load(os.path.join("topic modelling", "lsa_vectorizer.pkl"))
lsa_model = joblib.load(os.path.join("topic modelling", "lsa_model.pkl"))

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Sentiment prediction
def predict_sentiment(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "positive" if prediction == 1 else "negative"

def predict_topic(text, sentiment):
    cleaned = preprocess_text(text)
    tfidf_vec = lsa_vectorizer.transform([cleaned])
    lsa_vec = lsa_model.transform(tfidf_vec)  # 1 x n_topics

    csv_path = os.path.join(PROCESSED_DATA_DIR, f"topic_{sentiment}.csv")
    if not os.path.exists(csv_path):
        return "Unknown"

    topic_df = pd.read_csv(csv_path)

    topic_groups = topic_df.groupby("topic")
    topic_vectors = []
    topic_ids = []

    for topic_id, group in topic_groups:
        terms = group["term"].tolist()
        topic_text = " ".join(terms)
        topic_cleaned = preprocess_text(topic_text)
        topic_tfidf = lsa_vectorizer.transform([topic_cleaned])
        topic_lsa_vec = lsa_model.transform(topic_tfidf)
        topic_vectors.append(topic_lsa_vec)
        topic_ids.append(topic_id)

    if not topic_vectors:
        return "Unknown"

    topic_vectors = np.vstack(topic_vectors)
    similarities = cosine_similarity(lsa_vec, topic_vectors)[0]
    best_match_index = np.argmax(similarities)
    predicted_topic = topic_ids[best_match_index]

    return predicted_topic

# Load sentiment and topic data
def load_sentiment_data():
    path = os.path.join(PROCESSED_DATA_DIR, "sentiment.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def load_topic_data():
    pos_path = os.path.join(PROCESSED_DATA_DIR, "topic_positive.csv")
    neg_path = os.path.join(PROCESSED_DATA_DIR, "topic_negative.csv")
    pos_df = pd.read_csv(pos_path) if os.path.exists(pos_path) else pd.DataFrame()
    neg_df = pd.read_csv(neg_path) if os.path.exists(neg_path) else pd.DataFrame()
    return pos_df, neg_df

# Dash app setup
dash_app = Dash(__name__, server=server, url_base_pathname="/dashboard/")

dash_app.layout = html.Div([
    html.H1("Game Reviews NLP Dashboard (Original Dataset)", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", value="sentiment", children=[
        dcc.Tab(label="Sentiment Analysis", value="sentiment"),
        dcc.Tab(label="Topic Modeling", value="topic")
    ]),

    html.Div(id="sentiment-dashboard", children=[
        html.H2("Sentiment Summary"),

        html.Div([
            dcc.Graph(id="sentiment-bar-chart-bert", style={"width": "50%"}),
            dcc.Graph(id="sentiment-bar-chart-vader", style={"width": "50%"})
        ], style={"display": "flex"}),

        dcc.Interval(id="interval-component", interval=60 * 1000, n_intervals=0),
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
        dcc.Interval(id="interval-topic-component", interval=60 * 1000, n_intervals=0),
    ])
])

# Tab switch logic
@dash_app.callback(
    Output("sentiment-dashboard", "style"),
    Output("topic-dashboard", "style"),
    Input("tabs", "value")
)
def switch_tab(tab_value):
    if tab_value == "sentiment":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}

# Sentiment dashboard
@dash_app.callback(
    Output("sentiment-bar-chart-bert", "figure"),
    Output("sentiment-bar-chart-vader", "figure"),
    Input("interval-component", "n_intervals")
)
def update_sentiment_dashboard(_):
    df = load_sentiment_data()
    if df.empty:
        return {}, {}

    bert_summary = df.groupby("bert_sentiment").size().reset_index(name="count")
    bert_summary["bert_sentiment"] = bert_summary["bert_sentiment"].map({0: "Negative", 1: "Positive"})
    bert_fig = px.bar(bert_summary, x="bert_sentiment", y="count", color="bert_sentiment", title="BERT Sentiment")

    df['vader_sentiment'] = df['vader_sentiment'].apply(lambda x: "Positive" if x >= 0 else "Negative")
    vader_summary = df.groupby("vader_sentiment").size().reset_index(name="count")
    vader_fig = px.bar(vader_summary, x="vader_sentiment", y="count", color="vader_sentiment", title="VADER Sentiment")

    return bert_fig, vader_fig

# Topic dropdown + chart
@dash_app.callback(
    Output("topic-dropdown", "options"),
    Output("topic-dropdown", "value"),
    Input("sentiment-filter", "value"),
    Input("interval-topic-component", "n_intervals")
)
def update_topic_dropdown(sentiment, _):
    pos_df, neg_df = load_topic_data()
    df = pos_df if sentiment == "positive" else neg_df
    if df.empty:
        return [], None
    topics = sorted(df['topic'].unique())
    options = [{"label": f"Topic {t}", "value": t} for t in topics]
    return options, topics[0]

@dash_app.callback(
    Output("topic-bar-chart", "figure"),
    Input("sentiment-filter", "value"),
    Input("topic-dropdown", "value"),
    Input("interval-topic-component", "n_intervals")
)
def update_topic_chart(sentiment, topic, _):
    pos_df, neg_df = load_topic_data()
    df = pos_df if sentiment == "positive" else neg_df
    if df.empty or topic is None:
        return {}
    df = df[df['topic'] == topic]
    df['weightage'] = df['weightage'] * -1
    df = df.sort_values(by="weightage", ascending=False)
    fig = px.bar(
        df, x="term", y="weightage", color="term",
        title=f"Topic {topic} from {sentiment.capitalize()} Reviews",
        labels={"weightage": "Weight (Log Transformed)"}
    )
    return fig

# Prediction endpoint
@server.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("review_text", "")
        if not text:
            return jsonify({"error": "No review_text provided"}), 400

        sentiment = predict_sentiment(text)
        topic = predict_topic(text, sentiment)

        message = f"This review is predicted to have **{sentiment.upper()}** sentiment and belongs to **Topic {topic}**."
        return jsonify({
            "sentiment": sentiment,
            "topic": topic,
            "message": message
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    server.run(debug=True, port=8051)
