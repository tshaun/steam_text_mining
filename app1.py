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

# NLTK setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Flask app
server = Flask(__name__)

# Directory for processed data (original dataset)
PROCESSED_DATA_DIR = "processed_data1"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("./saved_model")
bert_model = BertForSequenceClassification.from_pretrained("./saved_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# Load or initialize LSA vectorizer & model
lsa_vectorizer = joblib.load(os.path.join("topic modelling", "lsa_vectorizer.pkl"))
lsa_model = joblib.load(os.path.join("topic modelling", "lsa_model.pkl"))

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
def predict_topic(text):
    cleaned_text = preprocess_text(text)
    tfidf_transformed = lsa_vectorizer.transform([cleaned_text])
    
    topic_vector = lsa_model.transform(tfidf_transformed).tolist()[0]
    
    most_relevant_topic = np.argmax(topic_vector)
    
    return int(most_relevant_topic + 1)

# Dash app for visualization (integrated with Flask server)
dash_app = Dash(__name__, server=server, url_base_pathname="/dashboard/")

# Load data for visualization from processed_data1 directory
def load_data():
    sentiment_path = os.path.join(PROCESSED_DATA_DIR, "sentiment.csv")
    topic_path = os.path.join(PROCESSED_DATA_DIR, "topic.csv")

    sentiment_df = pd.read_csv(sentiment_path) if os.path.exists(sentiment_path) else pd.DataFrame()
    topic_df = pd.read_csv(topic_path) if os.path.exists(topic_path) else pd.DataFrame()

    return sentiment_df, topic_df

dash_app.layout = html.Div([ 
    html.H1("Game Reviews NLP Dashboard (Original Dataset)", style={"textAlign": "center"}),

    html.Div([
        dcc.Tabs([
            dcc.Tab(label="Sentiment Analysis", value="sentiment"),
            dcc.Tab(label="Topic Modeling", value="topic")
        ], id="tabs", value="sentiment")
    ]),

    html.Div(id="sentiment-dashboard", children=[
        html.H2("Sentiment Summary"),
        dcc.Graph(id="sentiment-bar-chart-bert"),
        dcc.Graph(id="sentiment-bar-chart-vader"),
        dcc.Interval(id="interval-component", interval=60 * 1000, n_intervals=0),
    ]),

    html.Div(id="topic-dashboard", children=[
        html.H2("Topic Analysis"),
        dcc.Dropdown(id="topic-dropdown", options=[], value=None),
        dcc.Graph(id="topic-bar-chart"),
        dcc.Interval(id="interval-topic-component", interval=60 * 1000, n_intervals=0),
    ])
])

# Tab switch callback
@dash_app.callback(
    Output("sentiment-dashboard", "style"),
    Output("topic-dashboard", "style"),
    Input("tabs", "value")
)
def switch_tab(tab_value):
    if tab_value == "sentiment":
        return {"display": "block"}, {"display": "none"}
    elif tab_value == "topic":
        return {"display": "none"}, {"display": "block"}

# Sentiment dashboard update
@dash_app.callback(
    Output("sentiment-bar-chart-bert", "figure"),
    Output("sentiment-bar-chart-vader", "figure"),
    Input("interval-component", "n_intervals")
)
def update_sentiment_dashboard(n_intervals):
    sentiment_df, _ = load_data()

    if sentiment_df.empty:
        return {}, {}

    # BERT sentiment manipulation
    sentiment_bert_summary = sentiment_df.groupby("bert_sentiment").size().reset_index(name="count")
    sentiment_bert_summary["bert_sentiment"] = sentiment_bert_summary["bert_sentiment"].map({0: "Negative", 1: "Positive"})

    sentiment_bert_bar = px.bar(
        sentiment_bert_summary,
        x="bert_sentiment",
        y="count",
        color="bert_sentiment",
        title="BERT Sentiment Analysis"
    )

    # VADER sentiment manipulation
    sentiment_df['vader_sentiment'] = sentiment_df['vader_sentiment'].apply(lambda x: "Positive" if x >= 0 else "Negative")
    sentiment_vader_summary = sentiment_df.groupby("vader_sentiment").size().reset_index(name="count")

    sentiment_vader_bar = px.bar(
        sentiment_vader_summary,
        x="vader_sentiment",
        y="count",
        color="vader_sentiment",
        title="VADER Sentiment Analysis"
    )

    return sentiment_bert_bar, sentiment_vader_bar

# Topic dashboard update and dropdown options
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
    topic_df = topic_df[topic_df['weightage'].apply(lambda x: isinstance(x, (int, float)))]  # Ensure valid numeric weights

    if topic_df.empty:
        return [], {}

    topics = topic_df['topic'].unique()
    topic_options = [{'label': f"Topic {i+1}", 'value': i+1} for i in range(len(topics))]

    if selected_topic is None:
        selected_topic = topic_options[0]['value']

    topic_filtered = topic_df[topic_df['topic'] == selected_topic]

    topic_filtered['weightage'] = topic_filtered['weightage'] * -1

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

# Predict sentiment and topic using uploaded text
@server.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("review_text")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        sentiment = predict_sentiment(text)
        topic = predict_topic(text)

        return jsonify({"sentiment": sentiment, "topic": topic})

    except Exception as e:
        return jsonify({"error": f"Error processing the request: {str(e)}"}), 500

if __name__ == "__main__":
    server.run(debug=True, port=8051)
