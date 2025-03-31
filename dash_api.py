import os
import pandas as pd
import nltk
import re
import gensim
import pyLDAvis
import pyLDAvis.gensim
import plotly.express as px
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from textblob import TextBlob
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import dash
from dash import dcc, html, Input, Output, dash_table

nltk.download('stopwords')
nltk.download('punkt')

# Setup Flask app
app = Flask(__name__)
CORS(app)

# Constants
DATA_FOLDER = "processed_data"
# DEFAULT_FILE = "first_100_reviews.csv"
DEFAULT_FILE = "steam_reviews.csv"
SENTIMENT_CSV = os.path.join(DATA_FOLDER, "sentiment_results.csv")
LDA_TOPICS_CSV = os.path.join(DATA_FOLDER, "lda_topics.csv")
LDA_VIS_HTML = os.path.join(DATA_FOLDER, "lda_visualization.html")

os.makedirs(DATA_FOLDER, exist_ok=True)  # Ensure data folder exists

# Stopword list
stop_list = nltk.corpus.stopwords.words('english') + [
    "game", "play", "like", "good", "best", "one", "great", "really", "get", "time"
]

# Preprocess function
def preprocess_reviews(df):
    df = df.dropna(subset=['review_text'])
    df['tokens'] = df['review_text'].apply(nltk.word_tokenize)
    df['tokens'] = df['tokens'].apply(lambda x: [word.lower() for word in x if word.isalpha() and word.lower() not in stop_list])
    return df

# Sentiment analysis
def perform_sentiment_analysis(df):
    df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['sentiment_category'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
    df.to_csv(SENTIMENT_CSV, index=False)

# Topic Modeling with LDA
def perform_lda(df):
    docs = preprocess_reviews(df)['tokens']
    dictionary = corpora.Dictionary(docs)
    corpus_bow = [dictionary.doc2bow(doc) for doc in docs]
    lda_model = LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=10, passes=10, random_state=42)
    
    topic_words = []
    for idx, topic in lda_model.show_topics(num_topics=10, num_words=10, formatted=False):
        for word, weight in topic:
            topic_words.append({"topic": idx, "word": word, "weight": weight})
    
    pd.DataFrame(topic_words).to_csv(LDA_TOPICS_CSV, index=False)
    
    vis = pyLDAvis.gensim.prepare(lda_model, corpus_bow, dictionary)
    pyLDAvis.save_html(vis, LDA_VIS_HTML)

# Load default data
def load_default_data():
    if not os.path.exists(DEFAULT_FILE):
        print(f"Error: Default file {DEFAULT_FILE} not found.")
        return False
    
    print(f"Loading default dataset from {DEFAULT_FILE}...")
    df = pd.read_csv(DEFAULT_FILE)
    df = preprocess_reviews(df)
    perform_sentiment_analysis(df)
    perform_lda(df)
    print("Default dataset processed successfully.")
    return True

# Process new file upload
@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = os.path.join(DATA_FOLDER, "uploaded.csv")
    file.save(file_path)
    
    df = pd.read_csv(file_path)
    df = preprocess_reviews(df)
    perform_sentiment_analysis(df)
    perform_lda(df)
    
    return jsonify({"message": f"Processing complete. Sentiment analysis and topic modeling results saved to {SENTIMENT_CSV} and {LDA_TOPICS_CSV} respectively."})

# Route for serving LDA visualization HTML
@app.route('/lda_visualization', methods=['GET'])
def get_lda_visualization():
    return send_file(LDA_VIS_HTML)

# Create Dash app
dash_app = dash.Dash(__name__, server=app, routes_pathname_prefix='/dashboard/')

# Define Dash layout
dash_app.layout = html.Div([
    html.H1("Review Analysis Dashboard"),
    
    html.Div([
        html.H2("Sentiment Analysis"),
        dcc.Graph(id='sentiment-graph'),
        
        html.H2("Topic Modeling"),
        dash_table.DataTable(
            id='topic-table',
            columns=[
                {"name": "Topic", "id": "topic"},
                {"name": "Word", "id": "word"},
                {"name": "Weight", "id": "weight"}
            ],
            page_size=10,
            style_table={'overflowX': 'auto'}
        ),
        
        html.Div([
            html.H3("LDA Visualization"),
            html.Iframe(id='lda-iframe', src="/lda_visualization", style={"width": "100%", "height": "800px"})
        ])
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds, refresh every 5 seconds
        n_intervals=0
    )
])

# Callback to update sentiment graph
@dash_app.callback(
    Output('sentiment-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_sentiment_graph(n):
    df = pd.read_csv(SENTIMENT_CSV)
    sentiment_counts = df['sentiment_category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig = px.bar(sentiment_counts, x='Sentiment', y='Count', 
                 title='Sentiment Distribution',
                 color='Sentiment',
                 color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
    return fig

# Callback to update topic table
@dash_app.callback(
    Output('topic-table', 'data'),
    [Input('interval-component', 'n_intervals')]
)
def update_topic_table(n):
    df = pd.read_csv(LDA_TOPICS_CSV)
    return df.to_dict('records')

# Endpoint for sentiment prediction using the trained model
@app.route('/predict', methods=['POST'])
def predict():
    # Receive review text for prediction
    data = request.get_json()
    review_text = data.get('review_text')
    
    if not review_text:
        return jsonify({'error': 'No review text provided'}), 400
    
    # Preprocess review text
    cleaned_review = preprocess_reviews(pd.DataFrame({'review_text': [review_text]}))['review_text'].iloc[0]
    
    # Perform sentiment analysis directly
    sentiment = TextBlob(str(review_text)).sentiment.polarity
    prediction = 1 if sentiment > 0 else -1
    
    return jsonify({
        'prediction': prediction
    })

# Run the API
if __name__ == "__main__":
    # Ensure default data is loaded if needed
    if not os.path.exists(SENTIMENT_CSV) or not os.path.exists(LDA_TOPICS_CSV):
        success = load_default_data()
        if not success:
            print("Failed to load default data. Please ensure the default dataset exists.")
            exit(1)
    else:
        print("Using existing processed data files.")
    
    # Run the Flask app on port 8000
    app.run(debug=True, port=8000)