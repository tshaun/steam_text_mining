from flask import Flask, jsonify, request
import pandas as pd
import nltk
import re
import gensim
from gensim import corpora
from gensim.models import LdaModel
from flask_cors import CORS

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

stop_list = nltk.corpus.stopwords.words('english')
stop_list += ["game", "play", "like", "good", "best", "one", "great", "really", "get", "time"]

def preprocess_reviews(df, text_column='review_text'):
    df = df.dropna(subset=[text_column])
    docs = df[text_column].astype(str).apply(nltk.word_tokenize).tolist()
    docs = [[w.lower() for w in doc if w.isalpha() and w not in stop_list] for doc in docs]
    return docs

def train_lda(df, num_topics=5):
    docs = preprocess_reviews(df)
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, random_state=42)
    
    topics = []
    for idx, topic in lda_model.show_topics(num_words=5, formatted=False):
        topics.append({"topic": f"Topic {idx}", "words": [word for word, _ in topic]})
    
    return topics

@app.route('/topics', methods=['POST'])
def get_topics():
    file = request.files['file']
    df = pd.read_csv(file)
    topics = train_lda(df)
    return jsonify(topics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
