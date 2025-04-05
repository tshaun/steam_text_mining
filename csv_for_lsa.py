import pandas as pd
import joblib
import numpy as np

# Load the saved LSA model and vectorizer
lsa_model = joblib.load('lsa_model.pkl')
tfidf_vectorizer = joblib.load('lsa_vectorizer.pkl')

# Get the terms from the TF-IDF vectorizer
terms = tfidf_vectorizer.get_feature_names_out()

# Extract the topics and their respective terms with weights
topic_data = []

for i, topic in enumerate(lsa_model.components_):
    # Get the top 10 terms for each topic
    topic_terms = [terms[j] for j in topic.argsort()[:-11:-1]]  # Top 10 terms for each topic
    topic_weights = [topic[j] for j in topic.argsort()[:-11:-1]]  # Corresponding weights for the top terms
    
    # Apply log transformation to weights (adding a small constant to avoid log(0))
    topic_weights_log = [np.log(weight + 1e-10) for weight in topic_weights]  # Log transform with small constant

    for term, weight in zip(topic_terms, topic_weights_log):
        topic_data.append({
            "topic": i + 1,
            "term": term,
            "weightage": weight
        })

# Create a DataFrame for the topics and save it as a CSV file
topic_df = pd.DataFrame(topic_data)
topic_df.to_csv('topic.csv', index=False)

print("Topic model output with log-transformed weights saved to 'topic.csv'.")
