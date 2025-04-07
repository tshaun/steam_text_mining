import pandas as pd
import joblib
import numpy as np
import os

# Create a function to load model and generate topic CSV
def generate_topic_csv(label, output_filename):
    lsa_model = joblib.load(f'lsa_model_{label}.pkl')
    tfidf_vectorizer = joblib.load(f'lsa_vectorizer_{label}.pkl')

    terms = tfidf_vectorizer.get_feature_names_out()
    topic_data = []

    for i, topic in enumerate(lsa_model.components_):
        topic_terms = [terms[j] for j in topic.argsort()[:-11:-1]]
        topic_weights = [topic[j] for j in topic.argsort()[:-11:-1]]
        topic_weights_log = [np.log(weight + 1e-10) for weight in topic_weights]

        for term, weight in zip(topic_terms, topic_weights_log):
            topic_data.append({
                "topic": i + 1,
                "term": term,
                "weightage": weight
            })

    topic_df = pd.DataFrame(topic_data)
    topic_df.to_csv(output_filename, index=False)
    print(f"Saved {output_filename}")

# Generate both topic CSVs
generate_topic_csv('positive', 'topic_positive.csv')
generate_topic_csv('negative', 'topic_negative.csv')
