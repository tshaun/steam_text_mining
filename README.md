# Game Reviews NLP Dashboard

This project is a full-stack NLP dashboard that analyzes game reviews using BERT-based sentiment analysis and LSA-based topic modeling. It includes two Flask apps and a React frontend.

## Project Structure

```
├── app1.py                # Flask app for original dataset (static analysis)
├── app2.py                # Flask app for uploaded dataset (dynamic analysis)
├── saved_model/           # Directory containing fine-tuned BERT model
├── topic modelling/       # Directory containing fine-tuned LSA model
├── processed_data1/       # Output folder for app1.py
├── processed_data2/       # Output folder for app2.py
├── uploads/               # Temporary upload folder for CSVs (app2.py)
├── frontend/              # React frontend (runs on localhost:3000)
│   └── App.js             # React app entry point
```

## Features

- **Sentiment Analysis**: Uses BERT to classify reviews as Positive or Negative.
- **Topic Modeling**: Applies LSA (Latent Semantic Analysis) to extract top terms per topic.
- **Two Dashboards**:
  - `app1.py`: Preloads and analyzes a default dataset.
  - `app2.py`: Accepts uploaded CSVs and dynamically updates dashboard.
- **Live Text Prediction**: Both apps support real-time prediction of sentiment and topic from individual review text via the `/predict` endpoint.
- **React Frontend**: Sends review text to Flask backend for live prediction.

## Requirements

Install required Python packages:

```bash
pip install -r requirements.txt
```

Install NLTK dependencies:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

In the `frontend/` directory, install dependencies and start the React app:

```bash
cd frontend
npm install
npm start
```

## Usage

### 1. Run Flask App with Preloaded Dataset (app1.py)

```bash
python app1.py
```

- Dashboard: http://127.0.0.1:8051/dashboard
- API Predict Endpoint: http://127.0.0.1:8051/predict

### 2. Run Flask App with File Uploads (app2.py)

```bash
python app2.py
```

- Dashboard: http://127.0.0.1:8052/dashboard
- API Predict Endpoint: http://127.0.0.1:8052/predict
- Upload Endpoint: http://127.0.0.1:8052/upload (accepts CSVs with `review_text` column)

### 3. Run the React App (frontend)

```bash
cd frontend
npm start
```

- App: http://localhost:3000

The React app allows users to input a review and receive real-time sentiment and topic predictions.

## Notes

- Uploaded CSV files must contain a `review_text` column.
- Uploaded data is cleaned up on server exit.
- Predictions are powered by a local fine-tuned BERT model.

## Model Details

- **Sentiment Model**: `bert-base-uncased` fine-tuned for binary classification.
- **Topic Model**: TF-IDF + Truncated SVD (LSA) on preprocessed text.

## API Endpoints

### `POST /upload` (app2.py only)
Upload a CSV file with a `review_text` column.

### `POST /predict`
Predict sentiment and topic for a given review text:

```json
{
  "review_text": "This game was surprisingly fun and challenging."
}
```

### Response
```json
{
  "sentiment": "Positive",
  "topic": 3
}
```

