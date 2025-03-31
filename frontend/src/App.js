import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
    const [file, setFile] = useState(null);
    const [reviewText, setReviewText] = useState("");
    const [trainMessage, setTrainMessage] = useState("");
    const [predictMessage, setPredictMessage] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [dashboardUrl, setDashboardUrl] = useState("http://127.0.0.1:8000/dashboard/");

    useEffect(() => {
        // Check if dashboard is available when component mounts
        fetch("http://127.0.0.1:8000/dashboard/")
            .then(response => {
                if (response.ok) {
                    setTrainMessage("Dashboard loaded successfully with default data.");
                }
            })
            .catch(error => {
                setTrainMessage("Dashboard not available. Please ensure the server is running.");
            });
    }, []);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleReviewTextChange = (event) => {
        setReviewText(event.target.value);
    };

    const handleUpload = async () => {
        if (!file) {
            setTrainMessage("Please select a file to upload.");
            return;
        }
        setIsLoading(true);
        setTrainMessage("Processing file...");
        const formData = new FormData();
        formData.append("file", file);
        try {
            const response = await fetch("http://127.0.0.1:8000/process", {
                method: "POST",
                body: formData,
            });
            const result = await response.json();
            setTrainMessage(result.message);
            // Refresh the dashboard iframe to show new data
            setDashboardUrl("http://127.0.0.1:8000/dashboard/?t=" + new Date().getTime());
        } catch (error) {
            setTrainMessage("Error uploading file: " + error.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handlePredict = async () => {
        if (!reviewText) {
            setPredictMessage("Please enter review text.");
            return;
        }
        setIsLoading(true);
        setPredictMessage("Predicting sentiment...");
        try {
            const response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ review_text: reviewText })
            });
            const result = await response.json();
            if (result.error) {
                setPredictMessage(result.error);
            } else {
                let sentiment = result.prediction === 1 ? "Positive" : "Negative";
                setPredictMessage(`Predicted Sentiment: ${sentiment}`);
            }
        } catch (error) {
            setPredictMessage("Error predicting sentiment: " + error.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="app-container">
            <div className="form-container">
                <h1 className="text-center text-3xl font-bold">Game Reviews NLP Dashboard</h1>

                {/* Form to upload CSV file for training */}
                <div className="form-group">
                    <h2>Train Model with CSV File</h2>
                    <form>
                        <label>Upload your CSV file:</label>
                        <input type="file" onChange={handleFileChange} accept=".csv" />
                        <button type="button" onClick={handleUpload}>Train Model</button>
                    </form>
                    <div className="result">{trainMessage}</div>
                </div>

                {/* Form to predict sentiment */}
                <div className="form-group">
                    <h2>Predict Sentiment for Review Text</h2>
                    <form>
                        <label>Enter your review text:</label>
                        <input type="text" value={reviewText} onChange={handleReviewTextChange} />
                        <button type="button" onClick={handlePredict}>Predict Sentiment</button>
                    </form>
                    <div className="result">{predictMessage}</div>
                </div>
            </div>

            {/* Dashboard iframe */}
            <div className="dashboard-container">
                <iframe src={dashboardUrl} title="Dashboard" />
            </div>
        </div>
    );
}

export default App;
