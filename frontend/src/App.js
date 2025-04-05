import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
    const [file, setFile] = useState(null);
    const [reviewText, setReviewText] = useState("");
    const [trainMessage, setTrainMessage] = useState("");
    const [predictMessage, setPredictMessage] = useState("");
    const [topicMessage, setTopicMessage] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [dashboardUrl, setDashboardUrl] = useState("http://127.0.0.1:8051/dashboard/");
    const [activeTab, setActiveTab] = useState("train");
    const [selectedApp, setSelectedApp] = useState("app1");

    useEffect(() => {
        // Fetch the dashboard URL based on selectedApp
        const baseUrl = selectedApp === "app1" ? "http://127.0.0.1:8051" : "http://127.0.0.1:8052";
        fetch(`${baseUrl}/dashboard/`)
            .then(response => {
                if (response.ok) {
                    setTrainMessage("Dashboard loaded successfully with default data.");
                    setDashboardUrl(`${baseUrl}/dashboard/?t=${new Date().getTime()}`);
                }
            })
            .catch(error => {
                setTrainMessage("Dashboard not available. Please ensure the server is running.");
            });
    }, [selectedApp]);

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
        const baseUrl = selectedApp === "app1" ? "http://127.0.0.1:8051" : "http://127.0.0.1:8052";
        try {
            const response = await fetch(`${baseUrl}/upload`, {
                method: "POST",
                body: formData,
            });
            const result = await response.json();
            setTrainMessage(result.message);
            setDashboardUrl(`${baseUrl}/dashboard/?t=${new Date().getTime()}`);
        } catch (error) {
            setTrainMessage("Error uploading file: " + error.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handlePredictSentiment = async () => {
        if (!reviewText) {
            setPredictMessage("Please enter review text.");
            return;
        }
        setIsLoading(true);
        setPredictMessage("Predicting sentiment...");
        const baseUrl = selectedApp === "app1" ? "http://127.0.0.1:8051" : "http://127.0.0.1:8052";
        try {
            const response = await fetch(`${baseUrl}/predict`, {
                method: "POST",
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ review_text: reviewText }),
            });
            const result = await response.json();
            if (result.error) {
                setPredictMessage(result.error);
            } else {
                let sentiment = result.sentiment;
                setPredictMessage(`Predicted Sentiment: ${sentiment}`);
            }
        } catch (error) {
            setPredictMessage("Error predicting sentiment: " + error.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handlePredictTopic = async () => {
        if (!reviewText) {
            setTopicMessage("Please enter review text.");
            return;
        }
        setIsLoading(true);
        setTopicMessage("Predicting topic...");
        const baseUrl = selectedApp === "app1" ? "http://127.0.0.1:8051" : "http://127.0.0.1:8052";
        try {
            const response = await fetch(`${baseUrl}/predict`, {
                method: "POST",
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ review_text: reviewText }),
            });
            const result = await response.json();
            if (result.error) {
                setTopicMessage(result.error);
            } else {
                let topicVector = result.topic;
                let topicInfo = `Predicted Topic Vector: ${JSON.stringify(topicVector)}`;
                setTopicMessage(topicInfo);
            }
        } catch (error) {
            setTopicMessage("Error predicting topic: " + error.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="page-container">
            <div className="app-container">
                <div className="form-container">
                    <h1>Game Reviews NLP Dashboard</h1>

                    <div className="tab-navigation">
                        <button
                            className={`tab-button ${activeTab === "train" ? "active" : ""}`}
                            onClick={() => setActiveTab("train")}
                        >
                            Train Model
                        </button>
                        <button
                            className={`tab-button ${activeTab === "predict" ? "active" : ""}`}
                            onClick={() => setActiveTab("predict")}
                        >
                            Predict Sentiment & Topic
                        </button>
                    </div>

                    {/* App selection dropdown */}
                    <div className="app-selection">
                        <label>Select Backend App: </label>
                        <select value={selectedApp} onChange={(e) => setSelectedApp(e.target.value)}>
                            <option value="app1">App 1 (Port 8051)</option>
                            <option value="app2">App 2 (Port 8052)</option>
                        </select>
                    </div>

                    {activeTab === "train" && (
                        <div className="form-group">
                            <h2>Upload CSV File</h2>
                            <form>
                                <label>Upload your CSV file:</label>
                                <input type="file" onChange={handleFileChange} accept=".csv" />
                                <button type="button" onClick={handleUpload} disabled={isLoading}>
                                    {isLoading ? "Processing..." : "Upload File"}
                                </button>
                            </form>
                            <div className="result">{trainMessage}</div>
                        </div>
                    )}

                    {activeTab === "predict" && (
                        <div className="form-group">
                            <h2>Predict Sentiment & Topic for Review Text</h2>
                            <form>
                                <label>Enter your review text:</label>
                                <input type="text" value={reviewText} onChange={handleReviewTextChange} />
                                <button
                                    style={{ backgroundColor: "#34a853" }}
                                    type="button"
                                    onClick={handlePredictSentiment}
                                    disabled={isLoading}
                                >
                                    {isLoading ? "Processing..." : "Predict Sentiment"}
                                </button>
                                <button
                                    style={{ backgroundColor: "#4285f4", marginTop: "10px" }}
                                    type="button"
                                    onClick={handlePredictTopic}
                                    disabled={isLoading}
                                >
                                    {isLoading ? "Processing..." : "Predict Topic"}
                                </button>
                            </form>
                            <div className="result">{predictMessage}</div>
                            <div className="result">{topicMessage}</div>
                        </div>
                    )}
                </div>

                <div className="dashboard-container">
                    <iframe src={dashboardUrl} title="Dashboard" />
                </div>
            </div>

            <footer className="footer">
                <p>&copy; {new Date().getFullYear()} Game Reviews NLP Dashboard. All rights reserved.</p>
            </footer>
        </div>
    );
}

export default App;
