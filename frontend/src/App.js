import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [dashboardUrl, setDashboardUrl] = useState("http://127.0.0.1:5000/dashboard/");
    const [useDefault, setUseDefault] = useState(true);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
        setUseDefault(false);
    };

    const handleUpload = async () => {
        if (!file) {
            setMessage("Please select a file to upload.");
            return;
        }
        
        setIsLoading(true);
        setMessage("Processing file...");
        
        const formData = new FormData();
        formData.append("file", file);
        
        try {
            const response = await fetch("http://127.0.0.1:5000/process", {
                method: "POST",
                body: formData,
            });
            
            const result = await response.json();
            setMessage(result.message);
            
            // Refresh the dashboard iframe to show new data
            setDashboardUrl("http://127.0.0.1:5000/dashboard/?t=" + new Date().getTime());
        } catch (error) {
            setMessage("Error uploading file: " + error.message);
        } finally {
            setIsLoading(false);
        }
    };

    const useDefaultData = () => {
        setUseDefault(true);
        setMessage("Using default dataset (steam_reviews.csv)");
        setDashboardUrl("http://127.0.0.1:5000/dashboard/?t=" + new Date().getTime());
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Review Analysis Tool</h1>
            </header>
            
            <div className="control-panel">
                <h2>Upload Reviews CSV</h2>
                <p>Upload a CSV file with a 'review_text' column to analyze sentiment and topics</p>
                
                <div className="file-upload">
                    <input type="file" accept=".csv" onChange={handleFileChange} />
                    <button onClick={handleUpload} disabled={isLoading || !file}>
                        {isLoading ? "Processing..." : "Upload and Analyze"}
                    </button>
                    <button onClick={useDefaultData}>
                        Use Default Dataset
                    </button>
                </div>
                
                {message && <div className="message">{message}</div>}
            </div>
            
            <div className="dashboard-container">
                <h2>Analysis Dashboard</h2>
                <iframe 
                    src={dashboardUrl} 
                    title="Analysis Dashboard" 
                    width="100%" 
                    height="800px"
                    style={{border: "none"}}
                />
            </div>
        </div>
    );
}

export default App;
