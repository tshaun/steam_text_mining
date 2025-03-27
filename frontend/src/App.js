import React, { useState } from "react";
import Plot from "react-plotly.js";  // Plotly chart component

function App() {
  const [topics, setTopics] = useState([]);
  const [error, setError] = useState("");

  // Handle file upload and send data to backend
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:5000/topics", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setTopics(data);  // Store topic data for visualization
      } else {
        setError("Failed to fetch data from backend.");
      }
    } catch (error) {
      setError("Error uploading file.");
      console.error(error);
    }
  };

  // Render topics as Plotly bar chart
  const plotData = topics.map((topic, index) => ({
    type: "bar",
    x: topic.words,
    y: Array(topic.words.length).fill(index + 1),
    name: topic.topic,
  }));

  return (
    <div className="App">
      <h1>Steam Reviews NLP Dashboard</h1>
      <input type="file" onChange={handleFileUpload} />
      {error && <p>{error}</p>}

      {topics.length > 0 && (
        <div>
          <h2>Topic Modeling</h2>
          <Plot
            data={plotData}
            layout={{
              title: "LDA Topic Modeling",
              barmode: "stack",
              xaxis: { title: "Words" },
              yaxis: { title: "Topic" },
            }}
          />
        </div>
      )}
    </div>
  );
}

export default App;
