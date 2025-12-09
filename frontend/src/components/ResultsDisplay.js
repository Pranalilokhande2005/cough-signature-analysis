import React, { useEffect } from 'react';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend
} from 'chart.js';
import { Pie } from 'react-chartjs-2';
import './ResultsDisplay.css';

// register Chart.js components once
ChartJS.register(ArcElement, Tooltip, Legend);

const ResultsDisplay = ({ results }) => {
  const { noise_classification, cough_classification, audio_info } = results;

  /* ----------  pie-chart data  ---------- */
  const chartData = {
    labels: Object.keys(cough_classification.class_probabilities),
    datasets: [
      {
        data: Object.values(cough_classification.class_probabilities),
        backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56'],
        hoverBackgroundColor: ['#FF6384', '#36A2EB', '#FFCE56']
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'bottom' },
      title: { display: true, text: 'Disease Probability Distribution' }
    }
  };

  /* ----------  wrapper to avoid canvas reuse  ---------- */
  function PieChart({ data, options }) {
    useEffect(() => {
      /* destroy any existing chart on this canvas */
      const old = ChartJS.getChart('pie-canvas');
      if (old) old.destroy();
    }, []);
    return <Pie id="pie-canvas" data={data} options={options} />;
  }

  /* ----------  render  ---------- */
  return (
    <div className="results-display">
      <h2>Analysis Results</h2>

      <div className="result-section">
        <h3>Noise Classification</h3>
        <div className="classification-result">
          <p className={`status ${noise_classification.is_cough ? 'cough' : 'noise'}`}>
            {noise_classification.is_cough ? '✅ Cough Detected' : '⚠️ Noise Detected'}
          </p>
          <p className="confidence">
            Confidence: {(noise_classification.confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="result-section">
        <h3>Disease Classification</h3>
        <div className="prediction-main">
          <h4 className="prediction-label">{cough_classification.prediction}</h4>
          <p className="prediction-confidence">
            Confidence: {(cough_classification.confidence * 100).toFixed(1)}%
          </p>
        </div>

        <div className="chart-container">
          <PieChart data={chartData} options={chartOptions} />
        </div>

        <div className="top-predictions">
          <h4>Top 3 Predictions:</h4>
          <ul>
            {cough_classification.top_3_predictions.map((pred, index) => (
              <li key={index}>
                {pred.class}: {(pred.probability * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="result-section">
        <h3>Audio Information</h3>
        <div className="audio-info">
          <p>Duration: {audio_info.duration.toFixed(2)} seconds</p>
          <p>Sample Rate: {audio_info.sample_rate} Hz</p>
          <p>Features Extracted: {audio_info.features_extracted.length}</p>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;