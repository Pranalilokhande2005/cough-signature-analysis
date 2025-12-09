import React, { useState } from 'react';
import './App.css';
import AudioRecorder from './components/AudioRecorder';
import FileUploader from './components/FileUploader';
import ResultsDisplay from './components/ResultsDisplay';
import ExplanationPanel from './components/ExplanationPanel';
import { analyzeAudio } from './services/api';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAudioAnalysis = async (audioData, isFile = false) => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const data = await analyzeAudio(audioData, isFile);
      
      if (data.status === 'error') {
        setError(data.message);
      } else {
        setResults(data);
      }
    } catch (err) {
      setError('Failed to analyze audio. Please try again.');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Cough Analysis System</h1>
        <p>AI-powered cough classification with explainable AI</p>
      </header>

      <main className="App-main">
        <div className="input-section">
          <h2>Record or Upload Audio</h2>
          <div className="input-methods">
            <AudioRecorder onRecordingComplete={(audioData) => handleAudioAnalysis(audioData, false)} />
            <FileUploader onFileUpload={(file) => handleAudioAnalysis(file, true)} />
          </div>
        </div>

        {loading && (
          <div className="loading-section">
            <div className="spinner"></div>
            <p>Analyzing audio...</p>
          </div>
        )}

        {error && (
          <div className="error-section">
            <div className="error-message">
              <h3>Error</h3>
              <p>{error}</p>
            </div>
          </div>
        )}

        {results && (
          <div className="results-section">
            <ResultsDisplay results={results} />
            <ExplanationPanel explanation={results.explanation} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;