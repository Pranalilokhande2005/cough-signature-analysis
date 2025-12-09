import React, { useState } from 'react';
import './ExplanationPanel.css';

const ExplanationPanel = ({ explanation }) => {
  const [activeTab, setActiveTab] = useState('overview');

  if (!explanation) return null;

  //  ⬇️  simply drop “visualizations” from the destructuring
  const { prediction, feature_importance, clinical_reasoning, confidence_breakdown } = explanation;

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'features', label: 'Key Features' },
    { id: 'clinical', label: 'Clinical Reasoning' },
    { id: 'confidence', label: 'Confidence Analysis' }
  ];

  const renderOverview = () => (
    <div className="tab-content">
      <h3>Prediction Overview</h3>
      <div className="overview-card">
        <h4>Diagnosis: {prediction.prediction}</h4>
        <p className="confidence-score">
          Model Confidence: {(prediction.confidence * 100).toFixed(1)}%
        </p>
        <div className="explanation-summary">
          <p>
            The AI model has analyzed your cough audio and identified patterns consistent with 
            <strong> {prediction.prediction}</strong>. This conclusion is based on advanced 
            machine learning analysis of multiple audio features including spectral characteristics, 
            temporal patterns, and frequency distributions.
          </p>
        </div>
      </div>
    </div>
  );

  const renderFeatures = () => (
    <div className="tab-content">
      <h3>Key Features Analysis</h3>
      <div className="features-list">
        <h4>Top 5 Most Important Features:</h4>
        {feature_importance.top_features.map(([feature, importance], index) => (
          <div key={feature} className="feature-item">
            <div className="feature-header">
              <span className="feature-name">{feature}</span>
              <span className="feature-importance">
                {(importance * 100).toFixed(1)}%
              </span>
            </div>
            <div className="feature-bar">
              <div 
                className="feature-fill"
                style={{ width: `${importance * 100}%` }}
              ></div>
            </div>
            <p className="feature-description">
              {getFeatureDescription(feature, prediction.prediction)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );

  const renderClinical = () => (
    <div className="tab-content">
      <h3>Clinical Reasoning</h3>
      <div className="clinical-reasoning">
        <div className="reasoning-section">
          <h4>Primary Indicators:</h4>
          <ul>
            {clinical_reasoning.primary_indicators.map((indicator, index) => (
              <li key={index}>{indicator}</li>
            ))}
          </ul>
        </div>
        
        <div className="reasoning-section">
          <h4>Supporting Evidence:</h4>
          <ul>
            {clinical_reasoning.supporting_evidence.map((evidence, index) => (
              <li key={index}>{evidence}</li>
            ))}
          </ul>
        </div>
        
        <div className="reasoning-section">
          <h4>Recommendations:</h4>
          <ul>
            {clinical_reasoning.recommendations.map((recommendation, index) => (
              <li key={index}>{recommendation}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );

  const renderConfidence = () => (
    <div className="tab-content">
      <h3>Confidence Analysis</h3>
      <div className="confidence-breakdown">
        <div className="confidence-level">
          <h4>Confidence Level: {confidence_breakdown.prediction_strength.level}</h4>
          <p>{confidence_breakdown.prediction_strength.interpretation}</p>
        </div>
        
        <div className="feature-consistency">
          <h4>Feature Consistency</h4>
          <p>Overall Consistency: {(confidence_breakdown.feature_consistency.overall_consistency * 100).toFixed(1)}%</p>
          <div className="consistency-details">
            <p>MFCC Stability: {(confidence_breakdown.feature_consistency.mfcc_stability * 100).toFixed(1)}%</p>
            <p>Spectral Consistency: {(confidence_breakdown.feature_consistency.spectral_consistency * 100).toFixed(1)}%</p>
          </div>
        </div>
        
        <div className="model-reliability">
          <h4>Model Reliability</h4>
          <p>Validation Accuracy: {(confidence_breakdown.confidence_assessment.raw_confidence * 100).toFixed(1)}%</p>
          <p>Model Version: {confidence_breakdown.model_reliability.model_version}</p>
        </div>
      </div>
    </div>
  );

  const getFeatureDescription = (feature, prediction) => {
    const descriptions = {
      'COVID-19': {
        'MFCC_1': 'High first MFCC coefficient indicates strong low-frequency components typical of COVID-19 dry cough',
        'MFCC_2': 'Variable second MFCC coefficient suggests characteristic COVID-19 cough pattern',
        'Spectral_Centroid': 'Spectral centroid indicates the "brightness" of the cough sound',
        'Zero_Crossing_Rate': 'High zero-crossing rate indicates noisy, dry cough characteristics',
        'RMS_Energy': 'Energy distribution pattern consistent with COVID-19 cough'
      },
      'Asthma': {
        'MFCC_3': 'Elevated third MFCC coefficient indicates mid-frequency spectral characteristics of asthma',
        'Spectral_Rolloff': 'High spectral rolloff suggests wheezing components typical in asthma',
        'Chroma_1': 'Chroma features capturing harmonic content related to wheezing',
        'Chroma_2': 'Secondary chroma features supporting wheeze detection',
        'RMS_Energy': 'Energy patterns indicating airway obstruction'
      },
      'Healthy': {
        'MFCC_1': 'Stable first MFCC coefficient indicates regular cough pattern',
        'MFCC_2': 'Consistent second MFCC coefficient shows healthy cough characteristics',
        'MFCC_3': 'Normal third MFCC coefficient within healthy range',
        'Spectral_Centroid': 'Balanced spectral content typical of healthy cough',
        'Zero_Crossing_Rate': 'Low zero-crossing rate suggests smooth, non-irritated cough',
        'RMS_Energy': 'Regular energy patterns indicating healthy respiratory function',
        'Chroma_1': 'Normal harmonic content without pathological features',
        'Chroma_2': 'Balanced secondary chroma characteristics'
      }
    };
    
    return descriptions[prediction]?.[feature] || 'Feature contributing to classification';
  };

  return (
    <div className="explanation-panel">
      <h2>Explainable AI Analysis</h2>
      
      <div className="explanation-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="explanation-content">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'features' && renderFeatures()}
        {activeTab === 'clinical' && renderClinical()}
        {activeTab === 'confidence' && renderConfidence()}
      </div>
    </div>
  );
};

export default ExplanationPanel;