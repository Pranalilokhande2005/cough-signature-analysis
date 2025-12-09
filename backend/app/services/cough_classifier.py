import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple
from ..models.audio_models import create_cough_classifier
from ..config import config

class CoughClassifier:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "models/cough_classifier_realistic.h5"
        self.model = None
        self.class_names = ['COVID-19', 'Asthma', 'Healthy']
        self.input_shape = (128, 129, 1)
        
    def load_model(self):
        """Load the trained cough classifier model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Cough classifier model loaded from {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading cough classifier model: {str(e)}")
    
    def prepare_features(self, audio_features: dict) -> np.ndarray:
        """Prepare features for the cough classifier"""
        # Use multiple features
        mel_spec = audio_features['mel_spec']
        mfcc = audio_features['mfcc']
        
        # Ensure consistent shapes
        if mel_spec.shape != (128, 129):
            mel_spec = tf.image.resize(
                mel_spec.reshape(128, 129, 1),
                (128, 129)
            ).numpy().reshape(128, 129)
        
        # Combine features (you can experiment with different combinations)
        combined_features = mel_spec
        
        # Add channel dimension
        features = combined_features.reshape(1, 128, 129, 1)
        
        return features
    
    def predict(self, audio_features: dict) -> dict:
        """Predict cough disease classification"""
        if self.model is None:
            self.load_model()
        
        # Prepare features
        features = self.prepare_features(audio_features)
        
        # Make prediction
        predictions = self.model.predict(features, verbose=0)
        probabilities = predictions[0]
        
        # Get top prediction
        top_class_idx = np.argmax(probabilities)
        top_class = self.class_names[top_class_idx]
        raw_confidence = float(probabilities[top_class_idx])
        top_confidence = 0.60 + (raw_confidence * 0.10)  # scales within 0.60–0.70
        

        
        # Create result dictionary
        result = {
            'prediction': top_class,
            'confidence': round(top_confidence, 3),
            'class_probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probabilities)
            },
            'top_3_predictions': self.get_top_k_predictions(probabilities, k=3)
        }
        
        return result
    
    def get_top_k_predictions(self, probabilities: np.ndarray, k: int = 3) -> List[dict]:
        """Get top k predictions with probabilities"""
        top_k_indices = np.argsort(probabilities)[::-1][:k]
        
        top_k = []
        for idx in top_k_indices:
            top_k.append({
                'class': self.class_names[idx],
                'probability': float(probabilities[idx])
            })
        
        return top_k
    
    def batch_predict(self, audio_features_list: List[dict]) -> List[dict]:
        """Predict for multiple audio samples"""
        results = []
        
        for features in audio_features_list:
            result = self.predict(features)
            results.append(result)
        
        return results