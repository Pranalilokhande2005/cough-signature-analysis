import numpy as np
import tensorflow as tf
from typing import Tuple, List
from ..models.audio_models import create_noise_classifier
from ..config import config

class NoiseClassifier:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(config.NOISE_CLASSIFIER_PATH)
        self.model = None
        self.input_shape = (128, 129, 1)  # Adjust based on your feature extraction
        
    def load_model(self):
        """Load the trained noise classifier model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Noise classifier model loaded from {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading noise classifier model: {str(e)}")
    
    def prepare_features(self, audio_features: dict) -> np.ndarray:
        """Prepare features for the noise classifier"""
        # Use mel spectrogram as primary feature
        mel_spec = audio_features['mel_spec']
        
        # Ensure consistent shape
        if mel_spec.shape != (128, 129):
            # Resize if necessary
            mel_spec = tf.image.resize(
                mel_spec.reshape(128, 129, 1),
                (128, 129)
            ).numpy().reshape(128, 129)
        
        # Add channel dimension
        features = mel_spec.reshape(1, 128, 129, 1)
        
        return features
    
    def predict(self, audio_features: dict) -> dict:
        """Predict whether audio contains noise or cough"""
        if self.model is None:
            self.load_model()
        
        # Prepare features
        features = self.prepare_features(audio_features)
        
        # Make prediction
        prediction = self.model.predict(features, verbose=0)
        confidence = float(prediction[0][0])
        
        # Classify
        is_cough = confidence > 0.5
        confidence_score = confidence if is_cough else 1 - confidence
        
        result = {
            'is_cough': is_cough,
            'confidence': confidence_score,
            'raw_prediction': confidence,
            'class_probabilities': {
                'noise': 1 - confidence,
                'cough': confidence
            }
        }
        
        return result
    
    def classify_segments(self, segments: List[dict]) -> List[dict]:
        """Classify multiple audio segments"""
        classified_segments = []
        
        for segment in segments:
            # Extract features for this segment
            from .audio_processor import AudioProcessor
            processor = AudioProcessor()
            features = processor.extract_features(segment['segment'])
            
            # Classify
            classification = self.predict(features)
            
            # Add classification to segment
            segment['classification'] = classification
            classified_segments.append(segment)
        
        return classified_segments