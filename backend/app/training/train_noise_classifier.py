import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ..models.audio_models import create_noise_classifier, get_callbacks
from ..config import config
from ..services.audio_processor import AudioProcessor

def create_synthetic_data():
    """Create synthetic training data for noise classifier"""
    print("Creating synthetic training data...")
    
    # Parameters
    n_samples = 2000
    sample_rate = config.SAMPLE_RATE
    duration = config.DURATION
    
    # Generate synthetic cough sounds
    cough_samples = []
    for i in range(n_samples // 2):
        # Create synthetic cough-like signal
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Cough characteristics: impulsive, broadband
        cough = np.zeros_like(t)
        
        # Add multiple cough bursts
        n_bursts = np.random.randint(1, 4)
        for _ in range(n_bursts):
            burst_start = np.random.uniform(0, duration - 0.5)
            burst_duration = np.random.uniform(0.1, 0.3)
            burst_idx = int(burst_start * sample_rate)
            burst_end = int((burst_start + burst_duration) * sample_rate)
            
            if burst_end < len(t):
                # Create cough burst
                burst_t = t[burst_idx:burst_end] - t[burst_idx]
                burst = np.exp(-burst_t * 10) * np.sin(2 * np.pi * 100 * burst_t)
                burst += 0.5 * np.exp(-burst_t * 20) * np.sin(2 * np.pi * 200 * burst_t)
                
                # Add noise
                burst += 0.1 * np.random.randn(len(burst))
                
                cough[burst_idx:burst_end] += burst
        
        # Add some background noise
        cough += 0.01 * np.random.randn(len(cough))
        
        # Normalize
        cough = cough / (np.max(np.abs(cough)) + 1e-8)
        cough_samples.append(cough)
    
    # Generate synthetic noise samples
    noise_samples = []
    for i in range(n_samples // 2):
        noise_type = np.random.choice(['white', 'pink', 'brown', 'environmental'])
        
        if noise_type == 'white':
            noise = np.random.randn(int(sample_rate * duration)) * 0.3
        elif noise_type == 'pink':
            noise = np.random.randn(int(sample_rate * duration))
            noise = np.cumsum(noise) * 0.01
        elif noise_type == 'brown':
            noise = np.random.randn(int(sample_rate * duration))
            noise = np.cumsum(np.cumsum(noise)) * 0.001
        else:  # environmental
            noise = np.random.randn(int(sample_rate * duration)) * 0.1
            t = np.linspace(0, duration, int(sample_rate * duration))
            noise += 0.05 * np.sin(2 * np.pi * 50 * t)
            noise += 0.03 * np.sin(2 * np.pi * 120 * t)
        
        noise = noise / (np.max(np.abs(noise)) + 1e-8)
        noise_samples.append(noise)
    
    return cough_samples, noise_samples

def extract_features_from_audio(audio_samples):
    """Extract features from audio samples"""
    processor = AudioProcessor()
    features = []
    
    for audio in audio_samples:
        audio_features = processor.extract_features(audio)
        
        # Use mel spectrogram as primary feature
        mel_spec = audio_features['mel_spec']

        # -----------------------------
        # FIX APPLIED HERE (ONLY CHANGE)
        # -----------------------------
        # Ensure mel spectrogram has exactly 129 frames
        if mel_spec.shape[1] != 129:
            mel_spec = librosa.util.fix_length(mel_spec, size=129, axis=1)

        # Ensure final shape is exactly (128, 129)
        mel_spec = mel_spec.reshape(128, 129)
        # -----------------------------

        features.append(mel_spec)
    
    return np.array(features)

def train_noise_classifier():
    """Train the noise classifier model"""
    print("Starting noise classifier training...")
    
    # Create synthetic data
    cough_samples, noise_samples = create_synthetic_data()
    
    # Extract features
    print("Extracting features...")
    cough_features = extract_features_from_audio(cough_samples)
    noise_features = extract_features_from_audio(noise_samples)
    
    # Create labels
    X = np.concatenate([cough_features, noise_features])
    y = np.concatenate([np.ones(len(cough_features)), np.zeros(len(noise_features))])
    
    # Add channel dimension
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = create_noise_classifier(input_shape)
    
    print(f"Model created with input shape: {input_shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks('noise_classifier'),
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save model
    model_path = str(config.NOISE_CLASSIFIER_PATH)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return {
        'status': 'success',
        'test_accuracy': float(test_accuracy),
        'test_auc': float(test_auc),
        'model_path': model_path,
        'training_history': {
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
    }

if __name__ == "__main__":
    results = train_noise_classifier()
    print("Training completed!")
