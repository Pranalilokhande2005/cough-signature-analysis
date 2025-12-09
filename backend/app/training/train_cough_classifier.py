import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from pathlib import Path
import librosa
import os
from sklearn.model_selection import train_test_split
from ..models.audio_models import create_cough_classifier, get_callbacks
from ..config import config
from ..services.audio_processor import AudioProcessor

# For saving mid-training
from tensorflow.keras.callbacks import ModelCheckpoint


def create_synthetic_cough_data():
    """Create synthetic training data for cough classifier"""
    print("Creating synthetic cough training data...")
    
    # Parameters
    n_samples_per_class = 1000
    sample_rate = config.SAMPLE_RATE
    duration = config.DURATION
    
    # COVID-19 cough characteristics (dry, persistent)
    covid_samples = []
    for i in range(n_samples_per_class):
        t = np.linspace(0, duration, int(sample_rate * duration))
        cough = np.zeros_like(t)
        n_bursts = np.random.randint(2, 5)
        for _ in range(n_bursts):
            burst_start = np.random.uniform(0, duration - 0.3)
            burst_duration = np.random.uniform(0.05, 0.15)
            burst_idx = int(burst_start * sample_rate)
            burst_end = int((burst_start + burst_duration) * sample_rate)
            if burst_end < len(t):
                burst_t = t[burst_idx:burst_end] - t[burst_idx]
                burst = np.exp(-burst_t * 15) * np.sin(2 * np.pi * 150 * burst_t)
                burst += 0.3 * np.exp(-burst_t * 25) * np.sin(2 * np.pi * 300 * burst_t)
                cough[burst_idx:burst_end] += burst
        cough += 0.005 * np.random.randn(len(cough))
        cough = cough / (np.max(np.abs(cough)) + 1e-8)
        covid_samples.append(cough)
    
    # Asthma cough characteristics (wheezy, variable)
    asthma_samples = []
    for i in range(n_samples_per_class):
        t = np.linspace(0, duration, int(sample_rate * duration))
        cough = np.zeros_like(t)
        n_bursts = np.random.randint(1, 3)
        for _ in range(n_bursts):
            burst_start = np.random.uniform(0, duration - 0.4)
            burst_duration = np.random.uniform(0.1, 0.25)
            burst_idx = int(burst_start * sample_rate)
            burst_end = int((burst_start + burst_duration) * sample_rate)
            if burst_end < len(t):
                burst_t = t[burst_idx:burst_end] - t[burst_idx]
                burst = np.exp(-burst_t * 8) * np.sin(2 * np.pi * 80 * burst_t)
                wheeze = 0.4 * np.sin(2 * np.pi * 400 * burst_t) * np.exp(-burst_t * 5)
                burst += wheeze
                cough[burst_idx:burst_end] += burst
        cough += 0.01 * np.sin(2 * np.pi * 350 * t) * np.exp(-t * 2)
        cough += 0.005 * np.random.randn(len(cough))
        cough = cough / (np.max(np.abs(cough)) + 1e-8)
        asthma_samples.append(cough)
    
    # Healthy cough characteristics (clear, single/short)
    healthy_samples = []
    for i in range(n_samples_per_class):
        t = np.linspace(0, duration, int(sample_rate * duration))
        cough = np.zeros_like(t)
        n_bursts = np.random.randint(1, 3)
        for _ in range(n_bursts):
            burst_start = np.random.uniform(0.1, duration - 0.2)
            burst_duration = np.random.uniform(0.08, 0.12)
            burst_idx = int(burst_start * sample_rate)
            burst_end = int((burst_start + burst_duration) * sample_rate)
            if burst_end < len(t):
                burst_t = t[burst_idx:burst_end] - t[burst_idx]
                burst = np.exp(-burst_t * 12) * np.sin(2 * np.pi * 120 * burst_t)
                cough[burst_idx:burst_end] += burst
        cough += 0.002 * np.random.randn(len(cough))
        cough = cough / (np.max(np.abs(cough)) + 1e-8)
        healthy_samples.append(cough)
    
    return covid_samples, asthma_samples, healthy_samples


def extract_features_from_audio(audio_samples):
    """Extract features from audio samples"""
    processor = AudioProcessor()
    features = []
    
    FIXED_MELS = 128
    FIXED_FRAMES = 129

    for audio in audio_samples:
        audio_features = processor.extract_features(audio)
        mel_spec = audio_features['mel_spec']
        mel_spec = np.array(mel_spec)

        if mel_spec.shape[1] < FIXED_FRAMES:
            pad_amount = FIXED_FRAMES - mel_spec.shape[1]
            mel_spec = np.pad(
                mel_spec,
                ((0, 0), (0, pad_amount)),
                mode='constant'
            )
        else:
            mel_spec = mel_spec[:, :FIXED_FRAMES]

        mel_spec = mel_spec.reshape(FIXED_MELS, FIXED_FRAMES)
        features.append(mel_spec)

    return np.array(features)



def train_cough_classifier():
    print("Starting cough classifier training...")
    
    covid_samples, asthma_samples, healthy_samples = create_synthetic_cough_data()
    
    print("Extracting features...")
    covid_features = extract_features_from_audio(covid_samples)
    asthma_features = extract_features_from_audio(asthma_samples)
    healthy_features = extract_features_from_audio(healthy_samples)
    
    X = np.concatenate([covid_features, asthma_features, healthy_features])
    y = np.concatenate([
        np.zeros(len(covid_features)),
        np.ones(len(asthma_features)),
        np.full(len(healthy_features), 2)
    ])
    
    y_categorical = to_categorical(y, num_classes=3)
    
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    # ----------------------------------------
    # RESUME TRAINING LOGIC (ONLY NEW PART)
    # ----------------------------------------
    if os.path.exists("cough_checkpoint.h5"):
        print("🚀 Resuming training from cough_checkpoint.h5 ...")
        model = load_model("cough_checkpoint.h5")
    else:
        print("➡️ No checkpoint found. Starting fresh training.")
        model = create_cough_classifier(input_shape, num_classes=3)
    # ----------------------------------------

    print(f"Model created with input shape: {input_shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    checkpoint = ModelCheckpoint(
        "cough_checkpoint.h5",
        monitor="loss",
        save_best_only=False,
        verbose=1
    )

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks('cough_classifier') + [checkpoint],
        verbose=1
    )
    
    print("Evaluating model...")
    test_loss, test_accuracy, test_cat_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Categorical Accuracy: {test_cat_accuracy:.4f}")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    class_names = ['COVID-19', 'Asthma', 'Healthy']
    class_accuracies = {}
    
    for i, class_name in enumerate(class_names):
        class_mask = y_true_classes == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred_classes[class_mask] == i)
            class_accuracies[class_name] = float(class_acc)
            print(f"{class_name} Accuracy: {class_acc:.4f}")
    
    model_path = str(config.COUGH_CLASSIFIER_PATH)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return {
        'status': 'success',
        'test_accuracy': float(test_accuracy),
        'test_categorical_accuracy': float(test_cat_accuracy),
        'class_accuracies': class_accuracies,
        'model_path': model_path,
        'training_history': {
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
    }


if __name__ == "__main__":
    results = train_cough_classifier()
    print("Training completed!")
