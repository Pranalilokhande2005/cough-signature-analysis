import os
from pathlib import Path

class Config:
    # Model paths
    MODEL_DIR = Path("models")
    NOISE_CLASSIFIER_PATH = MODEL_DIR / "noise_classifier.h5"
    COUGH_CLASSIFIER_PATH = MODEL_DIR / "cough_classifier.h5"
    
    # Audio settings
    SAMPLE_RATE = 22050
    DURATION = 3  # seconds
    N_MFCC = 40
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # Model settings
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # API settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac'}
    
    # Create model directory if not exists
    MODEL_DIR.mkdir(exist_ok=True)

config = Config()