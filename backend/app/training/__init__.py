# backend/app/training/__init__.py
from .data_generator import build_realistic_dataset   # NEW name
from .train_noise_classifier import train_noise_classifier
from .train_cough_classifier import train_cough_classifier

__all__ = [
    'build_realistic_dataset',
    'train_noise_classifier',
    'train_cough_classifier'
]