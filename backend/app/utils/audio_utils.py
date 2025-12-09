import numpy as np
import librosa
from typing import Tuple, Optional, List
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

class AudioUtils:
    """Utility functions for audio processing"""
    
    @staticmethod
    def load_audio_safe(file_path: str, sample_rate: int = 22050) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Safely load audio file with error handling"""
        try:
            audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {str(e)}")
            return None, None
    
    @staticmethod
    def save_audio(audio: np.ndarray, file_path: str, sample_rate: int) -> bool:
        """Save audio file with error handling"""
        try:
            sf.write(file_path, audio, sample_rate)
            return True
        except Exception as e:
            print(f"Error saving audio file {file_path}: {str(e)}")
            return False
    
    @staticmethod
    def get_audio_info(file_path: str) -> dict:
        """Get audio file information"""
        try:
            info = sf.info(file_path)
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype
            }
        except Exception as e:
            print(f"Error getting audio info for {file_path}: {str(e)}")
            return {}
    
    @staticmethod
    def plot_waveform(audio: np.ndarray, sample_rate: int, title: str = "Waveform") -> plt.Figure:
        """Plot audio waveform"""
        fig, ax = plt.subplots(figsize=(12, 4))
        time = np.linspace(0, len(audio) / sample_rate, len(audio))
        ax.plot(time, audio)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_spectrogram(audio: np.ndarray, sample_rate: int, title: str = "Spectrogram") -> plt.Figure:
        """Plot spectrogram"""
        fig, ax = plt.subplots(figsize=(12, 6))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_synthetic_cough(duration: float = 1.0, sample_rate: int = 22050, cough_type: str = 'healthy') -> np.ndarray:
        """Generate synthetic cough for testing"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        cough = np.zeros_like(t)
        
        if cough_type == 'healthy':
            # Single clear cough
            burst_start = 0.2
            burst_duration = 0.1
            burst_idx = int(burst_start * sample_rate)
            burst_end = int((burst_start + burst_duration) * sample_rate)
            
            if burst_end < len(t):
                burst_t = t[burst_idx:burst_end] - t[burst_idx]
                burst = np.exp(-burst_t * 10) * np.sin(2 * np.pi * 100 * burst_t)
                cough[burst_idx:burst_end] = burst
                
        elif cough_type == 'covid':
            # Dry, persistent cough
            for i in range(3):
                burst_start = 0.1 + i * 0.3
                burst_duration = 0.08
                burst_idx = int(burst_start * sample_rate)
                burst_end = int((burst_start + burst_duration) * sample_rate)
                
                if burst_end < len(t):
                    burst_t = t[burst_idx:burst_end] - t[burst_idx]
                    burst = np.exp(-burst_t * 15) * np.sin(2 * np.pi * 150 * burst_t)
                    cough[burst_idx:burst_end] += burst
                    
        elif cough_type == 'asthma':
            # Wheezy cough
            burst_start = 0.2
            burst_duration = 0.2
            burst_idx = int(burst_start * sample_rate)
            burst_end = int((burst_start + burst_duration) * sample_rate)
            
            if burst_end < len(t):
                burst_t = t[burst_idx:burst_end] - t[burst_idx]
                burst = np.exp(-burst_t * 5) * np.sin(2 * np.pi * 80 * burst_t)
                wheeze = 0.3 * np.sin(2 * np.pi * 400 * burst_t) * np.exp(-burst_t * 3)
                cough[burst_idx:burst_end] = burst + wheeze
        
        # Normalize
        cough = cough / (np.max(np.abs(cough)) + 1e-8)
        return cough
    
    @staticmethod
    def add_noise(audio: np.ndarray, noise_level: float = 0.1, noise_type: str = 'white') -> np.ndarray:
        """Add noise to audio"""
        if noise_type == 'white':
            noise = np.random.randn(len(audio))
        elif noise_type == 'pink':
            noise = np.random.randn(len(audio))
            noise = np.cumsum(noise) * 0.01
        else:
            noise = np.random.randn(len(audio))
        
        noisy_audio = audio + noise_level * noise
        return noisy_audio / (np.max(np.abs(noisy_audio)) + 1e-8)