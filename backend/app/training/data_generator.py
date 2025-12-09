import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf 
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

SAMPLE_RATE = 22_050
DURATION    = 3.0
N_CLASSES   = 3
TARGET_SHAPE = (128, 129)

def add_noise(audio, snr_db=20):
    """Add white noise with given SNR"""
    p_sig = np.mean(audio**2)
    p_noise = p_sig / (10**(snr_db/10))
    noise = np.random.randn(len(audio)) * np.sqrt(p_noise)
    return audio + noise

def time_stretch(audio, rate_range=(0.8, 1.2)):
    rate = np.random.uniform(*rate_range)
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, n_steps_range=(-3, 3)):
    n_steps = np.random.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def volume_change(audio, db_range=(-6, 6)):
    db = np.random.uniform(*db_range)
    return audio * (10**(db/20))

def build_realistic_dataset(n_per_class=800):
    """Return X, y as numpy arrays ready for training"""
    X, y = [], []
    class_names = ['covid', 'asthma', 'healthy']

    for label_idx, cls in enumerate(class_names):
        print(f'Generating {cls} …')
        for i in range(n_per_class):
            # ---- 1. base synthetic signal (your old code) ----
            audio = synthetic_cough(cls)

            # ---- 2. augment ----
            audio = time_stretch(audio)
            audio = pitch_shift(audio, SAMPLE_RATE)
            audio = volume_change(audio)
            if np.random.rand() < 0.5:        # 50 % add noise
                audio = add_noise(audio, snr_db=np.random.randint(15, 30))

            # ---- 3. preprocess ----
            audio = trim_or_pad(audio, SAMPLE_RATE, DURATION)
            mel = librosa.power_to_db(
                librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128, hop_length=512)
            )
            # resize → 128×129
            if mel.shape[1] != TARGET_SHAPE[1]:
                mel = tf.image.resize(mel[..., np.newaxis], TARGET_SHAPE).numpy().squeeze(-1)
            X.append(mel)
            y.append(label_idx)

    X = np.array(X)[..., np.newaxis]   # (N, 128, 129, 1)
    y = to_categorical(y, N_CLASSES)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def synthetic_cough(cls: str) -> np.ndarray:
    """Your old synthetic generators merged & shortened"""
    sr = SAMPLE_RATE
    dur = DURATION
    t = np.linspace(0, dur, int(sr * dur))
    audio = np.zeros_like(t)

    if cls == 'covid':
        for i in range(np.random.randint(2, 5)):
            start = np.random.uniform(0.1, dur - 0.3)
            width = np.random.uniform(0.05, 0.15)
            burst = np.exp(-(t - start) * 15) * np.sin(2 * np.pi * 150 * (t - start))
            audio += np.where((t >= start) & (t <= start + width), burst, 0)

    elif cls == 'asthma':
        start = np.random.uniform(0.2, 0.5)
        width = np.random.uniform(0.15, 0.25)
        burst = np.exp(-(t - start) * 8) * np.sin(2 * np.pi * 80 * (t - start))
        wheeze = 0.4 * np.sin(2 * np.pi * 400 * (t - start)) * np.exp(-(t - start) * 5)
        audio += np.where((t >= start) & (t <= start + width), burst + wheeze, 0)

    else:  # healthy
        start = np.random.uniform(0.1, 0.4)
        width = np.random.uniform(0.08, 0.12)
        burst = np.exp(-(t - start) * 12) * np.sin(2 * np.pi * 120 * (t - start))
        audio += np.where((t >= start) & (t <= start + width), burst, 0)

    audio += 0.005 * np.random.randn(len(audio))
    return audio / (np.max(np.abs(audio)) + 1e-8)


def trim_or_pad(audio, sr, dur):
    target = int(sr * dur)
    if len(audio) > target:
        start = (len(audio) - target) // 2
        return audio[start:start + target]
    return np.pad(audio, (0, target - len(audio)))