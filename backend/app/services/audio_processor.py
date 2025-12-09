import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from typing import Tuple, List, Dict
import tensorflow as tf  # only used for fast resize


class AudioProcessor:
    def __init__(self, sample_rate: int = 22_050, duration: float = 3.0, n_mfcc: int = 40):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        # fixed parameters that give exactly 128 × 129
        self.n_fft = 2_048
        self.hop_length = 512
        self.n_mels = 128
        self.target_frames = 129

    # ------------------------------------------------------------------
    # 1.  audio I/O
    # ------------------------------------------------------------------
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
             print("ANALYZE ERROR:", str(e))   # ← added
  

    # ------------------------------------------------------------------
    # 2.  pre-processing chain  (unchanged)
    # ------------------------------------------------------------------
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = self.reduce_noise(audio)
        audio = self.normalize_audio(audio)
        audio = self.remove_silence(audio)
        audio = self.trim_or_pad(audio)
        return audio

    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        noise_samples = int(0.5 * self.sample_rate)
        return nr.reduce_noise(
            y=audio, y_noise=audio[:noise_samples], sr=self.sample_rate,
            stationary=True, prop_decrease=0.75
        )

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        return audio / (max_val + 1e-8)

    def remove_silence(self, audio: np.ndarray, top_db: int = 30) -> np.ndarray:
        intervals = librosa.effects.split(audio, top_db=top_db)
        if len(intervals) == 0:
            return audio
        return np.concatenate([audio[start:end] for start, end in intervals])

    def trim_or_pad(self, audio: np.ndarray) -> np.ndarray:
        target_len = int(self.sample_rate * self.duration)
        if len(audio) > target_len:
            start = (len(audio) - target_len) // 2
            return audio[start:start + target_len]
        else:
            return np.pad(audio, (0, target_len - len(audio)), mode='constant')

    # ------------------------------------------------------------------
    # 3.  FEATURE EXTRACTION  –  fixed-shape mel + aux + CHROMA
    # ------------------------------------------------------------------
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        # ---- mel spectrogram  (128 freq bins)  -----------------------
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # resize time axis → exactly 129 frames
        if mel_db.shape[1] != self.target_frames:
            mel_db = tf.image.resize(
                mel_db[..., np.newaxis], (self.n_mels, self.target_frames)
            ).numpy().squeeze(axis=-1)

        # ---- MFCC  (40 coeffs)  -------------------------------------
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=128)
        mfcc = librosa.util.fix_length(mfcc, size=self.target_frames, axis=1)

        # ---- CHROMA  (12 bins)  –  NEW  ------------------------------
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        # same time resize for chroma if needed
        if chroma.shape[1] != self.target_frames:
            chroma = tf.image.resize(
                chroma[..., np.newaxis], (12, self.target_frames)
            ).numpy().squeeze(axis=-1)

        # ---- spectral / temporal descriptors  -----------------------
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        spectral_rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        rms                = librosa.feature.rms(y=audio)

        return {
            'mel_spec': mel_db,              # (128, 129)
            'mfcc': mfcc,                    # (40, 129)
            'chroma': chroma,                # (12, 129)  ← added
            'spectral_centroids': spectral_centroids,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            'rms': rms
        }

    # ------------------------------------------------------------------
    # 4.  segmentation / cough detection  (unchanged)
    # ------------------------------------------------------------------
    def segment_audio(self, audio: np.ndarray, segment_length: float = 1.0) -> List[np.ndarray]:
        seg_samples = int(segment_length * self.sample_rate)
        hop = seg_samples // 2
        segments = []
        for start in range(0, len(audio) - seg_samples + 1, hop):
            segments.append(audio[start:start + seg_samples])
        return segments

    def detect_cough_segments(self, audio: np.ndarray) -> List[Dict]:
        segments = self.segment_audio(audio, 0.5)
        cough_segments = []
        for i, seg in enumerate(segments):
            feats = self.extract_features(seg)
            zcr   = np.mean(feats['zero_crossing_rate'])
            roll  = np.mean(feats['spectral_rolloff'])
            rms_mean = np.mean(feats['rms'])
            rms_std    = np.std(feats['rms'])

            score = (zcr > 0.1) * 0.3 + (roll > 5000) * 0.3 + (rms_std > 0.05) * 0.4
            if score > 0.5:
                cough_segments.append({
                    'segment': seg,
                    'start_time': i * 0.25,
                    'end_time': (i + 1) * 0.5,
                    'confidence': score
                })
        return cough_segments