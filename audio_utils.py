import os
import random
import librosa
import numpy as np
import torch
import joblib
from io import BytesIO
import torch.nn as nn
import torch.nn.functional as F

# ─────── Noise Function ───────
def add_noise(y, sr, dir="data/background_noises"):
    noises = [f for f in os.listdir(dir) if f.endswith(".wav")]
    if not noises:
        return y
    n, _ = librosa.load(os.path.join(dir, random.choice(noises)), sr=sr)
    if len(n) < len(y):
        n = np.tile(n, int(np.ceil(len(y) / len(n))))
    n = n[:len(y)]
    snr = random.uniform(5, 20)
    return librosa.util.normalize(y + np.sqrt(np.mean(y**2) / (10**(snr / 10)) / (np.mean(n**2) + 1e-6)) * n)

# ─────── Model Definition ───────
class EmotionCNN2D(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 10 * 16, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(self.dropout(x))

# ─────── Feature Extraction ───────
def extract_features_from_path(path, scaler, device, augment=False):
    try:
        # ADD offset to match training
        y, sr = librosa.load(path, sr=22050, duration=3, offset=0.5)
        y, _ = librosa.effects.trim(y)
        y = librosa.util.normalize(y)
        if augment:
            y = add_noise(y, sr)
            if random.random() < 0.3:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.choice([-2, 2]))
            if random.random() < 0.3:
                y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
        return _process_audio(y, sr, scaler, device)
    except Exception as e:
        print(f"⚠️ Error processing {path}: {e}")
        raise

def extract_features_from_bytes(audio_bytes, scaler, device, augment=False):
    try:
        # ADD offset to match training
        y, sr = librosa.load(BytesIO(audio_bytes), sr=22050, duration=3, offset=0.5)
        y, _ = librosa.effects.trim(y)
        y = librosa.util.normalize(y)
        if augment:
            y = add_noise(y, sr)
            if random.random() < 0.3:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.choice([-2, 2]))
            if random.random() < 0.3:
                y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
        return _process_audio(y, sr, scaler, device)
    except Exception as e:
        print(f"⚠️ Error processing audio bytes: {e}")
        raise

def _process_audio(y, sr, scaler, device):
    target_len = sr * 3
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512,
        win_length=1024,
        window='hann',
        center=True
    )
    delta = librosa.feature.delta(mfcc)
    combined = np.concatenate((mfcc, delta), axis=0)
    combined = librosa.util.fix_length(combined, size=130, axis=1)

    X_flat = combined.flatten().reshape(1, -1)
    X_scaled = scaler.transform(X_flat)
    X_scaled = X_scaled.reshape(1, 1, 80, 130)

    return torch.tensor(X_scaled, dtype=torch.float32).to(device)

# ─────── Load Model + Scaler ───────
def load_model_and_scaler(
    model_path="notebooks/emotion_recognition_model_final.pth",
    scaler_path="model/feature_scaler.pkl",
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNN2D(num_classes=5).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"⚠️ Error loading model from {model_path}: {e}")
        raise

    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"⚠️ Error loading scaler from {scaler_path}: {e}")
        raise

    return model, scaler

# ─────── Emotion Labels ───────
defined_emotions = ['angry', 'calm', 'fearful', 'happy', 'sad']
