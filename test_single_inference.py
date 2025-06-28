import numpy as np
import torch
from audio_utils import load_model_and_scaler, extract_features_from_path, defined_emotions

# ────────────── Config ──────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "notebooks/emotion_recognition_model_final.pth"
scaler_path = "model/feature_scaler.pkl"
file = "test_uploads/kaggle/angry/03-01-05-01-01-01-08.wav"  # angry sample

# ────────────── Load model & scaler ──────────────
try:
    model, scaler = load_model_and_scaler(
        model_path=model_path,
        scaler_path=scaler_path,
        device=device
    )
    print("✅ Model and scaler loaded.")
except Exception as e:
    print(f"❌ Failed to load model or scaler: {e}")
    exit(1)

# ────────────── Extract features ──────────────
try:
    features = extract_features_from_path(file, scaler, device=device, augment=False)
    if features.shape != (1, 1, 80, 130):
        raise ValueError(f"Invalid feature shape: {features.shape}")
    print("✅ Features extracted. Shape:", features.shape)
except Exception as e:
    print(f"❌ Failed to extract features: {e}")
    exit(1)

# ────────────── Inference ──────────────
model.eval()
with torch.no_grad():
    output = model(features)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    pred = defined_emotions[np.argmax(probs)]
    print(f"🎯 Predicted: {pred}")
    print("📊 Probabilities:", dict(zip(defined_emotions, np.round(probs, 3))))
