# 🚀 Enhanced Inference Script (Evaluate on saved val/test sets from train_model)

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from audio_utils import EmotionCNN2D, defined_emotions

# ─────────────── Config ───────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 PyTorch version: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

model_path = "notebooks/emotion_recognition_model_final.pth"
scaler_path = "model/feature_scaler.pkl"
val_data_path = "notebooks/X_val.npy"
val_labels_path = "notebooks/y_val.npy"
test_data_path = "notebooks/X_test.npy"
test_labels_path = "notebooks/y_test.npy"

# ─────────────── Load Model & Scaler ───────────────
try:
    model = EmotionCNN2D(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)
    print("✅ Loaded model and scaler.")
except Exception as e:
    print(f"❌ Failed to load model or scaler: {e}")
    exit()

# ─────────────── Load Datasets ───────────────
def load_set(data_path, labels_path):
    X = np.load(data_path)
    y = np.load(labels_path)
    return torch.tensor(X, dtype=torch.float32).to(device), y

# Validation set
X_val, y_val = load_set(val_data_path, val_labels_path)
# Test set
X_test, y_test = load_set(test_data_path, test_labels_path)

# ─────────────── Evaluate Function ───────────────
def evaluate(X, y, set_name="Set"):
    with torch.no_grad():
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    acc = (preds == y).mean()
    print(f"✅ {set_name} Accuracy: {acc:.2%}")
    print(f"{set_name} Classification Report:")
    print(classification_report(y, preds, target_names=defined_emotions, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=defined_emotions, yticklabels=defined_emotions)
    plt.title(f"{set_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"notebooks/{set_name.lower()}_confusion_matrix.png")
    plt.close()
    print(f"📸 Confusion matrix saved to notebooks/{set_name.lower()}_confusion_matrix.png\n")

# ─────────────── Evaluate on Val and Test ───────────────
evaluate(X_val, y_val, set_name="Validation")
evaluate(X_test, y_test, set_name="Test")
