# ✅ Fine-Tune Emotion CNN on User Feedback (Fixed for 5 Emotions)

import sys
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent folder to path to import audio_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_utils import extract_features_from_path, EmotionCNN2D, defined_emotions

# ─────────────── Config ───────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_root = os.path.abspath(os.path.dirname(__file__))
feedback_path = os.path.join(project_root, "feedback_data.csv")
feedback_audio_dir = os.path.join(project_root, "feedback_audios")
scaler_path = os.path.join(project_root, "..", "model", "feature_scaler.pkl")
label_encoder_path = os.path.join(project_root, "..", "model", "label_encoder.pkl")
model_path = os.path.join(project_root, "..", "notebooks", "emotion_recognition_model_final.pth")
notebooks_dir = os.path.join(project_root, "..", "notebooks")

os.makedirs(notebooks_dir, exist_ok=True)

# ───── Load Feedback Data ─────
if not os.path.exists(feedback_path):
    print(f"❌ No feedback data found at {feedback_path}.")
    sys.exit(1)

try:
    feedback_df = pd.read_csv(feedback_path)
    if 'filename' not in feedback_df.columns:
        feedback_df.columns = ["filename", "prediction", "label"]
    feedback_df = feedback_df[feedback_df['label'].isin(defined_emotions)]
except Exception as e:
    print(f"❌ Error loading feedback data: {e}")
    sys.exit(1)

if feedback_df.empty:
    print("❌ No valid labeled feedback samples.")
    sys.exit(1)

print("📄 Feedback samples found:", len(feedback_df))

# ───── Load scaler and encoder ─────
if not os.path.exists(scaler_path):
    print(f"❌ Scaler not found at {scaler_path}")
    sys.exit(1)
if not os.path.exists(label_encoder_path):
    print(f"❌ Label encoder not found at {label_encoder_path}")
    sys.exit(1)

scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

# ───── Extract Features ─────
X, y = [], []
processed_count = 0

for _, row in feedback_df.iterrows():
    path = os.path.join(feedback_audio_dir, row['filename'])
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        continue
    try:
        features = extract_features_from_path(path, scaler=scaler, device=device, augment=True)
        X.append(features.cpu().numpy())
        y.append(row['label'])
        print(f"✅ Processed {path}")
        processed_count += 1
    except Exception as e:
        print(f"⚠️ Failed to process {path}: {e}")

if not X:
    print("❌ No valid feedback audio samples processed.")
    sys.exit(1)

X = np.array(X)
y = np.array(y)

# ───── Encode Labels ─────
if not all(lbl in label_encoder.classes_ for lbl in np.unique(y)):
    print(f"❌ Some feedback labels not in model classes: {np.unique(y)} vs {label_encoder.classes_}")
    sys.exit(1)

y_encoded = label_encoder.transform(y)

# ───── Reshape ─────
if len(X.shape) == 5:  # (batch, 1, 1, 80, 130)
    X = X.squeeze(1)
elif len(X.shape) == 3:  # (batch, 80, 130)
    X = np.expand_dims(X, axis=1)
elif len(X.shape) != 4:
    print(f"❌ Unexpected X shape: {X.shape}")
    sys.exit(1)

# ───── Split ─────
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

# ───── Load Model ─────
try:
    model = EmotionCNN2D(num_classes=len(label_encoder.classes_)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.train()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# ───── Fine-tune ─────
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train.cpu()), y=y_train.cpu().numpy())
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)  # 🔥 lowered LR to reduce overfitting
epochs = 5  # 🔥 reduced epochs for small feedback

batch_size = 4
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True
)

print("🎯 Fine-tuning on feedback...")
train_losses, val_losses = [], []
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    train_losses.append(avg_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved best model to `{model_path}`")

    if len(val_losses) > 2 and val_loss > np.mean(val_losses[-2:]):
        print("⏹ Early stopping triggered.")
        break

# Save history
history_path = os.path.join(notebooks_dir, "fine_tuning_history.npz")
np.savez(history_path, train_losses=np.array(train_losses), val_losses=np.array(val_losses))
print(f"📊 Training history saved to {history_path}")

# ───── Evaluation ─────
model.eval()
with torch.no_grad():
    preds = model(X_val)
    preds = torch.argmax(preds, dim=1).cpu().numpy()

acc = accuracy_score(y_val.cpu(), preds)
print(f"✅ Validation Accuracy: {acc:.2%}")

# Fix for missing classes
unique_classes = np.unique(y_val.cpu().numpy())
idx_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}
present_labels = [idx_to_label[i] for i in unique_classes]

report = classification_report(
    y_val.cpu(),
    preds,
    labels=unique_classes,
    target_names=present_labels,
    zero_division=0
)
print("Validation Classification Report:\n", report)

# Save report
report_path = os.path.join(notebooks_dir, "fine_tuning_classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"📄 Classification report saved to `{report_path}`")
