# 🚀 Unified Training + Evaluation Pipeline (run_all.py)

import os
import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import torch
import joblib
import seaborn as sns
from sklearn.metrics import confusion_matrix
from audio_utils import EmotionCNN2D, defined_emotions

# ─────────────── Paths ───────────────
project_dir = os.path.abspath(os.path.dirname(__file__))
notebooks_dir = os.path.join(project_dir, "notebooks")
model_path = os.path.join(notebooks_dir, "emotion_recognition_model_final.pth")
scaler_path = os.path.join(project_dir, "model", "feature_scaler.pkl")
feedback_script = os.path.join(project_dir, "retrain_from_feedback.py")
train_script = os.path.join(project_dir, "train_model.py")
summary_output = os.path.join(notebooks_dir, "summary.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 PyTorch version: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

# ─────────────── Step 1: Clean old artifacts ───────────────
print("🧹 Cleaning old model and metadata files...")
files_to_remove = [
    scaler_path,
    model_path,
    os.path.join(notebooks_dir, "train_history.npz"),
    os.path.join(notebooks_dir, "fine_tuning_history.npz"),
    os.path.join(notebooks_dir, "fine_tuning_classification_report.txt"),
    os.path.join(notebooks_dir, "confusion_matrix.png"),
    os.path.join(notebooks_dir, "fine_tuning_confusion_matrix.png"),
    os.path.join(notebooks_dir, "X_test.npy"),
    os.path.join(notebooks_dir, "y_test.npy"),
    os.path.join(notebooks_dir, "X_val.npy"),
    os.path.join(notebooks_dir, "y_val.npy"),
    summary_output
]
for file_path in files_to_remove:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑 Removed {file_path}")
    except Exception as e:
        print(f"⚠️ Failed to remove {file_path}: {e}")

# ─────────────── Step 2: Train base model ───────────────
print("🧠 Training base model...")
try:
    subprocess.run([sys.executable, train_script], check=True)
    print(f"✅ Completed training: {train_script}")
except subprocess.CalledProcessError as e:
    print(f"❌ Training script failed: {e}")
    exit(1)

# ─────────────── Step 3: Fine-tune model ───────────────
print("🔁 Fine-tuning with feedback...")
try:
    subprocess.run([sys.executable, feedback_script], check=True)
    print(f"✅ Completed fine-tuning: {feedback_script}")
except subprocess.CalledProcessError as e:
    print(f"⚠️ Fine-tuning script failed: {e}. Continuing...")

# ─────────────── Step 4: Skip external test uploads ───────────────
print("🚫 Skipping test uploads evaluation step (focus on dataset accuracy)")

# ─────────────── Step 5: Evaluate internal val/test sets ───────────────
print("✅ Evaluating model on internal validation and test sets...")
try:
    model = EmotionCNN2D(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"❌ Failed to load model or scaler: {e}")
    exit(1)

try:
    X_val = np.load(os.path.join(notebooks_dir, "X_val.npy"))
    y_val = np.load(os.path.join(notebooks_dir, "y_val.npy"))
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs_val = model(X_val_tensor)
        preds_val = torch.argmax(outputs_val, dim=1)
        val_accuracy = (preds_val == y_val_tensor).float().mean().item()
        conf_matrix_val = confusion_matrix(y_val_tensor.cpu().numpy(), preds_val.cpu().numpy())
    print(f"✅ Validation Accuracy: {val_accuracy:.2%}")
except Exception as e:
    print(f"⚠️ Failed to evaluate validation set: {e}")
    val_accuracy, conf_matrix_val = 0.0, None

try:
    X_test = np.load(os.path.join(notebooks_dir, "X_test.npy"))
    y_test = np.load(os.path.join(notebooks_dir, "y_test.npy"))
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        preds_test = torch.argmax(outputs_test, dim=1)
        test_accuracy = (preds_test == y_test_tensor).float().mean().item()
        conf_matrix_test = confusion_matrix(y_test_tensor.cpu().numpy(), preds_test.cpu().numpy())
    print(f"✅ Test Accuracy: {test_accuracy:.2%}")
except Exception as e:
    print(f"⚠️ Failed to evaluate test set: {e}")
    test_accuracy, conf_matrix_test = 0.0, None

# ─────────────── Step 6: Load histories ───────────────
try:
    history_data = np.load(os.path.join(notebooks_dir, "train_history.npz"))
    train_losses = history_data['train_losses']
    val_losses = history_data['val_losses']
except Exception as e:
    print(f"❌ Failed to load train_history.npz: {e}")
    train_losses, val_losses = [], []

try:
    fine_tune_data = np.load(os.path.join(notebooks_dir, "fine_tuning_history.npz"))
    fine_tune_train_losses = fine_tune_data['train_losses']
    fine_tune_val_losses = fine_tune_data['val_losses']
except Exception as e:
    print(f"⚠️ Failed to load fine_tuning_history.npz: {e}")
    fine_tune_train_losses, fine_tune_val_losses = [], []

# ─────────────── Step 7: Skip test uploads accuracy ───────────────
test_uploads_accuracy = 0.0
fine_tuning_accuracy = 0.0

# ─────────────── Step 8: Summary Visualization ───────────────
print("📊 Generating summary visualization...")
fig, axs = plt.subplots(3, 2, figsize=(14, 18))
fig.suptitle("Model Training and Evaluation Summary", fontsize=20)

if conf_matrix_val is not None:
    sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues', ax=axs[0, 0],
                xticklabels=defined_emotions, yticklabels=defined_emotions)
    axs[0, 0].set_title("Confusion Matrix on Validation Set")
else:
    axs[0, 0].axis('off')
    axs[0, 0].set_title("Validation Set Confusion Matrix (Not Available)")

if conf_matrix_test is not None:
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', ax=axs[0, 1],
                xticklabels=defined_emotions, yticklabels=defined_emotions)
    axs[0, 1].set_title("Confusion Matrix on Test Set")
else:
    axs[0, 1].axis('off')
    axs[0, 1].set_title("Test Set Confusion Matrix (Not Available)")

if train_losses and val_losses:
    axs[1, 0].plot(train_losses, label="Train Loss (Base)")
    axs[1, 0].plot(val_losses, label="Validation Loss (Base)")
    axs[1, 0].set_title("Base Training Loss over Epochs")
    axs[1, 0].legend()
else:
    axs[1, 0].axis('off')
    axs[1, 0].set_title("Base Training Loss (Not Available)")

if fine_tune_train_losses and fine_tune_val_losses:
    axs[1, 1].plot(fine_tune_train_losses, label="Train Loss (Fine-Tune)")
    axs[1, 1].plot(fine_tune_val_losses, label="Validation Loss (Fine-Tune)")
    axs[1, 1].set_title("Fine-Tuning Loss over Epochs")
    axs[1, 1].legend()
else:
    axs[1, 1].axis('off')
    axs[1, 1].set_title("Fine-Tuning Loss (Not Available)")

axs[2, 0].axis('off')
axs[2, 0].set_title("Test Uploads Conf Matrix (Skipped)")

summary_text = f"""
Validation Accuracy:      {val_accuracy:.2% if val_accuracy else 'N/A'}
Test Accuracy:           {test_accuracy:.2% if test_accuracy else 'N/A'}
"""
axs[2, 1].axis('off')
axs[2, 1].text(0.5, 0.5, summary_text.strip(), ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
try:
    plt.savefig(summary_output)
    print(f"✅ Summary saved to `{summary_output}`")
except Exception as e:
    print(f"❌ Failed to save summary visualization: {e}")
