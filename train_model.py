import os
import random
import warnings
import joblib
import torch
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import kagglehub
from datasets import load_dataset

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ─────── Shared Noise Function ───────
def add_noise(y, sr, dir="data/background_noises"):
    noises = [f for f in os.listdir(dir) if f.endswith(".wav")]
    if not noises:
        return y
    n, _ = librosa.load(os.path.join(dir, random.choice(noises)), sr=sr)
    if len(n) < len(y):
        n = np.tile(n, int(np.ceil(len(y) / len(n))))
    n = n[:len(y)]
    snr = random.uniform(5, 20)
    return librosa.util.normalize(y + np.sqrt(np.mean(y**2) / (10**(snr/10)) / (np.mean(n**2)+1e-6)) * n)

# ─────── Feature Extraction ───────
def extract_features(path, augment=False):
    y, sr = librosa.load(path, sr=22050, duration=3, offset=0.5)
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)
    if augment:
        y = add_noise(y, sr)
        if random.random() < 0.3:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.choice([-2, 2]))
        if random.random() < 0.3:
            y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
    y = np.pad(y, (0, max(0, sr * 3 - len(y))))[:sr * 3]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512, win_length=1024)
    delta = librosa.feature.delta(mfcc)
    combined = librosa.util.fix_length(np.concatenate((mfcc, delta), axis=0), size=130, axis=1)
    return combined

# ─────── Main Execution Block ───────
if __name__ == "__main__":
    print(f"💪 PyTorch version: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("model", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)
    os.makedirs("data/background_noises", exist_ok=True)

    # ─────── Load and prepare datasets ───────
    print("\n📅 Loading RAVDESS...")
    ravdess_path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    ravdess_dir = os.path.expanduser(ravdess_path)
    emotion_map = {'02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful'}
    ravdess = [
        {'source': os.path.join(root, file), 'label': emotion_map[file.split('-')[2]]}
        for root, _, files in os.walk(ravdess_dir)
        for file in files if file.endswith('.wav') and file.split('-')[2] in emotion_map
    ]
    print(f"🔹 RAVDESS: {len(ravdess)} samples")

    print("📅 Loading TESS...")
    tess_ds = load_dataset("AbstractTTS/TESS", split="train")
    tess = [
        {'source': r['audio']['path'], 'label': {'angry': 'angry', 'fear': 'fearful', 'happy': 'happy', 'sad': 'sad', 'neutral': 'calm'}[r['emotion']]}
        for r in tess_ds if r['emotion'] in ['angry', 'fear', 'happy', 'sad', 'neutral']
    ]
    print(f"🔹 TESS: {len(tess)} samples")

    print("📅 Loading feedback...")
    feedback = []
    if os.path.exists("feedback/feedback_data.csv"):
        fb_df = pd.read_csv("feedback/feedback_data.csv")
        fb_df = fb_df[fb_df['label'].isin(emotion_map.values())]
        fb_df['source'] = fb_df['filename'].apply(lambda f: os.path.join("feedback/feedback_audios", f))
        feedback = fb_df[['source', 'label']].to_dict(orient='records')
    print(f"🔹 Feedback: {len(feedback)} samples")

    all_data = pd.DataFrame(ravdess + tess + feedback)
    label_encoder = LabelEncoder().fit(['happy', 'sad', 'angry', 'fearful', 'calm'])
    all_data['label_id'] = label_encoder.transform(all_data['label'])
    joblib.dump(label_encoder, "model/label_encoder.pkl")
    print("✅ Saved label_encoder to model/label_encoder.pkl")
    print(f"📊 Total samples: {len(all_data)}")

    # ─────── Extract Features ───────
    X, y = [], []
    print(f"🎷 Extracting features...")
    for _, row in all_data.iterrows():
        path, label = row['source'], row['label_id']
        if not os.path.exists(path):
            continue
        try:
            X.append(extract_features(path, augment=False))
            X.append(extract_features(path, augment=True))
            y.extend([label, label])
        except Exception as e:
            print(f"❌ {path}: {e}")

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "model/feature_scaler.pkl")
    print("✅ Saved feature_scaler to model/feature_scaler.pkl")

    X_scaled = X_scaled.reshape(-1, 1, 80, 130)

    # ─────── Split Data ───────
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    np.save("notebooks/X_test.npy", X_test.cpu().numpy())
    np.save("notebooks/y_test.npy", y_test.cpu().numpy())
    np.save("notebooks/X_val.npy", X_val.cpu().numpy())
    np.save("notebooks/y_val.npy", y_val.cpu().numpy())

    # ─────── Model ───────
    class EmotionCNN2D(nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(128 * 10 * 16, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)
            return self.fc(self.dropout(x))

    model = EmotionCNN2D(num_classes=5).to(device)

    # ─────── Training ───────
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train.cpu()), y=y_train.cpu().numpy())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(100):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        train_losses.append(total_loss / len(loader))
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train={train_losses[-1]:.4f} | Val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "notebooks/emotion_recognition_model_final.pth")

        if len(val_losses) > 10 and val_loss > np.mean(val_losses[-10:]):
            print("⏹ Early stopping triggered.")
            break

    np.savez("notebooks/train_history.npz", train_losses=np.array(train_losses), val_losses=np.array(val_losses))

    # ─────── Evaluate ───────
    model.eval()
    for split_name, X_eval, y_eval in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        with torch.no_grad():
            preds = torch.argmax(model(X_eval), dim=1).cpu().numpy()
            acc = (preds == y_eval.cpu().numpy()).mean()
            print(f"✅ {split_name} Accuracy: {acc:.2%}")
            print(f"{split_name} Classification Report:")
            print(classification_report(y_eval.cpu(), preds, target_names=label_encoder.classes_))
            cm = confusion_matrix(y_eval.cpu(), preds)
            ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot()
            plt.title(f"{split_name} Confusion Matrix")
            plt.savefig(f"notebooks/{split_name.lower()}_confusion_matrix.png")
            plt.close()

