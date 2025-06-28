# 🚨 Emotion Monitoring App (Final, urgent sorting updated on feedback)

import streamlit as st
st.set_page_config(layout="wide")

import numpy as np
import pandas as pd
import torch
import io, os, uuid
import joblib
from audio_utils import load_model_and_scaler, extract_features_from_bytes

# ────────── Styling ──────────
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stButton>button { padding: 0.2rem 0.5rem; font-size: 0.8rem; }
    .stFileUploader { margin-bottom: 0; }
    </style>
""", unsafe_allow_html=True)

# ────────── Load Model + Scaler ──────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model, scaler = load_model_and_scaler(
        model_path="notebooks/emotion_recognition_model_final.pth",
        scaler_path="model/feature_scaler.pkl",
        device=device
    )
    st.success("✅ Model and scaler loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model or scaler: {e}")
    st.stop()

# ────────── Load Label Encoder ──────────
try:
    label_encoder = joblib.load("model/label_encoder.pkl")
    emotion_classes = list(label_encoder.classes_)
    emotion_map = {i: emo for i, emo in enumerate(emotion_classes)}
except Exception as e:
    st.error(f"❌ Failed to load label encoder: {e}")
    st.stop()

urgent_emotions = {'angry', 'sad', 'fearful'}
feedback_path = "feedback/feedback_data.csv"
feedback_dir = "feedback/feedback_audios"
os.makedirs(feedback_dir, exist_ok=True)

# ───── Instructions ─────
st.info("Upload a clear 3-second speech audio (WAV preferred). Avoid background noise. Use the same style as training data.")

augment_uploads = st.checkbox("Apply slight augmentation to uploaded audio (experimental)", value=False)

if 'employees' not in st.session_state:
    st.session_state.employees = {}

# ────────── UI ──────────
st.title("📞 Emotion Monitoring Dashboard (Training-aligned)")

# ───── Add Employee ─────
with st.expander("➕ Add Another Employee"):
    new_name = st.text_input("Employee Name")
    if st.button("Add Employee") and new_name.strip():
        if new_name not in st.session_state.employees:
            st.session_state.employees[new_name] = {"urgent": False}
            st.rerun()
        else:
            st.warning("⚠️ Employee already exists.")

# ───── Urgent Calls Summary ─────
urgent_summary = [
    (name, emp["emotion"], emp["confidence"])
    for name, emp in st.session_state.employees.items()
    if emp.get("urgent")
]

st.markdown("## 🧠 Urgent Call Summary")
if urgent_summary:
    st.error("🚨 Urgent Calls:")
    for n, e, c in urgent_summary:
        st.markdown(f"- **{n}**: {e.upper()} ({c:.1f}%)")
else:
    st.success("✅ No urgent calls at the moment.")


# ───── Call Monitoring ─────
st.markdown("## 👥 Call Monitoring Dashboard")

if not st.session_state.employees:
    st.info("No employees yet.")
else:
    # Sort: urgent first
    sorted_names = sorted(
        st.session_state.employees.keys(),
        key=lambda name: not st.session_state.employees[name].get("urgent", False)
    )

    for name in sorted_names:
        emp = st.session_state.employees[name]
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 3, 3, 3, 2])

            # ─── Reset
            with col1:
                if st.button(f"🗑", key=f"reset_{name}"):
                    del st.session_state.employees[name]
                    st.rerun()

            # ─── Upload + Predict
            with col2:
                st.markdown(f"**👤 {name}**")
                audio = st.file_uploader("Upload audio file", type=["wav", "mp3"], key=f"upload_{name}")
                if audio:
                    try:
                        audio_bytes = audio.read()
                        features = extract_features_from_bytes(audio_bytes, scaler, device, augment=augment_uploads)

                        if features.shape != (1, 1, 80, 130):
                            st.error(f"⚠️ Feature shape mismatch: {features.shape}")
                            continue

                        with torch.no_grad():
                            output = model(features)
                            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

                        pred_idx = np.argmax(probs)
                        pred = emotion_map[pred_idx]
                        confidence = probs[pred_idx] * 100
                        urgent = pred in urgent_emotions

                        temp_data = {
                            "audio": audio_bytes,
                            "emotion": pred,
                            "confidence": confidence,
                            "urgent": urgent,
                            "probs": probs
                        }
                        st.session_state.employees[name] = temp_data
                        st.success(f"✅ Prediction: {pred} ({confidence:.1f}%)")

                        st.json({emo: f"{prob*100:.1f}%" for emo, prob in zip(emotion_classes, probs)})

                    except Exception as e:
                        st.error(f"⚠️ Error processing audio for {name}: {e}")
                else:
                    temp_data = emp

            # ─── Prediction + Playback
            with col3:
                if temp_data.get("emotion"):
                    dot = "🔴" if temp_data.get("urgent") else "🟢"
                    st.markdown(f"{dot} **{temp_data['emotion']}** ({temp_data['confidence']:.1f}%)")
                    st.audio(temp_data["audio"])

            # ─── Feedback
            # ─── Feedback
            with col4:
                if temp_data.get("audio"):
                    correct_label = st.selectbox("Correct emotion", emotion_classes, key=f"correct_{name}")
                    feedback_file = f"feedback_{name}_{str(uuid.uuid4())[:8]}.wav"
                    feedback_path_full = os.path.join(feedback_dir, feedback_file)

                    # Check if we previously saved feedback to show success after rerun
                    if st.session_state.get(f"feedback_saved_{name}"):
                        st.success("✅ Feedback saved")
                        # Reset so it does not stay forever on next rerun
                        st.session_state[f"feedback_saved_{name}"] = False

                    if st.button("📤 Submit", key=f"submit_{name}"):
                        try:
                            with open(feedback_path_full, "wb") as f:
                                f.write(temp_data["audio"])
                            header = not os.path.exists(feedback_path) or os.path.getsize(feedback_path) == 0
                            pd.DataFrame([[feedback_file, temp_data["emotion"], correct_label]],
                                         columns=["filename", "prediction", "label"]).to_csv(
                                feedback_path, mode='a', header=header, index=False)

                            # ✅ Save "saved" flag to session state so it persists after rerun
                            st.session_state[f"feedback_saved_{name}"] = True

                            # ✅ Update urgent status based on corrected label
                            st.session_state.employees[name]["urgent"] = correct_label in urgent_emotions
                            st.rerun()  # Refresh UI immediately after saving

                        except Exception as e:
                            st.error(f"⚠️ Error saving feedback for {name}: {e}")

            with col5:
                st.empty()
