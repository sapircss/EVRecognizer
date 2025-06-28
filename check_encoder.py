import joblib

encoder_path = "model/label_encoder.pkl"
encoder = joblib.load(encoder_path)

print("Classes in label encoder:")
print(list(encoder.classes_))
