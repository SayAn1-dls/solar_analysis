import joblib
import os

model = joblib.load("models/solar_model.pkl")
joblib.dump(model, "models/solar_model.pkl", compress=("zlib", 3))
print("Compressed model size:", os.path.getsize("models/solar_model.pkl") / 1024**2, "MB")
