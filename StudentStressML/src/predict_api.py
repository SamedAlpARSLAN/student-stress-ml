import os
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# --- Yollar ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}. Önce train_model.py dosyasını çalıştır.")

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(f"Özellik listesi bulunamadı: {FEATURES_PATH}. Önce train_model.py dosyasını çalıştır.")

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_names = json.load(f)

# 0,1,2 -> isim (gerekirse değiştir)
STRESS_LABELS = {
    0: "low",
    1: "medium",
    2: "high",
}

app = Flask(__name__)


def make_input_dataframe(data: dict) -> pd.DataFrame:
    """
    data: JSON içinden gelen tek gözlem (dict).
    Eksik feature'lar 0 ile doldurulur.
    Fazla key'ler yok sayılır.
    """
    row = []
    for col in feature_names:
        row.append(data.get(col, 0))
    df = pd.DataFrame([row], columns=feature_names)
    return df


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "Student stress prediction API"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Beklenen JSON örneği:
    {
        "anxiety_level": 14,
        "self_esteem": 20,
        ...
    }
    Feature isimleri train_model.py ile kaydedilen feature_names.json ile aynı olmalı.
    """
    if not request.is_json:
        return jsonify({"error": "JSON body bekleniyor."}), 400

    data = request.get_json()
    df_input = make_input_dataframe(data)
    pred = model.predict(df_input)[0]

    # Olasılık varsa ekle
    proba = None
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(df_input)[0]
        proba = {
            int(cls): float(p)
            for cls, p in zip(sorted(np.unique(model.classes_)), proba_arr)
        }

    label = STRESS_LABELS.get(int(pred), str(pred))

    resp = {
        "predicted_class": int(pred),
        "predicted_label": label,
    }
    if proba is not None:
        resp["probabilities"] = proba

    return jsonify(resp)


if __name__ == "__main__":
    # Geliştirme için
    app.run(host="0.0.0.0", port=5000, debug=True)
