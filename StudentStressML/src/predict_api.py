import os
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# --- Yol ayarları ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model bulunamadı: {MODEL_PATH}. Önce train_model.py dosyasını çalıştır."
    )

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(
        f"Özellik listesi bulunamadı: {FEATURES_PATH}. Önce train_model.py dosyasını çalıştır."
    )

# Model ve özellik isimlerini yükle
model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_names = json.load(f)

# Flask uygulaması (templates klasörünü belirt)
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
)


def make_input_dataframe(data: dict) -> pd.DataFrame:
    """
    JSON'dan gelen tek gözlemi DataFrame'e çevir.
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


@app.route("/ui", methods=["GET"])
def ui():
    """
    Basit web arayüzü.
    feature_names -> index.html içinde form alanları olarak kullanılıyor.
    """
    return render_template("index.html", feature_names=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Beklenen JSON örneği:
    {
        "feature1": 1.2,
        "feature2": 3.4,
        ...
    }
    Anahtar isimleri feature_names.json ile aynı olmalı.
    """
    if not request.is_json:
        return jsonify({"error": "JSON body bekleniyor."}), 400

    data = request.get_json()
    df_input = make_input_dataframe(data)

    # Tahmin
    pred = model.predict(df_input)[0]

    # Olasılıklar (varsa)
    proba = None
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(df_input)[0]
        proba = {
            str(cls): float(p)
            for cls, p in zip(model.classes_, proba_arr)
        }

    resp = {
        "predicted_label": str(pred),
    }
    if proba is not None:
        resp["probabilities"] = proba

    return jsonify(resp)


if __name__ == "__main__":
    # Geliştirme için
    app.run(host="0.0.0.0", port=5000, debug=True)
