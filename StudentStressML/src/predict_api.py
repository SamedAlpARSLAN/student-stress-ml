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
RANGES_PATH = os.path.join(MODELS_DIR, "feature_ranges.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model bulunamadı: {MODEL_PATH}. Önce train_model.py dosyasını çalıştır."
    )

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(
        f"Özellik listesi bulunamadı: {FEATURES_PATH}. Önce train_model.py dosyasını çalıştır."
    )

if not os.path.exists(RANGES_PATH):
    raise FileNotFoundError(
        f"Özellik aralıkları bulunamadı: {RANGES_PATH}. Önce train_model.py dosyasını çalıştır."
    )

# Model ve metadata'yı yükle
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_names = json.load(f)

with open(RANGES_PATH, "r", encoding="utf-8") as f:
    feature_ranges = json.load(f)

# Flask uygulaması (templates klasörünü belirt)
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
)


def scale_ui_to_dataset(data: dict) -> dict:
    """
    Arayüzden gelen 0–100 değerlerini, veri setindeki gerçek min–max
    aralıklarına ölçekler.
    """
    scaled = {}

    for col in feature_names:
        raw_val = data.get(col, 0)

        try:
            v = float(raw_val)
        except (TypeError, ValueError):
            v = 0.0

        # 0–100 aralığına sıkıştır
        if v < 0:
            v = 0.0
        if v > 100:
            v = 100.0

        ranges = feature_ranges.get(col)
        if ranges is not None:
            min_v = float(ranges.get("min", 0.0))
            max_v = float(ranges.get("max", 1.0))
            if max_v > min_v:
                # 0–100'den [min,max] aralığına lineer ölçekleme
                v = min_v + (max_v - min_v) * (v / 100.0)

        scaled[col] = v

    return scaled


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

    # 0–100'leri veri setinin aralığına ölçekle
    scaled_data = scale_ui_to_dataset(data)
    df_input = make_input_dataframe(scaled_data)

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
