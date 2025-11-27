import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

# --- Yollar ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "StressLevelDataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURE_RANGES_PATH = os.path.join(MODELS_DIR, "feature_ranges.json")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)

    if "stress_level" not in df.columns:
        raise ValueError("Beklenen 'stress_level' sütunu veride yok.")

    # Hedef ve özellikleri ayır
    y = df["stress_level"]
    X = df.drop("stress_level", axis=1)

    # Sayısal olmayan kolonları sayısala çevir (gerekirse)
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Eksik değerleri 0 ile doldur (basit ama güvenli)
    X = X.fillna(0)

    # Sınıf dağılımını bir kez yazdır (kontrol için)
    print("=== Sınıf dağılımı (y) ===")
    print(y.value_counts(normalize=True).round(3))

    return X, y


def build_models():
    """
    Dengesiz sınıflar için class_weight='balanced' kullanan modeller.
    KNN'de class_weight yok, onu olduğu gibi bırakıyoruz.
    """
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        multi_class="multinomial",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced",
        ),
    }
    return models


def main():
    print("→ Veri yükleniyor...")
    X, y = load_data()
    feature_names = list(X.columns)

    # Her feature için min–max bilgilerini kaydet
    desc = X.describe()
    feature_ranges = {}
    for col in feature_names:
        col_stats = desc[col]
        feature_ranges[col] = {
            "min": float(col_stats["min"]),
            "max": float(col_stats["max"]),
        }

    with open(FEATURE_RANGES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_ranges, f, ensure_ascii=False, indent=2)
    print(f"Özellik min/max değerleri kaydedildi: {FEATURE_RANGES_PATH}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    results = []

    best_name = None
    best_f1 = -np.inf
    best_model = None

    for name, model in models.items():
        print(f"\n=== Model: {name} ===")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        print(f"Accuracy:   {acc:.4f}")
        print(f"Macro F1:   {macro_f1:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred))

        results.append(
            {
                "model": name,
                "accuracy": acc,
                "macro_f1": macro_f1,
            }
        )

        # "En iyi" modeli macro F1'e göre seç
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name
            best_model = model

        # Her modeli ayrı kaydet
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        # DİKKAT: yukarıdaki satırda süslü parantez hatası olmasın, doğru hali:
        # model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"Kaydedildi: {model_path}")

    # Metrikleri CSV'ye yaz
    metrics_df = pd.DataFrame(results)
    metrics_path = os.path.join(MODELS_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nTüm metrikler kaydedildi: {metrics_path}")

    # En iyi modeli ayrıca kaydet
    if best_model is not None:
        best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        joblib.dump(best_model, best_model_path)
        print(f"\nEn iyi model: {best_name} (macro_f1={best_f1:.4f})")
        print(f"En iyi model kaydedildi: {best_model_path}")

        # Özellik isimlerini kaydet (frontend/API için)
        feat_path = os.path.join(MODELS_DIR, "feature_names.json")
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)
        print(f"Özellik isimleri kaydedildi: {feat_path}")


if __name__ == "__main__":
    main()
