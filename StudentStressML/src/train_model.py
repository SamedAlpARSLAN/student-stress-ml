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
from sklearn.metrics import accuracy_score, classification_report

# --- Yollar ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "StressLevelDataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    if "stress_level" not in df.columns:
        raise ValueError("Beklenen 'stress_level' sütunu veride yok.")
    X = df.drop("stress_level", axis=1)
    y = df["stress_level"]
    return X, y


def build_models():
    # 4 algoritma:
    # 1) Logistic Regression
    # 2) KNN
    # 3) SVM (RBF)
    # 4) Random Forest
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial"))
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5))
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True))
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
    }
    return models


def main():
    print("→ Veri yükleniyor...")
    X, y = load_data()
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models()
    results = []

    best_name = None
    best_acc = -np.inf
    best_model = None

    for name, model in models.items():
        print(f"\n=== Model: {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred))

        results.append({"model": name, "accuracy": acc})

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

        # Her modeli ayrı kaydet
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
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
        print(f"\nEn iyi model: {best_name} (acc={best_acc:.4f})")
        print(f"En iyi model kaydedildi: {best_model_path}")

        # Özellik isimlerini de kaydet (frontend/API için lazım olacak)
        feat_path = os.path.join(MODELS_DIR, "feature_names.json")
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)
        print(f"Özellik isimleri kaydedildi: {feat_path}")


if __name__ == "__main__":
    main()
