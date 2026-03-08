"""
Chennai Traffic Accident Prediction Model
Ensemble: XGBoost + RandomForest with SMOTE oversampling
Run: python train_model.py
Outputs: models/accident_model.pkl, models/model_meta.json
"""

import os
import json
import pickle
import sys
import numpy as np
import pandas as pd

# ── Data generation (inline, no network needed) ─────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../data"))
from generate_dataset import generate_dataset

DATA_DIR   = os.path.join(os.path.dirname(__file__), "../data")
MODELS_DIR = os.path.dirname(__file__)

FEATURE_COLS = [
    "hour", "is_weekend",
    "temperature_c", "humidity_percent", "wind_speed_kmh",
    "visibility_km", "rainfall_mm",
    "congestion_percent", "avg_speed_kmh",
    "is_junction", "near_school", "flood_risk",
    "weather_encoded", "road_encoded", "time_encoded",
]

WEATHER_MAP = {
    "Clear": 0, "Drizzle": 1, "Light Rain": 2,
    "Heavy Rain": 3, "Fog": 4, "Thunderstorm": 5,
}
ROAD_MAP = {
    "Highway": 0, "Arterial": 1, "Urban": 2,
    "Coastal": 3, "Industrial": 4, "Flyover": 5,
}
TIME_MAP = {
    "Early Morning": 0, "Morning Peak": 1, "Afternoon": 2,
    "Night": 3, "Evening Peak": 4,
}


def load_or_generate():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chennai_accident_dataset.csv")
    if not os.path.exists(csv_path):
        print("Generating synthetic dataset …")
        df = generate_dataset()
    else:
        df = pd.read_csv(csv_path)
    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["weather_encoded"] = df["weather_condition"].map(WEATHER_MAP).fillna(0).astype(int)
    df["road_encoded"]    = df["road_type"].map(ROAD_MAP).fillna(2).astype(int)
    df["time_encoded"]    = df["time_period"].map(TIME_MAP).fillna(0).astype(int)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["accident_occurred"].values.astype(int)
    return X, y


def train():
    try:
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import classification_report, roc_auc_score
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run:  pip install xgboost scikit-learn pandas numpy")
        sys.exit(1)

    print("── Loading data ──")
    df = load_or_generate()
    X, y = preprocess(df)
    print(f"   Samples: {len(X)}  |  Positives: {y.sum()} ({y.mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale for RF
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── XGBoost ─────────────────────────────────────────────────────────────
    print("\n── Training XGBoost ──")
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)

    # ── Random Forest ────────────────────────────────────────────────────────
    print("── Training RandomForest ──")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train)

    # ── Soft-voting Ensemble ──────────────────────────────────────────────────
    print("── Building Ensemble ──")

    class ScaledRF:
        """Thin wrapper so RF uses scaled features in VotingClassifier."""
        def __init__(self, rf, scaler):
            self.rf = rf
            self.scaler = scaler
            self.classes_ = rf.classes_

        def predict_proba(self, X):
            return self.rf.predict_proba(self.scaler.transform(X))

        def predict(self, X):
            return self.rf.predict(self.scaler.transform(X))

        def fit(self, X, y):          # required by sklearn interface
            return self

    srf = ScaledRF(rf, scaler)

    # Manual ensemble (weighted average)
    xgb_prob  = xgb.predict_proba(X_test)[:, 1]
    rf_prob   = srf.predict_proba(X_test)[:, 1]
    ens_prob  = 0.55 * xgb_prob + 0.45 * rf_prob
    ens_pred  = (ens_prob >= 0.45).astype(int)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n── Evaluation (test set) ──")
    print(classification_report(y_test, ens_pred, target_names=["No Accident", "Accident"]))
    auc = roc_auc_score(y_test, ens_prob)
    print(f"ROC-AUC: {auc:.4f}")

    # Feature importances from XGBoost
    fi = dict(zip(FEATURE_COLS, xgb.feature_importances_.tolist()))

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_bundle = {
        "xgb":    xgb,
        "rf":     rf,
        "scaler": scaler,
        "feature_cols":  FEATURE_COLS,
        "weather_map":   WEATHER_MAP,
        "road_map":      ROAD_MAP,
        "time_map":      TIME_MAP,
    }
    pkl_path  = os.path.join(MODELS_DIR, "accident_model.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")

    with open(pkl_path, "wb") as f:
        pickle.dump(model_bundle, f)

    meta = {
        "roc_auc":       round(auc, 4),
        "feature_importances": fi,
        "n_train":       int(len(X_train)),
        "n_test":        int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
        "xgb_weight":    0.55,
        "rf_weight":     0.45,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Model saved  → {pkl_path}")
    print(f"✅ Meta saved   → {meta_path}")
    return model_bundle


if __name__ == "__main__":
    train()
