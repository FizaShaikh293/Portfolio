# src/ml_module.py
# Fully repaired version with improved anomaly separation, feature scaling,
# score_samples(), and upgraded behavioural features.

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers


# --------------------------------------------------------------
# FEATURE AUGMENTATION (NEW FIX)
# --------------------------------------------------------------

def enhance_features(df):
    """Add extra behavioural features that improve anomaly separation."""

    df = df.copy()

    # Day of week
    df["day_of_week"] = pd.to_datetime(df["timestamp"], unit="s").dt.dayofweek

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Z-scores
    df["iat_z"] = (df["iat"] - df["iat"].mean()) / (df["iat"].std() + 1e-9)
    df["amount_z"] = (df["amount_log"] - df["amount_log"].mean()) / (df["amount_log"].std() + 1e-9)

    # Rolling behaviour signals (simple behavioural windows)
    df["rolling_iat_mean"] = df["iat"].rolling(window=20, min_periods=1).mean()
    df["rolling_amount_mean"] = df["amount_log"].rolling(window=20, min_periods=1).mean()

    return df


# --------------------------------------------------------------
# ISOLATION FOREST (FIXED)
# --------------------------------------------------------------

def train_isolation_forest(X: pd.DataFrame, random_state=42):
    """Train a stronger Isolation Forest using score_samples (correct scale)."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Auto-adjust contamination based on data variation
    contamination = 0.02 if X.std().mean() < 0.5 else 0.01

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        bootstrap=True,
        warm_start=True
    )

    model.fit(X_scaled)
    return model, scaler


# --------------------------------------------------------------
# AUTOENCODER
# --------------------------------------------------------------

def build_small_autoencoder(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(max(8, input_dim // 2), activation="relu")(inp)
    x = layers.Dense(max(4, input_dim // 4), activation="relu")(x)
    x = layers.Dense(max(8, input_dim // 2), activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear")(x)

    auto = models.Model(inp, out)
    auto.compile(optimizer="adam", loss="mse")
    return auto


def train_autoencoder(X, epochs=25, batch_size=128):
    auto = build_small_autoencoder(X.shape[1])
    auto.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=1)
    return auto


# --------------------------------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ML models on Monero features")
    parser.add_argument(
        "--input",
        default="data/processed/features_1500000_1530000.csv",
        help="Path to feature CSV"
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="Where to save trained models"
    )

    args = parser.parse_args()

    print(f"[+] Loading dataset: {args.input}")
    df = pd.read_csv(args.input)

    # Enhance features (fix)
    df = enhance_features(df)

    # Final feature set
    feature_cols = [
        "iat", "amount_log", "ring_size", "txs_per_day", "hour",
        "day_of_week", "hour_sin", "hour_cos",
        "iat_z", "amount_z",
        "rolling_iat_mean", "rolling_amount_mean"
    ]

    X = df[feature_cols].values

    os.makedirs(args.model_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Isolation Forest
    # ----------------------------------------------------------
    print("[+] Training Isolation Forest...")
    iso_model, scaler = train_isolation_forest(df[feature_cols])
    joblib.dump(iso_model, f"{args.model_dir}/isolation_forest.pkl")
    joblib.dump(scaler, f"{args.model_dir}/scaler.pkl")
    print("[✓] Isolation Forest saved.")

    # Compute anomaly scores correctly
    X_scaled = scaler.transform(df[feature_cols])
    iso_scores = iso_model.score_samples(X_scaled)  # ← FIXED

    # Normalize anomaly scale for stability
    iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)

    df["iso_score"] = iso_scores

    df.to_csv("data/processed/features_with_scores.csv", index=False)
    print("[✓] Anomaly scores written to data/processed/features_with_scores.csv")

    # ----------------------------------------------------------
    # Autoencoder
    # ----------------------------------------------------------
    print("[+] Training Autoencoder...")
    auto = train_autoencoder(X_scaled)
    auto.save(f"{args.model_dir}/autoencoder.h5")
    print("[✓] Autoencoder saved.")

    print("[✓] Training complete.")
