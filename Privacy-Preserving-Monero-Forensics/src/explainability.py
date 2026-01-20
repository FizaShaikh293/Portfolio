# src/explainability.py
# SHAP and LIME explainability for the trained Monero models.

import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


def run_shap_for_isolation_forest(model, X_scaled: np.ndarray, feature_names, out_dir: str):
    """
    Run SHAP TreeExplainer on the Isolation Forest model and save plots.
    """
    print("[+] Running SHAP for Isolation Forest...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    os.makedirs(out_dir, exist_ok=True)

    # Summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False)
    summary_path = os.path.join(out_dir, "shap_summary_beeswarm.png")
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] SHAP summary beeswarm saved to {summary_path}")

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, plot_type="bar", show=False)
    bar_path = os.path.join(out_dir, "shap_summary_bar.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()
    print(f"[✓] SHAP summary bar saved to {bar_path}")


def run_lime_for_instance(model, X_scaled: np.ndarray, feature_names, instance_index: int, out_dir: str):
    """
    Run LIME for a single instance of the Isolation Forest decision function.
    Saves an HTML file with the explanation.
    """
    print(f"[+] Running LIME for instance index {instance_index}...")

    explainer = LimeTabularExplainer(
        training_data=X_scaled,
        feature_names=feature_names,
        mode="regression"
    )

    def predict_fn(x):
        # Isolation Forest decision_function gives anomaly scores
        scores = model.decision_function(x)
        # LIME for regression expects 2D output
        return scores.reshape(-1, 1)

    explanation = explainer.explain_instance(
        data_row=X_scaled[instance_index],
        predict_fn=predict_fn
    )

    os.makedirs(out_dir, exist_ok=True)
    lime_path = os.path.join(out_dir, f"lime_instance_{instance_index}.html")
    explanation.save_to_file(lime_path)
    print(f"[✓] LIME explanation saved to {lime_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SHAP + LIME explainability for Monero models")
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default="data/processed/features_with_scores.csv",
        help="CSV with features and iso_score"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=False,
        default="models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=False,
        default="explainability",
        help="Directory to save SHAP/LIME outputs"
    )
    parser.add_argument(
        "--instance_index",
        type=int,
        required=False,
        default=0,
        help="Row index for LIME explanation"
    )

    args = parser.parse_args()

    print(f"[+] Loading feature data from {args.input}")
    df = pd.read_csv(args.input)

    feature_cols = ["iat", "amount_log", "ring_size", "txs_per_day", "hour"]
    X = df[feature_cols].values

    print(f"[+] Loading models from {args.model_dir}")
    iso_path = os.path.join(args.model_dir, "isolation_forest.pkl")
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")

    iso_model = joblib.load(iso_path)
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(df[feature_cols])

    # SHAP
    run_shap_for_isolation_forest(
        model=iso_model,
        X_scaled=X_scaled,
        feature_names=feature_cols,
        out_dir=args.out_dir
    )

    # LIME
    if args.instance_index < 0 or args.instance_index >= X_scaled.shape[0]:
        print("[!] instance_index out of range, defaulting to 0")
        idx = 0
    else:
        idx = args.instance_index

    run_lime_for_instance(
        model=iso_model,
        X_scaled=X_scaled,
        feature_names=feature_cols,
        instance_index=idx,
        out_dir=args.out_dir
    )

    print("[✓] Explainability (SHAP + LIME) complete.")
