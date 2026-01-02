# app/streamlit_app.py
#
# Streamlit dashboard for privacy-preserving behavioural risk scoring
# on Monero transactions.
#
# The lookup pipeline is:
#   1) Try to find the transaction in the offline 30,001-tx dataset
#   2) If not present, try local monerod RPC (if running)
#   3) If that fails, fall back to public block explorers
#
# The user only sees the final result (risk score + classification +
# explanation). They do not see internal messages about where the
# transaction was found.

import os
import json
import requests
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------------------------------------------
# Basic configuration
# -------------------------------------------------------------------

# Paths relative to project root
FEATURES_CSV = "data/processed/features_with_scores.csv"
MODEL_DIR = "models"
ISO_MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Monerod RPC URL. If monerod is not running this will simply fail
# and the app will silently fall back to public explorers.
MONEROD_RPC_URL = "http://127.0.0.1:18081"


# -------------------------------------------------------------------
# Cached data / model loading helpers
# -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_feature_table() -> pd.DataFrame:
    """
    Load the offline feature dataset once and cache it.
    The table should already contain:
        txid, timestamp, iat, amount_log, ring_size,
        txs_per_day, hour, iso_score
    """
    df = pd.read_csv(FEATURES_CSV)

    # Make sure txid is treated as string
    df["txid"] = df["txid"].astype(str)

    # Defensive: ensure timestamp exists in a usable form
    if "timestamp" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "timestamp"})

    return df


@st.cache_resource(show_spinner=False)
def load_models_and_explainer():
    """
    Load the Isolation Forest model, scaler, and SHAP explainer once.

    The SHAP explainer is created on top of the Isolation Forest so that
    we can provide per-feature explanations for each transaction's
    anomaly score.
    """
    iso_model = joblib.load(ISO_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Build SHAP TreeExplainer for the Isolation Forest
    explainer = shap.TreeExplainer(iso_model)

    return iso_model, scaler, explainer


@st.cache_data(show_spinner=False)
def compute_global_threshold(df: pd.DataFrame, percentile: float = 5.0) -> float:
    """
    Compute a global anomaly threshold using the chosen percentile of
    Isolation Forest scores. Lower scores are more anomalous.
    """
    if "iso_score" not in df.columns:
        return float(df["iso_score"].median())  # fallback, should not happen

    return float(np.percentile(df["iso_score"].values, percentile))


# -------------------------------------------------------------------
# Monerod RPC helper (best effort only)
# -------------------------------------------------------------------

def fetch_tx_from_monerod(txid: str) -> Optional[Dict[str, Any]]:
    """
    Try to fetch a transaction from a local monerod instance.
    This function is deliberately simple and defensive. If anything
    goes wrong (node offline, timeout, unexpected response format),
    we just return None so that the caller can try other options.

    We use the /get_transactions RPC endpoint.
    """
    try:
        url = f"{MONEROD_RPC_URL}/get_transactions"
        payload = {
            "txs_hashes": [txid],
            "decode_as_json": True,
            "prune": False
        }

        resp = requests.post(url, json=payload, timeout=6)
        resp.raise_for_status()
        data = resp.json()

        # Basic sanity checks
        if "txs" not in data or not data["txs"]:
            return None

        tx_entry = data["txs"][0]

        # Some daemons return "in_pool" entries for mempool txs,
        # others may require "block_timestamp" for confirmed txs.
        # We handle both in a very gentle way.
        timestamp = tx_entry.get("block_timestamp") or tx_entry.get("timestamp")

        # tx JSON can be retrieved as string; sometimes nested values
        # must be parsed again. Here we just keep it simple and
        # only use fields we can reasonably access.
        ring_size = tx_entry.get("ring_size", None)

        # Amount is not directly visible due to RingCT — we treat amount
        # as 0 in a privacy-preserving way. The model is still able to
        # use other features (timing, ring size, frequency).
        amount = 0.0

        return {
            "timestamp": timestamp,
            "ring_size": ring_size,
            "amount": amount,
        }

    except Exception:
        return None


# -------------------------------------------------------------------
# Public explorer fallback helper (best effort only)
# -------------------------------------------------------------------

def fetch_tx_from_explorer(txid: str) -> Optional[Dict[str, Any]]:
    """
    Fallback when monerod is not available.
    We query public Monero explorers. Since these APIs are not
    standardized and may change, this is written as a best-effort,
    very defensive function.

    If we can extract a timestamp and ring size, that is already
    enough for the model to produce a sensible risk score.
    """

    explorer_urls = [
        # xmrchain.net API (example; actual path may differ)
        f"https://xmrchain.net/api/transaction/{txid}",
        # Monero.com / Cake Wallet explorer API (example)
        f"https://monero.com/api/transaction/{txid}",
    ]

    for url in explorer_urls:
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                continue

            data = resp.json()

            # Different explorers may wrap the tx in a "tx" object.
            tx_data = data.get("tx", data)

            timestamp = tx_data.get("timestamp")
            ring_size = tx_data.get("ring_size", None)

            # Again, RingCT hides the actual amount; we keep this 0.0
            amount = tx_data.get("amount", 0.0)

            if timestamp is None:
                # If timestamp is missing we cannot build time-based features.
                # Skip this explorer and try the next one.
                continue

            return {
                "timestamp": timestamp,
                "ring_size": ring_size,
                "amount": amount,
            }

        except Exception:
            # Network error, invalid JSON, etc. — we simply try the next one.
            continue

    return None


# -------------------------------------------------------------------
# Feature construction for live transactions
# -------------------------------------------------------------------

FEATURE_COLUMNS = ["iat", "amount_log", "ring_size", "txs_per_day", "hour"]


def build_features_for_offline_row(row: pd.Series) -> pd.DataFrame:
    """
    Take a row from the offline dataset and reshape it into the feature
    DataFrame expected by the model.
    """
    return pd.DataFrame([{
        "iat": float(row["iat"]),
        "amount_log": float(row["amount_log"]),
        "ring_size": float(row["ring_size"]),
        "txs_per_day": float(row["txs_per_day"]),
        "hour": float(row["hour"]),
    }])


def build_features_for_live_tx(tx: Dict[str, Any],
                               reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct feature values for a transaction that was fetched either
    from monerod or from a public explorer.

    We do not have the full chain of historical transactions here,
    so some values (like inter-arrival time and txs_per_day) are
    approximated in a simple and privacy-preserving way.
    """

    ts = tx.get("timestamp")
    if ts is None:
        # Fall back to median timestamp of the dataset
        ts = reference_df["timestamp"].median()

    # Normalise timestamp to integer seconds
    try:
        ts_int = int(ts)
    except Exception:
        ts_int = int(reference_df["timestamp"].median())

    # Hour-of-day in local time
    hour = datetime.utcfromtimestamp(ts_int).hour

    # Ring size and amount (amount is 0 because of RingCT)
    ring_size = tx.get("ring_size") or reference_df["ring_size"].median()
    amount = tx.get("amount", 0.0)
    amount_log = float(np.log1p(amount))

    # For live txs we do not have exact inter-arrival time or per-day count,
    # so we use simple, safe defaults based on the dataset median.
    iat_default = float(reference_df["iat"].median())
    txs_per_day_default = float(reference_df["txs_per_day"].median())

    return pd.DataFrame([{
        "iat": iat_default,
        "amount_log": amount_log,
        "ring_size": float(ring_size),
        "txs_per_day": txs_per_day_default,
        "hour": float(hour),
    }])


# -------------------------------------------------------------------
# Risk scoring and SHAP explanation
# -------------------------------------------------------------------

def compute_risk_score_and_explanation(
    features: pd.DataFrame,
    iso_model,
    scaler,
    explainer,
    threshold: float
) -> Tuple[float, str, pd.DataFrame]:
    """
    Given a 1-row features DataFrame, compute:
        - Isolation Forest anomaly score
        - Human-readable classification string
        - SHAP local explanation as a small DataFrame

    Returns:
        score, label_text, shap_df
    """

    # Scale using the same StandardScaler as during training
    X_scaled = scaler.transform(features[FEATURE_COLUMNS])

    # Isolation Forest decision function: higher = more normal,
    # lower = more anomalous
    score = float(iso_model.decision_function(X_scaled)[0])

    # Classification relative to global threshold
    if score < threshold:
        label = "Suspicious / anomalous"
    else:
        label = "Normal"

    # SHAP values for the single sample
    shap_values = explainer.shap_values(X_scaled)

    # TreeExplainer may return a list for multi-output models.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_row = shap_values[0]

    shap_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "shap_value": shap_row,
        "abs_value": np.abs(shap_row),
    }).sort_values("abs_value", ascending=False)

    return score, label, shap_df


def plot_feature_importance(shap_df: pd.DataFrame):
    """
    Simple horizontal bar chart of |SHAP| values to show which features
    were most important for this particular transaction.
    All text is forced to black for readability.
    """

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.barh(shap_df["feature"], shap_df["abs_value"])
    ax.invert_yaxis()

    ax.set_xlabel("Feature impact (|SHAP value|)", color="black")
    ax.set_ylabel("Feature", color="black")
    ax.set_title("Feature importance for this transaction", color="black")

    # Make tick labels black as well
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")

    fig.tight_layout()
    return fig


# -------------------------------------------------------------------
# Streamlit layout
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Monero Behavioural Risk Scoring",
        page_icon="🕵️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Main title and short description
    st.title("Monero Behavioural Risk Scoring Dashboard")
    st.write(
        "This dashboard uses a privacy-preserving behavioural model "
        "to assign a risk score to Monero transactions. "
        "The model is trained on a 30,001-transaction window from the "
        "Monero blockchain and supports both offline and live lookups."
    )

    # Load dataset and models up front
    df = load_feature_table()
    iso_model, scaler, explainer = load_models_and_explainer()
    global_threshold = compute_global_threshold(df, percentile=5.0)

    # Quick summary metrics at the top
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total transactions in offline window", f"{len(df):,}")

    with col2:
        try:
            t_min = datetime.utcfromtimestamp(int(df["timestamp"].min()))
            t_max = datetime.utcfromtimestamp(int(df["timestamp"].max()))
            st.metric(
                "Time range covered (offline dataset)",
                f"{t_min.date()} → {t_max.date()}",
            )
        except Exception:
            st.metric(
                "Time range covered (offline dataset)",
                "Not available",
            )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Transaction lookup section
    # ------------------------------------------------------------------

    st.subheader("Transaction Risk Lookup")

    st.write(
        "Paste a Monero transaction hash (txid) below. "
        "The system will search the offline dataset first, "
        "and if necessary retrieve live data from the blockchain. "
        "You only see the final risk score and explanation."
    )

    txid_input = st.text_input(
        "Enter transaction ID (txid)",
        value="",
        placeholder="e.g. 73a90b7c091d1db41e3d5f54b3e4da3c8e32eb4f53be1bbd5e5d2aa7af6bba09",
    ).strip()

    if st.button("Lookup"):
        if not txid_input:
            st.warning("Please enter a transaction ID.")
            return

        # ------------------ 1) Try offline dataset -------------------
        tx_row = None
        if txid_input in df["txid"].values:
            tx_row = df[df["txid"] == txid_input].iloc[0]
            features = build_features_for_offline_row(tx_row)
            source = "offline"

        else:
            # -------------- 2) Try monerod RPC (best effort) ----------
            tx_live = fetch_tx_from_monerod(txid_input)

            if tx_live is None:
                # ---------- 3) Try public explorers as fallback -------
                tx_live = fetch_tx_from_explorer(txid_input)

            if tx_live is None:
                st.error(
                    "This transaction could not be retrieved from the "
                    "offline dataset, local node, or public explorers."
                )
                return

            features = build_features_for_live_tx(tx_live, df)
            source = "live"

        # ------------------------------------------------------------------
        # Compute risk score and explanation
        # ------------------------------------------------------------------
        score, label, shap_df = compute_risk_score_and_explanation(
            features, iso_model, scaler, explainer, global_threshold
        )

        # ------------------------------------------------------------------
        # Present results
        # ------------------------------------------------------------------

        st.markdown("### Risk result")

        # Risk score and classification in one compact block
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                label="Risk score (Isolation Forest)",
                value=f"{score:.4f}",
            )
        with col_b:
            st.metric(
                label="Classification",
                value=label,
            )

        # Short natural-language explanation of the score direction
        if score < global_threshold:
            st.info(
                f"The anomaly score **{score:.4f}** is below the global "
                f"5th percentile threshold ({global_threshold:.4f}), "
                "so this transaction is treated as behaviourally unusual."
            )
        else:
            st.success(
                f"The anomaly score **{score:.4f}** is above the global "
                f"5th percentile threshold ({global_threshold:.4f}), "
                "so this transaction is treated as behaviourally normal "
                "relative to the training window."
            )

        # Show the features actually fed into the model
        st.markdown("### Behavioural features used for this transaction")
        st.dataframe(features[FEATURE_COLUMNS], use_container_width=True)

        # SHAP-based feature importance
        st.markdown("### Feature importance for this transaction")
        fig = plot_feature_importance(shap_df)
        st.pyplot(fig)

        # Textual explanation from SHAP values
        st.markdown("### Explanation of key behavioural drivers")

        bullets = []
        for _, row in shap_df.head(5).iterrows():
            feat = row["feature"]
            val = float(row["shap_value"])
            direction = "higher" if val > 0 else "lower"
            importance = "strongly" if abs(val) > shap_df["abs_value"].mean() else "slightly"

            bullets.append(
                f"- **{feat}** {importance} pushed the score towards "
                f"being **{direction} (more normal)**."
                if val > 0
                else f"- **{feat}** {importance} pushed the score towards "
                     f"being **lower (more anomalous)**."
            )

        st.markdown("\n".join(bullets))


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
