# src/feature_engineering.py
# Convert raw transaction-level data into privacy-preserving behavioural features.

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from privacy_utils import apply_dp_to_series


def create_basic_features(tx_df: pd.DataFrame, dp_epsilon: float = 1.0) -> pd.DataFrame:
    """
    Produce behavioural features from Monero transaction data.

    Expected columns:
        - txid
        - timestamp (UNIX seconds)
        - amount
        - ring_size
        - outputs_count
    """

    df = tx_df.copy()

    # Ensure timestamps are numeric UNIX seconds
    if not np.issubdtype(df["timestamp"].dtype, np.number):
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9

    # Sort chronologically
    df = df.sort_values("timestamp")

    # Inter-arrival time
    df["iat"] = df["timestamp"].diff().fillna(0.0)

    # Convert timestamp to datetime for rolling window calculations
    df["ts_dt"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("ts_dt")

    # Rolling tx frequency per 24-hour window
    df["txs_per_day"] = df["txid"].rolling("1D").count().fillna(0)
    df = df.reset_index(drop=True)

    # Log amount transform
    df["amount_log"] = np.log1p(df["amount"])

    # Convert ring size
    df["ring_size"] = df["ring_size"].astype(int)

    # Hour of day feature
    df["hour"] = pd.to_datetime(df["timestamp"], unit="s").dt.hour

    # Differential privacy
    df["amount_log_dp"] = apply_dp_to_series(
        df["amount_log"].values, sensitivity=1.0, epsilon=dp_epsilon
    )

    df["txs_per_day_dp"] = apply_dp_to_series(
        df["txs_per_day"].values, sensitivity=1.0, epsilon=dp_epsilon
    )

    # Final features
    final_df = df[
        [
            "txid",
            "timestamp",
            "iat",
            "amount_log_dp",
            "ring_size",
            "txs_per_day_dp",
            "hour",
        ]
    ].copy()

    final_df = final_df.rename(
        columns={
            "amount_log_dp": "amount_log",
            "txs_per_day_dp": "txs_per_day",
        }
    )

    return final_df


# --------------------------------------------------------------
# MAIN EXECUTION BLOCK (THIS MAKES THE FILE RUNNABLE)
# --------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature engineering for Monero data")
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default="data/raw_blocks/headers_1500000_1530000.json",
        help="Path to extracted block headers JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="data/processed/features_1500000_1530000.csv",
        help="Output CSV for engineered features",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Differential privacy epsilon",
    )

    args = parser.parse_args()

    print(f"[+] Loading raw data from {args.input}")

    # Load JSON
    with open(args.input, "r") as f:
        headers = json.load(f)

    # Convert headers to DataFrame
    # Note: block headers contain tx_hashes, timestamps, etc.
    # For simplicity we treat tx_hash as one "transaction"
    # Real pipeline would expand raw tx details, but this fits your lightweight approach.
    tx_records = []

    for h in headers:
        tx_records.append({
            "txid": h.get("hash", ""),
            "timestamp": h.get("timestamp", 0),
            "amount": 0.0,           # Placeholder (Monero hides amounts)
            "ring_size": h.get("depth", 11),  # Lightweight behavioural proxy
            "outputs_count": len(h.get("tx_hashes", [])),
        })

    tx_df = pd.DataFrame(tx_records)

    print(f"[+] Loaded {len(tx_df)} transactions")
    print("[+] Creating behavioural features...")

    feat_df = create_basic_features(tx_df, dp_epsilon=args.epsilon)

    # Save output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"[+] Saving engineered features to {args.output}")
    feat_df.to_csv(args.output, index=False)

    print("[✓] Feature engineering complete.")
