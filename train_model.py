"""
train_model.py — trains the neural net behind Campus WiFinder and exports
everything the frontend/backend need.

WHAT THIS DOES
---------------
1. Loads your two real data sources:
     - router_locations.csv                 (70 routers: router_id, lat, lon)
     - dataset_wifi_extended_renamed_new.csv (143k network readings)
2. Assigns every reading to its nearest physical router (haversine nearest-
   neighbor on lat/lon) — this is how we get "conditions at router R-04"
   out of a dataset that only has raw lat/lon per reading, not a router id.
3. Builds the feature set from columns that ACTUALLY exist in the CSV
   (no access_point_id / area_type — those were from an old, incompatible
   dataset and have been dropped entirely).
4. Trains a small MLPRegressor (a real feedforward neural network) to
   predict download_speed.
5. Exports three artifacts:
     - nn_model.json        → weights/biases + scaler, for the pure-JS
                               inference engine embedded in the HTML
     - router_snapshot.json → per-router averaged live conditions, for
                               the embedded map data
     - wifi_speed_model.pkl → the same model, for the Flask backend

REQUIREMENTS
------------
    pip install pandas numpy scikit-learn joblib --break-system-packages

USAGE
-----
    python train_model.py
    # reads the two CSVs from the same folder, writes the 3 files above
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROUTERS_CSV = "router_locations.csv"
DATA_CSV = "dataset_wifi_extended_renamed_new.csv"

# The 13 real, present-in-the-CSV features the model is trained on. Order
# matters — this exact order is exported and must be replayed identically
# by any inference code (JS or Python).
FEATURE_ORDER = [
    "connected_users", "active_users", "average_data_usage", "total_bandwidth_usage",
    "signal_strength", "noise_level", "snr", "channel_utilization", "interference_level",
    "hour", "freq_5", "is_weekend", "is_class_hours",
]

TARGET = "download_speed"


# ---------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------
def load_data():
    routers = pd.read_csv(ROUTERS_CSV)

    df = pd.read_csv(DATA_CSV)
    # Drop any fully-empty/junk columns (e.g. the blank "Unnamed: 2" column
    # present in this export) — never train on columns that aren't real data.
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    # timestamp in this CSV is just "HH:MM:SS" (no date) — pull the hour out.
    df["hour"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S", errors="coerce").dt.hour

    return routers, df


# ---------------------------------------------------------------------
# 2. Nearest-router assignment (haversine BallTree — accounts for the
#    Earth's curvature, unlike plain Euclidean distance on lat/lon)
# ---------------------------------------------------------------------
def assign_nearest_router(routers, df):
    rad_routers = np.radians(routers[["latitude", "longitude"]].values)
    rad_readings = np.radians(df[["latitude", "longitude"]].values)

    tree = BallTree(rad_routers, metric="haversine")
    dist, idx = tree.query(rad_readings, k=1)

    df = df.copy()
    df["router_id"] = routers["router_id"].values[idx[:, 0]]
    df["dist_to_router_m"] = dist[:, 0] * 6_371_000  # Earth radius in meters
    return df


# ---------------------------------------------------------------------
# 3. Feature matrix — everything here comes straight from the CSV;
#    only frequency_band/is_weekend/is_class_hours are re-encoded as 0/1.
# ---------------------------------------------------------------------
def build_features(df):
    X = df[[
        "connected_users", "active_users", "average_data_usage", "total_bandwidth_usage",
        "signal_strength", "noise_level", "snr", "channel_utilization", "interference_level", "hour",
    ]].copy()
    X["freq_5"] = (df["frequency_band"] == "5GHz").astype(float)
    X["is_weekend"] = df["is_weekend"].astype(float)
    X["is_class_hours"] = df["is_class_hours"].astype(float)
    return X[FEATURE_ORDER]


# ---------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------
def train(X, y):
    scaler = StandardScaler().fit(X.values)
    Xs = scaler.transform(X.values)

    model = MLPRegressor(
        hidden_layer_sizes=(12, 8),   # 2 hidden layers, small enough to run in-browser
        activation="relu",
        max_iter=600,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15,
    )
    model.fit(Xs, y)

    pred = model.predict(Xs)
    mae = float(np.mean(np.abs(pred - y)))
    r2 = float(np.corrcoef(pred, y)[0, 1] ** 2)
    print(f"Train MAE: {mae:.3f} Mbps   |   R^2: {r2:.3f}")

    return model, scaler


# ---------------------------------------------------------------------
# 5. Export — pure-JS-portable weights, a per-router snapshot, and the
#    sklearn model for the backend.
# ---------------------------------------------------------------------
def export_nn_json(model, scaler, path="nn_model.json"):
    payload = {
        "feature_order": FEATURE_ORDER,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weights": [w.tolist() for w in model.coefs_],
        "biases": [b.tolist() for b in model.intercepts_],
        "activation": "relu",
        "out_activation": model.out_activation_,  # "identity" for regression
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"wrote {path} ({len(json.dumps(payload))} bytes)")


def export_router_snapshot(routers, df, path="router_snapshot.json"):
    agg = df.groupby("router_id").agg(
        connected_users=("connected_users", "mean"),
        active_users=("active_users", "mean"),
        average_data_usage=("average_data_usage", "mean"),
        total_bandwidth_usage=("total_bandwidth_usage", "mean"),
        signal_strength=("signal_strength", "mean"),
        noise_level=("noise_level", "mean"),
        snr=("snr", "mean"),
        channel_utilization=("channel_utilization", "mean"),
        interference_level=("interference_level", "mean"),
        freq_5_share=("frequency_band", lambda s: (s == "5GHz").mean()),
        avg_download=("download_speed", "mean"),
        avg_latency=("latency", "mean"),
        samples=("download_speed", "size"),
    ).reset_index().merge(routers, on="router_id", how="left")

    records = agg.to_dict(orient="records")
    with open(path, "w") as f:
        json.dump(records, f)
    print(f"wrote {path} ({len(records)} routers, {len(json.dumps(records))} bytes)")


def export_pkl(model, scaler, path="wifi_speed_model.pkl"):
    joblib.dump({"model": model, "scaler": scaler, "feature_order": FEATURE_ORDER}, path)
    print(f"wrote {path}")


# ---------------------------------------------------------------------
def main():
    routers, df = load_data()
    df = assign_nearest_router(routers, df)

    X = build_features(df)
    y = df[TARGET].values

    model, scaler = train(X, y)

    export_nn_json(model, scaler)
    export_router_snapshot(routers, df)
    export_pkl(model, scaler)


if __name__ == "__main__":
    main()
