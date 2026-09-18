"""
Campus WiFinder backend — serves live neural-net predictions for the 70
real routers in router_locations.csv, plus on-demand predictions for any
arbitrary lat/lng the frontend sends (used by the "pick on map" mode).

This replaces the old zone-based backend: the previous dataset had
access_point_id / area_type columns that don't exist in the CSVs you're
using now, so that logic has been dropped. Everything below is derived
only from router_locations.csv and dataset_wifi_extended_renamed_new.csv.

Run:
    pip install flask pandas scikit-learn --break-system-packages
    python app.py

The frontend (campus_wifinder.html) already runs fully client-side with
an embedded snapshot + copy of this same neural net, so this server is
optional — use it if you want the map to reflect fresh data without
re-exporting the HTML each time. It serves the same two things the page
embeds statically: a per-router snapshot and the NN weights, plus a live
/api/predict-point endpoint for arbitrary coordinates.
"""

import os
import json
import math

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from sklearn.neighbors import BallTree
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTERS_PATH = os.path.join(BASE_DIR, "router_locations.csv")
DATA_PATH = os.path.join(BASE_DIR, "dataset_wifi_extended_renamed_new.csv")

app = Flask(__name__)


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


# ---------------------------------------------------------------------
# Load + prep, once at startup
# ---------------------------------------------------------------------
routers = pd.read_csv(ROUTERS_PATH)

df = pd.read_csv(DATA_PATH)
# "Unnamed: 2" is a fully empty column in the source CSV — not real data.
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
df["hour"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S", errors="coerce").dt.hour

# Assign every reading to its nearest real router (haversine nearest-neighbor)
_rad_routers = np.radians(routers[["latitude", "longitude"]].values)
_rad_readings = np.radians(df[["latitude", "longitude"]].values)
_tree = BallTree(_rad_routers, metric="haversine")
_dist, _idx = _tree.query(_rad_readings, k=1)
df["router_id"] = routers["router_id"].values[_idx[:, 0]]

FEATURE_ORDER = [
    "connected_users", "active_users", "average_data_usage", "total_bandwidth_usage",
    "signal_strength", "noise_level", "snr", "channel_utilization", "interference_level",
    "hour", "freq_5", "is_weekend", "is_class_hours",
]


def _model_frame(source_df):
    X = source_df[[
        "connected_users", "active_users", "average_data_usage", "total_bandwidth_usage",
        "signal_strength", "noise_level", "snr", "channel_utilization", "interference_level", "hour",
    ]].copy()
    X["freq_5"] = (source_df["frequency_band"] == "5GHz").astype(float)
    X["is_weekend"] = source_df["is_weekend"].astype(float)
    X["is_class_hours"] = source_df["is_class_hours"].astype(float)
    return X[FEATURE_ORDER]


X = _model_frame(df)
y = df["download_speed"].values

_scaler = StandardScaler().fit(X.values)
_Xs = _scaler.transform(X.values)

_nn = MLPRegressor(hidden_layer_sizes=(12, 8), activation="relu", max_iter=600,
                    random_state=42, early_stopping=True, n_iter_no_change=15)
_nn.fit(_Xs, y)


def nn_predict_rows(feature_rows):
    """feature_rows: DataFrame with FEATURE_ORDER columns."""
    Xs = _scaler.transform(feature_rows[FEATURE_ORDER].values)
    return np.maximum(0, _nn.predict(Xs))


# Per-router aggregated "current conditions" — the same numbers the model
# reasons over, just averaged from real readings assigned to that router.
_agg = df.groupby("router_id").agg(
    connected_users=("connected_users", "mean"),
    active_users=("active_users", "mean"),
    average_data_usage=("average_data_usage", "mean"),
    total_bandwidth_usage=("total_bandwidth_usage", "mean"),
    signal_strength=("signal_strength", "mean"),
    noise_level=("noise_level", "mean"),
    snr=("snr", "mean"),
    channel_utilization=("channel_utilization", "mean"),
    interference_level=("interference_level", "mean"),
    freq_5=("frequency_band", lambda s: (s == "5GHz").mean()),
    avg_download=("download_speed", "mean"),
    avg_latency=("latency", "mean"),
    samples=("download_speed", "size"),
).reset_index().merge(routers, on="router_id", how="left")


def _time_context(now=None):
    now = now or pd.Timestamp.now()
    is_weekend = now.dayofweek >= 5
    is_class_hours = (9 <= now.hour < 17) and not is_weekend
    return {"hour": now.hour, "is_weekend": is_weekend, "is_class_hours": is_class_hours}


def haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@app.route("/api/router-stats")
def router_stats():
    """Every real router with a live, time-aware predicted speed."""
    ctx = _time_context()
    rows = _agg.copy()
    for k, v in ctx.items():
        rows[k] = v
    rows["download_mbps"] = nn_predict_rows(rows).round(1)
    rows["latency_ms"] = (42 - rows["download_mbps"]).clip(lower=5).round(1)
    out = rows[["router_id", "latitude", "longitude", "download_mbps", "latency_ms", "samples"]]
    return jsonify(out.to_dict(orient="records"))


@app.route("/api/predict-point")
def predict_point():
    """AI estimate for ANY coordinate — inverse-distance blend of the k
    nearest routers' live conditions, fed through the same neural net."""
    try:
        lat = float(request.args.get("lat"))
        lng = float(request.args.get("lng"))
    except (TypeError, ValueError):
        return jsonify({"error": "pass ?lat=..&lng=.."}), 400
    k = min(3, len(_agg))

    dists = _agg.apply(lambda r: haversine_m(lat, lng, r["latitude"], r["longitude"]), axis=1)
    nearest = dists.nsmallest(k).index
    weights = 1 / dists.loc[nearest].clip(lower=1)
    w = weights / weights.sum()

    blend_cols = ["connected_users", "active_users", "average_data_usage", "total_bandwidth_usage",
                  "signal_strength", "noise_level", "snr", "channel_utilization", "interference_level", "freq_5"]
    blended = (_agg.loc[nearest, blend_cols].T * w.values).T.sum()

    ctx = _time_context()
    row = pd.DataFrame([{**blended.to_dict(), **ctx}])
    speed = float(nn_predict_rows(row)[0])

    return jsonify({
        "lat": lat, "lng": lng,
        "download_mbps": round(speed, 1),
        "nearest_router": _agg.loc[dists.idxmin(), "router_id"],
        "nearest_dist_m": round(dists.min()),
    })


@app.route("/")
def health():
    return jsonify({"status": "ok", "routers": len(_agg)})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
