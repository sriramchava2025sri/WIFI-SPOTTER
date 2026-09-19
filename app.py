"""
app.py — CAMPUS WIFINDER API v2 (simulator-driven)

Routes are now THIN. Every one of them is a few lines: parse input, call the
simulator, jsonify. No model code, no pandas, no feature engineering lives
here any more. If a route is longer than ten lines, the logic is in the wrong
file.

Run:
    pip install flask pandas scikit-learn joblib --break-system-packages
    python app.py

Backwards compatibility promise: GET /api/zone-stats still returns a JSON
array of 8 objects each containing `download_mbps` and `latency_ms`, in zone
order. Your existing campus_wifinder.html works against this server with zero
edits — it just won't use the new fields yet. Deploy the backend first, then
upgrade the frontend at your own pace.
"""

from __future__ import annotations

import os
import logging
import time

from flask import Flask, jsonify, request

from predictor import load_predictor
from simulator import CampusSimulator, build_zones, Tuning

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset_wifi_updated.csv")

FEATURE_COLS = [
    "access_point_id", "area_type", "latitude", "longitude",
    "connected_users", "active_users", "average_data_usage",
    "total_bandwidth_usage", "signal_strength", "noise_level", "snr",
    "channel", "channel_utilization", "interference_level",
    "frequency_band", "day_of_week", "is_weekend", "is_holiday",
    "is_class_hours", "hour", "minute",
]

app = Flask(__name__)


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Client-Id"
    return resp


# ---------------------------------------------------------------------------
# BOOT — load once, freeze baselines, start the tick loop
# ---------------------------------------------------------------------------
def _load_dataframe():
    try:
        import pandas as pd
        df = pd.read_csv(DATA_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        log.info("dataset loaded: %d rows", len(df))
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning("dataset unavailable (%s) — using synthetic baselines", exc)
        return None


predictor = load_predictor()
ZONES = build_zones(_load_dataframe(), FEATURE_COLS)
sim = CampusSimulator(predictor, ZONES)
sim.start()


def _client_id() -> str:
    """Identify a browser without auth. Header first, then IP."""
    return request.headers.get("X-Client-Id") or request.remote_addr or "anon"


# ===========================================================================
# READ ENDPOINTS
# ===========================================================================
@app.route("/api/zone-stats")
def zone_stats():
    """UNCHANGED CONTRACT. Array of 8, `download_mbps` + `latency_ms` present.
    New fields added alongside. Old frontend keeps working."""
    return jsonify(sim.snapshot())


@app.route("/api/zones")
def zones_static():
    """Static geometry — polygons, centroids, capacity. Fetch once on load.
    Makes the backend the single source of truth for zone definitions so you
    stop maintaining the same coordinates in two files."""
    return jsonify([
        {
            "zone_id": z.id, "name": z.name, "polygon": z.polygon,
            "lat": z.lat, "lng": z.lng,
            "capacity_users": z.capacity_users, "ap_ids": z.ap_ids,
        }
        for z in sim.zones.values()
    ])


@app.route("/api/recommend")
def recommend():
    """
    THE LOAD-BALANCING ENDPOINT. Server-side ranking, because only the server
    can see global load. Returns zones sorted best-first with scores, ranks,
    distances and a human-readable reason.

        /api/recommend?lat=12.9715&lng=79.16&radius=600
    """
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    radius = request.args.get("radius", default=600, type=float)

    ranked = sim.rank(lat, lng, radius)
    in_range = [z for z in ranked if z.get("distance_m", 0) <= radius] if lat else ranked
    best = in_range[0] if in_range else (ranked[0] if ranked else None)

    reason = None
    if best:
        if best["headroom"] > 5:
            reason = f"{best['download_mbps']} Mbps with {best['headroom']} slots free"
        else:
            reason = f"{best['download_mbps']} Mbps but filling up fast"

    return jsonify({
        "best": best,
        "reason": reason,
        "alternatives": in_range[1:4],
        "all": ranked,
        "version": sim.version,
    })


@app.route("/api/state")
def state():
    """Everything a dashboard needs in one call: zones, best, events, version."""
    return jsonify({
        "version": sim.version,
        "tick_seconds": Tuning.TICK_SECONDS,
        "last_tick": sim.last_tick,
        "best_zone": sim.best_zone_id(),
        "zones": sim.snapshot(),
        "events": sim.events[-8:],
    })


@app.route("/api/events")
def events():
    """Reroute feed — 'Zone 1 dropped, redirecting to Zone 6'. Pure demo gold."""
    return jsonify(sim.events[-20:])


# ===========================================================================
# WRITE ENDPOINTS — the crowd moves
# ===========================================================================
@app.route("/api/join", methods=["POST", "OPTIONS"])
def join():
    """A real user commits to a zone. They now count against its capacity."""
    if request.method == "OPTIONS":
        return ("", 204)
    body = request.get_json(silent=True) or {}
    zone_id = body.get("zone_id")
    if zone_id not in sim.zones:
        return jsonify({"error": "unknown zone_id"}), 400
    sim.join(zone_id, _client_id())
    return jsonify({"ok": True, "zone": sim.zones[zone_id].to_public()})


@app.route("/api/leave", methods=["POST", "OPTIONS"])
def leave():
    if request.method == "OPTIONS":
        return ("", 204)
    sim.leave(_client_id())
    return jsonify({"ok": True})


@app.route("/api/sim/inject", methods=["POST", "OPTIONS"])
def sim_inject():
    """
    THE DEMO BUTTON. POST {"zone_id": "zone-1", "users": 25} and watch that
    zone collapse and the recommendation jump elsewhere. Negative values
    remove users.
    """
    if request.method == "OPTIONS":
        return ("", 204)
    body = request.get_json(silent=True) or {}
    zone_id = body.get("zone_id")
    users = int(body.get("users", 10))
    if zone_id not in sim.zones:
        return jsonify({"error": "unknown zone_id"}), 400
    before = sim.zones[zone_id].predicted_mbps
    sim.inject(zone_id, users)
    after = sim.zones[zone_id].predicted_mbps
    return jsonify({
        "ok": True,
        "zone_id": zone_id,
        "before_mbps": round(before, 1),
        "after_mbps": round(after, 1),
        "new_best": sim.best_zone_id(),
    })


@app.route("/api/sim/reset", methods=["POST", "OPTIONS"])
def sim_reset():
    if request.method == "OPTIONS":
        return ("", 204)
    sim.reset()
    return jsonify({"ok": True})


# ===========================================================================
# OPS
# ===========================================================================
@app.route("/api/model-info")
def model_info():
    """Which backend is live, and what does it expect? Saves you 20 minutes of
    confusion on integration day."""
    return jsonify({
        **predictor.describe(),
        "env": os.getenv("MODEL_BACKEND", "legacy_ap"),
        "zones": len(sim.zones),
        "tuning": {k: v for k, v in vars(Tuning).items() if not k.startswith("_")},
    })


@app.route("/")
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "backend": predictor.name,
        "version": sim.version,
        "uptime_ticks": sim.version,
        "server_time": time.time(),
    })


if __name__ == "__main__":
    # threaded=True matters: the tick loop and requests run concurrently.
    app.run(port=5000, debug=False, threaded=True)
