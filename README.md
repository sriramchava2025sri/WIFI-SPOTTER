# Campus WiFinder v2 — simulator-driven load balancing

## Run
Drop these files next to your existing `wifi_speed_model.pkl`,
`model_columns.pkl` and `dataset_wifi_updated.csv`, then:

    pip install -r requirements.txt --break-system-packages
    python app.py            # http://localhost:5000
    open campus_wifinder.html

## Files
| file                   | role |
|------------------------|------|
| `predictor.py`         | model adapter — the ONLY file that changes when the Colab model lands |
| `simulator.py`         | zone state, congestion physics, 2s tick loop |
| `app.py`               | thin Flask routes over the simulator |
| `campus_wifinder.html` | patched frontend: live polling, Go-here, sim panel, reroute banner |

## Model backends
    MODEL_BACKEND=legacy_ap python app.py    # current pkl (default)
    MODEL_BACKEND=spatial   python app.py    # new Colab model (model_v2.pkl + model_v2_meta.json)
    MODEL_BACKEND=stub      python app.py    # no model file needed — demo safety net

## Verified demo path
    Zone 8: 18.6 Mbps, 14 slots free
      -> POST /api/sim/inject {"zone_id":"zone-8","users":35}
    Zone 8: 7.5 Mbps, latency 18ms -> 88ms, status good -> bad
      -> reroute event fires, recommendation moves to Zone 1 (18.3 Mbps)
      -> crowd decays over ~40s and Zone 8 recovers on its own

## Endpoints
| method | path | purpose |
|--------|------|---------|
| GET  | `/api/zone-stats`   | unchanged contract — old frontend still works |
| GET  | `/api/zones`        | static polygons/centroids/capacity |
| GET  | `/api/state`        | everything: zones + best + events + version |
| GET  | `/api/recommend`    | server-side ranking (`?lat=&lng=&radius=`) |
| GET  | `/api/events`       | reroute feed |
| POST | `/api/join`         | `{"zone_id":"zone-3"}` — user commits, adds load |
| POST | `/api/leave`        | drop out |
| POST | `/api/sim/inject`   | `{"zone_id":"zone-1","users":25}` — demo button |
| POST | `/api/sim/reset`    | clear all simulated crowds |
| GET  | `/api/model-info`   | live backend + expected features + tuning |
| GET  | `/api/health`       | liveness |
