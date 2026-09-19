"""
simulator.py — ZONE STATE, CONGESTION PHYSICS, AND THE TICK LOOP.

This is the new brain. It holds the answer to "how fast is Zone 3 *right now*,
given that 20 people just walked into it", which your current system cannot
answer because it has no concept of "right now" or "people".

--------------------------------------------------------------------------
WHY THERE IS A PHYSICS LAYER AT ALL (read this before you tune anything)
--------------------------------------------------------------------------
I measured your trained RandomForest's sensitivity to crowding. Holding one
real Zone-1 row fixed and sweeping a single feature:

    connected_users      1 -> 150   speed 15.97 -> 15.67 Mbps   (0.3 Mbps!)
    active_users         1 -> 100   speed 16.10 -> 15.61 Mbps   (0.5 Mbps)
    channel_utilization  5 ->  95   speed 16.46 -> 12.30 Mbps   (4.2 Mbps)

Feature importances say the same thing: connected_users is 0.0017, while
snr + channel + frequency_band together are ~0.80.

Read that again, because it decides your whole architecture: **your model did
not learn congestion.** If you simply pipe "20 more users" into it and re-
predict, the number moves by a third of a megabit and your load-balancing demo
silently does nothing. The naive design fails.

So the system splits the question in two:

    CAPACITY   what can this radio link deliver?          <- the ML model
    SHARE      how does that get divided among a crowd?   <- this file

    delivered_mbps = capacity_mbps  x  fair_share(load)  x  efficiency(load)

The share/efficiency terms are textbook CSMA/CA behaviour: airtime is a shared
medium, so per-user throughput falls roughly as 1/N once you pass the point
where the AP can serve everyone, and protocol overhead (collisions, retries,
management frames) eats extra capacity on top. This is honest, explainable
engineering, not a fudge factor — and when a judge asks "is that the model or
a formula?", the answer is a good one: "the model sizes the pipe, a contention
model splits it, and we can show you both numbers separately."

It also means the whole demo survives a model swap. When the Colab model lands
and *does* learn congestion, you lower CONTENTION_EXP toward 0 and let the
model take over. Nothing else changes.
"""

from __future__ import annotations

import math
import os
import threading
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

import pandas as pd

log = logging.getLogger("simulator")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# TUNABLES — every magic number in the system lives here, nowhere else.
# ---------------------------------------------------------------------------
class Tuning:
    #: extra channel utilisation (%) contributed per connected user
    UTIL_PER_USER = 1.6
    #: fraction of connected users actively transmitting at any moment
    ACTIVE_FRACTION = 0.65
    #: how hard per-user throughput falls once load exceeds capacity.
    #: 1.0 = perfect 1/N fair share. 0.85 = slightly gentler, reads better on stage.
    CONTENTION_EXP = 0.85
    #: max fraction of capacity lost to protocol overhead at full congestion
    EFFICIENCY_LOSS = 0.22
    #: latency grows quadratically with load (queueing delay)
    LATENCY_BASE_MS = 14.0
    LATENCY_QUEUE_K = 26.0
    #: simulated users drift away on their own, so the demo self-heals
    DECAY_PER_TICK = 0.04
    #: seconds between recomputes
    TICK_SECONDS = 2.0
    #: status thresholds (Mbps)
    GOOD_MBPS = 15.0
    OKAY_MBPS = 9.0


# ---------------------------------------------------------------------------
# ZONE STATE — the core data structure you asked about (question 3)
# ---------------------------------------------------------------------------
@dataclass
class ZoneState:
    """
    One hotspot zone. Split deliberately into three tiers, because they have
    three different lifetimes and conflating them is how these systems rot:

      STATIC     never changes after boot  (identity, geometry, capacity)
      BASELINE   frozen snapshot of the dataset (the "empty campus" condition)
      DYNAMIC    changes every tick        (who's here, how fast it is now)

    Keeping BASELINE frozen is not a detail — it is the fix for the single
    worst bug in the current app.py. Today `predict_zone()` calls
    `sub[col].sample(1)` per request, so every refresh redraws random features
    and the predicted speed jitters by several Mbps for no reason. I checked
    the dataset: all eight hotspot zones have a median download_speed between
    17.14 and 17.59 Mbps — statistically identical. So the zone ranking your
    map shows right now is *pure sampling noise*, not signal. Freeze the
    baseline and the only thing that can move a zone's number is the crowd,
    which is exactly the story you want to tell.
    """

    # --- STATIC -----------------------------------------------------------
    id: str
    name: str
    polygon: list[list[float]]
    lat: float
    lng: float
    ap_ids: list[str] = field(default_factory=list)      # legacy model binding
    router_ids: list[str] = field(default_factory=list)  # new model binding
    capacity_users: int = 30       # comfortable concurrent users before pain

    # --- BASELINE (frozen at boot) ---------------------------------------
    baseline_features: dict[str, Any] = field(default_factory=dict)
    baseline_users: int = 14
    baseline_util: float = 12.3

    # --- DYNAMIC ----------------------------------------------------------
    sim_users: float = 0.0         # placed by the simulator / "Go here" taps
    committed_users: set = field(default_factory=set)  # real client ids
    capacity_mbps: float = 0.0     # raw model output
    predicted_mbps: float = 0.0    # after contention — what the UI shows
    latency_ms: float = 0.0
    status: str = "good"
    trend: str = "flat"            # up | down | flat — drives the UI arrow
    _prev_mbps: float = 0.0

    # --- derived ----------------------------------------------------------
    @property
    def total_users(self) -> float:
        return self.baseline_users + self.sim_users + len(self.committed_users)

    @property
    def load_ratio(self) -> float:
        return self.total_users / max(1, self.capacity_users)

    @property
    def headroom(self) -> int:
        """Seats left before this zone starts degrading. Great UI copy."""
        return max(0, int(self.capacity_users - self.total_users))

    def to_public(self) -> dict:
        """
        The wire format. NOTE the first two keys: `download_mbps` and
        `latency_ms` are exactly what your existing frontend already reads.
        Everything else is additive, so the current campus_wifinder.html keeps
        working byte-for-byte even if you deploy this backend first. Never
        rename a field the client depends on — add beside it.
        """
        return {
            "download_mbps": round(self.predicted_mbps, 1),
            "latency_ms": round(self.latency_ms, 1),
            # --- additive, new ---
            "zone_id": self.id,
            "name": self.name,
            "lat": self.lat,
            "lng": self.lng,
            "capacity_mbps": round(self.capacity_mbps, 1),
            "users": int(self.total_users),
            "capacity_users": self.capacity_users,
            "headroom": self.headroom,
            "load_pct": round(min(1.5, self.load_ratio) * 100),
            "channel_utilization": round(self.effective_util(), 1),
            "status": self.status,
            "trend": self.trend,
        }

    # --- physics ----------------------------------------------------------
    def effective_util(self) -> float:
        """Users occupy airtime. This is the lever the model actually responds to."""
        util = self.baseline_util + Tuning.UTIL_PER_USER * (
            self.sim_users + len(self.committed_users)
        )
        return float(min(95.0, max(2.0, util)))

    def model_row(self) -> dict:
        """
        Build the feature dict handed to the predictor: the frozen baseline,
        overridden with the crowd-derived features. This is the one and only
        place the simulator touches model inputs.
        """
        row = dict(self.baseline_features)
        row["channel_utilization"] = self.effective_util()
        row["connected_users"] = int(self.total_users)
        row["active_users"] = int(self.total_users * Tuning.ACTIVE_FRACTION)
        now = time.localtime()
        row["hour"] = now.tm_hour
        row["minute"] = now.tm_min
        return row

    def apply_capacity(self, capacity_mbps: float) -> None:
        """Turn raw model capacity into delivered per-user speed + latency."""
        self.capacity_mbps = capacity_mbps
        lr = self.load_ratio

        # 1. FAIR SHARE — below capacity everyone gets full speed; above it,
        #    throughput divides roughly as 1/N^exp.
        share = 1.0 if lr <= 1.0 else 1.0 / (lr ** Tuning.CONTENTION_EXP)

        # 2. EFFICIENCY — collisions/retries waste airtime as the cell fills.
        #    Ramps in from 60% load, maxes out at 200% load.
        over = min(1.0, max(0.0, (lr - 0.6) / 1.4))
        efficiency = 1.0 - Tuning.EFFICIENCY_LOSS * over

        delivered = capacity_mbps * share * efficiency

        self._prev_mbps = self.predicted_mbps
        self.predicted_mbps = max(0.4, delivered)

        # 3. LATENCY — queueing delay grows with the square of load.
        self.latency_ms = Tuning.LATENCY_BASE_MS + Tuning.LATENCY_QUEUE_K * (lr ** 2)
        self.latency_ms = min(320.0, self.latency_ms)

        # 4. STATUS + TREND for the UI
        self.status = (
            "good" if self.predicted_mbps >= Tuning.GOOD_MBPS
            else "okay" if self.predicted_mbps >= Tuning.OKAY_MBPS
            else "bad"
        )
        delta = self.predicted_mbps - self._prev_mbps
        self.trend = "up" if delta > 0.3 else "down" if delta < -0.3 else "flat"


# ---------------------------------------------------------------------------
# THE SIMULATOR
# ---------------------------------------------------------------------------
class CampusSimulator:
    """
    Owns all zone state, runs the tick loop, and is the ONLY thing that calls
    the predictor. Flask routes just read `sim.snapshot()` — they never touch
    the model. That separation is what lets you swap the model, change the
    physics, or bolt on a websocket later without rewriting routes.

    Thread safety: one lock around all mutation. A hackathon Flask dev server
    is threaded, so two simultaneous /api/join calls will race without it.
    """

    def __init__(self, predictor, zones: list[ZoneState]):
        self.predictor = predictor
        self.zones = {z.id: z for z in zones}
        self.lock = threading.RLock()
        self.version = 0           # bumps every tick — lets clients poll cheaply
        self.last_tick = 0.0
        self.events: list[dict] = []   # rerouting events, for the demo feed
        self._stop = threading.Event()
        self.recompute()

    # -- core ------------------------------------------------------------
    def recompute(self) -> None:
        """One batched predict for ALL zones. Never predict in a loop."""
        with self.lock:
            ordered = list(self.zones.values())
            rows = [z.model_row() for z in ordered]
            try:
                caps = self.predictor.predict_capacity(rows)
            except Exception as exc:  # noqa: BLE001
                log.error("predict failed: %s", exc)
                caps = [z.capacity_mbps or 17.0 for z in ordered]

            prev_best = self.best_zone_id()
            for zone, cap in zip(ordered, caps):
                zone.apply_capacity(cap)

            new_best = self.best_zone_id()
            # version==0 is the boot recompute, when every capacity is still
            # 0.0 and "best" is meaningless — don't log a phantom reroute.
            if self.version > 0 and prev_best and new_best and prev_best != new_best:
                self.events.append({
                    "t": time.time(),
                    "type": "reroute",
                    "from": self.zones[prev_best].name,
                    "to": self.zones[new_best].name,
                    "reason": f"{self.zones[prev_best].name} dropped to "
                              f"{self.zones[prev_best].predicted_mbps:.1f} Mbps",
                })
                self.events = self.events[-20:]

            self.version += 1
            self.last_tick = time.time()

    def tick(self) -> None:
        """Called every TICK_SECONDS by the background thread."""
        with self.lock:
            for z in self.zones.values():
                # Simulated crowds disperse over time -> the system self-heals
                # on stage without you touching anything. Very good demo optics.
                if z.sim_users > 0:
                    z.sim_users = max(0.0, z.sim_users * (1 - Tuning.DECAY_PER_TICK) - 0.05)
        self.recompute()

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001
                    log.error("tick error: %s", exc)
                self._stop.wait(Tuning.TICK_SECONDS)

        threading.Thread(target=loop, daemon=True, name="sim-tick").start()
        log.info("simulator tick loop started (%.1fs)", Tuning.TICK_SECONDS)

    def stop(self) -> None:
        self._stop.set()

    # -- mutations -------------------------------------------------------
    def inject(self, zone_id: str, users: int) -> None:
        with self.lock:
            if zone_id in self.zones:
                z = self.zones[zone_id]
                z.sim_users = max(0.0, z.sim_users + users)
        self.recompute()

    def join(self, zone_id: str, client_id: str) -> None:
        """A real user taps 'Go here'. Their presence loads the zone."""
        with self.lock:
            for z in self.zones.values():
                z.committed_users.discard(client_id)   # one zone at a time
            if zone_id in self.zones:
                self.zones[zone_id].committed_users.add(client_id)
        self.recompute()

    def leave(self, client_id: str) -> None:
        with self.lock:
            for z in self.zones.values():
                z.committed_users.discard(client_id)
        self.recompute()

    def reset(self) -> None:
        with self.lock:
            for z in self.zones.values():
                z.sim_users = 0.0
                z.committed_users.clear()
            self.events.clear()
        self.recompute()

    # -- reads -----------------------------------------------------------
    def best_zone_id(self) -> str | None:
        if not self.zones:
            return None
        return max(self.zones.values(), key=lambda z: z.predicted_mbps).id

    def snapshot(self) -> list[dict]:
        with self.lock:
            return [z.to_public() for z in self.zones.values()]

    def rank(self, lat=None, lng=None, radius_m=None,
             speed_w=0.65, dist_w=0.35) -> list[dict]:
        """
        Server-side ranking. Moving this off the client is what makes the
        system a *load balancer* rather than eight independent readouts: only
        the server knows every zone's live load, so only the server can spread
        people out.
        """
        with self.lock:
            out = []
            max_mbps = max((z.predicted_mbps for z in self.zones.values()), default=1.0) or 1.0
            for z in self.zones.values():
                pub = z.to_public()
                if lat is not None and lng is not None:
                    d = _haversine_m(lat, lng, z.lat, z.lng)
                    pub["distance_m"] = int(d)
                    prox = max(0.0, 1 - d / radius_m) if radius_m else 0.5
                else:
                    prox = 0.5
                speed_score = min(1.0, z.predicted_mbps / max_mbps)
                # headroom bonus: prefer zones that can absorb people. This is
                # the actual load-balancing term — without it you stampede.
                headroom_score = min(1.0, z.headroom / max(1, z.capacity_users))
                pub["score"] = round(
                    speed_w * speed_score + dist_w * prox + 0.15 * headroom_score, 4
                )
                out.append(pub)
            out.sort(key=lambda p: p["score"], reverse=True)
            for i, p in enumerate(out):
                p["rank"] = i + 1
            return out


def _haversine_m(lat1, lng1, lat2, lng2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# ZONE REGISTRY — build ZoneState objects from your existing constants + CSV
# ---------------------------------------------------------------------------
# Same 8 polygons as campus_wifinder.html, same order. Single source of truth:
# the frontend should fetch these from /api/zones rather than keeping its own
# copy, so a zone edit is a one-file change.
HOTSPOT_POLYGONS = [
    [[12.972028, 79.156694], [12.971417, 79.156556], [12.972028, 79.158333]],
    [[12.968889, 79.155361], [12.968861, 79.156444], [12.969639, 79.156472], [12.969639, 79.155361]],
    [[12.970389, 79.155000], [12.969361, 79.154778], [12.970389, 79.155333]],
    [[12.969639, 79.156528], [12.969611, 79.157139], [12.969111, 79.157111], [12.969111, 79.156556]],
    [[12.969583, 79.157472], [12.968861, 79.157389], [12.969528, 79.158028]],
    [[12.971500, 79.163167], [12.970889, 79.163222], [12.970639, 79.164111], [12.971444, 79.164222]],
    [[12.972306, 79.166250], [12.970556, 79.166528], [12.970944, 79.167583], [12.972389, 79.168278]],
    [[12.972861, 79.157389], [12.972667, 79.157389], [12.972917, 79.163722], [12.973111, 79.163722]],
]

# Per-zone comfortable capacity. Varying these is free realism: a library zone
# holds more people than a walkway. It also guarantees the zones differentiate
# under load even though the dataset says they're identical when empty.
ZONE_CAPACITIES = [26, 34, 22, 30, 24, 38, 32, 28]


def build_zones(df: pd.DataFrame | None, feature_cols: list[str]) -> list[ZoneState]:
    """
    Freeze one baseline feature row per zone from the dataset.

    Numeric -> median (robust, deterministic). Categorical -> mode.
    Computed ONCE at boot. Contrast with today's `sample(1)` per request.
    """
    zones: list[ZoneState] = []
    for i, poly in enumerate(HOTSPOT_POLYGONS, start=1):
        zid = f"zone-{i}"
        lat = sum(p[0] for p in poly) / len(poly)
        lng = sum(p[1] for p in poly) / len(poly)
        ap_ids = [f"R-{((i-1)*5+j):02d}" for j in range(1,6)]

        baseline: dict[str, Any] = {}
        base_users, base_util = 14, 12.3

        if df is not None:
            sub = df[
            (abs(df["latitude"] - lat) < 0.0008) &
            (abs(df["longitude"] - lng) < 0.0008)
            ]
            print(f"{zid}: matched {len(sub)} rows")
            print(zid, ap_ids)
            print("Rows matched:", len(sub))
            print(
                f"{zid}: rows={len(sub)} "
                f"users={sub['connected_users'].median() if not sub.empty else 'NONE'} "
                f"util={sub['channel_utilization'].median() if not sub.empty else 'NONE'}"
            )
            if not sub.empty:
                for col in feature_cols:
                    if col not in sub.columns:
                        continue
                    s = sub[col]
                    if pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
                        baseline[col] = float(s.median())
                    else:
                        baseline[col] = s.mode().iloc[0]
                baseline["access_point_id"] = ap_ids[0]
                base_users = int(sub["connected_users"].median())
                base_util = float(sub["channel_utilization"].median())

                print(
                f"{zid}: rows={len(sub)} "
                f"users={base_users} "
                f"util={base_util}"
                )

                print(
                f"{zid}: "
                f"download={sub['download_speed'].median():.1f} "
                f"latency={sub['latency'].median():.1f}"
                )

        zones.append(ZoneState(
            id=zid,
            name=f"Hotspot Zone {i}",
            polygon=poly,
            lat=lat,
            lng=lng,
            ap_ids=ap_ids,
            capacity_users=ZONE_CAPACITIES[i - 1],
            baseline_features=baseline,
            baseline_users=base_users,
            baseline_util=base_util,
        ))
    return zones
