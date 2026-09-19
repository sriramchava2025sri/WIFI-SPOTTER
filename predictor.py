"""
predictor.py — THE SEAM BETWEEN YOUR APP AND WHATEVER MODEL IS CURRENTLY WINNING.

This is the single most important file for future-proofing. Nothing else in the
codebase imports joblib, knows what a one-hot column is, or knows that the
current model needs `access_point_id`. When your teammate's Colab model lands,
you write ONE new class here, flip an env var, and ship. No route changes, no
simulator changes, no frontend changes.

    MODEL_BACKEND=legacy_ap   python app.py     # today's model (default)
    MODEL_BACKEND=spatial     python app.py     # teammate's new model
    MODEL_BACKEND=stub        python app.py     # no model at all (demo safety net)

THE CONTRACT every backend must honour:

    predict_capacity(rows: list[dict]) -> list[float]

    rows  : one dict per zone. Plain python/np scalars, feature names in
            SNAKE_CASE, exactly as the dataset spells them.
    return: link CAPACITY in Mbps for each row, same order.

Note the word CAPACITY, not "speed". This is the deliberate architectural
split explained in simulator.py: the model estimates what the radio link can
do; the contention layer decides how that gets divided among a crowd. Keeping
those two concerns apart is what lets you swap models without re-tuning the
demo.
"""

from __future__ import annotations

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("predictor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------
class SpeedPredictor(ABC):
    """Every model backend implements exactly this. Nothing more."""

    name: str = "abstract"
    #: feature keys this backend actually consumes — used by /api/model-info
    #: so you can see at a glance whether the simulator is feeding it correctly.
    expects: list[str] = []

    @abstractmethod
    def predict_capacity(self, rows: list[dict[str, Any]]) -> list[float]:
        ...

    def describe(self) -> dict:
        return {"backend": self.name, "expects": self.expects}


# ---------------------------------------------------------------------------
# Backend 1 — TODAY'S MODEL (wifi_speed_model.pkl + model_columns.pkl)
# ---------------------------------------------------------------------------
class LegacyAPPredictor(SpeedPredictor):
    """
    Wraps the RandomForest you already trained.

    Its quirk — and the reason this wrapper exists — is that it was trained on
    `pd.get_dummies()` output, so it expects ~100 columns including one-hots
    like `access_point_id_AP-HOTSPOT3-2`. That coupling is a liability: the new
    `wifi_dataset_pure.csv` has no `access_point_id` column at all. By hiding
    the reindex dance in here, the rest of the app never learns about it and
    never has to unlearn it.
    """

    name = "legacy_ap"

    def __init__(self, model_path: str, columns_path: str):
        import joblib

        self.model = joblib.load(model_path)
        self.columns = list(joblib.load(columns_path))
        self.expects = [
            "access_point_id", "area_type", "latitude", "longitude",
            "connected_users", "active_users", "average_data_usage",
            "total_bandwidth_usage", "signal_strength", "noise_level", "snr",
            "channel", "channel_utilization", "interference_level",
            "frequency_band", "day_of_week", "is_weekend", "is_holiday",
            "is_class_hours", "hour", "minute",
        ]
        log.info("LegacyAPPredictor ready — %d training columns", len(self.columns))

    def predict_capacity(self, rows):
        if not rows:
            return []
        frame = pd.DataFrame(rows)
        # Keep only what the model was trained on, then one-hot exactly as
        # training did, then force the column set to match 1:1. `fill_value=0`
        # is what makes unseen categories harmless instead of fatal.
        frame = frame[[c for c in self.expects if c in frame.columns]]
        frame = pd.get_dummies(frame)
        frame = frame.reindex(columns=self.columns, fill_value=0)
        return [float(v) for v in self.model.predict(frame)]


# ---------------------------------------------------------------------------
# Backend 2 — THE NEW COLAB MODEL (fill this in when it arrives)
# ---------------------------------------------------------------------------
class SpatialRouterPredictor(SpeedPredictor):
    """
    Placeholder for the model being trained on `wifi_dataset_pure.csv` +
    `router_locations_pure.csv`.

    That dataset is keyed by `user_id` and lat/lon with NO access_point_id and
    NO area_type, so zone identity has to come from geography (nearest router
    / point-in-polygon) rather than from an ID string. Your teammate should
    hand you two things and nothing else:

        model_v2.pkl        — the fitted estimator
        model_v2_meta.json  — {"features": [...], "target": "download_speed",
                               "categorical": [...], "trained_at": "...",
                               "metrics": {"mae": ..., "r2": ...}}

    Read the feature list from the metadata instead of hardcoding it. That one
    habit is what turns "integration day" into a five-minute job: if they add
    or drop a feature, this class adapts without a code edit.
    """

    name = "spatial"

    def __init__(self, model_path: str, meta_path: str):
        import joblib

        self.model = joblib.load(model_path)
        with open(meta_path) as fh:
            self.meta = json.load(fh)
        self.expects = list(self.meta["features"])
        self.categorical = list(self.meta.get("categorical", []))
        self.columns = self.meta.get("dummy_columns")  # None if pipeline handles it
        log.info("SpatialRouterPredictor ready — %d features", len(self.expects))

    def predict_capacity(self, rows):
        if not rows:
            return []
        frame = pd.DataFrame(rows).reindex(columns=self.expects)
        # Only dummy-encode if the training side did. If your teammate exports
        # a proper sklearn Pipeline with a ColumnTransformer (strongly
        # recommended — tell them this), `dummy_columns` stays null and raw
        # categoricals pass straight through.
        if self.columns:
            frame = pd.get_dummies(frame)
            frame = frame.reindex(columns=self.columns, fill_value=0)
        return [float(v) for v in self.model.predict(frame)]


# ---------------------------------------------------------------------------
# Backend 3 — STUB (your demo insurance policy)
# ---------------------------------------------------------------------------
class StubPredictor(SpeedPredictor):
    """
    Returns a deterministic pseudo-capacity derived from the features, with no
    model file at all. Two uses, both real:

      1. Unit tests and laptop dev without loading a 103 MB pickle.
      2. 3 a.m. on demo day when the pickle won't load / sklearn version
         mismatches. `MODEL_BACKEND=stub` and the crowd-balancing demo still
         runs perfectly, because the interesting behaviour lives in the
         contention layer, not the model.
    """

    name = "stub"
    expects = ["channel_utilization", "snr", "latitude"]

    def predict_capacity(self, rows):
        out = []
        for r in rows:
            util = float(r.get("channel_utilization", 15.0))
            snr = float(r.get("snr", 20.0))
            # ~24 Mbps clean, decaying with airtime occupancy and poor SNR.
            cap = 24.0 - 0.09 * util - max(0.0, (22.0 - snr)) * 0.35
            out.append(float(np.clip(cap, 1.5, 26.0)))
        return out


# ---------------------------------------------------------------------------
# Factory — the only function the rest of the app calls
# ---------------------------------------------------------------------------
def load_predictor() -> SpeedPredictor:
    backend = os.getenv("MODEL_BACKEND", "legacy_ap").strip().lower()

    try:
        if backend == "spatial":
            return SpatialRouterPredictor(
                os.path.join(BASE_DIR, "model_v2.pkl"),
                os.path.join(BASE_DIR, "model_v2_meta.json"),
            )
        if backend == "stub":
            return StubPredictor()
        return LegacyAPPredictor(
            os.path.join(BASE_DIR, "wifi_speed_model.pkl"),
            os.path.join(BASE_DIR, "model_columns.pkl"),
        )
    except Exception as exc:  # noqa: BLE001
        # Never let a bad pickle take down the demo.
        log.error("Backend %r failed to load (%s) — falling back to stub", backend, exc)
        return StubPredictor()
