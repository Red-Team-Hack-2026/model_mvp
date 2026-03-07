# associator.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


def parse_iso8601(ts: str) -> datetime:
    # Handles ...Z
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def dt_ms(a: str, b: str) -> float:
    ta = parse_iso8601(a)
    tb = parse_iso8601(b)
    return abs((ta - tb).total_seconds() * 1000.0)


def cosine(a: List[float], b: List[float]) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-12
    return float(np.dot(aa, bb) / denom)


@dataclass
class AssocGroup:
    group_id: int
    observations: List[Dict[str, Any]] = field(default_factory=list)
    created_ts: Optional[str] = None
    last_ts: Optional[str] = None

    def add(self, obs: Dict[str, Any]) -> None:
        self.observations.append(obs)
        if self.created_ts is None:
            self.created_ts = obs["timestamp"]
        self.last_ts = obs["timestamp"]

    def receivers(self) -> set[str]:
        return {o["receiver_id"] for o in self.observations}

    def size(self) -> int:
        return len(self.observations)

    def observation_ids(self) -> List[str]:
        out = []
        for o in self.observations:
            if "observation_id" in o and o["observation_id"] is not None:
                out.append(str(o["observation_id"]))
        return out

    def centroid_embedding(self) -> Optional[List[float]]:
        embs = [
            o["embedding"] for o in self.observations if o.get("embedding") is not None
        ]
        if not embs:
            return None
        arr = np.asarray(embs, dtype=np.float32)
        mu = arr.mean(axis=0)
        mu /= np.linalg.norm(mu) + 1e-12
        return mu.tolist()

    def dominant_label_id(self) -> Optional[int]:
        if not self.observations:
            return None
        counts = {}
        for o in self.observations:
            counts[o["pred_label_id"]] = counts.get(o["pred_label_id"], 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def dominant_unknown(self) -> bool:
        vals = [bool(o.get("ood_unknown", False)) for o in self.observations]
        return sum(vals) >= max(1, len(vals) // 2)

    def dominant_label_name(self) -> Optional[str]:
        label_id = self.dominant_label_id()
        if label_id is None:
            return None
        for o in self.observations:
            if o.get("pred_label_id") == label_id:
                return o.get("pred_label_name")
        return None

    def is_civilian(self) -> bool:
        name = self.dominant_label_name()
        if not name:
            return False
        nl = str(name).lower()
        return ("am radio" in nl) or ("am-dsb" in nl) or ("am_dsb" in nl)

    def mean_confidence(self) -> Optional[float]:
        vals = [
            o.get("confidence")
            for o in self.observations
            if o.get("confidence") is not None
        ]
        if not vals:
            return None
        return float(sum(float(v) for v in vals) / len(vals))

    def mean_nearest_centroid_cosine_sim(self) -> Optional[float]:
        vals = [
            o.get("nearest_centroid_cosine_sim")
            for o in self.observations
            if o.get("nearest_centroid_cosine_sim") is not None
        ]
        if not vals:
            return None
        return float(sum(float(v) for v in vals) / len(vals))

    def affiliation(self) -> str:
        if self.is_civilian():
            return "civilian"
        return "hostile" if self.dominant_unknown() else "friendly"

    def assurance_pct(self) -> Optional[float]:
        aff = self.affiliation()
        if aff == "friendly":
            c = self.mean_confidence()
            return None if c is None else float(max(0.0, min(1.0, c)) * 100.0)

        s = self.mean_nearest_centroid_cosine_sim()
        return None if s is None else float(max(0.0, min(1.0, s)) * 100.0)

    def summary(self) -> Dict[str, Any]:
        label_id = self.dominant_label_id()
        label_name = None
        for o in self.observations:
            if o["pred_label_id"] == label_id:
                label_name = o["pred_label_name"]
                break

        aff = self.affiliation()

        return {
            "group_id": self.group_id,
            "start_timestamp": self.created_ts,
            "end_timestamp": self.last_ts,
            "num_observations": len(self.observations),
            "observation_ids": self.observation_ids(),
            "receiver_ids": sorted(list(self.receivers())),
            "pred_label_id": label_id,
            "pred_label_name": label_name,
            "ood_unknown": self.dominant_unknown(),
            "affiliation": aff,
            "assurance_pct": self.assurance_pct(),
            "mean_confidence": self.mean_confidence(),
            "mean_nearest_centroid_cosine_sim": self.mean_nearest_centroid_cosine_sim(),
            "observations": self.observations,
        }


class ObservationAssociator:
    def __init__(
        self,
        max_dt_ms: float = 1.0,  # change from 100.0
        min_score: float = 0.55,
        flush_age_ms: float = 2.0,  # change from 120.0
        same_receiver_penalty: float = 0.30,
    ):
        self.max_dt_ms = max_dt_ms
        self.min_score = min_score
        self.flush_age_ms = flush_age_ms
        self.same_receiver_penalty = same_receiver_penalty
        self.groups: List[AssocGroup] = []
        self.next_group_id = 1

    def _time_score(self, obs: Dict[str, Any], group: AssocGroup) -> float:
        if not group.last_ts:
            return 1.0
        d = dt_ms(obs["timestamp"], group.last_ts)
        if d > self.max_dt_ms:
            return 0.0
        return max(0.0, 1.0 - d / self.max_dt_ms)

    def _label_score(self, obs: Dict[str, Any], group: AssocGroup) -> float:
        g_label = group.dominant_label_id()
        if g_label is None:
            return 1.0

        # unknown matches unknown well; known-vs-known prefers same label
        if obs["ood_unknown"] and group.dominant_unknown():
            return 0.8
        if obs["ood_unknown"] != group.dominant_unknown():
            return 0.25
        return 1.0 if obs["pred_label_id"] == g_label else 0.10

    def _embedding_score(self, obs: Dict[str, Any], group: AssocGroup) -> float:
        g_emb = group.centroid_embedding()
        if g_emb is None or obs.get("embedding") is None:
            return 0.5
        sim = cosine(obs["embedding"], g_emb)  # [-1,1], usually [0,1] here
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

    def _rssi_score(self, obs: Dict[str, Any], group: AssocGroup) -> float:
        # weak sanity check only; geolocation will do the real geometry later
        rssis = [o["rssi_dbm"] for o in group.observations]
        if not rssis:
            return 1.0
        mu = sum(rssis) / len(rssis)
        diff = abs(obs["rssi_dbm"] - mu)
        # 0 score at 30 dB away or more
        return max(0.0, 1.0 - diff / 30.0)

    def _toa_score(self, obs: Dict[str, Any], group: AssocGroup) -> float:
        toa = obs.get("time_of_arrival_ns")
        if toa is None:
            return 0.5
        vals = [
            o.get("time_of_arrival_ns")
            for o in group.observations
            if o.get("time_of_arrival_ns") is not None
        ]
        if not vals:
            return 0.5
        mu = sum(vals) / len(vals)
        diff = abs(float(toa) - mu)
        # broad placeholder; tune later when you know network geometry
        # return max(0.0, 1.0 - diff / 1_000_000.0)
        return max(0.0, 1.0 - diff / 10_000)  # try 10_000 instead of 1m

    def _receiver_penalty(self, obs: Dict[str, Any], group: AssocGroup) -> float:
        # return (
        #     self.same_receiver_penalty
        #     if obs["receiver_id"] in group.receivers()
        #     else 0.0
        # )

        # try rejecting instead of penalizing
        if obs["receiver_id"] in group.receivers():
            return 1.0
        return 0.0

    def _compatibility(self, obs: Dict[str, Any], group: AssocGroup) -> float:
        # try rejecting
        if obs["receiver_id"] in group.receivers():
            return -1.0

        s = (
            0.35 * self._time_score(obs, group)
            + 0.25 * self._label_score(obs, group)
            + 0.25 * self._embedding_score(obs, group)
            + 0.10 * self._rssi_score(obs, group)
            + 0.05 * self._toa_score(obs, group)
        )
        s -= self._receiver_penalty(obs, group)
        return s

    def _should_flush(self, newest_obs_ts: str, group: AssocGroup) -> bool:
        if group.last_ts is None:
            return False
        return dt_ms(newest_obs_ts, group.last_ts) > self.flush_age_ms

    def add(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        ready = []

        # flush stale groups first
        still_open = []
        for g in self.groups:
            if self._should_flush(obs["timestamp"], g):
                ready.append(g.summary())
            else:
                still_open.append(g)
        self.groups = still_open

        # pick best matching group
        best_group = None
        best_score = -1.0
        for g in self.groups:
            score = self._compatibility(obs, g)
            if score > best_score:
                best_score = score
                best_group = g

        if best_group is not None and best_score >= self.min_score:
            best_group.add(obs)
        else:
            g = AssocGroup(group_id=self.next_group_id)
            self.next_group_id += 1
            g.add(obs)
            self.groups.append(g)

        return ready

    def flush_all(self) -> List[Dict[str, Any]]:
        out = [g.summary() for g in self.groups]
        self.groups = []
        return out
