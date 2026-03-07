# track_manager.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def parse_iso8601(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def dt_seconds(a: str, b: str) -> float:
    return abs((parse_iso8601(a) - parse_iso8601(b)).total_seconds())


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


@dataclass
class Track:
    track_id: int
    created_at: str
    updated_at: str
    last_seen: str

    latitude: float
    longitude: float
    uncertainty_radius_m: float

    pred_label_id: Optional[int]
    pred_label_name: Optional[str]
    ood_unknown: bool

    affiliation: Optional[str] = None
    assurance_pct: Optional[float] = None
    mean_confidence: Optional[float] = None
    mean_nearest_centroid_cosine_sim: Optional[float] = None

    observation_ids: List[str] = field(default_factory=list)

    num_updates: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self, stale_after_s: float, now_ts: Optional[str] = None) -> Dict[str, Any]:
        ref_ts = now_ts or self.last_seen
        age_s = dt_seconds(ref_ts, self.last_seen)

        return {
            "track_id": self.track_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_seen": self.last_seen,
            "age_s": age_s,
            "is_stale": age_s > stale_after_s,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "uncertainty_radius_m": self.uncertainty_radius_m,
            "pred_label_id": self.pred_label_id,
            "pred_label_name": self.pred_label_name,
            "ood_unknown": self.ood_unknown,
            "affiliation": self.affiliation,
            "assurance_pct": self.assurance_pct,
            "mean_confidence": self.mean_confidence,
            "mean_nearest_centroid_cosine_sim": self.mean_nearest_centroid_cosine_sim,
            "observation_ids": self.observation_ids,
            "num_updates": self.num_updates,
            "history": self.history,
        }


class TrackManager:
    """
    Consumes geolocation fixes and maintains persistent tracks.

    Expected input fix schema:
    {
      "group_id": 1,
      "geolocation_ok": true,
      "pred_label_id": 1,
      "pred_label_name": "BPSK | SATCOM",
      "ood_unknown": false,
      "start_timestamp": "...",
      "end_timestamp": "...",
      "num_observations": 3,
      "receiver_ids": ["rx1","rx2","rx3"],
      "latitude": 49.27,
      "longitude": -123.11,
      "uncertainty_radius_m": 87.4,
      "debug": {...}
    }
    """

    def __init__(
        self,
        match_distance_m: float = 250.0,
        max_time_gap_s: float = 10.0,
        stale_after_s: float = 30.0,
        drop_after_s: float = 120.0,
        alpha_pos: float = 0.35,
        alpha_uncertainty: float = 0.30,
        enemy_after_updates: int = 3,
    ):
        self.match_distance_m = match_distance_m
        self.max_time_gap_s = max_time_gap_s
        self.stale_after_s = stale_after_s
        self.drop_after_s = drop_after_s
        self.alpha_pos = alpha_pos
        self.alpha_uncertainty = alpha_uncertainty
        self.enemy_after_updates = int(enemy_after_updates)

        self.tracks: List[Track] = []
        self.next_track_id = 1

    def _compute_affiliation(self, ood_unknown: bool, num_updates: int) -> str:
        if not bool(ood_unknown):
            return "friendly"
        return "enemy" if int(num_updates) >= self.enemy_after_updates else "unknown"

    def _label_compatible(self, fix: Dict[str, Any], track: Track) -> bool:
        # Unknown can match unknown. Known should usually match same label.
        fix_unknown = bool(fix.get("ood_unknown", False))

        if fix_unknown and track.ood_unknown:
            return True
        if fix_unknown != track.ood_unknown:
            return False

        fix_label = fix.get("pred_label_id")
        if fix_label is None or track.pred_label_id is None:
            return True
        return int(fix_label) == int(track.pred_label_id)

    def _time_compatible(self, fix: Dict[str, Any], track: Track) -> bool:
        ts = fix.get("end_timestamp") or fix.get("start_timestamp")
        if not ts:
            return False
        return dt_seconds(ts, track.last_seen) <= self.max_time_gap_s

    def _distance_to_track_m(self, fix: Dict[str, Any], track: Track) -> float:
        return haversine_m(
            float(fix["latitude"]),
            float(fix["longitude"]),
            track.latitude,
            track.longitude,
        )

    def _gate(self, fix: Dict[str, Any], track: Track) -> bool:
        if not self._label_compatible(fix, track):
            return False
        if not self._time_compatible(fix, track):
            return False

        dist_m = self._distance_to_track_m(fix, track)

        # Dynamic gate: base gate + uncertainty allowance
        gate_m = max(
            self.match_distance_m,
            track.uncertainty_radius_m + float(fix["uncertainty_radius_m"])
        )
        return dist_m <= gate_m

    def _match_score(self, fix: Dict[str, Any], track: Track) -> float:
        dist_m = self._distance_to_track_m(fix, track)
        dt_s = dt_seconds((fix.get("end_timestamp") or fix.get("start_timestamp")), track.last_seen)

        # Normalize to 0..1 where higher is better
        dist_score = max(0.0, 1.0 - dist_m / max(self.match_distance_m, 1.0))
        time_score = max(0.0, 1.0 - dt_s / max(self.max_time_gap_s, 1e-6))

        conf = 0.5
        if "debug" in fix and isinstance(fix["debug"], dict):
            pass

        return 0.70 * dist_score + 0.30 * time_score

    def _create_track(self, fix: Dict[str, Any]) -> Track:
        ts = fix.get("end_timestamp") or fix.get("start_timestamp")
        ood_unknown = bool(fix.get("ood_unknown", False))
        affiliation = self._compute_affiliation(ood_unknown=ood_unknown, num_updates=1)
        t = Track(
            track_id=self.next_track_id,
            created_at=ts,
            updated_at=ts,
            last_seen=ts,
            latitude=float(fix["latitude"]),
            longitude=float(fix["longitude"]),
            uncertainty_radius_m=float(fix["uncertainty_radius_m"]),
            pred_label_id=fix.get("pred_label_id"),
            pred_label_name=fix.get("pred_label_name"),
            ood_unknown=ood_unknown,
            affiliation=affiliation,
            assurance_pct=fix.get("assurance_pct"),
            mean_confidence=fix.get("mean_confidence"),
            mean_nearest_centroid_cosine_sim=fix.get("mean_nearest_centroid_cosine_sim"),
            observation_ids=[str(x) for x in fix.get("observation_ids", []) if x is not None],
            num_updates=1,
            history=[self._history_entry_from_fix(fix)],
        )
        self.next_track_id += 1
        self.tracks.append(t)
        return t

    def _history_entry_from_fix(self, fix: Dict[str, Any]) -> Dict[str, Any]:
        ood_unknown = bool(fix.get("ood_unknown", False))
        # Note: affiliation is computed at track-level for persistence
        affiliation = fix.get("affiliation")
        if affiliation is None:
            affiliation = "unknown" if ood_unknown else "friendly"
        return {
            "group_id": fix.get("group_id"),
            "timestamp": fix.get("end_timestamp") or fix.get("start_timestamp"),
            "latitude": float(fix["latitude"]),
            "longitude": float(fix["longitude"]),
            "uncertainty_radius_m": float(fix["uncertainty_radius_m"]),
            "pred_label_id": fix.get("pred_label_id"),
            "pred_label_name": fix.get("pred_label_name"),
            "ood_unknown": ood_unknown,
            "affiliation": affiliation,
            "assurance_pct": fix.get("assurance_pct"),
            "mean_confidence": fix.get("mean_confidence"),
            "mean_nearest_centroid_cosine_sim": fix.get("mean_nearest_centroid_cosine_sim"),
            "observation_ids": [str(x) for x in fix.get("observation_ids", []) if x is not None],
            "num_observations": fix.get("num_observations"),
            "receiver_ids": fix.get("receiver_ids", []),
        }

    def _update_track(self, track: Track, fix: Dict[str, Any]) -> None:
        # Simple smoothing
        new_lat = float(fix["latitude"])
        new_lon = float(fix["longitude"])
        new_unc = float(fix["uncertainty_radius_m"])

        track.latitude = (1.0 - self.alpha_pos) * track.latitude + self.alpha_pos * new_lat
        track.longitude = (1.0 - self.alpha_pos) * track.longitude + self.alpha_pos * new_lon
        track.uncertainty_radius_m = (
            (1.0 - self.alpha_uncertainty) * track.uncertainty_radius_m
            + self.alpha_uncertainty * new_unc
        )

        track.updated_at = fix.get("end_timestamp") or fix.get("start_timestamp")
        track.last_seen = track.updated_at
        track.num_updates += 1

        # Keep latest classification if compatible
        track.pred_label_id = fix.get("pred_label_id")
        track.pred_label_name = fix.get("pred_label_name")
        track.ood_unknown = bool(fix.get("ood_unknown", False))

        track.affiliation = self._compute_affiliation(
            ood_unknown=track.ood_unknown,
            num_updates=track.num_updates,
        )
        track.assurance_pct = fix.get("assurance_pct")
        track.mean_confidence = fix.get("mean_confidence")
        track.mean_nearest_centroid_cosine_sim = fix.get("mean_nearest_centroid_cosine_sim")

        incoming_ids = [str(x) for x in fix.get("observation_ids", []) if x is not None]
        if incoming_ids:
            seen = set(track.observation_ids)
            for oid in incoming_ids:
                if oid not in seen:
                    track.observation_ids.append(oid)
                    seen.add(oid)

        track.history.append(self._history_entry_from_fix(fix))

    def _purge_old_tracks(self, now_ts: str) -> List[Dict[str, Any]]:
        removed = []
        keep = []
        for t in self.tracks:
            age_s = dt_seconds(now_ts, t.last_seen)
            if age_s > self.drop_after_s:
                removed.append({
                    "event": "track_dropped",
                    "track": t.summary(self.stale_after_s, now_ts=now_ts),
                })
            else:
                keep.append(t)
        self.tracks = keep
        return removed

    def process_fix(self, fix: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns events such as:
          - track_created
          - track_updated
          - track_dropped
        """
        events: List[Dict[str, Any]] = []

        if not fix.get("geolocation_ok", False):
            return events

        ts = fix.get("end_timestamp") or fix.get("start_timestamp")
        if not ts:
            return events

        events.extend(self._purge_old_tracks(ts))

        best_track: Optional[Track] = None
        best_score = -1.0

        for t in self.tracks:
            if not self._gate(fix, t):
                continue
            score = self._match_score(fix, t)
            if score > best_score:
                best_score = score
                best_track = t

        if best_track is None:
            t = self._create_track(fix)
            events.append({
                "event": "track_created",
                "track": t.summary(self.stale_after_s, now_ts=ts),
                "source_fix": fix,
            })
        else:
            self._update_track(best_track, fix)
            events.append({
                "event": "track_updated",
                "track": best_track.summary(self.stale_after_s, now_ts=ts),
                "source_fix": fix,
                "match_score": best_score,
            })

        return events

    def get_tracks(self, now_ts: Optional[str] = None) -> List[Dict[str, Any]]:
        if now_ts is None:
            if not self.tracks:
                return []
            now_ts = max(t.last_seen for t in self.tracks)
        return [t.summary(self.stale_after_s, now_ts=now_ts) for t in self.tracks]

    def flush_all(self, now_ts: Optional[str] = None) -> List[Dict[str, Any]]:
        if now_ts is None:
            if not self.tracks:
                return []
            now_ts = max(t.last_seen for t in self.tracks)

        out = []
        for t in self.tracks:
            out.append({
                "event": "track_final",
                "track": t.summary(self.stale_after_s, now_ts=now_ts),
            })
        self.tracks = []
        return out


def main():
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--match-distance-m", type=float, default=250.0)
    ap.add_argument("--max-time-gap-s", type=float, default=10.0)
    ap.add_argument("--stale-after-s", type=float, default=30.0)
    ap.add_argument("--drop-after-s", type=float, default=120.0)
    ap.add_argument("--alpha-pos", type=float, default=0.35)
    ap.add_argument("--alpha-uncertainty", type=float, default=0.30)
    ap.add_argument("--enemy-after-updates", type=int, default=3)
    args = ap.parse_args()

    tm = TrackManager(
        match_distance_m=args.match_distance_m,
        max_time_gap_s=args.max_time_gap_s,
        stale_after_s=args.stale_after_s,
        drop_after_s=args.drop_after_s,
        alpha_pos=args.alpha_pos,
        alpha_uncertainty=args.alpha_uncertainty,
        enemy_after_updates=args.enemy_after_updates,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        fix = json.loads(line)
        events = tm.process_fix(fix)
        for e in events:
            print(json.dumps(e), flush=True)

    for e in tm.flush_all():
        print(json.dumps(e), flush=True)


if __name__ == "__main__":
    main()