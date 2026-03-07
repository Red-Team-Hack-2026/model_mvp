# geolocate.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


EARTH_RADIUS_M = 6371000.0


def latlon_to_local_xy_m(
    lat: float,
    lon: float,
    lat0: float,
    lon0: float,
) -> Tuple[float, float]:
    """
    Equirectangular local projection around reference point (lat0, lon0).
    Good enough for hackathon-scale areas.
    Returns x, y in meters.
    """
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

    x = (lon_rad - lon0_rad) * math.cos(lat0_rad) * EARTH_RADIUS_M
    y = (lat_rad - lat0_rad) * EARTH_RADIUS_M
    return x, y


def local_xy_to_latlon_m(
    x: float,
    y: float,
    lat0: float,
    lon0: float,
) -> Tuple[float, float]:
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

    lat_rad = y / EARTH_RADIUS_M + lat0_rad
    lon_rad = x / (EARTH_RADIUS_M * math.cos(lat0_rad)) + lon0_rad

    return math.degrees(lat_rad), math.degrees(lon_rad)


def rssi_to_distance_m(
    rssi_dbm: float,
    rssi_ref_dbm: float,
    d_ref_m: float,
    path_loss_exponent: float,
) -> float:
    """
    Invert:
      RSSI = RSSI_ref - 10 * n * log10(d / d_ref)
    =>
      d = d_ref * 10^((RSSI_ref - RSSI)/(10n))
    """
    exp = (rssi_ref_dbm - rssi_dbm) / (10.0 * path_loss_exponent)
    return d_ref_m * (10.0 ** exp)


def weighted_centroid_initial_guess(
    receiver_xy: np.ndarray,
    distances_m: np.ndarray,
) -> np.ndarray:
    """
    Cheap initial guess:
    closer estimated distances get higher weight.
    """
    weights = 1.0 / np.maximum(distances_m, 1.0)
    weights = weights / np.sum(weights)
    return np.sum(receiver_xy * weights[:, None], axis=0)


def solve_position_weighted_least_squares(
    receiver_xy: np.ndarray,
    distances_m: np.ndarray,
    sigma_rssi_db: float = 4.0,
    max_iters: int = 50,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Solve for x,y minimizing:
      sum_i w_i * ( ||p - r_i|| - d_i )^2

    Uses Gauss-Newton.
    Returns:
      p_hat: np.array([x,y])
      uncertainty_radius_m: rough uncertainty estimate
      debug: dict
    """
    if len(receiver_xy) < 3:
        raise ValueError("Need at least 3 receiver observations for RSSI trilateration.")

    p = weighted_centroid_initial_guess(receiver_xy, distances_m)

    # Heuristic weights: closer observations matter more
    w = 1.0 / np.maximum(distances_m, 1.0)
    w = w / np.max(w)

    last_cost = None

    for it in range(max_iters):
        residuals = []
        J_rows = []

        for i in range(len(receiver_xy)):
            rx = receiver_xy[i]
            dx = p[0] - rx[0]
            dy = p[1] - rx[1]
            pred_dist = math.sqrt(dx * dx + dy * dy) + 1e-9

            residual = pred_dist - distances_m[i]
            residuals.append(math.sqrt(w[i]) * residual)

            J_rows.append([
                math.sqrt(w[i]) * (dx / pred_dist),
                math.sqrt(w[i]) * (dy / pred_dist),
            ])

        r = np.asarray(residuals, dtype=np.float64)
        J = np.asarray(J_rows, dtype=np.float64)

        cost = float(np.dot(r, r))

        # Solve GN step: (J^T J) delta = -J^T r
        H = J.T @ J
        g = J.T @ r

        # Small damping for stability
        H = H + 1e-6 * np.eye(2)

        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        p = p + delta

        if last_cost is not None and abs(last_cost - cost) < tol:
            break
        if np.linalg.norm(delta) < tol:
            break

        last_cost = cost

    # Final residuals
    final_res = []
    for i in range(len(receiver_xy)):
        pred = np.linalg.norm(p - receiver_xy[i])
        final_res.append(pred - distances_m[i])
    final_res = np.asarray(final_res, dtype=np.float64)

    rmse = float(np.sqrt(np.mean(final_res ** 2)))

    # Rough uncertainty radius:
    # combine fit residual + geometry condition
    try:
        cov = np.linalg.inv(J.T @ J + 1e-6 * np.eye(2))
        geom_scale = float(np.sqrt(np.trace(cov)))
    except np.linalg.LinAlgError:
        geom_scale = 1000.0

    uncertainty_radius_m = max(10.0, rmse + 25.0 * geom_scale)

    debug = {
        "rmse_m": rmse,
        "iterations_used": it + 1,
        "weights": w.tolist(),
        "estimated_distances_m": distances_m.tolist(),
    }

    return p, uncertainty_radius_m, debug


class RSSIGeolocator:
    def __init__(self, receivers_json: str, path_loss_json: str):
        self.receivers_path = Path(receivers_json)
        self.path_loss_path = Path(path_loss_json)

        self.receivers = self._load_receivers()
        self.path_loss = self._load_path_loss()

    def _load_receivers(self) -> Dict[str, Dict[str, Any]]:
        data = json.loads(self.receivers_path.read_text())

        # supports either list of receivers or {"receivers":[...]}
        if isinstance(data, dict) and "receivers" in data:
            items = data["receivers"]
        elif isinstance(data, list):
            items = data
        else:
            raise ValueError("Receiver config must be a list or {'receivers': [...]}")

        out = {}
        for r in items:
            out[str(r["receiver_id"])] = {
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "sensitivity_dbm": float(r.get("sensitivity_dbm", -120.0)),
                "timing_accuracy_ns": float(r.get("timing_accuracy_ns", 0.0)),
            }
        return out

    def _load_path_loss(self) -> Dict[str, float]:
        data = json.loads(self.path_loss_path.read_text())
        return {
            "rssi_ref_dbm": float(data["rssi_ref_dbm"]),
            "d_ref_m": float(data["d_ref_m"]),
            "path_loss_exponent": float(data["path_loss_exponent"]),
            "rssi_noise_std_db": float(data.get("rssi_noise_std_db", 4.0)),
        }

    def geolocate_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        obs = group.get("observations", [])
        if len(obs) < 3:
            return {
                "group_id": group.get("group_id"),
                "geolocation_ok": False,
                "reason": "need_at_least_3_observations",
            }

        best_by_receiver: Dict[str, Dict[str, Any]] = {}
        for o in obs:
            rid = str(o["receiver_id"])
            if rid not in self.receivers:
                continue
            prev = best_by_receiver.get(rid)
            if prev is None:
                best_by_receiver[rid] = o
                continue
            try:
                if float(o.get("rssi_dbm", -1e9)) > float(prev.get("rssi_dbm", -1e9)):
                    best_by_receiver[rid] = o
            except Exception:
                continue

        usable = list(best_by_receiver.values())

        if len(usable) < 3:
            return {
                "group_id": group.get("group_id"),
                "geolocation_ok": False,
                "reason": "fewer_than_3_known_receivers",
            }

        # Reference origin = mean receiver lat/lon in this group
        lats = [self.receivers[str(o["receiver_id"])]["latitude"] for o in usable]
        lons = [self.receivers[str(o["receiver_id"])]["longitude"] for o in usable]
        lat0 = float(np.mean(lats))
        lon0 = float(np.mean(lons))

        receiver_xy = []
        distances_m = []
        receiver_debug = []

        for o in usable:
            rid = str(o["receiver_id"])
            rmeta = self.receivers[rid]

            x, y = latlon_to_local_xy_m(
                rmeta["latitude"], rmeta["longitude"], lat0, lon0
            )

            d_est = rssi_to_distance_m(
                rssi_dbm=float(o["rssi_dbm"]),
                rssi_ref_dbm=self.path_loss["rssi_ref_dbm"],
                d_ref_m=self.path_loss["d_ref_m"],
                path_loss_exponent=self.path_loss["path_loss_exponent"],
            )
            d_est = float(max(10.0, min(5000.0, d_est)))

            receiver_xy.append([x, y])
            distances_m.append(d_est)

            receiver_debug.append({
                "receiver_id": rid,
                "latitude": rmeta["latitude"],
                "longitude": rmeta["longitude"],
                "rssi_dbm": float(o["rssi_dbm"]),
                "distance_estimate_m": d_est,
            })

        receiver_xy = np.asarray(receiver_xy, dtype=np.float64)
        distances_m = np.asarray(distances_m, dtype=np.float64)

        if len(distances_m) >= 4:
            med = float(np.median(distances_m))
            if med > 0:
                lo = med / 5.0
                hi = med * 5.0
                keep = (distances_m >= lo) & (distances_m <= hi)
                if int(np.sum(keep)) >= 3:
                    receiver_xy = receiver_xy[keep]
                    distances_m = distances_m[keep]
                    receiver_debug = [d for d, k in zip(receiver_debug, keep.tolist()) if k]

        try:
            p_hat, uncertainty_radius_m, solver_debug = solve_position_weighted_least_squares(
                receiver_xy=receiver_xy,
                distances_m=distances_m,
                sigma_rssi_db=self.path_loss["rssi_noise_std_db"],
            )
        except ValueError as e:
            return {
                "group_id": group.get("group_id"),
                "geolocation_ok": False,
                "reason": str(e),
            }

        est_lat, est_lon = local_xy_to_latlon_m(
            p_hat[0], p_hat[1], lat0, lon0
        )

        return {
            "group_id": group.get("group_id"),
            "geolocation_ok": True,
            "pred_label_id": group.get("pred_label_id"),
            "pred_label_name": group.get("pred_label_name"),
            "ood_unknown": group.get("ood_unknown"),
            "affiliation": group.get("affiliation"),
            "assurance_pct": group.get("assurance_pct"),
            "mean_confidence": group.get("mean_confidence"),
            "mean_nearest_centroid_cosine_sim": group.get("mean_nearest_centroid_cosine_sim"),
            "observation_ids": group.get("observation_ids", []),
            "start_timestamp": group.get("start_timestamp"),
            "end_timestamp": group.get("end_timestamp"),
            "num_observations": len(usable),
            "receiver_ids": sorted([str(o["receiver_id"]) for o in usable]),
            "latitude": est_lat,
            "longitude": est_lon,
            "uncertainty_radius_m": uncertainty_radius_m,
            "debug": {
                "reference_origin": {"latitude": lat0, "longitude": lon0},
                "receivers": receiver_debug,
                "solver": solver_debug,
            },
        }


def main():
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--receivers", required=True, help="Path to receiver network JSON")
    ap.add_argument("--path-loss", required=True, help="Path to path-loss JSON")
    args = ap.parse_args()

    geolocator = RSSIGeolocator(args.receivers, args.path_loss)

    # Reads associated groups as JSONL from stdin, prints geolocation fixes as JSONL
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        group = json.loads(line)
        fix = geolocator.geolocate_group(group)
        print(json.dumps(fix), flush=True)


if __name__ == "__main__":
    main()