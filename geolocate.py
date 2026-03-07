# geolocate.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EARTH_RADIUS_M = 6371000.0
C_M_PER_NS = 0.299792458


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
    return d_ref_m * (10.0**exp)


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
        raise ValueError(
            "Need at least 3 receiver observations for RSSI trilateration."
        )

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

            J_rows.append(
                [
                    math.sqrt(w[i]) * (dx / pred_dist),
                    math.sqrt(w[i]) * (dy / pred_dist),
                ]
            )

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

    rmse = float(np.sqrt(np.mean(final_res**2)))

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


def choose_reference_observation(
    observations: List[Dict[str, Any]],
    receivers: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return min(
        observations,
        key=lambda o: (
            receivers[str(o["receiver_id"])]["timing_accuracy_ns"],
            -float(o.get("snr_estimate_db", 0.0)),
            abs(float(o.get("rssi_dbm", -999.0))),
        ),
    )


def centroid_initial_guess(receiver_xy: np.ndarray) -> np.ndarray:
    return np.mean(receiver_xy, axis=0)


def solve_position_tdoa_gauss_newton(
    ref_xy: Tuple[float, float],
    other_receiver_xy: np.ndarray,
    delta_d_m: np.ndarray,
    sigma_d_m: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    if len(other_receiver_xy) < 2:
        raise ValueError("Need at least 3 receivers total for 2D TDoA.")

    ref_xy_arr = np.asarray(ref_xy, dtype=np.float64)
    all_xy = np.vstack([ref_xy_arr[None, :], other_receiver_xy])
    p = centroid_initial_guess(all_xy)

    last_cost = None
    it = 0

    for it in range(max_iters):
        residuals = []
        J_rows = []

        dx_ref = p[0] - ref_xy_arr[0]
        dy_ref = p[1] - ref_xy_arr[1]
        d_ref = math.sqrt(dx_ref * dx_ref + dy_ref * dy_ref) + 1e-9

        for i in range(len(other_receiver_xy)):
            rx = other_receiver_xy[i]

            dx = p[0] - rx[0]
            dy = p[1] - rx[1]
            d_i = math.sqrt(dx * dx + dy * dy) + 1e-9

            pred_dd = d_i - d_ref
            resid = (pred_dd - delta_d_m[i]) / sigma_d_m[i]
            residuals.append(resid)

            jx = (dx / d_i - dx_ref / d_ref) / sigma_d_m[i]
            jy = (dy / d_i - dy_ref / d_ref) / sigma_d_m[i]
            J_rows.append([jx, jy])

        r = np.asarray(residuals, dtype=np.float64)
        J = np.asarray(J_rows, dtype=np.float64)

        cost = float(np.dot(r, r))

        H = J.T @ J
        g = J.T @ r
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

    dx_ref = p[0] - ref_xy_arr[0]
    dy_ref = p[1] - ref_xy_arr[1]
    d_ref = math.sqrt(dx_ref * dx_ref + dy_ref * dy_ref) + 1e-9

    final_res = []
    final_J = []

    for i in range(len(other_receiver_xy)):
        rx = other_receiver_xy[i]

        dx = p[0] - rx[0]
        dy = p[1] - rx[1]
        d_i = math.sqrt(dx * dx + dy * dy) + 1e-9

        pred_dd = d_i - d_ref
        resid = pred_dd - delta_d_m[i]
        final_res.append(resid)

        jx = (dx / d_i - dx_ref / d_ref) / sigma_d_m[i]
        jy = (dy / d_i - dy_ref / d_ref) / sigma_d_m[i]
        final_J.append([jx, jy])

    final_res = np.asarray(final_res, dtype=np.float64)
    final_J = np.asarray(final_J, dtype=np.float64)

    rmse_m = float(np.sqrt(np.mean(final_res**2))) if len(final_res) else float("inf")

    try:
        cov = np.linalg.inv(final_J.T @ final_J + 1e-6 * np.eye(2))
        geom_scale = float(np.sqrt(np.trace(cov)))
    except np.linalg.LinAlgError:
        geom_scale = 1000.0

    uncertainty_radius_m = max(5.0, rmse_m + 15.0 * geom_scale)

    debug = {
        "rmse_range_difference_m": rmse_m,
        "iterations_used": it + 1,
        "delta_d_m": delta_d_m.tolist(),
        "sigma_d_m": sigma_d_m.tolist(),
    }

    return p, uncertainty_radius_m, debug


class BaseGeolocator:
    def __init__(self, receivers_json: str, path_loss_json: str):
        self.receivers_path = Path(receivers_json)
        self.path_loss_path = Path(path_loss_json)

        self.receivers = self._load_receivers()
        self.path_loss = self._load_path_loss()

    def _load_receivers(self) -> Dict[str, Dict[str, Any]]:
        data = json.loads(self.receivers_path.read_text())

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
                "timing_accuracy_ns": float(r.get("timing_accuracy_ns", 50.0)),
                "timing_bias_ns": float(r.get("timing_bias_ns", 0.0)),
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

    def _get_usable_observations(self, group: Dict[str, Any]) -> List[Dict[str, Any]]:
        obs = group.get("observations", [])
        usable = []

        for o in obs:
            rid = str(o["receiver_id"])
            if rid in self.receivers:
                usable.append(o)

        return usable

    def _compute_reference_origin(
        self,
        usable: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        lats = [self.receivers[str(o["receiver_id"])]["latitude"] for o in usable]
        lons = [self.receivers[str(o["receiver_id"])]["longitude"] for o in usable]
        return float(np.mean(lats)), float(np.mean(lons))

    def _build_common_success_result(
        self,
        group: Dict[str, Any],
        usable: List[Dict[str, Any]],
        est_lat: float,
        est_lon: float,
        uncertainty_radius_m: float,
        method: str,
        debug: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "group_id": group.get("group_id"),
            "geolocation_ok": True,
            "method": method,
            "pred_label_id": group.get("pred_label_id"),
            "pred_label_name": group.get("pred_label_name"),
            "ood_unknown": group.get("ood_unknown"),
            "affiliation": group.get("affiliation"),
            "assurance_pct": group.get("assurance_pct"),
            "mean_confidence": group.get("mean_confidence"),
            "mean_nearest_centroid_cosine_sim": group.get(
                "mean_nearest_centroid_cosine_sim"
            ),
            "observation_ids": group.get("observation_ids", []),
            "start_timestamp": group.get("start_timestamp"),
            "end_timestamp": group.get("end_timestamp"),
            "num_observations": len(usable),
            "receiver_ids": sorted([str(o["receiver_id"]) for o in usable]),
            "latitude": est_lat,
            "longitude": est_lon,
            "uncertainty_radius_m": uncertainty_radius_m,
            "debug": debug,
        }

    def _build_failure_result(
        self,
        group: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "group_id": group.get("group_id"),
            "geolocation_ok": False,
            "reason": reason,
        }

    def geolocate_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class RSSIGeolocator(BaseGeolocator):
    def geolocate_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        usable = self._get_usable_observations(group)
        if len(usable) < 3:
            return self._build_failure_result(group, "fewer_than_3_known_receivers")

        lat0, lon0 = self._compute_reference_origin(usable)

        receiver_xy = []
        distances_m = []
        receiver_debug = []

        for o in usable:
            rid = str(o["receiver_id"])
            rmeta = self.receivers[rid]

            x, y = latlon_to_local_xy_m(
                rmeta["latitude"],
                rmeta["longitude"],
                lat0,
                lon0,
            )

            d_est = rssi_to_distance_m(
                rssi_dbm=float(o["rssi_dbm"]),
                rssi_ref_dbm=self.path_loss["rssi_ref_dbm"],
                d_ref_m=self.path_loss["d_ref_m"],
                path_loss_exponent=self.path_loss["path_loss_exponent"],
            )

            receiver_xy.append([x, y])
            distances_m.append(d_est)

            receiver_debug.append(
                {
                    "receiver_id": rid,
                    "latitude": rmeta["latitude"],
                    "longitude": rmeta["longitude"],
                    "rssi_dbm": float(o["rssi_dbm"]),
                    "distance_estimate_m": d_est,
                }
            )

        receiver_xy = np.asarray(receiver_xy, dtype=np.float64)
        distances_m = np.asarray(distances_m, dtype=np.float64)

        try:
            p_hat, uncertainty_radius_m, solver_debug = (
                solve_position_weighted_least_squares(
                    receiver_xy=receiver_xy,
                    distances_m=distances_m,
                    sigma_rssi_db=self.path_loss["rssi_noise_std_db"],
                )
            )
        except ValueError as e:
            return self._build_failure_result(group, str(e))

        est_lat, est_lon = local_xy_to_latlon_m(p_hat[0], p_hat[1], lat0, lon0)

        return self._build_common_success_result(
            group=group,
            usable=usable,
            est_lat=est_lat,
            est_lon=est_lon,
            uncertainty_radius_m=uncertainty_radius_m,
            method="rssi",
            debug={
                "reference_origin": {"latitude": lat0, "longitude": lon0},
                "receivers": receiver_debug,
                "solver": solver_debug,
            },
        )


class TDoAGeolocator(BaseGeolocator):
    def geolocate_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        usable = self._get_usable_observations(group)
        usable = [o for o in usable if "time_of_arrival_ns" in o]

        if len(usable) < 3:
            return self._build_failure_result(
                group, "fewer_than_3_time_of_arrival_observations"
            )

        lat0, lon0 = self._compute_reference_origin(usable)

        ref_obs = choose_reference_observation(usable, self.receivers)
        ref_rid = str(ref_obs["receiver_id"])
        ref_meta = self.receivers[ref_rid]

        ref_x, ref_y = latlon_to_local_xy_m(
            ref_meta["latitude"],
            ref_meta["longitude"],
            lat0,
            lon0,
        )

        ref_toa_ns = float(ref_obs["time_of_arrival_ns"]) - float(
            ref_meta.get("timing_bias_ns", 0.0)
        )
        ref_sigma_ns = float(ref_meta.get("timing_accuracy_ns", 50.0))

        other_xy = []
        delta_d_m = []
        sigma_d_m = []
        measurement_debug = []

        for o in usable:
            rid = str(o["receiver_id"])
            if rid == ref_rid:
                continue

            rmeta = self.receivers[rid]
            x, y = latlon_to_local_xy_m(
                rmeta["latitude"],
                rmeta["longitude"],
                lat0,
                lon0,
            )

            toa_ns = float(o["time_of_arrival_ns"]) - float(
                rmeta.get("timing_bias_ns", 0.0)
            )
            delta_t_ns = toa_ns - ref_toa_ns
            delta_range_m = delta_t_ns * C_M_PER_NS

            rx_sigma_ns = float(rmeta.get("timing_accuracy_ns", 50.0))
            total_sigma_ns = math.sqrt(ref_sigma_ns**2 + rx_sigma_ns**2)
            total_sigma_m = max(total_sigma_ns * C_M_PER_NS, 1e-3)

            other_xy.append([x, y])
            delta_d_m.append(delta_range_m)
            sigma_d_m.append(total_sigma_m)

            measurement_debug.append(
                {
                    "receiver_id": rid,
                    "latitude": rmeta["latitude"],
                    "longitude": rmeta["longitude"],
                    "time_of_arrival_ns": float(o["time_of_arrival_ns"]),
                    "corrected_time_of_arrival_ns": toa_ns,
                    "delta_t_ns": delta_t_ns,
                    "delta_d_m": delta_range_m,
                    "sigma_d_m": total_sigma_m,
                    "rssi_dbm": float(o.get("rssi_dbm", 0.0)),
                    "snr_estimate_db": float(o.get("snr_estimate_db", 0.0)),
                }
            )

        other_xy = np.asarray(other_xy, dtype=np.float64)
        delta_d_m = np.asarray(delta_d_m, dtype=np.float64)
        sigma_d_m = np.asarray(sigma_d_m, dtype=np.float64)

        try:
            p_hat, uncertainty_radius_m, solver_debug = (
                solve_position_tdoa_gauss_newton(
                    ref_xy=(ref_x, ref_y),
                    other_receiver_xy=other_xy,
                    delta_d_m=delta_d_m,
                    sigma_d_m=sigma_d_m,
                )
            )
        except ValueError as e:
            return self._build_failure_result(group, str(e))

        est_lat, est_lon = local_xy_to_latlon_m(p_hat[0], p_hat[1], lat0, lon0)

        return self._build_common_success_result(
            group=group,
            usable=usable,
            est_lat=est_lat,
            est_lon=est_lon,
            uncertainty_radius_m=uncertainty_radius_m,
            method="tdoa",
            debug={
                "reference_origin": {"latitude": lat0, "longitude": lon0},
                "reference_receiver_id": ref_rid,
                "reference_receiver": {
                    "latitude": ref_meta["latitude"],
                    "longitude": ref_meta["longitude"],
                    "time_of_arrival_ns": float(ref_obs["time_of_arrival_ns"]),
                    "corrected_time_of_arrival_ns": ref_toa_ns,
                    "timing_accuracy_ns": ref_sigma_ns,
                },
                "tdoa_measurements": measurement_debug,
                "solver": solver_debug,
            },
        )


def main():
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--receivers", required=True, help="Path to receiver network JSON")
    ap.add_argument("--path-loss", required=True, help="Path to path-loss JSON")
    args = ap.parse_args()

    # geolocator = RSSIGeolocator(args.receivers, args.path_loss)
    geolocator = TDoAGeolocator(args.receivers, args.path_loss)

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
