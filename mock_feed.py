# mock_feed.py
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import h5py
import numpy as np


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_iq_from_h5(h5_path: str, key: str) -> List[float]:
    with h5py.File(h5_path, "r") as f:
        return f[key][:].astype(np.float32).tolist()


def first_matching_keys(h5_path: str, contains: List[str], limit: int = 10) -> List[str]:
    contains = [c.lower() for c in contains]
    out = []
    with h5py.File(h5_path, "r") as f:
        for k in f.keys():
            kl = k.lower()
            if all(c in kl for c in contains):
                out.append(k)
                if len(out) >= limit:
                    break
    return out


def random_iq_snapshot(seed: Optional[int] = None) -> List[float]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(256,)).astype(np.float32)
    return x.tolist()


def make_obs(
    observation_id: str,
    timestamp: str,
    receiver_id: str,
    rssi_dbm: float,
    iq_snapshot: List[float],
    snr_estimate_db: float,
    time_of_arrival_ns: Optional[float],
) -> Dict[str, Any]:
    return {
        "observation_id": observation_id,
        "timestamp": timestamp,
        "receiver_id": receiver_id,
        "rssi_dbm": float(rssi_dbm),
        "iq_snapshot": [float(x) for x in iq_snapshot],
        "snr_estimate_db": float(snr_estimate_db),
        "time_of_arrival_ns": None if time_of_arrival_ns is None else float(time_of_arrival_ns),
    }


def emit_jsonl(observations: List[Dict[str, Any]], delay_s: float = 0.0) -> None:
    for obs in observations:
        print(json.dumps(obs), flush=True)
        if delay_s > 0:
            time.sleep(delay_s)


def build_dataset_mode_observations(
    h5_path: str,
    key: str,
    num_receivers: int = 3,
    base_receiver_ids: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    id_prefix: str = "obs",
) -> List[Dict[str, Any]]:
    """
    Emits a small cluster of near-simultaneous observations that should
    be associated as one emitter event.
    """
    if base_receiver_ids is None:
        base_receiver_ids = [f"rx{i+1}" for i in range(num_receivers)]

    iq = load_iq_from_h5(h5_path, key)
    t0 = start_time or datetime.now(timezone.utc)

    # Slightly different RSSI/ToA per receiver
    rssi_values = [-64.0, -68.0, -72.0, -75.0, -78.0]
    toa_values = [1000.0, 1015.0, 1030.0, 1050.0, 1080.0]
    snr_values = [6.0, 4.5, 3.0, 1.5, 0.5]

    obs = []
    for i, rid in enumerate(base_receiver_ids):
        ts = iso_z(t0 + timedelta(milliseconds=20 * i))
        obs.append(
            make_obs(
                observation_id=f"{id_prefix}_{i+1}",
                timestamp=ts,
                receiver_id=rid,
                rssi_dbm=rssi_values[i],
                iq_snapshot=iq,
                snr_estimate_db=snr_values[i],
                time_of_arrival_ns=toa_values[i],
            )
        )
    return obs


def build_two_emitter_dataset_mode(
    h5_path: str,
    key1: str,
    key2: str,
    receiver_ids: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    id_prefix: str = "emit",
) -> List[Dict[str, Any]]:
    """
    Builds 2 emitters interleaved in time so you can test association splitting.
    """
    if receiver_ids is None:
        receiver_ids = ["rx1", "rx2", "rx3"]

    iq1 = load_iq_from_h5(h5_path, key1)
    iq2 = load_iq_from_h5(h5_path, key2)
    t0 = start_time or datetime.now(timezone.utc)

    observations = []

    # emitter A
    for i, rid in enumerate(receiver_ids):
        observations.append(
            make_obs(
                observation_id=f"{id_prefix}_A_{i+1}",
                timestamp=iso_z(t0 + timedelta(milliseconds=10 * i)),
                receiver_id=rid,
                rssi_dbm=[-63.0, -67.0, -71.0][i],
                iq_snapshot=iq1,
                snr_estimate_db=[7.0, 5.5, 3.5][i],
                time_of_arrival_ns=[1000.0, 1012.0, 1024.0][i],
            )
        )

    # emitter B slightly later but overlapping
    for i, rid in enumerate(receiver_ids):
        observations.append(
            make_obs(
                observation_id=f"{id_prefix}_B_{i+1}",
                timestamp=iso_z(t0 + timedelta(milliseconds=55 + 10 * i)),
                receiver_id=rid,
                rssi_dbm=[-70.0, -66.0, -74.0][i],
                iq_snapshot=iq2,
                snr_estimate_db=[2.5, 4.0, 1.5][i],
                time_of_arrival_ns=[2000.0, 2015.0, 2032.0][i],
            )
        )

    observations.sort(key=lambda o: o["timestamp"])
    return observations


def build_random_mode_observations(
    num_observations: int = 6,
    receiver_ids: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    id_prefix: str = "rand",
) -> List[Dict[str, Any]]:
    if receiver_ids is None:
        receiver_ids = ["rx1", "rx2", "rx3"]

    t0 = start_time or datetime.now(timezone.utc)
    out = []

    for i in range(num_observations):
        rid = receiver_ids[i % len(receiver_ids)]
        out.append(
            make_obs(
                observation_id=f"{id_prefix}_{i+1}",
                timestamp=iso_z(t0 + timedelta(milliseconds=25 * i)),
                receiver_id=rid,
                rssi_dbm=random.uniform(-85, -55),
                iq_snapshot=random_iq_snapshot(seed=i),
                snr_estimate_db=random.uniform(-5, 8),
                time_of_arrival_ns=1000.0 + i * 12.0,
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["dataset", "dataset-two", "random"], default="dataset")
    ap.add_argument("--h5", help="Path to HDF5 dataset")
    ap.add_argument("--key", help="Single HDF5 key for dataset mode")
    ap.add_argument("--key2", help="Second HDF5 key for dataset-two mode")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="Sleep between emitted observations")
    ap.add_argument("--find", nargs="*",
                    help="Search substrings for a key, e.g. bpsk Satcom -10")
    ap.add_argument("--repeat", action="store_true",
                    help="Repeat emitting observation bursts forever")
    ap.add_argument("--repeat-delay", type=float, default=0.5,
                    help="Sleep between repeated bursts")
    args = ap.parse_args()

    if args.mode in {"dataset", "dataset-two"} and not args.h5:
        raise SystemExit("--h5 is required for dataset modes")

    if args.find and not args.key:
        matches = first_matching_keys(args.h5, args.find, limit=20)
        print("# matching keys", file=sys.stderr)
        for m in matches:
            print(m, file=sys.stderr)
        if not matches:
            raise SystemExit("No matching keys found")
        args.key = matches[0]

    run_idx = 0
    while True:
        run_idx += 1
        prefix = f"run{run_idx}"

        if args.mode == "dataset":
            if not args.key:
                raise SystemExit("--key is required for dataset mode")
            observations = build_dataset_mode_observations(
                args.h5, args.key, id_prefix=prefix)

        elif args.mode == "dataset-two":
            if not args.key or not args.key2:
                raise SystemExit(
                    "--key and --key2 are required for dataset-two mode")
            observations = build_two_emitter_dataset_mode(
                args.h5, args.key, args.key2, id_prefix=prefix)

        else:
            observations = build_random_mode_observations(id_prefix=prefix)

        emit_jsonl(observations, delay_s=args.delay)

        if not args.repeat:
            break

        if args.repeat_delay > 0:
            time.sleep(args.repeat_delay)


if __name__ == "__main__":
    import sys
    main()

