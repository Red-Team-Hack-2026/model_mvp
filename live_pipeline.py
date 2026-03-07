from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from rfml.model import IQCNN
from associator import ObservationAssociator
from unknown_labeler import UnknownSignalLabeler


def preprocess_iq(x256: np.ndarray, normalize: bool = True) -> np.ndarray:
    x256 = x256.astype(np.float32)
    I = x256[:128] - x256[:128].mean()
    Q = x256[128:] - x256[128:].mean()
    if normalize:
        rms = np.sqrt(np.mean(I * I + Q * Q) + 1e-12)
        I = I / rms
        Q = Q / rms
    return np.stack([I, Q], axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


class LiveInferenceEngine:
    def __init__(self, ckpt_path: str, ood_thresh: float = 0.70, device: Optional[str] = None):
        self.ckpt_path = Path(ckpt_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ood_thresh = ood_thresh
        self.unknown_labeler = UnknownSignalLabeler()

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.model = IQCNN(num_classes=ckpt["num_classes"], emb_dim=ckpt["emb_dim"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)

        centroids_path = self.ckpt_path.with_name("centroids.npy")
        self.centroids = None
        if centroids_path.exists():
            self.centroids = np.load(centroids_path, allow_pickle=True).item()

        meta_path = self.ckpt_path.with_name("meta.json")
        self.id_to_signal_type = {}
        self.id_to_modulation = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            signal_type_overrides = {
                "SATCOM": "Satcom",
            }
            for d in meta["labels"]:
                lid = int(d["label_id"])
                modulation = str(d["modulation"])
                signal_type = str(d["signal_name"])
                signal_type = signal_type_overrides.get(signal_type, signal_type)

                self.id_to_signal_type[lid] = signal_type
                self.id_to_modulation[lid] = modulation

    @torch.no_grad()
    def infer_one_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        iq = np.asarray(obs["iq_snapshot"], dtype=np.float32)
        x = preprocess_iq(iq)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)

        logits, emb = self.model(xt)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        e = emb.cpu().numpy()[0]

        is_unknown = False
        nearest = None
        if self.centroids is not None:
            sims = {c: cosine(e, mu) for c, mu in self.centroids.items()}
            nearest = max(sims.items(), key=lambda kv: kv[1])
            if nearest[1] < self.ood_thresh:
                is_unknown = True

        rule_decision = None
        if is_unknown:
            try:
                rule_decision = self.unknown_labeler.label_iq_snapshot(obs["iq_snapshot"])
            except Exception:
                rule_decision = None
            if rule_decision is not None:
                pred = int(rule_decision.pred_label_id)
                conf = float(rule_decision.confidence)

        pred_signal_type = self.id_to_signal_type.get(pred, str(pred))
        pred_modulation = self.id_to_modulation.get(pred)
        if rule_decision is not None:
            pred_signal_type = rule_decision.pred_label_name
            pred_modulation = rule_decision.pred_modulation

        return {
            "observation_id": str(obs["observation_id"]),
            "timestamp": str(obs["timestamp"]),
            "receiver_id": str(obs["receiver_id"]),
            "rssi_dbm": float(obs["rssi_dbm"]),
            "snr_estimate_db": float(obs["snr_estimate_db"]),
            "time_of_arrival_ns": obs.get("time_of_arrival_ns"),
            "pred_label_id": pred,
            "pred_label_name": pred_signal_type,
            "pred_modulation": pred_modulation,
            "confidence": conf,
            "ood_unknown": is_unknown,
            "ood_thresh": float(self.ood_thresh),
            "nearest_centroid_cosine_sim": None if nearest is None else float(nearest[1]),
            "nearest_centroid": None if nearest is None else {
                "label_id": int(nearest[0]),
                "cosine_sim": float(nearest[1]),
            },
            "embedding": e.astype(np.float32).tolist(),
        }


def main():
    """
    Reads raw observations as JSON lines from stdin.
    Emits associated groups as JSON lines to stdout.
    """
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ood-thresh", type=float, default=0.70)
    ap.add_argument("--assoc-window-ms", type=float, default=100.0)
    ap.add_argument("--assoc-score-thresh", type=float, default=0.55)
    args = ap.parse_args()

    infer_engine = LiveInferenceEngine(args.ckpt, ood_thresh=args.ood_thresh)
    associator = ObservationAssociator(
        max_dt_ms=args.assoc_window_ms,
        min_score=args.assoc_score_thresh,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        raw_obs = json.loads(line)
        enriched = infer_engine.infer_one_observation(raw_obs)

        ready_groups = associator.add(enriched)
        for g in ready_groups:
            print(json.dumps(g), flush=True)

    # flush leftovers on shutdown
    for g in associator.flush_all():
        print(json.dumps(g), flush=True)


if __name__ == "__main__":
    main()