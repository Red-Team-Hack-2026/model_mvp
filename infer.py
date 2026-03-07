import argparse
import json
from pathlib import Path
import ast

import h5py
import numpy as np
import torch

from rfml.model import IQCNN


def preprocess_iq(x256: np.ndarray, normalize: bool = True) -> np.ndarray:
    x256 = x256.astype(np.float32)
    I = x256[:128] - x256[:128].mean()
    Q = x256[128:] - x256[128:].mean()
    if normalize:
        rms = np.sqrt(np.mean(I*I + Q*Q) + 1e-12)
        I = I / rms
        Q = Q / rms
    return np.stack([I, Q], axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best.pt")
    ap.add_argument("--h5", required=True, help="HDF5 file to sample from")
    ap.add_argument("--key", required=True, help="Dataset key inside the HDF5 to run inference on")
    ap.add_argument("--ood-thresh", type=float, default=0.70, help="Cosine similarity threshold to nearest friendly centroid.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = IQCNN(num_classes=ckpt["num_classes"], emb_dim=ckpt["emb_dim"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(args.device)

    # Load centroids if present
    centroids_path = Path(args.ckpt).with_name("centroids.npy")
    centroids = None
    if centroids_path.exists():
        centroids = np.load(centroids_path, allow_pickle=True).item()

    # Load label metadata if present
    meta_path = Path(args.ckpt).with_name("meta.json")
    id_to_name = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        id_to_name = {int(d["label_id"]): f'{d["modulation"]} | {d["signal_name"]}' for d in meta["labels"]}

    with h5py.File(args.h5, "r") as f:
        x256 = f[args.key][:]

    x = preprocess_iq(x256)
    xt = torch.from_numpy(x).unsqueeze(0).to(args.device)
    with torch.no_grad():
        logits, emb = model(xt)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        e = emb.cpu().numpy()[0]

    # OOD heuristic: nearest centroid similarity
    is_unknown = False
    nearest = None
    if centroids is not None:
        sims = {c: cosine(e, mu) for c, mu in centroids.items()}
        nearest = max(sims.items(), key=lambda kv: kv[1])
        if nearest[1] < args.ood_thresh:
            is_unknown = True

    name = id_to_name.get(pred, str(pred)) if id_to_name else str(pred)
    print(json.dumps({
        "pred_label_id": pred,
        "pred_label_name": name,
        "confidence": conf,
        "ood_unknown": is_unknown,
        "nearest_centroid": None if nearest is None else {"label_id": int(nearest[0]), "cosine_sim": float(nearest[1])},
    }, indent=2))


if __name__ == "__main__":
    main()
