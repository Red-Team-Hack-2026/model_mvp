import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from rfml.dataset import build_index, H5IQDataset
from rfml.model import IQCNN
from rfml.utils import seed_everything, stratified_split, accuracy, compute_class_centroids

def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", nargs="+", required=True, help="One or more .hdf5 files for training (friendly-labeled).")
    ap.add_argument("--out", default="runs/rf_model", help="Output folder for checkpoints + metadata.")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--freeze-epochs", type=int, default=1, help="When resuming, freeze feature extractor for N epochs (default 1).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Resume / fine-tune
    ap.add_argument("--resume", default=None, help="Path to a checkpoint (best.pt) to continue training from.")
    ap.add_argument("--lr-resume", type=float, default=5e-4, help="LR to use when resuming (usually smaller).")

    args = ap.parse_args()
    seed_everything(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Build index + labels
    # ----------------------------
    refs, label_map, meta = build_index(args.h5, label_map=None)
    with (out / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    labels = [r.label for r in refs]
    indices = list(range(len(refs)))
    tr_idx, va_idx = stratified_split(indices, labels, val_frac=args.val_frac, seed=args.seed)

    ds = H5IQDataset(refs, normalize=True)
    dl_tr = DataLoader(Subset(ds, tr_idx), batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    dl_va = DataLoader(Subset(ds, va_idx), batch_size=args.batch, shuffle=False, num_workers=0)

    num_classes = len(label_map)

    # ----------------------------
    # Build model
    # ----------------------------
    model = IQCNN(num_classes=num_classes, emb_dim=args.emb_dim).to(args.device)

    # If resuming, load weights from previous best.pt
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")

        # Safety checks: architecture must match
        if ckpt.get("num_classes") != num_classes:
            raise ValueError(
                f"Checkpoint num_classes={ckpt.get('num_classes')} but current label_map has {num_classes}. "
                f"Make sure your friendly label set didn't change."
            )
        if ckpt.get("emb_dim") != args.emb_dim:
            raise ValueError(
                f"Checkpoint emb_dim={ckpt.get('emb_dim')} but you requested emb_dim={args.emb_dim}."
            )

        model.load_state_dict(ckpt["model"])
        print(f"Resumed weights from: {args.resume}")

    # ----------------------------
    # Loss (class weights)
    # ----------------------------
    counts = np.bincount(np.array(labels), minlength=num_classes).astype(np.float32)
    weights = (counts.sum() / (counts + 1e-6))
    weights = weights / weights.mean()
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=args.device))

    # ----------------------------
    # Optimizer / schedule
    # ----------------------------
    lr = args.lr_resume if args.resume else args.lr
    print(f"Using learning rate: {lr}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    # ----------------------------
    # Train loop
    # ----------------------------
    best = -1.0

    for ep in range(1, args.epochs + 1):
        # ---- Freeze/unfreeze schedule (only when resuming) ----
        if args.resume and args.freeze_epochs > 0:
            if ep <= args.freeze_epochs:
                # Freeze feature extractor
                set_requires_grad(model.feat, False)
                set_requires_grad(model.emb, True)
                set_requires_grad(model.cls, True)
                if ep == 1:
                    print(f"[resume] Freezing model.feat for {args.freeze_epochs} epoch(s); training emb+cls only.")
            else:
                # Unfreeze everything
                set_requires_grad(model.feat, True)
                set_requires_grad(model.emb, True)
                set_requires_grad(model.cls, True)
                if ep == args.freeze_epochs + 1:
                    print("[resume] Unfreezing all layers; fine-tuning full model.")
        
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        n_tr = 0

        for x, y, snr in dl_tr:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits, emb = model(x)
            loss = ce(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tr_loss += loss.item() * x.size(0)
            tr_acc += accuracy(logits, y) * x.size(0)
            n_tr += x.size(0)

        # Validation
        model.eval()
        va_loss = 0.0
        va_acc = 0.0
        n_va = 0
        all_emb = []
        all_y = []

        with torch.no_grad():
            for x, y, snr in dl_va:
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                logits, emb = model(x)
                loss = ce(logits, y)

                va_loss += loss.item() * x.size(0)
                va_acc += accuracy(logits, y) * x.size(0)
                n_va += x.size(0)

                all_emb.append(emb.detach().cpu().numpy())
                all_y.append(y.detach().cpu().numpy())

        tr_loss /= max(1, n_tr)
        tr_acc /= max(1, n_tr)
        va_loss /= max(1, n_va)
        va_acc /= max(1, n_va)

        print(
            f"epoch {ep:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

        # Save best checkpoint + centroids
        if va_acc > best:
            best = va_acc

            ckpt_out = {
                "model": model.state_dict(),
                "num_classes": num_classes,
                "emb_dim": args.emb_dim,
                "label_map": {f"{k[0]}|{k[1]}": v for k, v in label_map.items()},
                "seed": args.seed,
            }
            torch.save(ckpt_out, out / "best.pt")

            emb_np = np.concatenate(all_emb, axis=0)
            y_np = np.concatenate(all_y, axis=0)
            centroids = compute_class_centroids(emb_np, y_np)
            np.save(out / "centroids.npy", centroids, allow_pickle=True)

        sched.step()

    # Final save
    torch.save(
        {"model": model.state_dict(), "num_classes": num_classes, "emb_dim": args.emb_dim},
        out / "last.pt",
    )
    ds.close()
    print(f"done. best val acc={best:.4f}. artifacts in: {out}")


if __name__ == "__main__":
    main()