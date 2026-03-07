from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import urllib.request
except Exception:  # pragma: no cover
    urllib = None  # type: ignore

try:
    import certifi  # type: ignore
except Exception:  # pragma: no cover
    certifi = None  # type: ignore

from associator import ObservationAssociator
from geolocate import RSSIGeolocator
from live_pipeline import LiveInferenceEngine

BATCH_URL = "https://findmyforce.online/submissions/batch"


def submit_batch(rows: List[Dict[str, Any]], api_key: str, verify_ssl: bool = False) -> None:
    """POST up to 100 classifications via the batch endpoint."""
    if not rows:
        return
    payload = json.dumps({"submissions": rows}).encode()
    req = urllib.request.Request(BATCH_URL, data=payload, method="POST")  # type: ignore[attr-defined]
    req.add_header("X-API-Key", api_key)
    req.add_header("Content-Type", "application/json")
    ctx = ssl._create_unverified_context() if not verify_ssl else None
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:  # type: ignore[attr-defined]
            result = json.loads(resp.read())
        acc = result.get("accepted_count", 0)
        rej = result.get("rejected_count", 0)
        print(f"[BATCH] sent={len(rows)} accepted={acc} rejected={rej}", flush=True)
        for r in result.get("results", []):
            tag = "ACCEPTED" if r.get("accepted") else "REJECTED"
            print(f"  [{tag}] obs={r.get('observation_id', '?')[:12]}... {r.get('message', '')}", flush=True)
    except Exception as e:
        print(f"[ERROR] batch of {len(rows)} failed: {e}", flush=True)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ObservationSource:
    def __iter__(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError


class StdinJSONLSource(ObservationSource):
    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class HttpPollingJSONLSource(ObservationSource):
    def __init__(
        self,
        url: str,
        poll_interval_s: float = 0.25,
        headers: Optional[List[str]] = None,
        timeout_s: float = 10.0,
        verify_ssl: bool = True,
    ):
        self.url = url
        self.poll_interval_s = poll_interval_s
        self.headers = headers or []
        self.timeout_s = timeout_s
        self.verify_ssl = bool(verify_ssl)

    def _fetch(self) -> Any:
        req = urllib.request.Request(self.url)  # type: ignore[attr-defined]
        for h in self.headers:
            k, v = h.split(":", 1)
            req.add_header(k.strip(), v.strip())
        context = None
        if not self.verify_ssl:
            context = ssl._create_unverified_context()
        else:
            if certifi is not None:
                context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=self.timeout_s, context=context) as resp:  # type: ignore[attr-defined]
            payload = resp.read().decode("utf-8")
        return json.loads(payload)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        if urllib is None:
            raise RuntimeError("urllib is not available in this Python environment")

        for i in range(0, 100):
            data = self._fetch()
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(data, dict):
                if "observations" in data and isinstance(data.get("observations"), list):
                    for item in data["observations"]:
                        if isinstance(item, dict):
                            yield item
                else:
                    yield data
            time.sleep(self.poll_interval_s)

    def fetch_batch(self) -> List[Dict[str, Any]]:
        """Return all observations from a single fetch as a list."""
        items: List[Dict[str, Any]] = []
        data = self._fetch()
        if isinstance(data, list):
            items = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            if "observations" in data and isinstance(data.get("observations"), list):
                items = [x for x in data["observations"] if isinstance(x, dict)]
            else:
                items = [data]
        return items


def open_output(out: str):
    if out == "-":
        return None
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path.open("a", encoding="utf-8", buffering=1)


def emit_jsonl(out_f, obj: Dict[str, Any]) -> None:
    line = json.dumps(obj)
    if out_f is None:
        print(line, flush=True)
    else:
        out_f.write(line)
        out_f.write("\n")
        out_f.flush()


# Map model labels -> API-accepted classification_label values (friendly)
_LABEL_MAP = {
    "Satcom": "Satcom",
    "Radar-Altimeter": "Radar-Altimeter",
    "short-range": "short-range",
}

# Labels the model may predict that the API doesn't accept
_INVALID_LABELS = {"Bluetooth", "IEEE802.15.4", "IEEE802.11bg"}

# Valid hostile labels the API accepts
_HOSTILE_LABELS = {"Airbourne-detection", "Airbourne-range", "Air-Ground-MTI", "EW-Jammer"}


def _label_short(label: str) -> str:
    if "|" in label:
        name = label.split("|", 1)[1].strip()
    else:
        name = label
    mapped = _LABEL_MAP.get(name, name)
    return mapped


def _to_analysis_rows(fix: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not fix.get("geolocation_ok"):
        return []

    obs_ids = fix.get("observation_ids") or []
    if not obs_ids:
        return []

    shared = {
        "classification_label": _label_short(fix.get("pred_label_name")),
        "confidence": fix.get("mean_confidence"),
        "estimated_latitude": fix.get("latitude"),
        "estimated_longitude": fix.get("longitude"),
    }
    return [{"observation_id": oid, **shared} for oid in obs_ids]


def main() -> None:
    ap = argparse.ArgumentParser()

    here = Path(__file__).resolve().parent
    default_ckpt = str(here / "runs" / "friendly_cnn" / "best.pt")
    default_receivers = str(here / "receivers.json")
    default_path_loss = str(here / "path_loss.json")

    ap.add_argument("--ckpt", default=default_ckpt)
    ap.add_argument("--receivers", default=default_receivers)
    ap.add_argument("--path-loss", default=default_path_loss)

    ap.add_argument("--ood-thresh", type=float, default=0.70)
    ap.add_argument("--assoc-window-ms", type=float, default=100.0)
    ap.add_argument("--assoc-score-thresh", type=float, default=0.55)

    ap.add_argument("--source", choices=["stdin", "http"], default="stdin")
    ap.add_argument("--api-url", default=None)
    ap.add_argument("--api-header", action="append", default=[])
    ap.add_argument("--poll-interval-s", type=float, default=0.25)
    ap.add_argument("--no-ssl-verify", action="store_true")
    ap.add_argument("--run-seconds", type=float, default=None)

    ap.add_argument("--out", default="-", help="Output JSONL (use '-' for stdout)")
    ap.add_argument("--include-failed", action="store_true", help="Emit rows even when geolocation fails")
    ap.add_argument("--submit-api-key", default=None,
                    help="If set, POST each classification to the submissions API")

    args = ap.parse_args()

    if args.source == "stdin":
        source: ObservationSource = StdinJSONLSource()
    else:
        if not args.api_url:
            raise SystemExit("--api-url is required when --source http")
        source = HttpPollingJSONLSource(
            url=args.api_url,
            poll_interval_s=args.poll_interval_s,
            headers=args.api_header,
            verify_ssl=not bool(args.no_ssl_verify),
        )

    out_f = open_output(args.out)
    infer_engine = LiveInferenceEngine(args.ckpt, ood_thresh=args.ood_thresh)
    associator = ObservationAssociator(max_dt_ms=args.assoc_window_ms, min_score=args.assoc_score_thresh)
    geolocator = RSSIGeolocator(args.receivers, args.path_loss)
    submitted_ids: set[str] = set()

    start_t = time.time()

    try:
        if args.submit_api_key and isinstance(source, HttpPollingJSONLSource):
            # Single-batch mode: fetch once, infer, submit in chunks of 100, then exit
            raw_batch = source.fetch_batch()
            print(f"[INFO] Fetched {len(raw_batch)} observations", flush=True)
            obs_info: Dict[str, Dict[str, Any]] = {}

            for raw_obs in raw_batch:
                enriched = infer_engine.infer_one_observation(raw_obs)
                obs_id = enriched.get("observation_id") or raw_obs.get("observation_id")
                cls_label = _label_short(enriched.get("pred_label_name"))
                if obs_id and obs_id not in submitted_ids and cls_label is not None:
                    submitted_ids.add(obs_id)
                    obs_info[obs_id] = {
                        "observation_id": obs_id,
                        "classification_label": cls_label,
                        "confidence": enriched.get("confidence"),
                        "estimated_latitude": None,
                        "estimated_longitude": None,
                    }

                for g in associator.add(enriched):
                    fix = geolocator.geolocate_group(g)
                    rows = _to_analysis_rows(fix)
                    if not rows:
                        if args.include_failed:
                            emit_jsonl(out_f, {
                                "timestamp": iso_z(datetime.now(timezone.utc)),
                                "geolocation_ok": False,
                                "fix": fix,
                            })
                        continue
                    for r in rows:
                        emit_jsonl(out_f, r)
                        oid = r.get("observation_id")
                        if oid in obs_info:
                            if r.get("estimated_latitude") is not None:
                                obs_info[oid]["estimated_latitude"] = r["estimated_latitude"]
                            if r.get("estimated_longitude") is not None:
                                obs_info[oid]["estimated_longitude"] = r["estimated_longitude"]

            # Flush remaining association groups and merge geo
            for g in associator.flush_all():
                fix = geolocator.geolocate_group(g)
                rows = _to_analysis_rows(fix)
                if not rows:
                    if args.include_failed:
                        emit_jsonl(out_f, {
                            "timestamp": iso_z(datetime.now(timezone.utc)),
                            "geolocation_ok": False,
                            "fix": fix,
                        })
                    continue
                for r in rows:
                    emit_jsonl(out_f, r)
                    oid = r.get("observation_id")
                    if oid in obs_info:
                        if r.get("estimated_latitude") is not None:
                            obs_info[oid]["estimated_latitude"] = r["estimated_latitude"]
                        if r.get("estimated_longitude") is not None:
                            obs_info[oid]["estimated_longitude"] = r["estimated_longitude"]

            # Submit in chunks of 100 (skip items with None lat/lon values in payload)
            pending_batch = []
            for item in obs_info.values():
                row = {k: v for k, v in item.items() if v is not None}
                pending_batch.append(row)
            for i in range(0, len(pending_batch), 100):
                chunk = pending_batch[i:i + 100]
                print(f"[INFO] Submitting chunk {i // 100 + 1} ({len(chunk)} items)", flush=True)
                submit_batch(chunk, args.submit_api_key,
                             verify_ssl=not bool(args.no_ssl_verify))
            print(f"[INFO] Done. Processed {len(obs_info)} unique observations.", flush=True)
        else:
            # Streaming mode (stdin or no submission key)
            for raw_obs in source:
                if args.run_seconds is not None and (time.time() - start_t) >= args.run_seconds:
                    break

                enriched = infer_engine.infer_one_observation(raw_obs)

                for g in associator.add(enriched):
                    fix = geolocator.geolocate_group(g)
                    rows = _to_analysis_rows(fix)
                    if not rows:
                        if args.include_failed:
                            emit_jsonl(out_f, {
                                "timestamp": iso_z(datetime.now(timezone.utc)),
                                "geolocation_ok": False,
                                "fix": fix,
                            })
                        continue
                    for r in rows:
                        emit_jsonl(out_f, r)

    except KeyboardInterrupt:
        pass
    finally:
        for g in associator.flush_all():
            fix = geolocator.geolocate_group(g)
            rows = _to_analysis_rows(fix)
            if not rows:
                if args.include_failed:
                    emit_jsonl(out_f, {
                        "timestamp": iso_z(datetime.now(timezone.utc)),
                        "geolocation_ok": False,
                        "fix": fix,
                    })
                continue
            for r in rows:
                emit_jsonl(out_f, r)

        if out_f is not None:
            out_f.close()


if __name__ == "__main__":
    main()