from __future__ import annotations

import argparse
import json
import signal
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
from unknown_labeler import UnknownSignalLabeler

# BATCH_URL = "https://findmyforce.online/submissions/batch"
BATCH_URL = "https://findmyforce.online/evaluate/submit"


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
        # Evaluate endpoint returns scores directly
        if "total_score" in result:
            try:
                print(f"[EVAL] sent={len(rows)}", flush=True)
                print(f"  total_score={result.get('total_score')}", flush=True)
                print(f"  classification_score={result.get('classification_score')}", flush=True)
                print(f"  geolocation_score={result.get('geolocation_score')}", flush=True)
                print(f"  novelty_detection_score={result.get('novelty_detection_score')}", flush=True)
                print(f"  correct_classifications={result.get('correct_classifications')}", flush=True)
                print(f"  coverage={result.get('coverage')}%", flush=True)
                print(f"  average_cep_meters={result.get('average_cep_meters')}", flush=True)
                print(f"  attempt={result.get('attempt_number')} is_best={result.get('is_best')} best_total={result.get('best_total_score')}", flush=True)
                for cls in result.get("per_class_scores", []):
                    print(f"    {cls['label']}: P={cls['precision']} R={cls['recall']} F1={cls['f1']} n={cls['count']}", flush=True)
            except BrokenPipeError:
                raise SystemExit(0)
        else:
            # Batch submissions endpoint
            acc = result.get("accepted_count", 0)
            rej = result.get("rejected_count", 0)
            try:
                print(f"[BATCH] sent={len(rows)} accepted={acc} rejected={rej}", flush=True)
                for r in result.get("results", []):
                    tag = "ACCEPTED" if r.get("accepted") else "REJECTED"
                    print(f"  [{tag}] obs={r.get('observation_id', '?')[:12]}... {r.get('message', '')}", flush=True)
            except BrokenPipeError:
                raise SystemExit(0)
    except Exception as e:
        try:
            print(f"[ERROR] batch of {len(rows)} failed: {e}", flush=True)
        except BrokenPipeError:
            raise SystemExit(0)


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
        try:
            print(line, flush=True)
        except BrokenPipeError:
            raise SystemExit(0)
    else:
        try:
            out_f.write(line)
            out_f.write("\n")
            out_f.flush()
        except BrokenPipeError:
            raise SystemExit(0)


# Map model labels -> API-accepted classification_label values (friendly)
_LABEL_MAP = {
    "Satcom": "Satcom",
    "Radar-Altimeter": "Radar-Altimeter",
    "short-range": "short-range",
}

# Labels the model may predict that the API doesn't accept
_INVALID_LABELS = {"Bluetooth", "IEEE802.15.4", "IEEE802.11bg"}

# Valid hostile labels the API accepts
_HOSTILE_LABELS = {"Airborne-detection", "Airborne-range", "Air-Ground-MTI", "EW-Jammer"}

# Valid civilian labels the API accepts
_CIVILIAN_LABELS = {"AM radio"}

_ALLOWED_LABELS = set(_LABEL_MAP.values()) | _HOSTILE_LABELS | _CIVILIAN_LABELS


def _normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None

    # live_pipeline.py now emits Signal Type directly (no "modulation | name"),
    # but keep a minimal guard in case older checkpoints emit legacy formatting.
    name = label.split("|", 1)[1].strip() if "|" in label else label

    mapped = _LABEL_MAP.get(name, name)

    # If it's a known-invalid friendly class, drop it to avoid API rejections.
    if mapped in _INVALID_LABELS:
        return None

    return mapped if mapped in _ALLOWED_LABELS else None


def _load_receiver_sensitivity(receivers_json_path: str) -> Dict[str, float]:
    try:
        data = json.loads(Path(receivers_json_path).read_text())
    except Exception:
        return {}

    if isinstance(data, dict) and "receivers" in data:
        items = data["receivers"]
    elif isinstance(data, list):
        items = data
    else:
        return {}

    out: Dict[str, float] = {}
    for r in items:
        try:
            rid = str(r["receiver_id"])
            out[rid] = float(r.get("sensitivity_dbm", -120.0))
        except Exception:
            continue
    return out


def _to_analysis_rows(fix: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not fix.get("geolocation_ok"):
        return []

    obs_ids = fix.get("observation_ids") or []
    if not obs_ids:
        return []

    cls_label = _normalize_label(fix.get("pred_label_name"))
    if cls_label is None:
        return []

    shared = {
        "classification_label": cls_label,
        "confidence": fix.get("mean_confidence"),
        "estimated_latitude": fix.get("latitude"),
        "estimated_longitude": fix.get("longitude"),
    }
    return [{"observation_id": oid, **shared} for oid in obs_ids]


def main() -> None:
    # Avoid noisy stack traces when piping output into commands like `head`.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        pass

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

    ap.add_argument("--geo-sensitivity-margin-db", type=float, default=0.0,
                    help="Only use observations for geolocation if rssi_dbm >= (receiver_sensitivity_dbm + margin).")
    ap.add_argument("--max-uncertainty-m", type=float, default=999999.0,
                    help="Only attach lat/lon to submissions if uncertainty_radius_m <= this value.")

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
    unknown_labeler = UnknownSignalLabeler()
    receiver_sens = _load_receiver_sensitivity(args.receivers)
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

                # Classification label selection:
                # 1) Use model label if it is API-allowed
                # 2) Otherwise, fall back to rule-based unknown labeler (hostile/civilian)
                cls_label = _normalize_label(enriched.get("pred_label_name"))
                if cls_label is None:
                    try:
                        decision = unknown_labeler.label_iq_snapshot(raw_obs["iq_snapshot"])
                    except Exception:
                        decision = None
                    if decision is not None:
                        cls_label = _normalize_label(decision.pred_label_name)

                if obs_id and obs_id not in submitted_ids and cls_label is not None:
                    submitted_ids.add(obs_id)
                    obs_info[obs_id] = {
                        "observation_id": obs_id,
                        "classification_label": cls_label,
                        "confidence": enriched.get("confidence"),
                        "estimated_latitude": None,
                        "estimated_longitude": None,
                    }

                # Only feed observations into association/geolocation if they pass RSSI sensitivity gating.
                rid = str(enriched.get("receiver_id"))
                rssi = enriched.get("rssi_dbm")
                sens = receiver_sens.get(rid)
                usable_for_geo = True
                if sens is not None and rssi is not None:
                    usable_for_geo = float(rssi) >= float(sens) + float(args.geo_sensitivity_margin_db)

                if not usable_for_geo:
                    continue

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

                            # Only attach geolocation if uncertainty is acceptable.
                            unc = fix.get("uncertainty_radius_m")
                            ok_unc = (unc is None) or (float(unc) <= float(args.max_uncertainty_m))
                            if ok_unc:
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
                        unc = fix.get("uncertainty_radius_m")
                        ok_unc = (unc is None) or (float(unc) <= float(args.max_uncertainty_m))
                        if ok_unc:
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