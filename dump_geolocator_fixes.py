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

        while True:
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


def _label_short(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    if "|" in label:
        return label.split("|", 1)[1].strip()
    return label


def _to_analysis_row(fix: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not fix.get("geolocation_ok"):
        return None

    obs_ids = fix.get("observation_ids") or []
    observation_id = obs_ids[0] if isinstance(obs_ids, list) and obs_ids else None

    return {
        "observation_id": observation_id,
        "classification_label": _label_short(fix.get("pred_label_name")),
        "confidence": fix.get("mean_confidence"),
        "estimated_latitude": fix.get("latitude"),
        "estimated_longitude": fix.get("longitude"),
    }


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

    start_t = time.time()

    try:
        for raw_obs in source:
            if args.run_seconds is not None and (time.time() - start_t) >= args.run_seconds:
                break

            enriched = infer_engine.infer_one_observation(raw_obs)
            for g in associator.add(enriched):
                fix = geolocator.geolocate_group(g)
                row = _to_analysis_row(fix)
                if row is None:
                    if args.include_failed:
                        emit_jsonl(out_f, {
                            "timestamp": iso_z(datetime.now(timezone.utc)),
                            "geolocation_ok": False,
                            "fix": fix,
                        })
                    continue
                emit_jsonl(out_f, row)

    except KeyboardInterrupt:
        pass
    finally:
        for g in associator.flush_all():
            fix = geolocator.geolocate_group(g)
            row = _to_analysis_row(fix)
            if row is None:
                if args.include_failed:
                    emit_jsonl(out_f, {
                        "timestamp": iso_z(datetime.now(timezone.utc)),
                        "geolocation_ok": False,
                        "fix": fix,
                    })
                continue
            emit_jsonl(out_f, row)

        if out_f is not None:
            out_f.close()


if __name__ == "__main__":
    main()
