from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import urllib.request
except Exception:  # pragma: no cover
    urllib = None  # type: ignore

try:
    import certifi  # type: ignore
except Exception:  # pragma: no cover
    certifi = None  # type: ignore

from associator import ObservationAssociator
from geolocate import TDoAGeolocator
from live_pipeline import LiveInferenceEngine
from track_manager import TrackManager


def iso_z(dt: datetime) -> str:
    return (
        dt.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def parse_iso8601(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


@dataclass
class PipelineConfig:
    ckpt: str
    receivers_json: str
    path_loss_json: str

    ood_thresh: float
    assoc_window_ms: float
    assoc_score_thresh: float

    track_match_distance_m: float
    track_max_time_gap_s: float
    track_stale_after_s: float
    track_drop_after_s: float
    track_alpha_pos: float
    track_alpha_uncertainty: float
    enemy_after_updates: int
    flush_age_ms: float


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
    """Very small stub for tomorrow's API.

    Assumption: endpoint returns JSON (either a list of observations or a single observation).

    You can adapt this once the provider publishes the schema:
    - auth headers
    - pagination / since cursor
    - websocket / SSE streaming
    """

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
        with urllib.request.urlopen(
            req, timeout=self.timeout_s, context=context
        ) as resp:  # type: ignore[attr-defined]
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
                # Common schema: {"observations": [...], "count": N, "has_more": bool}
                if "observations" in data and isinstance(
                    data.get("observations"), list
                ):
                    for item in data["observations"]:
                        if isinstance(item, dict):
                            yield item
                else:
                    yield data
            time.sleep(self.poll_interval_s)


def collect_for_duration(
    source: ObservationSource,
    duration_s: float,
    max_observations: Optional[int] = None,
) -> List[Dict[str, Any]]:
    obs: List[Dict[str, Any]] = []
    t0 = time.time()
    for o in source:
        obs.append(o)
        if max_observations is not None and len(obs) >= max_observations:
            break
        if time.time() - t0 >= duration_s:
            break
    return obs


def open_output(out: str):
    if out == "-":
        return None
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path.open("a", encoding="utf-8", buffering=1)


def emit_event(out_f, event: Dict[str, Any]) -> None:
    line = json.dumps(event)
    if out_f is None:
        print(line, flush=True)
    else:
        out_f.write(line)
        out_f.write("\n")
        out_f.flush()


def run_pipeline_batch(
    observations: List[Dict[str, Any]], cfg: PipelineConfig
) -> List[Dict[str, Any]]:
    infer_engine = LiveInferenceEngine(cfg.ckpt, ood_thresh=cfg.ood_thresh)
    associator = ObservationAssociator(
        max_dt_ms=cfg.assoc_window_ms,
        min_score=cfg.assoc_score_thresh,
    )
    geolocator = TDoAGeolocator(cfg.receivers_json, cfg.path_loss_json)
    tm = TrackManager(
        match_distance_m=cfg.track_match_distance_m,
        max_time_gap_s=cfg.track_max_time_gap_s,
        stale_after_s=cfg.track_stale_after_s,
        drop_after_s=cfg.track_drop_after_s,
        alpha_pos=cfg.track_alpha_pos,
        alpha_uncertainty=cfg.track_alpha_uncertainty,
        enemy_after_updates=cfg.enemy_after_updates,
    )

    events: List[Dict[str, Any]] = []

    # 1) Inference + association: produce groups
    for raw_obs in observations:
        enriched = infer_engine.infer_one_observation(raw_obs)
        ready_groups = associator.add(enriched)
        for g in ready_groups:
            fix = geolocator.geolocate_group(g)
            events.extend(tm.process_fix(fix))

    # flush leftover groups
    for g in associator.flush_all():
        fix = geolocator.geolocate_group(g)
        events.extend(tm.process_fix(fix))

    # finalize tracks
    events.extend(tm.flush_all())
    return events


def run_pipeline_live(
    source: ObservationSource,
    cfg: PipelineConfig,
    out_f,
    snapshot_interval_s: float,
    run_seconds: Optional[float],
) -> None:
    infer_engine = LiveInferenceEngine(cfg.ckpt, ood_thresh=cfg.ood_thresh)
    associator = ObservationAssociator(
        max_dt_ms=cfg.assoc_window_ms,
        min_score=cfg.assoc_score_thresh,
        flush_age_ms=cfg.assoc_flush_age_ms,
    )
    geolocator = TDoAGeolocator(cfg.receivers_json, cfg.path_loss_json)
    tm = TrackManager(
        match_distance_m=cfg.track_match_distance_m,
        max_time_gap_s=cfg.track_max_time_gap_s,
        stale_after_s=cfg.track_stale_after_s,
        drop_after_s=cfg.track_drop_after_s,
        alpha_pos=cfg.track_alpha_pos,
        alpha_uncertainty=cfg.track_alpha_uncertainty,
        enemy_after_updates=cfg.enemy_after_updates,
    )

    last_snapshot_t = time.time()
    start_t = time.time()

    def maybe_snapshot() -> None:
        nonlocal last_snapshot_t
        if snapshot_interval_s <= 0:
            return
        now = time.time()
        if now - last_snapshot_t >= snapshot_interval_s:
            last_snapshot_t = now
            emit_event(
                out_f,
                {
                    "event": "tracks_snapshot",
                    "timestamp": iso_z(datetime.now(timezone.utc)),
                    "tracks": tm.get_tracks(),
                },
            )

    try:
        for raw_obs in source:
            if run_seconds is not None and (time.time() - start_t) >= run_seconds:
                break

            enriched = infer_engine.infer_one_observation(raw_obs)
            ready_groups = associator.add(enriched)
            for g in ready_groups:
                fix = geolocator.geolocate_group(g)
                for ev in tm.process_fix(fix):
                    emit_event(out_f, ev)

            maybe_snapshot()

    except KeyboardInterrupt:
        pass

    # flush leftover groups
    for g in associator.flush_all():
        fix = geolocator.geolocate_group(g)
        for ev in tm.process_fix(fix):
            emit_event(out_f, ev)

    # emit finals for anything still tracked
    for ev in tm.flush_all():
        emit_event(out_f, ev)


def main() -> None:
    ap = argparse.ArgumentParser()

    here = Path(__file__).resolve().parent
    default_ckpt = str(here / "runs" / "friendly_cnn" / "best.pt")
    default_receivers = str(here / "receivers.json")
    default_path_loss = str(here / "path_loss.json")
    default_out = str(here / "output.jsonl")

    # Model / configs
    ap.add_argument("--ckpt", default=default_ckpt, help="Path to runs/.../best.pt")
    ap.add_argument(
        "--receivers", default=default_receivers, help="Path to receivers.json"
    )
    ap.add_argument(
        "--path-loss", default=default_path_loss, help="Path to path_loss.json"
    )

    # Inference / association
    ap.add_argument("--ood-thresh", type=float, default=0.70)
    ap.add_argument("--assoc-window-ms", type=float, default=1.0)
    ap.add_argument("--assoc-score-thresh", type=float, default=0.55)
    ap.add_argument("--assoc-flush-age-ms", type=float, default=2.0)

    # Track policy
    ap.add_argument("--enemy-after-updates", type=int, default=3)

    # Track tuning
    ap.add_argument("--match-distance-m", type=float, default=250.0)
    ap.add_argument("--max-time-gap-s", type=float, default=10.0)
    ap.add_argument("--stale-after-s", type=float, default=30.0)
    ap.add_argument("--drop-after-s", type=float, default=120.0)
    ap.add_argument("--alpha-pos", type=float, default=0.35)
    ap.add_argument("--alpha-uncertainty", type=float, default=0.30)

    # Ingest mode
    ap.add_argument("--source", choices=["stdin", "http"], default="stdin")
    ap.add_argument(
        "--api-url",
        default=None,
        help="HTTP endpoint for live observations (when --source http)",
    )
    ap.add_argument(
        "--api-header",
        action="append",
        default=[],
        help='HTTP header like "Authorization: Bearer ..."',
    )
    ap.add_argument("--poll-interval-s", type=float, default=0.25)
    ap.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable SSL certificate verification for HTTP source",
    )

    # Batch window
    ap.add_argument(
        "--collect-seconds",
        type=float,
        default=120.0,
        help="How long to collect observations before processing",
    )
    ap.add_argument("--max-observations", type=int, default=None)

    ap.add_argument(
        "--out",
        default=default_out,
        help="Output JSONL path for pipeline events (use '-' for stdout)",
    )

    ap.add_argument("--mode", choices=["live", "batch"], default="live")
    ap.add_argument(
        "--snapshot-interval-s",
        type=float,
        default=2.0,
        help="Emit tracks_snapshot every N seconds in live mode (0 disables)",
    )
    ap.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Optional: stop live mode after N seconds",
    )

    args = ap.parse_args()

    cfg = PipelineConfig(
        ckpt=args.ckpt,
        receivers_json=args.receivers,
        path_loss_json=args.path_loss,
        ood_thresh=args.ood_thresh,
        assoc_window_ms=args.assoc_window_ms,
        assoc_score_thresh=args.assoc_score_thresh,
        track_match_distance_m=args.match_distance_m,
        track_max_time_gap_s=args.max_time_gap_s,
        track_stale_after_s=args.stale_after_s,
        track_drop_after_s=args.drop_after_s,
        track_alpha_pos=args.alpha_pos,
        track_alpha_uncertainty=args.alpha_uncertainty,
        enemy_after_updates=args.enemy_after_updates,
        flush_age_ms=args.assoc_flush_age_ms,
    )

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
    try:
        if args.mode == "batch":
            observations = collect_for_duration(
                source,
                duration_s=args.collect_seconds,
                max_observations=args.max_observations,
            )
            events = run_pipeline_batch(observations, cfg)
            for e in events:
                emit_event(out_f, e)
        else:
            run_pipeline_live(
                source=source,
                cfg=cfg,
                out_f=out_f,
                snapshot_interval_s=args.snapshot_interval_s,
                run_seconds=args.run_seconds,
            )
    finally:
        if out_f is not None:
            out_f.close()


if __name__ == "__main__":
    main()
