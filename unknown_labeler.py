from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UnknownLabelDecision:
    pred_label_id: int
    pred_label_name: str
    confidence: float
    debug: Dict[str, Any]


def _iq_to_complex(iq_snapshot: List[float]) -> np.ndarray:
    x = np.asarray(iq_snapshot, dtype=np.float32)
    if x.shape[0] != 256:
        raise ValueError(f"Expected 256 IQ floats (I[0:128],Q[128:256]); got {x.shape[0]}")
    i = x[:128]
    q = x[128:]
    return i.astype(np.float32) + 1j * q.astype(np.float32)


def _spectral_features(z: np.ndarray) -> Tuple[float, float]:
    # Returns (spectral_flatness, occupied_bw_frac)
    w = np.hanning(len(z)).astype(np.float32)
    Z = np.fft.fftshift(np.fft.fft(z * w))
    p = (np.abs(Z) ** 2).astype(np.float64) + 1e-24

    # Spectral flatness: exp(mean(log(P))) / mean(P)
    sf = float(np.exp(np.mean(np.log(p))) / np.mean(p))

    # Occupied bandwidth fraction: bins containing 99% of energy
    p_norm = p / np.sum(p)
    order = np.argsort(p_norm)[::-1]
    csum = np.cumsum(p_norm[order])
    k = int(np.searchsorted(csum, 0.99, side="left") + 1)
    occ_frac = float(k / len(p_norm))

    return sf, occ_frac


def _envelope_features(z: np.ndarray) -> Tuple[float, float, float]:
    # Returns (impulsiveness, peak_to_rms, acf_peak)
    env = np.abs(z).astype(np.float64) + 1e-12
    rms = float(np.sqrt(np.mean(env * env)))
    peak = float(np.max(env))
    peak_to_rms = float(peak / (rms + 1e-12))

    # Impulsiveness proxy via kurtosis of envelope
    mu = float(np.mean(env))
    s2 = float(np.mean((env - mu) ** 2)) + 1e-12
    s4 = float(np.mean((env - mu) ** 4))
    kurt = float(s4 / (s2 * s2))

    # Autocorr peak excluding lag 0 (detect periodic pulsing)
    x = env - mu
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac) // 2 :]
    if len(ac) <= 1 or ac[0] <= 0:
        acf_peak = 0.0
    else:
        acf_peak = float(np.max(ac[1:]) / (ac[0] + 1e-12))

    return kurt, peak_to_rms, acf_peak


class UnknownSignalLabeler:
    # Stable IDs for rule-based unknown labels (kept separate from the trained classifier label space).
    LABEL_ID_EW_JAMMER = 100
    LABEL_ID_AIRBORNE_DETECTION = 101
    LABEL_ID_AIRBORNE_RANGE = 102
    LABEL_ID_AIR_GROUND_MTI = 103

    def __init__(
        self,
        jammer_spectral_flatness_thresh: float = 0.35,
        jammer_occupied_bw_frac_thresh: float = 0.25,
        pulsed_kurtosis_thresh: float = 6.0,
        pulsed_peak_to_rms_thresh: float = 3.0,
        periodic_acf_peak_thresh: float = 0.35,
    ):
        self.jammer_spectral_flatness_thresh = jammer_spectral_flatness_thresh
        self.jammer_occupied_bw_frac_thresh = jammer_occupied_bw_frac_thresh
        self.pulsed_kurtosis_thresh = pulsed_kurtosis_thresh
        self.pulsed_peak_to_rms_thresh = pulsed_peak_to_rms_thresh
        self.periodic_acf_peak_thresh = periodic_acf_peak_thresh

    def label_iq_snapshot(self, iq_snapshot: List[float]) -> Optional[UnknownLabelDecision]:
        z = _iq_to_complex(iq_snapshot)

        sf, occ_frac = _spectral_features(z)
        kurt, p2r, acf_peak = _envelope_features(z)

        debug = {
            "spectral_flatness": sf,
            "occupied_bw_frac_99": occ_frac,
            "envelope_kurtosis": kurt,
            "envelope_peak_to_rms": p2r,
            "envelope_acf_peak": acf_peak,
        }

        # 1) Jammer: broadband-ish, spectrally flat
        if sf >= self.jammer_spectral_flatness_thresh and occ_frac >= self.jammer_occupied_bw_frac_thresh:
            conf = float(min(0.99, 0.5 + 0.6 * (sf - self.jammer_spectral_flatness_thresh)))
            return UnknownLabelDecision(
                pred_label_id=self.LABEL_ID_EW_JAMMER,
                pred_label_name="Jamming | EW-Jammer",
                confidence=conf,
                debug=debug,
            )

        # 2) Pulsed radar family: envelope is spiky/impulsive
        if kurt >= self.pulsed_kurtosis_thresh or p2r >= self.pulsed_peak_to_rms_thresh:
            # Use periodicity proxy to pick subtype
            if acf_peak >= self.periodic_acf_peak_thresh:
                label_id = self.LABEL_ID_AIR_GROUND_MTI
                name = "Pulsed | Air-Ground-MTI"
            else:
                # Default split: range-finding tends to be sparse (single strong pulse)
                # If extremely peaky, call it range; otherwise airborne detection.
                if p2r >= (self.pulsed_peak_to_rms_thresh + 1.5):
                    label_id = self.LABEL_ID_AIRBORNE_RANGE
                    name = "Pulsed | Airborne-range"
                else:
                    label_id = self.LABEL_ID_AIRBORNE_DETECTION
                    name = "Pulsed | Airborne-detection"

            conf = float(min(0.95, 0.45 + 0.08 * max(0.0, kurt - self.pulsed_kurtosis_thresh) + 0.08 * max(0.0, p2r - self.pulsed_peak_to_rms_thresh)))
            return UnknownLabelDecision(pred_label_id=label_id, pred_label_name=name, confidence=conf, debug=debug)

        return None
