# rfml/dataset.py (PATCH)
import ast
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRef:
    path: str
    key: str
    label: int
    snr_db: int


def _to_int(x) -> int:
    """Handle ints stored as int or as string like '0'."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(x)
    if isinstance(x, str):
        return int(float(x))
    return int(x)


def parse_key_tuple(key: str) -> Tuple[str, str, int, int]:
    """
    Parse keys that look like:
      ('bpsk', 'Satcom', -10, 42)
      ('AM-DSB', 'AM radio', '0', '1000')
    """
    t = ast.literal_eval(key)
    mod = str(t[0])
    sig = str(t[1])
    snr = _to_int(t[2])
    idx = _to_int(t[3])
    return mod, sig, snr, idx


def canonical_pair(mod: str, sig: str) -> Tuple[str, str]:
    """
    Normalize different spellings/casing into one canonical label.
    This prevents duplicates like:
      'BPSK' vs 'bpsk'
      'SATCOM' vs 'Satcom'
      'Radar Altimeter' vs 'Radar-Altimeter'
      'AM-DSB' vs 'amdsb'  (we'll ignore AM anyway, but normalize consistently)
    """
    m = mod.strip().lower().replace("-", "_")
    s = sig.strip().lower().replace("-", "_").replace(" ", "_")

    # Normalize known modulations
    mod_map = {
        "fmcw": "FMCW",
        "bpsk": "BPSK",
        "gfsk": "GFSK",
        "dsss_oqpsk": "DSSS-OQPSK",
        "dsss_cck": "DSSS-CCK",
        "ask": "ASK",

        # Non-friendly (still normalize so filtering is consistent)
        "am_dsb": "AM-DSB",
        "amdsb": "AM-DSB",
        "am_ssb": "AM-SSB",
        "amssb": "AM-SSB",
        "pulsed": "PULSED",
    }

    # Normalize known signal names
    sig_map = {
        "satcom": "SATCOM",
        "satcom_": "SATCOM",
        "satcom__": "SATCOM",

        "radar_altimeter": "Radar-Altimeter",
        "radar_altimeter_": "Radar-Altimeter",
        "radar_altimeter__": "Radar-Altimeter",

        "bluetooth": "Bluetooth",
        "ieee802.15.4": "IEEE802.15.4",
        "ieee80215.4": "IEEE802.15.4",
        "ieee802.11bg": "IEEE802.11bg",
        "ieee80211bg": "IEEE802.11bg",

        "short_range": "short-range",
        "short-range": "short-range",

        # Non-friendly
        "am_radio": "AM radio",
        "air_ground_mti": "Air-Ground-MTI",
        "airborne_detection": "Airborne-detection",
        "airborne_range": "Airborne-range",
        "ground_mapping": "Ground mapping",
        "raw": "Raw",
    }

    mod_c = mod_map.get(m, mod.strip())
    sig_c = sig_map.get(s, sig.strip())
    return mod_c, sig_c


# --- Define EXACT friendly whitelist (this is the important part) ---
FRIENDLY_CLASSES = [
    ("FMCW", "Radar-Altimeter"),
    ("BPSK", "SATCOM"),
    ("GFSK", "Bluetooth"),
    ("DSSS-OQPSK", "IEEE802.15.4"),
    ("DSSS-CCK", "IEEE802.11bg"),
    ("ASK", "short-range"),
]

FRIENDLY_LABEL_MAP: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(FRIENDLY_CLASSES)}


def build_index(h5_paths: List[str], label_map: Optional[Dict[Tuple[str, str], int]] = None):
    """
    Build SampleRefs for ONLY the labels in label_map.
    If label_map is None, defaults to FRIENDLY_LABEL_MAP.
    """
    if label_map is None:
        label_map = FRIENDLY_LABEL_MAP

    refs: List[SampleRef] = []
    skipped = 0

    for p in h5_paths:
        with h5py.File(p, "r") as f:
            for k in f.keys():
                mod, sig, snr, _ = parse_key_tuple(k)
                mod_c, sig_c = canonical_pair(mod, sig)
                pair = (mod_c, sig_c)

                if pair not in label_map:
                    skipped += 1
                    continue

                refs.append(SampleRef(path=p, key=k, label=label_map[pair], snr_db=snr))

    meta = {
        "labels": [{"label_id": label_map[p], "modulation": p[0], "signal_name": p[1]} for p in FRIENDLY_CLASSES],
        "num_samples": len(refs),
        "skipped_samples_not_in_label_map": skipped,
    }
    return refs, label_map, meta


class H5IQDataset(Dataset):
    def __init__(self, refs: List[SampleRef], normalize: bool = True):
        self.refs = refs
        self.normalize = normalize
        self._handles = {}

    def __len__(self):
        return len(self.refs)

    def _get_handle(self, path: str) -> h5py.File:
        h = self._handles.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._handles[path] = h
        return h

    def __getitem__(self, idx: int):
        ref = self.refs[idx]
        f = self._get_handle(ref.path)
        x = f[ref.key][:].astype(np.float32)

        I = x[:128]
        Q = x[128:]

        # DC remove
        I = I - I.mean()
        Q = Q - Q.mean()

        # RMS normalize
        if self.normalize:
            rms = np.sqrt(np.mean(I * I + Q * Q) + 1e-12)
            I = I / rms
            Q = Q / rms

        xi = np.stack([I, Q], axis=0)  # (2,128)
        return torch.from_numpy(xi), int(ref.label), int(ref.snr_db)

    def close(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()