"""Shared helpers for the COMPASS skill runner scripts."""
import json
from pathlib import Path

import numpy as np


def load_data(path: str) -> dict:
    """Load a data dict from .npz (arrays) or .json (lists/scalars)."""
    p = Path(path)
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            out = {}
            for k in z.files:
                v = z[k]
                out[k] = v.item() if v.shape == () else v
            return out
    if p.suffix == ".json":
        raw = json.loads(p.read_text())
        return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in raw.items()}
    raise SystemExit(f"unsupported data file {path!r}: use .npz or .json")


def load_model(path: str) -> str:
    src = Path(path).read_text()
    if "def model(" not in src:
        raise SystemExit(f"{path}: must define a function named `model`")
    return src


def dump_result(obj, out: str | None) -> None:
    text = json.dumps(obj, indent=1, default=str)
    if out:
        Path(out).write_text(text)
        print(f"saved -> {out}")
    print(text[:4000])
