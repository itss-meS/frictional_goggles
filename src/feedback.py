from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .detector import Detection


def save_false_positives(root: str | Path, frame: np.ndarray, detections: Sequence[Detection], source: str) -> dict:
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "crops").mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    image_rel = f"images/{stamp}.jpg"
    cv2.imwrite(str(root / image_rel), frame)
    h, w = frame.shape[:2]
    items = []
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d.box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        crop_rel = None
        if x2 > x1 and y2 > y1:
            crop_rel = f"crops/{stamp}_{i}.jpg"
            cv2.imwrite(str(root / crop_rel), frame[y1:y2, x1:x2])
        items.append(
            {
                "class": d.name,
                "group": d.group,
                "conf": round(d.conf, 4),
                "box": [x1, y1, x2, y2],
                "crop": crop_rel,
            }
        )
    record = {
        "id": stamp,
        "time": now.isoformat(timespec="seconds"),
        "source": source,
        "image": image_rel,
        "detections": items,
    }
    with open(root / "meta.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_records(root: str | Path) -> list[dict]:
    path = Path(root) / "meta.jsonl"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_zip(root: str | Path) -> bytes:
    root = Path(root)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())
    return buffer.getvalue()
