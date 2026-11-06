from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from ultralytics import RTDETR

from .classes import group_of, is_camo

PALETTE = [
    (75, 180, 60),
    (230, 120, 40),
    (40, 90, 220),
    (190, 60, 150),
    (30, 170, 200),
    (160, 100, 30),
    (90, 90, 90),
    (60, 60, 200),
]


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    conf: float
    cls_id: int
    name: str
    group: str
    camo: bool
    track_id: Optional[int] = None

    def row(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "class": self.name,
            "group": self.group,
            "camo": self.camo,
            "conf": round(self.conf, 3),
            "track": self.track_id,
            "x1": round(x1),
            "y1": round(y1),
            "x2": round(x2),
            "y2": round(y2),
        }


def available_devices() -> list[str]:
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def color_for(group: str) -> tuple[int, int, int]:
    return PALETTE[zlib.crc32(group.encode()) % len(PALETTE)]


def draw(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    thickness = max(1, round(min(h, w) / 400))
    scale = max(0.4, min(h, w) / 1200)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        color = color_for(d.group)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{d.name} {d.conf:.2f}"
        if d.track_id is not None:
            label += f" #{d.track_id}"
        (tw, th), base = cv2.getTextSize(label, font, scale, max(1, thickness - 1))
        ty = max(y1, th + base + 2)
        cv2.rectangle(out, (x1, ty - th - base - 2), (x1 + tw + 4, ty), color, -1)
        cv2.putText(out, label, (x1 + 2, ty - base - 1), font, scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)
    return out


class Detector:
    def __init__(self, weights: str, device: str = "auto", imgsz: int = 640, tracker: str = "bytetrack.yaml"):
        self.weights = str(weights)
        self.device = resolve_device(device)
        self.imgsz = int(imgsz) if Path(self.weights).suffix == ".pt" else None
        self.tracker = tracker
        self.model = RTDETR(self.weights)
        self.names = {int(k): v for k, v in self.model.names.items()}
        self._mode: Optional[str] = None

    def _kwargs(self, conf: float, classes: Optional[Sequence[int]]) -> dict:
        kwargs = {
            "conf": float(conf),
            "classes": list(classes) if classes else None,
            "device": self.device,
            "verbose": False,
        }
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        return kwargs

    def _enter(self, mode: str) -> None:
        if self._mode != mode:
            self.model.predictor = None
            self._mode = mode

    def reset_tracker(self) -> None:
        self.model.predictor = None
        self._mode = None

    def predict(self, frame: np.ndarray, conf: float = 0.25, classes: Optional[Sequence[int]] = None):
        self._enter("predict")
        result = self.model.predict(frame, **self._kwargs(conf, classes))[0]
        return self._parse(result), dict(result.speed)

    def track(self, frame: np.ndarray, conf: float = 0.25, classes: Optional[Sequence[int]] = None):
        self._enter("track")
        result = self.model.track(frame, persist=True, tracker=self.tracker, **self._kwargs(conf, classes))[0]
        return self._parse(result), dict(result.speed)

    def _parse(self, result) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        confs = boxes.conf.cpu().numpy().tolist()
        classes = boxes.cls.cpu().numpy().astype(int).tolist()
        ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else [None] * len(classes)
        detections = []
        for box, conf, cls_id, track_id in zip(xyxy, confs, classes, ids):
            name = self.names.get(cls_id, str(cls_id))
            detections.append(
                Detection(
                    box=tuple(box),
                    conf=conf,
                    cls_id=cls_id,
                    name=name,
                    group=group_of(name),
                    camo=is_camo(name),
                    track_id=track_id,
                )
            )
        return detections
