from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frames: int


def decode_image(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError(f"Could not open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not fps or fps != fps or fps <= 0 or fps > 240:
        fps = 25.0
    return VideoInfo(fps=float(fps), width=width, height=height, frames=max(0, frames))


def iter_frames(path: str | Path, stride: int = 1, max_frames: Optional[int] = None) -> Iterator[tuple[int, np.ndarray]]:
    stride = max(1, int(stride))
    cap = cv2.VideoCapture(str(path))
    try:
        index = 0
        yielded = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if index % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield index, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            index += 1
    finally:
        cap.release()


def read_frame(path: str | Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {index}")
    return frame


def open_capture(source: str) -> cv2.VideoCapture:
    text = source.strip()
    cap = cv2.VideoCapture(int(text) if text.isdigit() else text)
    if not cap.isOpened():
        cap.release()
        raise ConnectionError(f"Cannot open source: {source}")
    return cap


class Mp4Writer:
    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        self.shape = (height, width)
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps:.3f}",
            "-i", "-",
            "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != self.shape:
            frame = cv2.resize(frame, (self.shape[1], self.shape[0]))
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with code {code}")
