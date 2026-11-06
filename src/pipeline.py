from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

from .detector import Detector, draw
from .media import Mp4Writer, iter_frames, probe


def process_video(
    detector: Detector,
    src: str | Path,
    dst: str | Path,
    conf: float = 0.25,
    classes: Optional[Sequence[int]] = None,
    stride: int = 1,
    max_frames: Optional[int] = None,
    track: bool = True,
    progress: Optional[Callable[[float], None]] = None,
) -> dict:
    stride = max(1, int(stride))
    info = probe(src)
    expected = math.ceil(info.frames / stride) if info.frames else None
    if max_frames is not None:
        expected = min(expected, max_frames) if expected else max_frames
    writer = Mp4Writer(dst, max(info.fps / stride, 1.0), info.width, info.height)
    if track:
        detector.reset_tracker()
    run = detector.track if track else detector.predict
    counts: Counter = Counter()
    tracks: dict[str, set] = defaultdict(set)
    total_ms = 0.0
    processed = 0
    try:
        for _, frame in iter_frames(src, stride, max_frames):
            detections, speed = run(frame, conf, classes)
            writer.write(draw(frame, detections))
            for d in detections:
                counts[d.group] += 1
                if d.track_id is not None:
                    tracks[d.group].add(d.track_id)
            total_ms += sum(speed.values())
            processed += 1
            if progress and expected:
                progress(min(1.0, processed / expected))
    finally:
        writer.close()
    return {
        "frames": processed,
        "avg_ms": total_ms / processed if processed else 0.0,
        "groups": {g: {"detections": counts[g], "tracks": len(tracks.get(g, ()))} for g in sorted(counts)},
        "fps_out": max(info.fps / stride, 1.0),
        "source_frames": info.frames,
    }
