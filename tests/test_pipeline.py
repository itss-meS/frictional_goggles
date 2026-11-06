from src.media import probe
from src.pipeline import process_video


def test_process_video(detector, pan_video, tmp_path):
    out = tmp_path / "out.mp4"
    seen = []
    summary = process_video(detector, pan_video, out, conf=0.5, stride=2, max_frames=None, track=True, progress=seen.append)
    info = probe(out)
    assert summary["frames"] == 6 == info.frames
    assert (info.width, info.height) == (640, 640)
    assert info.fps == 5.0
    assert summary["groups"]["person"]["tracks"] == 3
    assert summary["groups"]["person"]["detections"] >= 10
    assert summary["groups"]["vehicle"]["tracks"] == 1
    assert summary["avg_ms"] > 0
    assert seen and seen[-1] == 1.0 and seen == sorted(seen)


def test_process_video_max_frames_no_track(detector, pan_video, tmp_path):
    out = tmp_path / "out.mp4"
    summary = process_video(detector, pan_video, out, conf=0.5, stride=1, max_frames=4, track=False)
    assert summary["frames"] == 4 == probe(out).frames
    assert summary["groups"]["person"]["tracks"] == 0
