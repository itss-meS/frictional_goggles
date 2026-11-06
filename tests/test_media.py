import cv2
import numpy as np
import pytest

from src.media import Mp4Writer, decode_image, iter_frames, open_capture, probe, read_frame


def test_decode_image_roundtrip(bus_image):
    ok, buf = cv2.imencode(".png", bus_image)
    assert ok
    assert np.array_equal(decode_image(buf.tobytes()), bus_image)


def test_decode_image_invalid():
    with pytest.raises(ValueError):
        decode_image(b"not an image")


def test_probe_and_iter(pan_video):
    info = probe(pan_video)
    assert (info.width, info.height, info.frames) == (640, 640, 12)
    assert info.fps == pytest.approx(10.0)
    assert [i for i, _ in iter_frames(pan_video, 1)] == list(range(12))
    assert [i for i, _ in iter_frames(pan_video, 5)] == [0, 5, 10]
    assert [i for i, _ in iter_frames(pan_video, 2, max_frames=3)] == [0, 2, 4]


def test_read_frame(pan_video):
    assert read_frame(pan_video, 7).shape == (640, 640, 3)
    with pytest.raises(ValueError):
        read_frame(pan_video, 500)


def test_probe_missing_file(tmp_path):
    with pytest.raises(ValueError):
        probe(tmp_path / "nope.mp4")


def test_writer_pads_odd_sizes(tmp_path):
    path = tmp_path / "odd.mp4"
    writer = Mp4Writer(path, 5, 101, 77)
    for _ in range(4):
        writer.write(np.full((77, 101, 3), 120, np.uint8))
    writer.close()
    info = probe(path)
    assert (info.width, info.height, info.frames) == (102, 78, 4)


def test_open_capture_errors():
    with pytest.raises(ConnectionError):
        open_capture("/definitely/not/a/video.mp4")


def test_open_capture_file(pan_video):
    cap = open_capture(str(pan_video))
    assert cap.read()[0]
    cap.release()
