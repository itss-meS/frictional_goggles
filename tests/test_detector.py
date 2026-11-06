import numpy as np

from src.detector import Detection, available_devices, color_for, draw, resolve_device


def test_devices():
    assert available_devices()[:2] == ["auto", "cpu"]
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "mps"} or resolve_device("auto").startswith("cuda")


def test_predict_bus(detector, bus_image):
    dets, speed = detector.predict(bus_image, 0.5)
    names = [d.name for d in dets]
    assert "bus" in names
    assert names.count("person") >= 3
    h, w = bus_image.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= w + 1 and 0 <= y1 < y2 <= h + 1
        assert d.track_id is None
    assert {"preprocess", "inference", "postprocess"} <= set(speed)
    bus = next(d for d in dets if d.name == "bus")
    assert bus.group == "vehicle" and not bus.camo


def test_class_filter(detector, bus_image):
    person_id = next(i for i, n in detector.names.items() if n == "person")
    dets, _ = detector.predict(bus_image, 0.25, [person_id])
    assert dets and {d.name for d in dets} == {"person"}


def test_high_conf_gives_fewer(detector, bus_image):
    low, _ = detector.predict(bus_image, 0.1)
    high, _ = detector.predict(bus_image, 0.9)
    assert len(high) < len(low)
    assert all(d.conf >= 0.9 for d in high)


def test_draw_does_not_modify_input(bus_image):
    d = Detection(box=(10.0, 0.0, 200.0, 300.0), conf=0.8, cls_id=0, name="person", group="person", camo=False, track_id=3)
    before = bus_image.copy()
    out = draw(bus_image, [d])
    assert out.shape == bus_image.shape
    assert np.array_equal(bus_image, before)
    assert not np.array_equal(out, before)
    assert color_for("person") == color_for("person")


def test_track_ids_and_mode_switch(detector, pan_video, bus_image):
    from src.media import iter_frames

    runs = []
    for _ in range(2):
        detector.reset_tracker()
        frames = []
        for _, frame in iter_frames(pan_video, 2):
            dets, _ = detector.track(frame, 0.5)
            frames.append({(d.name, d.track_id) for d in dets})
        runs.append(frames)
    assert runs[0] == runs[1]
    first, later = runs[0][0], runs[0][1:]
    assert ("bus", 1) in first
    assert all(("bus", 1) in f for f in later)
    assert all({t for n, t in f if n == "person"} == {2, 3} for f in later)
    assert {t for n, t in first if n == "person"} >= {2, 3}
    dets, _ = detector.predict(bus_image, 0.5)
    assert all(d.track_id is None for d in dets)
