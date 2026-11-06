import json

import numpy as np

from src.detector import Detection
from src.feedback import load_records, make_zip, save_false_positives


def det(box, name="person"):
    return Detection(box=box, conf=0.9, cls_id=0, name=name, group="person", camo=False)


def test_save_and_load(tmp_path):
    frame = np.zeros((100, 200, 3), np.uint8)
    dets = [det((10, 10, 60, 80)), det((-20, -5, 30, 40)), det((150, 50, 150, 90))]
    record = save_false_positives(tmp_path, frame, dets, "clip.mp4#3")
    assert (tmp_path / record["image"]).is_file()
    assert record["detections"][0]["crop"] and (tmp_path / record["detections"][0]["crop"]).is_file()
    assert record["detections"][1]["box"] == [0, 0, 30, 40]
    assert record["detections"][2]["crop"] is None
    record2 = save_false_positives(tmp_path, frame, dets[:1], "b.jpg")
    assert record2["id"] != record["id"]
    loaded = load_records(tmp_path)
    assert [r["source"] for r in loaded] == ["clip.mp4#3", "b.jpg"]
    assert json.loads((tmp_path / "meta.jsonl").read_text().splitlines()[0])["id"] == record["id"]


def test_empty_root(tmp_path):
    assert load_records(tmp_path / "missing") == []


def test_zip_contains_files(tmp_path):
    import io
    import zipfile

    frame = np.zeros((50, 50, 3), np.uint8)
    save_false_positives(tmp_path, frame, [det((5, 5, 20, 20))], "x")
    names = zipfile.ZipFile(io.BytesIO(make_zip(tmp_path))).namelist()
    assert "meta.jsonl" in names
    assert any(n.startswith("images/") for n in names)
    assert any(n.startswith("crops/") for n in names)
