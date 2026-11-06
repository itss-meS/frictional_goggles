import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.classes import CLASS_NAMES
from src.dataprep import build_dataset, clamp_box, label_to_box, parse_mapping, parse_source

ROOT = Path(__file__).resolve().parent.parent


def write_image(path, w=100, h=80):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 90, np.uint8))


@pytest.fixture
def roboflow(tmp_path):
    src = tmp_path / "soldier-civilian"
    (src / "data.yaml").parent.mkdir(parents=True)
    (src / "data.yaml").write_text(yaml.safe_dump({"names": ["civilian", "soldier", "junk"], "nc": 3}))
    for split, n in (("train", 3), ("valid", 2), ("test", 1)):
        for i in range(n):
            write_image(src / split / "images" / f"{i}.jpg")
            labels = src / split / "labels"
            labels.mkdir(parents=True, exist_ok=True)
            lines = ["1 0.5 0.5 0.4 0.4", "0 0.2 0.2 0.2 0.2", "2 0.7 0.7 0.1 0.1"]
            if i == 1:
                lines = ["1 0.1 0.1 0.2 0.2 0.3 0.1 0.3 0.4 0.1 0.4"]
            if i == 2:
                lines = []
            (labels / f"{i}.txt").write_text("\n".join(lines))
    write_image(src / "train" / "images" / "nolabel.jpg")
    return src


@pytest.fixture
def coco(tmp_path):
    root = tmp_path / "coco"
    images, annotations = [], []
    for i in range(6):
        write_image(root / "train2017" / f"{i:012d}.jpg", 200, 100)
        images.append({"id": i, "file_name": f"{i:012d}.jpg", "width": 200, "height": 100})
    categories = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}, {"id": 18, "name": "dog"}]
    annotations += [
        {"id": 1, "image_id": 0, "category_id": 1, "bbox": [10, 10, 50, 40], "iscrowd": 0},
        {"id": 2, "image_id": 0, "category_id": 3, "bbox": [100, 20, 80, 60], "iscrowd": 0},
        {"id": 3, "image_id": 1, "category_id": 18, "bbox": [10, 10, 50, 40], "iscrowd": 0},
        {"id": 4, "image_id": 2, "category_id": 1, "bbox": [10, 10, 50, 40], "iscrowd": 1},
        {"id": 5, "image_id": 3, "category_id": 1, "bbox": [150, 50, 100, 100], "iscrowd": 0},
        {"id": 6, "image_id": 4, "category_id": 1, "bbox": [10, 10, 1, 1], "iscrowd": 0},
    ]
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "instances_train2017.json").write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return root


def read_labels(path):
    return [line.split() for line in path.read_text().splitlines() if line]


def test_parse_mapping():
    assert parse_mapping("soldier=soldier_woodland, civilian=person,junk=skip") == {"soldier": "soldier_woodland", "civilian": "person", "junk": None}
    with pytest.raises(ValueError):
        parse_mapping("soldier=tank")
    with pytest.raises(ValueError):
        parse_mapping("soldier")
    path, mapping = parse_source(r"C:\data\rf::soldier=soldier_desert")
    assert str(path) == r"C:\data\rf" and mapping == {"soldier": "soldier_desert"}


def test_label_helpers():
    assert label_to_box(["3", "0.5", "0.5", "0.2", "0.4"]) == (3, 0.5, 0.5, 0.2, 0.4)
    cls, cx, cy, w, h = label_to_box("0 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4".split())
    assert (cls, round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)) == (0, 0.2, 0.25, 0.2, 0.3)
    with pytest.raises(ValueError):
        label_to_box(["0", "0.1", "0.2"])
    assert clamp_box(0.95, 0.5, 0.2, 0.2) == pytest.approx((0.925, 0.5, 0.15, 0.2))
    assert clamp_box(1.5, 0.5, 0.2, 0.2) is None


def test_roboflow_and_coco(tmp_path, roboflow, coco):
    out = tmp_path / "camo"
    mapping = {"civilian": "person", "soldier": "soldier_woodland", "junk": None}
    yaml_path, counts = build_dataset(out, [(roboflow, mapping)], coco, 100, 0, False)
    config = yaml.safe_load(yaml_path.read_text())
    assert Path(config["path"]) == out.resolve()
    assert [config["names"][i] for i in range(15)] == CLASS_NAMES
    n_train = len(list((out / "images" / "train").iterdir()))
    assert n_train == len(list((out / "labels" / "train").iterdir())) == 4 + 2
    assert counts["train"]["_images"] == n_train and counts["val"]["_images"] == 2 and counts["test"]["_images"] == 1
    train_labels = sorted((out / "labels" / "train").glob("0_*_0.txt"))[0]
    assert [r[0] for r in read_labels(train_labels)] == [str(CLASS_NAMES.index("soldier_woodland")), "0"]
    polygon = sorted((out / "labels" / "train").glob("0_*_1.txt"))[0]
    assert read_labels(polygon)[0][0] == "8"
    assert not any("junk" in str(r) for r in read_labels(train_labels)) and len(read_labels(train_labels)) == 2
    assert (out / "labels" / "train" / next(p.name for p in (out / "labels" / "train").glob("0_*_nolabel.txt"))).read_text() == ""
    coco_labels = {p.name: read_labels(p) for p in (out / "labels" / "train").glob("coco_*.txt")}
    assert set(coco_labels) == {"coco_000000000000.txt", "coco_000000000003.txt"}
    first = coco_labels["coco_000000000000.txt"]
    assert [r[0] for r in first] == ["0", "1"]
    assert [round(float(v), 4) for v in first[0][1:]] == [0.175, 0.3, 0.25, 0.4]
    assert [round(float(v), 4) for v in coco_labels["coco_000000000003.txt"][0][1:]] == [0.875, 0.75, 0.25, 0.5]
    assert counts["train"]["car"] == 1 and counts["train"]["dog"] == 0


def test_coco_max_images_is_deterministic(tmp_path, roboflow, coco):
    mapping = {"civilian": "person", "soldier": "soldier_woodland", "junk": None}
    names = []
    for run in range(2):
        out = tmp_path / f"o{run}"
        build_dataset(out, [(roboflow, mapping)], coco, 1, 7, False)
        names.append(sorted(p.name for p in (out / "images" / "train").glob("coco_*")))
    assert names[0] == names[1] and len(names[0]) == 1


def test_errors(tmp_path, roboflow, coco):
    with pytest.raises(ValueError, match="no mapping"):
        build_dataset(tmp_path / "a", [(roboflow, {"soldier": "soldier_woodland"})], None, 1, 0, False)
    with pytest.raises(ValueError, match="unknown source class"):
        build_dataset(tmp_path / "b", [(roboflow, {"civilian": "person", "soldier": "soldier_woodland", "junk": None, "x": None})], None, 1, 0, False)
    with pytest.raises(ValueError, match="No validation"):
        build_dataset(tmp_path / "c", [], coco, 5, 0, False)
    with pytest.raises(ValueError, match="Nothing to do"):
        build_dataset(tmp_path / "d", [], None, 5, 0, False)
    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path / "e", [(tmp_path / "missing", {})], None, 1, 0, False)
    mapping = {"civilian": "person", "soldier": "soldier_woodland", "junk": None}
    out = tmp_path / "f"
    build_dataset(out, [(roboflow, mapping)], None, 1, 0, False)
    with pytest.raises(FileExistsError):
        build_dataset(out, [(roboflow, mapping)], None, 1, 0, False)
    build_dataset(out, [(roboflow, mapping)], None, 1, 0, True)


def test_cli_and_ultralytics_accepts_output(tmp_path, roboflow):
    out = tmp_path / "camo"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_data.py"), "--out", str(out), "--roboflow", f"{roboflow}::civilian=person,soldier=soldier_desert,junk=skip"],
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "soldier_desert" in result.stdout and "train: 4 images" in result.stdout
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(out / "data.yaml"), autodownload=False)
    assert data["nc"] == 15 and data["names"][9] == "soldier_desert"
    assert Path(data["train"]).is_dir() and Path(data["val"]).is_dir()
