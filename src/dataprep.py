from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import yaml

from .classes import CLASS_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
COCO_MAP = {name: name for name in ("person", "car", "truck", "bus", "motorcycle", "bicycle", "boat")}


def parse_mapping(text: str) -> dict[str, Optional[str]]:
    mapping: dict[str, Optional[str]] = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        if "=" not in item:
            raise ValueError(f"Bad mapping '{item}', expected source=target")
        source, target = (s.strip() for s in item.split("=", 1))
        if target == "skip":
            mapping[source] = None
        elif target in CLASS_NAMES:
            mapping[source] = target
        else:
            raise ValueError(f"Unknown target class '{target}'. Valid: {', '.join(CLASS_NAMES)} or skip")
    return mapping


def parse_source(spec: str) -> tuple[Path, dict[str, Optional[str]]]:
    if "::" not in spec:
        raise ValueError(f"Bad --roboflow value '{spec}', expected PATH::source=target,source=target")
    path, mapping = spec.split("::", 1)
    return Path(path), parse_mapping(mapping)


def read_source_names(src: Path) -> list[str]:
    config = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def label_to_box(parts: list[str]) -> tuple[int, float, float, float, float]:
    cls = int(float(parts[0]))
    values = [float(v) for v in parts[1:]]
    if len(values) == 4:
        cx, cy, w, h = values
        return cls, cx, cy, w, h
    if len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    raise ValueError(f"Unsupported label line: {' '.join(parts)}")


def clamp_box(cx: float, cy: float, w: float, h: float) -> Optional[tuple[float, float, float, float]]:
    x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def format_label(cls_id: int, box: tuple[float, float, float, float]) -> str:
    return f"{cls_id} " + " ".join(f"{v:.6f}" for v in box)


def tag_for(path: Path, index: int) -> str:
    return f"{index}_{re.sub(r'[^A-Za-z0-9]+', '-', path.name).strip('-')[:24]}"


def prepare_output(out: Path, overwrite: bool) -> None:
    existing = [out / "images", out / "labels"]
    if any(p.exists() and any(p.iterdir()) for p in existing):
        if not overwrite:
            raise FileExistsError(f"{out} already contains data, pass --overwrite to replace it")
        for p in existing:
            shutil.rmtree(p, ignore_errors=True)
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            (out / kind / split).mkdir(parents=True, exist_ok=True)


def convert_roboflow(src: Path, out: Path, mapping: dict[str, Optional[str]], tag: str, counts: dict) -> None:
    names = read_source_names(src)
    unmapped = [n for n in names if n not in mapping]
    if unmapped:
        raise ValueError(f"{src.name}: no mapping for classes {unmapped}. Use source=target or source=skip")
    for bad in mapping:
        if bad not in names:
            raise ValueError(f"{src.name}: mapping refers to unknown source class '{bad}', available: {names}")
    target_ids = {i: (CLASS_NAMES.index(mapping[n]) if mapping[n] else None) for i, n in enumerate(names)}
    found = False
    for folder, split in SPLIT_ALIASES.items():
        images_dir = src / folder / "images"
        if not images_dir.is_dir():
            continue
        found = True
        for image in sorted(images_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_file = src / folder / "labels" / f"{image.stem}.txt"
            lines = []
            if label_file.is_file():
                for raw in label_file.read_text(encoding="utf-8").splitlines():
                    parts = raw.split()
                    if not parts:
                        continue
                    cls, cx, cy, w, h = label_to_box(parts)
                    if cls not in target_ids:
                        raise ValueError(f"{label_file}: class id {cls} out of range")
                    target = target_ids[cls]
                    box = clamp_box(cx, cy, w, h)
                    if target is None or box is None:
                        continue
                    lines.append(format_label(target, box))
                    counts[split][CLASS_NAMES[target]] += 1
            stem = f"{tag}_{image.stem}"
            shutil.copy2(image, out / "images" / split / f"{stem}{image.suffix.lower()}")
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts[split]["_images"] += 1
    if not found:
        raise FileNotFoundError(f"{src}: no train/valid/test folders with an images directory (export as YOLOv8 format)")


def coco_annotations_by_image(data: dict, wanted: dict) -> dict[int, list]:
    per_image: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd") or ann["category_id"] not in wanted:
            continue
        per_image[ann["image_id"]].append(ann)
    return per_image


def coco_image_order(per_image: dict, seed: int) -> list[int]:
    """The order in which convert_coco considers images; the downloader fetches the same prefix."""
    ids = sorted(per_image)
    random.Random(seed).shuffle(ids)
    return ids


def coco_candidate_files(annotations: Path, seed: int) -> list[str]:
    data = json.loads(annotations.read_text(encoding="utf-8"))
    category = {c["id"]: c["name"] for c in data["categories"]}
    wanted = {cid: COCO_MAP[n] for cid, n in category.items() if n in COCO_MAP}
    info = {im["id"]: im["file_name"] for im in data["images"]}
    return [info[i] for i in coco_image_order(coco_annotations_by_image(data, wanted), seed)]


def convert_coco(root: Path, out: Path, max_images: int, seed: int, counts: dict) -> None:
    annotations = root / "annotations" / "instances_train2017.json"
    images_dir = root / "train2017"
    if not annotations.is_file() or not images_dir.is_dir():
        raise FileNotFoundError(f"Expected {annotations} and {images_dir}")
    data = json.loads(annotations.read_text(encoding="utf-8"))
    category = {c["id"]: c["name"] for c in data["categories"]}
    wanted = {cid: COCO_MAP[n] for cid, n in category.items() if n in COCO_MAP}
    info = {im["id"]: im for im in data["images"]}
    per_image = coco_annotations_by_image(data, wanted)
    ids = coco_image_order(per_image, seed)
    accepted = 0
    for image_id in ids:
        if accepted >= max_images:
            break
        im = info[image_id]
        source = images_dir / im["file_name"]
        if not source.is_file():
            continue
        W, H = im["width"], im["height"]
        lines = []
        for ann in per_image[image_id]:
            x, y, w, h = ann["bbox"]
            box = clamp_box((x + w / 2) / W, (y + h / 2) / H, w / W, h / H)
            if box is None or w < 2 or h < 2:
                continue
            target = CLASS_NAMES.index(wanted[ann["category_id"]])
            lines.append(format_label(target, box))
            counts["train"][CLASS_NAMES[target]] += 1
        if not lines:
            continue
        stem = f"coco_{source.stem}"
        shutil.copy2(source, out / "images" / "train" / f"{stem}{source.suffix.lower()}")
        (out / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        counts["train"]["_images"] += 1
        accepted += 1


def write_data_yaml(out: Path) -> Path:
    config = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    path = out / "data.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def build_dataset(out: Path, roboflow: list[tuple[Path, dict]], coco: Optional[Path], coco_images: int, seed: int, overwrite: bool) -> tuple[Path, dict]:
    if not roboflow and coco is None:
        raise ValueError("Nothing to do: pass --roboflow and/or --coco")
    prepare_output(out, overwrite)
    counts: dict = {s: Counter() for s in ("train", "val", "test")}
    for index, (src, mapping) in enumerate(roboflow):
        convert_roboflow(src, out, mapping, tag_for(src, index), counts)
    if coco is not None:
        convert_coco(coco, out, coco_images, seed, counts)
    if counts["val"]["_images"] == 0:
        raise ValueError("No validation images produced. Roboflow exports need a valid/ split; COCO is used for training only")
    return write_data_yaml(out), counts


def format_report(counts: dict) -> str:
    lines = []
    for split in ("train", "val", "test"):
        c = counts[split]
        lines.append(f"{split}: {c['_images']} images")
        for name in CLASS_NAMES:
            if c[name]:
                lines.append(f"    {name:<18}{c[name]}")
    return "\n".join(lines)
