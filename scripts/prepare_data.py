import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataprep import build_dataset, format_report, parse_source  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Merge datasets into the unified YOLO layout used for training")
    p.add_argument("--out", default="data/camo")
    p.add_argument("--roboflow", action="append", default=[], metavar="PATH::src=target,src=target", help="Roboflow YOLOv8 export folder with class mapping, repeatable")
    p.add_argument("--coco", default=None, help="COCO root containing annotations/ and train2017/")
    p.add_argument("--coco-images", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    sources = [parse_source(s) for s in a.roboflow]
    yaml_path, counts = build_dataset(Path(a.out), sources, Path(a.coco) if a.coco else None, a.coco_images, a.seed, a.overwrite)
    print(format_report(counts))
    print(f"\nTrain with: python scripts/train.py --data {yaml_path}")


if __name__ == "__main__":
    main()
