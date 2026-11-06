"""Download the datasets, put them under datasets/, and build the merged training set in data/camo/.

    python scripts/download_data.py --api-key YOUR_ROBOFLOW_KEY

Safe to re-run: finished downloads are skipped, only the merge is redone.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataprep import build_dataset, format_report  # noqa: E402
from src.download import ROBOFLOW_DATASETS, build_mapping, describe_mapping, fetch_coco, fetch_roboflow, log  # noqa: E402


def parse_roboflow(spec: str) -> tuple[str, str, int | None, str]:
    parts = spec.strip("/").split("/")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"'{spec}' should look like workspace/project or workspace/project/version")
    version = int(parts[2]) if len(parts) == 3 else None
    return parts[0], parts[1], version, parts[1]


def parse_args():
    p = argparse.ArgumentParser(description="Download datasets and prepare data/camo/ in one step")
    p.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"), help="Roboflow API key (or set ROBOFLOW_API_KEY)")
    p.add_argument("--roboflow", action="append", type=parse_roboflow, metavar="WORKSPACE/PROJECT[/VERSION]", help="Roboflow dataset to use instead of the defaults, repeatable")
    p.add_argument("--datasets-dir", default="datasets")
    p.add_argument("--out", default="data/camo")
    p.add_argument("--coco-images", type=int, default=2000, help="COCO images to download and merge, 0 to skip COCO")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--download-only", action="store_true", help="Download but do not merge")
    p.add_argument("--overwrite", action="store_true", help="Rebuild data/camo/ if it already exists")
    return p.parse_args()


def main():
    a = parse_args()
    datasets_dir = Path(a.datasets_dir)
    specs = a.roboflow or ROBOFLOW_DATASETS
    folders = fetch_roboflow(datasets_dir, a.api_key, specs)
    coco = fetch_coco(datasets_dir, a.coco_images, a.seed) if a.coco_images > 0 else None
    if a.download_only:
        log("\nDownload finished.")
        return
    log("\nClass mapping (classes not listed in src/download.py CLASS_MAP are skipped):")
    sources = []
    for folder in folders:
        mapping = build_mapping(folder)
        log(describe_mapping(folder, mapping))
        sources.append((folder, mapping))
    yaml_path, counts = build_dataset(Path(a.out), sources, coco, a.coco_images, a.seed, a.overwrite)
    log("\n" + format_report(counts))
    log(f"\nDone. Train with: python scripts/train.py --data {yaml_path}")


if __name__ == "__main__":
    main()
