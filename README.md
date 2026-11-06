# Camouflage Detector

Streamlit app and training/export scripts for RT-DETR based detection of persons, vehicles and drones, including camouflage-specific classes (see `src/classes.py`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision        # use the CUDA build from pytorch.org on GPU machines
pip install -r requirements.txt
```

Tested with Python 3.13, torch 2.15, ultralytics 8.4.202, streamlit 1.66.

## Run the app

```bash
streamlit run app.py
```

Weights `rtdetr-l.pt` / `rtdetr-x.pt` are downloaded into `weights/` on first use. Fine-tuned `.pt`, exported `.onnx` and `.engine` files placed in `weights/` appear in the sidebar, or give any path under "Custom weights path".

Tabs: **Image**, **Video** (annotated H.264 output with ByteTrack IDs), **Hard negatives** (false positives flagged in the other tabs, downloadable as a zip), **Live stream** (camera index, RTSP/HTTP URL or file path).

## Get the data (one command)

```bash
pip install -r requirements.txt
python scripts/download_data.py --api-key YOUR_ROBOFLOW_KEY      # or set ROBOFLOW_API_KEY
```

This downloads the two Roboflow soldier datasets (YOLOv8 export) into `datasets/`, downloads the COCO annotations plus only the 2000 COCO images it needs (not the 18 GB archive), maps the classes and writes `data/camo/`. A free Roboflow key is at https://app.roboflow.com/settings/api. Re-running skips finished downloads. Options: `--coco-images 0` to skip COCO, `--download-only`, `--overwrite`, and `--roboflow WORKSPACE/PROJECT[/VERSION]` (repeatable) to use other datasets. The class mapping (`soldier` to `soldier_woodland`, `civilian` to `person`, anything else skipped) is `CLASS_MAP` in `src/download.py`.

## Prepare data manually

Download each Roboflow dataset with the **YOLOv8** export format and unzip it (it has `train/`, `valid/`, `test/` and `data.yaml`). Map its class names to ours; every source class must be mapped or marked `skip`.

```bash
python scripts/prepare_data.py \
  --roboflow "datasets/soldier-civilian-detection::soldier=soldier_woodland,civilian=person" \
  --roboflow "datasets/camouflaged-soldiers::soldier=soldier_woodland" \
  --coco datasets/coco --coco-images 2000
```

`--coco` expects `annotations/instances_train2017.json` and `train2017/`. COCO adds person, car, truck, bus, motorcycle, bicycle and boat to the training split only, so validation stays camouflage-only. The output is `data/camo/` plus `data/camo/data.yaml`, and the per-class counts are printed. Add `--overwrite` to rebuild.

## Train

```bash
python scripts/train.py --epochs 5 --imgsz 640 --batch 4      # sanity run
python scripts/train.py --epochs 100 --imgsz 640 --batch 16   # real run, lower --batch if out of memory
```

Results are written to `runs/train/exp/` (`weights/best.pt`, curves, `results.csv`). On Windows use `--workers 0` if the dataloader errors.

## Export

```bash
python scripts/export.py --weights runs/train/exp/weights/best.pt --format onnx --imgsz 640
python scripts/export.py --weights runs/train/exp/weights/best.pt --format engine --half --device 0
```

## Tests

```bash
pytest
```

The tests run the real `rtdetr-l` model on CPU (about 35 s).

## Layout

```
app.py               Streamlit UI
src/classes.py       class taxonomy and super-class groups
src/detector.py      RT-DETR wrapper: predict, track, draw
src/media.py         image/video IO, H.264 writer
src/pipeline.py      video processing job
src/dataprep.py      dataset conversion and merging
src/download.py      dataset downloads (Roboflow, COCO subset)
src/feedback.py      hard-negative store
scripts/             prepare_data.py, train.py, export.py
configs/data.yaml    dataset definition
```
