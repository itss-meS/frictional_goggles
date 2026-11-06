import argparse

from ultralytics import RTDETR


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune RT-DETR on the camouflage dataset")
    p.add_argument("--data", default="data/camo/data.yaml")
    p.add_argument("--model", default="weights/rtdetr-l.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="exp")
    p.add_argument("--lr0", type=float, default=1e-4)
    p.add_argument("--close-mosaic", type=int, default=15)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    model = RTDETR(a.model)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        project=a.project,
        name=a.name,
        resume=a.resume,
        optimizer="AdamW",
        lr0=a.lr0,
        lrf=0.01,
        weight_decay=1e-4,
        warmup_epochs=5,
        cos_lr=True,
        close_mosaic=a.close_mosaic,
        patience=a.patience,
        mosaic=1.0,
        mixup=0.2,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )


if __name__ == "__main__":
    main()
