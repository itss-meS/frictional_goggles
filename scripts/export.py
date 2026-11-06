import argparse

from ultralytics import RTDETR


def parse_args():
    p = argparse.ArgumentParser(description="Export RT-DETR weights for edge runtimes")
    p.add_argument("--weights", default="weights/rtdetr-l.pt")
    p.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "torchscript"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--data", default="configs/data.yaml")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    kwargs = {"format": a.format, "imgsz": a.imgsz, "half": a.half, "batch": 1, "device": a.device}
    if a.format == "onnx":
        kwargs["simplify"] = True
    if a.int8:
        kwargs.update(int8=True, data=a.data)
    print(RTDETR(a.weights).export(**kwargs))


if __name__ == "__main__":
    main()
