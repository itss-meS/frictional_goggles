from pathlib import Path

import cv2
import pytest
import ultralytics

from src.detector import Detector
from src.media import Mp4Writer
from src.weights import resolve


@pytest.fixture(scope="session")
def bus_image():
    path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    return cv2.imread(str(path))


@pytest.fixture(scope="session")
def detector():
    return Detector(resolve("rtdetr-l.pt"), "cpu", 640)


@pytest.fixture(scope="session")
def pan_video(tmp_path_factory, bus_image):
    path = tmp_path_factory.mktemp("video") / "pan.mp4"
    h, w = bus_image.shape[:2]
    writer = Mp4Writer(path, 10, 640, 640)
    for i in range(12):
        writer.write(cv2.resize(bus_image[:, i * 12 : i * 12 + (w - 150)], (640, 640)))
    writer.close()
    return path
