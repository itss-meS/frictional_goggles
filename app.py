import hashlib
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src import feedback, media, weights
from src.detector import Detector, available_devices, draw
from src.pipeline import process_video

HARD_NEG_DIR = Path(__file__).resolve().parent / "data" / "hard_negatives"

st.set_page_config(page_title="Camouflage Detector", layout="wide")


@st.cache_resource(show_spinner="Loading model")
def load_detector(path: str, device: str, imgsz: int) -> Detector:
    return Detector(path, device, imgsz)


@st.cache_data(show_spinner="Running detection")
def run_image(_detector, model_key, data, conf, classes):
    frame = media.decode_image(data)
    return _detector.predict(frame, conf, classes)


@st.cache_data(show_spinner="Running detection")
def run_video_frame(_detector, model_key, path, index, conf, classes):
    frame = media.read_frame(path, index)
    return _detector.predict(frame, conf, classes)


def detections_table(detections):
    if not detections:
        st.info("No detections at this confidence threshold.")
        return
    st.dataframe(pd.DataFrame([d.row() for d in detections]), width="stretch", hide_index=True)


def group_metrics(detections, speed):
    groups = {}
    for d in detections:
        groups[d.group] = groups.get(d.group, 0) + 1
    cols = st.columns(max(2, len(groups) + 1))
    cols[0].metric("Detections", len(detections))
    cols[1].metric("Latency", f"{sum(speed.values()):.0f} ms")
    for col, (group, count) in zip(cols[2:], sorted(groups.items())):
        col.metric(group, count)


def flag_ui(frame, detections, source, key):
    if not detections:
        return
    labels = [f"#{i} {d.name} {d.conf:.2f}" for i, d in enumerate(detections)]
    chosen = st.multiselect("Mark false positives", labels, key=f"{key}_sel")
    if st.button("Save as hard negatives", key=f"{key}_btn", disabled=not chosen):
        picked = [detections[labels.index(label)] for label in chosen]
        record = feedback.save_false_positives(HARD_NEG_DIR, frame, picked, source)
        st.success(f"Saved {len(picked)} detection(s) to {record['image']}")


def clear_video_state():
    previous = st.session_state.pop("video_result", None)
    if previous:
        shutil.rmtree(previous["dir"], ignore_errors=True)


with st.sidebar:
    st.title("Camouflage Detector")
    choice = st.selectbox("Weights", weights.discover())
    custom = st.text_input("Custom weights path", placeholder="runs/train/exp/weights/best.pt")
    device = st.selectbox("Device", available_devices())
    imgsz = st.select_slider("Input size", options=[320, 480, 640, 800, 1024], value=640)
    conf = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)

try:
    weights_path = weights.resolve(custom.strip() or choice)
    detector = load_detector(weights_path, device, imgsz)
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

model_key = f"{weights_path}|{device}|{imgsz}"

with st.sidebar:
    selected = st.multiselect("Class filter (empty = all)", sorted(set(detector.names.values())))
    classes = tuple(sorted(i for i, n in detector.names.items() if n in selected)) or None
    st.caption(f"{Path(weights_path).name} on {detector.device}, {len(detector.names)} classes")

tab_image, tab_video, tab_negatives, tab_stream = st.tabs(["Image", "Video", "Hard negatives", "Live stream"])

with tab_image:
    upload = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="image_upload")
    if upload:
        data = upload.getvalue()
        frame = media.decode_image(data)
        detections, speed = run_image(detector, model_key, data, conf, classes)
        group_metrics(detections, speed)
        left, right = st.columns([3, 2])
        left.image(draw(frame, detections), channels="BGR", width="stretch")
        with right:
            detections_table(detections)
            flag_ui(frame, detections, upload.name, f"img_{hashlib.md5(data).hexdigest()}_{model_key}_{conf}_{classes}")

with tab_video:
    video = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv", "webm"], key="video_upload")
    c1, c2, c3 = st.columns(3)
    stride = c1.number_input("Process every Nth frame", 1, 60, 3)
    max_frames = c2.number_input("Max frames processed", 1, 5000, 200)
    track = c3.toggle("Track objects (ByteTrack)", value=True)
    if st.button("Process video", disabled=video is None, type="primary"):
        clear_video_state()
        workdir = Path(tempfile.mkdtemp(prefix="camo_"))
        src = workdir / f"input{Path(video.name).suffix.lower()}"
        src.write_bytes(video.getvalue())
        dst = workdir / "annotated.mp4"
        bar = st.progress(0.0, text="Processing")
        try:
            started = time.time()
            summary = process_video(
                detector, src, dst, conf, classes, int(stride), int(max_frames), track,
                progress=lambda p: bar.progress(p, text=f"Processing {p:.0%}"),
            )
            summary["seconds"] = time.time() - started
            st.session_state["video_result"] = {
                "dir": str(workdir), "src": str(src), "dst": str(dst), "name": video.name, "summary": summary,
            }
        except Exception as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            st.error(f"Processing failed: {exc}")
        finally:
            bar.empty()

    result = st.session_state.get("video_result")
    if result:
        summary = result["summary"]
        st.video(result["dst"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Frames processed", summary["frames"])
        m2.metric("Avg latency", f"{summary['avg_ms']:.0f} ms")
        m3.metric("Wall time", f"{summary['seconds']:.1f} s")
        if summary["groups"]:
            st.dataframe(
                pd.DataFrame(
                    [{"group": g, **v} for g, v in summary["groups"].items()]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No detections in the processed frames.")
        with open(result["dst"], "rb") as f:
            st.download_button("Download annotated video", f.read(), file_name=f"annotated_{Path(result['name']).stem}.mp4", mime="video/mp4")
        with st.expander("Review a frame"):
            last = max(0, summary["source_frames"] - 1)
            index = st.slider("Source frame", 0, last, 0) if last > 0 else 0
            detections, speed = run_video_frame(detector, model_key, result["src"], index, conf, classes)
            frame = media.read_frame(result["src"], index)
            group_metrics(detections, speed)
            st.image(draw(frame, detections), channels="BGR", width="stretch")
            flag_ui(frame, detections, f"{result['name']}#{index}", f"vid_{result['dir']}_{index}_{conf}_{classes}")

with tab_negatives:
    records = feedback.load_records(HARD_NEG_DIR)
    total = sum(len(r["detections"]) for r in records)
    a, b = st.columns(2)
    a.metric("Flagged frames", len(records))
    b.metric("Flagged detections", total)
    if records:
        st.download_button("Download hard-negative set (zip)", feedback.make_zip(HARD_NEG_DIR), file_name="hard_negatives.zip", mime="application/zip")
        crops = [(HARD_NEG_DIR / det["crop"], f"{det['class']} {det['conf']:.2f}") for r in reversed(records) for det in r["detections"] if det["crop"]]
        for row_start in range(0, min(len(crops), 24), 6):
            cols = st.columns(6)
            for col, (path, caption) in zip(cols, crops[row_start:row_start + 6]):
                col.image(str(path), caption=caption, width="stretch")
        st.dataframe(
            pd.DataFrame([{"time": r["time"], "source": r["source"], "detections": len(r["detections"]), "image": r["image"]} for r in reversed(records)]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Nothing flagged yet. Mark false positives in the Image or Video tab.")

with tab_stream:
    source = st.text_input("Source", value="0", help="Camera index, RTSP/HTTP URL or video file path")
    s1, s2 = st.columns(2)
    stream_track = s1.toggle("Track objects", value=True, key="stream_track")
    run = s2.toggle("Run stream", key="stream_run")
    placeholder = st.empty()
    stats = st.empty()
    if run:
        cap = None
        try:
            cap = media.open_capture(source)
            if stream_track:
                detector.reset_tracker()
            infer = detector.track if stream_track else detector.predict
            while True:
                ok, frame = cap.read()
                if not ok:
                    st.info("Stream ended.")
                    break
                detections, speed = infer(frame, conf, classes)
                placeholder.image(draw(frame, detections), channels="BGR", width="stretch")
                stats.caption(f"{len(detections)} detections, {sum(speed.values()):.0f} ms per frame")
        except Exception as exc:
            st.error(str(exc))
        finally:
            if cap is not None:
                cap.release()
