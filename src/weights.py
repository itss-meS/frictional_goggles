from pathlib import Path

from ultralytics.utils.downloads import attempt_download_asset

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
OFFICIAL = ("rtdetr-l.pt", "rtdetr-x.pt")
SUFFIXES = {".pt", ".onnx", ".engine"}


def discover() -> list[str]:
    local = []
    if WEIGHTS_DIR.is_dir():
        local = sorted(p.name for p in WEIGHTS_DIR.iterdir() if p.suffix in SUFFIXES)
    extra = [n for n in OFFICIAL if n not in local]
    return local + extra


def resolve(choice: str) -> str:
    path = Path(choice).expanduser()
    if path.is_file():
        return str(path)
    candidate = WEIGHTS_DIR / path.name
    if candidate.is_file():
        return str(candidate)
    if path.name in OFFICIAL:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(candidate)
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Weights not found: {choice}")
