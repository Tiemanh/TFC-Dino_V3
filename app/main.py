"""FastAPI backend for the TF-C + DINOv3 prediction product."""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[1]
TFC_DIR = ROOT / "code" / "TFC"
sys.path.insert(0, str(TFC_DIR))
from app.data_io import parse_xyz_bytes  # noqa: E402


MODEL_PATH = Path(os.environ.get("MODEL_BUNDLE_PATH", ROOT / "artifacts" / "gesture_model" / "model_bundle.pt"))
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLE_FILE = ROOT / "sample_data" / "gesture_example.csv"
app = FastAPI(title="TF-C + DINOv3 Gesture Prediction", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_predictor = None
_load_error = None
logger = logging.getLogger(__name__)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, _error):
    """Return a stable JSON contract for malformed multipart requests."""
    return JSONResponse(status_code=400, content={"detail": "Yêu cầu không hợp lệ hoặc thiếu file dữ liệu"})


def get_predictor():
    global _predictor, _load_error
    # Retry when a model appears after the web server was started.
    if _predictor is None:
        if not MODEL_PATH.exists():
            _load_error = (
                "Chưa có model AI. Hãy huấn luyện dữ liệu và đặt model_bundle.pt "
                "vào thư mục artifacts/gesture_model."
            )
        else:
            try:
                # Importing PyTorch/DINOv3 is expensive. Keep lightweight routes
                # and API tests fast by loading the AI runtime only on demand.
                from inference import GesturePredictor

                _predictor = GesturePredictor(MODEL_PATH, device="cpu")
                _load_error = None
            except Exception as error:  # surfaced through health/predict endpoints
                _load_error = f"Không thể nạp model: {error}"
    return _predictor


def parse_xyz(contents: bytes):
    try:
        return parse_xyz_bytes(contents)
    except Exception as error:
        raise HTTPException(400, str(error)) from error


async def read_sensor_upload(file: UploadFile) -> np.ndarray:
    """Validate an upload before visualization or inference."""
    filename = (file.filename or "").strip()
    if not filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(400, "Chỉ chấp nhận file CSV hoặc TXT dạng bảng")
    return parse_xyz(await file.read())


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/style.css", include_in_schema=False)
async def stylesheet():
    """Keep the UI styled when index.html uses portable sibling asset paths."""
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
async def frontend_script():
    """Serve the browser application from the same path as index.html."""
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


@app.get("/sample-data/gesture_example.csv", include_in_schema=False)
async def sample_data():
    """Provide a real InGesture CSV so users can verify the upload flow."""
    return FileResponse(SAMPLE_FILE, media_type="text/csv", filename="gesture_example.csv")


@app.get("/api/health")
async def health():
    predictor = get_predictor()
    info = predictor.training_info if predictor else {}
    message = "Model đã sẵn sàng" if predictor else _load_error
    if info.get("initialization") == "random_full_vits16":
        message = (
            "Bản thử nghiệm: TF-C + DINOv3 ViT-S/16 học từ đầu trên InGesture, "
            "chưa dùng trọng số pretrained. "
            f"Test accuracy: {predictor.metrics.get('test_accuracy', 0):.1%}; "
            f"macro F1: {predictor.metrics.get('macro_f1', 0):.1%}."
        )
    return {
        "ready": predictor is not None,
        "model_path": str(MODEL_PATH),
        "message": message,
        "training_info": info,
        "evaluation": predictor.metrics if predictor else {},
        "class_names": predictor.class_names if predictor else [],
        "required_samples": int(predictor.metadata["window_size"]) if predictor else 206,
    }


@app.get("/api/live", include_in_schema=False)
async def live():
    """Lightweight platform health check that does not load the AI model."""
    return {"status": "ok"}


@app.post("/api/preview")
async def preview(file: UploadFile = File(...)):
    xyz = await read_sensor_upload(file)
    display = xyz[-min(len(xyz), 500):]
    spectrum = np.abs(np.fft.rfft(display[:, 0]))
    return {
        "filename": file.filename,
        "sample_count": int(len(xyz)),
        "time": {"x": display[:, 0].tolist(), "y": display[:, 1].tolist(), "z": display[:, 2].tolist()},
        "frequency": spectrum[: min(len(spectrum), 100)].tolist(),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(503, _load_error)
    xyz = await read_sensor_upload(file)
    required = int(predictor.metadata["window_size"])
    if len(xyz) < required:
        raise HTTPException(400, f"Cần ít nhất {required} mẫu, file hiện có {len(xyz)}")
    try:
        result = predictor.predict(xyz)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(500, "Model không thể xử lý dữ liệu này") from error
    result["sample_count"] = int(len(xyz))
    return result
