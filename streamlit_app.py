"""Streamlit deployment for the TF-C + DINOv3 gesture classifier."""

from __future__ import annotations

import hashlib
import html
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
TFC_DIR = ROOT / "code" / "TFC"
sys.path.insert(0, str(TFC_DIR))

from inference import GesturePredictor  # noqa: E402
from app.data_io import parse_xyz_bytes  # noqa: E402


MODEL_PATH = Path(
    os.environ.get("MODEL_BUNDLE_PATH", ROOT / "artifacts" / "gesture_model" / "model_bundle.pt")
)
SAMPLE_PATH = ROOT / "sample_data" / "gesture_example.csv"
COLORS = {"X": "#78838c", "Y": "#81a8e7", "Z": "#9c8fc4"}


st.set_page_config(
    page_title="TF-C Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .stApp { background: #f7f7f5; color: #202a38; }
    .block-container { max-width: 1780px; padding: 1rem 1.2rem 2rem; }
    .hero { background:#eee9f6; border:3px solid #858d93; border-radius:24px;
            padding:20px 30px; margin-bottom:14px; }
    .hero h1 { margin:0; font-size:2.55rem; color:#202a38; }
    .hero p { margin:.35rem 0 0; color:#666f79; }
    .section-title { padding:12px 16px; border-radius:14px; margin:0 0 14px;
                     font-size:1.65rem; font-weight:800; color:#202a38; }
    .input-title { background:#f7dcdc; }
    .preview-title { background:#d9e5fb; }
    .result-title { background:#dcedd7; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-width:2px; border-radius:20px; }
    div[data-testid="stFileUploaderDropzone"] { min-height:170px; border:2px dashed #99a1a7;
                                                  border-radius:16px; background:white; }
    .note, .model-state, .result-card { background:white; border:2px solid #c9ced2;
                                       border-radius:14px; padding:16px; margin:10px 0; }
    .note code { display:block; background:#f7f3f3; padding:10px; margin:10px 0; }
    .predicted { font-size:1.55rem; font-weight:850; overflow-wrap:anywhere; }
    .confidence { color:#328442; font-size:2rem; font-weight:850; }
    .prob-row { display:grid; grid-template-columns:minmax(130px,1fr) 1.5fr 58px;
                gap:9px; align-items:center; margin:9px 0; font-size:.9rem; }
    .prob-track { height:12px; background:#e5eae4; border-radius:4px; overflow:hidden; }
    .prob-fill { height:100%; background:#95c388; }
    .status-ok { color:#2f8542; font-weight:700; }
    .status-warn { color:#b05b45; font-weight:700; }
    footer { visibility:hidden; }
    @media (max-width: 900px) { .hero h1 { font-size:2rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Đang nạp model TF-C + DINOv3…")
def load_predictor(model_path: str) -> GesturePredictor:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {path}")
    # A Git LFS pointer is only a few bytes and cannot be loaded by PyTorch.
    if path.stat().st_size < 1024 and path.read_bytes().startswith(b"version https://git-lfs"):
        raise RuntimeError("Git LFS chưa tải nội dung model_bundle.pt trên máy chủ")
    return GesturePredictor(path, device="cpu")


def reset_prediction() -> None:
    st.session_state.pop("prediction", None)
    st.session_state.pop("prediction_digest", None)


def select_uploaded_file() -> None:
    st.session_state["use_sample"] = False
    reset_prediction()


def format_label(label: str) -> str:
    return label.replace("_", " ")


def probability_rows(probabilities: dict[str, float]) -> str:
    rows = []
    for label, value in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        safe_label = html.escape(format_label(label))
        percentage = max(0.0, min(100.0, float(value) * 100.0))
        rows.append(
            '<div class="prob-row">'
            f'<span title="{safe_label}">{safe_label}</span>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{percentage:.2f}%"></div></div>'
            f'<strong>{percentage:.1f}%</strong></div>'
        )
    return "".join(rows)


st.markdown(
    '<div class="hero"><h1>TF-C Prediction</h1>'
    '<p>Nhận dạng 8 cử chỉ bằng TF-C + DINOv3</p></div>',
    unsafe_allow_html=True,
)

try:
    predictor = load_predictor(str(MODEL_PATH))
    model_error = None
except Exception as error:  # show a useful deployment error instead of crashing the page
    predictor = None
    model_error = str(error)

required_samples = int(predictor.metadata["window_size"]) if predictor else 206
left, middle, right = st.columns([0.92, 1.18, 0.96], gap="medium")

with left:
    with st.container(border=True):
        st.markdown('<div class="section-title input-title">Input Data</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Chọn file dữ liệu",
            type=["csv", "txt"],
            help="File cần có các cột timestamp, x, y, z. Timestamp có thể bỏ qua khi dự đoán.",
            key="sensor_file",
            on_change=select_uploaded_file,
        )
        sample_col, download_col = st.columns(2)
        if sample_col.button("Dùng CSV mẫu", use_container_width=True):
            st.session_state["use_sample"] = True
            reset_prediction()
        if SAMPLE_PATH.is_file():
            download_col.download_button(
                "Tải CSV mẫu",
                data=SAMPLE_PATH.read_bytes(),
                file_name="gesture_example.csv",
                mime="text/csv",
                use_container_width=True,
            )
        st.markdown(
            '<div class="note"><strong>Định dạng yêu cầu</strong>'
            '<code>timestamp, x, y, z</code>'
            f'<span>Tối thiểu {required_samples} mẫu cảm biến.</span></div>',
            unsafe_allow_html=True,
        )

        file_bytes = None
        file_name = None
        if st.session_state.get("use_sample") and SAMPLE_PATH.is_file():
            file_bytes = SAMPLE_PATH.read_bytes()
            file_name = SAMPLE_PATH.name
            st.success("Đang sử dụng CSV mẫu InGesture.")
        elif uploaded is not None:
            file_bytes = uploaded.getvalue()
            file_name = uploaded.name
            st.caption(f"Đã chọn: {file_name} · {len(file_bytes) / 1024:.1f} KB")

xyz = None
parse_error = None
if file_bytes is not None:
    try:
        xyz = parse_xyz_bytes(file_bytes)
    except ValueError as error:
        parse_error = str(error)

with middle:
    with st.container(border=True):
        st.markdown('<div class="section-title preview-title">Signal Preview</div>', unsafe_allow_html=True)
        if parse_error:
            st.error(parse_error)
        elif xyz is None:
            st.info("Hãy tải file CSV/TXT hoặc chọn CSV mẫu để bắt đầu.")
        else:
            display = xyz[-min(len(xyz), 500):]
            time_frame = pd.DataFrame(display, columns=["X", "Y", "Z"])
            st.markdown("#### Time Series (Time Domain)")
            st.line_chart(time_frame, color=[COLORS["X"], COLORS["Y"], COLORS["Z"]], height=290)

            spectrum = np.abs(np.fft.rfft(display[:, 0]))[:100]
            st.markdown("#### Frequency Spectrum")
            st.caption("FFT magnitude, derived internally from trục X")
            st.bar_chart(pd.DataFrame({"Magnitude": spectrum}), color="#81a8e7", height=230)

            digest = hashlib.sha256(file_bytes).hexdigest()
            if len(xyz) < required_samples:
                st.error(f"Cần ít nhất {required_samples} mẫu, file hiện có {len(xyz)}.")
            elif predictor is None:
                st.error("Model chưa sẵn sàng nên không thể dự đoán.")
            elif st.button("▶ Run Prediction", type="primary", use_container_width=True):
                with st.spinner("DINOv3 đang phân tích time và frequency views…"):
                    st.session_state["prediction"] = predictor.predict(xyz)
                    st.session_state["prediction_digest"] = digest
            st.caption(f"Đã đọc {len(xyz):,} mẫu từ {file_name}.")

with right:
    with st.container(border=True):
        st.markdown('<div class="section-title result-title">Prediction Result</div>', unsafe_allow_html=True)
        current_digest = hashlib.sha256(file_bytes).hexdigest() if file_bytes is not None else None
        result = st.session_state.get("prediction")
        if st.session_state.get("prediction_digest") != current_digest:
            result = None

        if result:
            confidence = float(result["confidence"]) * 100.0
            st.markdown(
                '<div class="result-card"><strong>Predicted</strong>'
                f'<div class="predicted">{html.escape(format_label(result["label"]))}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="result-card"><strong>Confidence</strong>'
                f'<div class="confidence">{confidence:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, max(0.0, float(result["confidence"]))))
            st.markdown("#### Class Probabilities")
            st.markdown(probability_rows(result["probabilities"]), unsafe_allow_html=True)
            st.caption(f"Dự đoán trung bình trên {result['windows_used']} cửa sổ tín hiệu.")
        else:
            st.markdown(
                '<div class="result-card"><strong>Predicted</strong>'
                '<div class="predicted">—</div></div>',
                unsafe_allow_html=True,
            )
            st.info("Kết quả sẽ xuất hiện sau khi model chạy.")

        if predictor:
            metrics = predictor.metrics
            st.markdown(
                '<div class="model-state"><span class="status-ok">● Model đã sẵn sàng</span><br>'
                'Baseline TF-C + DINOv3 ViT-S/16 học từ đầu trên InGesture; '
                f'accuracy {metrics.get("test_accuracy", 0):.1%}, '
                f'macro F1 {metrics.get("macro_f1", 0):.1%}. '
                'Chưa dùng trọng số pretrained.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="model-state"><span class="status-warn">● Chưa nạp được model</span><br>'
                f'{html.escape(model_error or "Không rõ lỗi")}</div>',
                unsafe_allow_html=True,
            )

st.caption(
    "Dự đoán được tính từ model thật. Confidence là xác suất của model, không phải độ chính xác kiểm thử. "
    "Baseline InGesture sử dụng gia tốc đơn vị g và tần số khoảng 200 Hz."
)
