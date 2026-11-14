FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_BUNDLE_PATH=/app/artifacts/gesture_model/model_bundle.pt \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Keep dependencies in a reusable layer when only code or model changes.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

RUN useradd --create-home --uid 10001 appuser

# Production inference files only; training datasets stay outside the image.
COPY --chown=appuser:appuser streamlit_app.py ./
COPY --chown=appuser:appuser run_web.py ./
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser code/TFC ./code/TFC
COPY --chown=appuser:appuser sample_data ./sample_data
COPY --chown=appuser:appuser artifacts/gesture_model ./artifacts/gesture_model

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/live', timeout=4)" || exit 1

CMD ["python", "run_web.py"]
