FROM python:3.11-slim
WORKDIR /app
COPY requirements-dinov3.txt requirements-web.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-web.txt
COPY . .
ENV MODEL_BUNDLE_PATH=/app/artifacts/gesture_model/model_bundle.pt
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
