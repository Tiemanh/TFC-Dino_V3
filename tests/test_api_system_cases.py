"""Ten functional cases covering the frontend, API, AI contract, and mocks."""

from __future__ import annotations

import asyncio
import io
import sys
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from app.main import app

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def sensor_csv(rows: int = 206) -> bytes:
    output = io.StringIO()
    output.write("timestamp,x,y,z\n")
    for index in range(rows):
        output.write(f"{index},{index / 1000:.4f},{index / 900:.4f},{1 + index / 800:.4f}\n")
    return output.getvalue().encode("utf-8")


class FakePredictor:
    metadata = {"window_size": 206}
    class_names = [f"gesture_{index}" for index in range(8)]
    training_info = {"initialization": "pretrained_test_double"}
    metrics = {"test_accuracy": 0.5, "macro_f1": 0.2}

    def predict(self, xyz):
        assert isinstance(xyz, np.ndarray)
        probabilities = {name: 0.05 for name in self.class_names}
        probabilities["gesture_0"] = 0.65
        return {
            "label": "gesture_0",
            "confidence": 0.65,
            "probabilities": probabilities,
            "windows_used": 1,
        }


class ApiSystemCases(unittest.IsolatedAsyncioTestCase):
    """Exactly ten cases required by the functional test specification."""

    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_01_frontend_is_served(self):
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("TF-C Prediction", response.text)

    async def test_02_frontend_assets_are_served(self):
        self.assertEqual((await self.client.get("/style.css")).status_code, 200)
        script = await self.client.get("/app.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn("/api/predict", script.text)

    @patch("app.main.get_predictor", return_value=FakePredictor())
    async def test_03_health_reports_ready_model(self, _mock):
        live = await self.client.get("/api/live")
        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json(), {"status": "ok"})
        response = await self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ready"])
        self.assertEqual(response.json()["required_samples"], 206)

    @patch("app.main.get_predictor", return_value=None)
    async def test_04_health_reports_unavailable_model(self, _mock):
        response = await self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ready"])

    async def test_05_sample_csv_is_downloadable(self):
        response = await self.client.get("/sample-data/gesture_example.csv")
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.content), 100)

    async def test_06_preview_returns_time_and_frequency_data(self):
        response = await self.client.post(
            "/api/preview", files={"file": ("gesture.csv", sensor_csv(), "text/csv")}
        )
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["sample_count"], 206)
        self.assertEqual(set(payload["time"]), {"x", "y", "z"})
        self.assertGreater(len(payload["frequency"]), 0)

    async def test_07_preview_rejects_wrong_extension(self):
        response = await self.client.post(
            "/api/preview", files={"file": ("gesture.exe", sensor_csv(), "application/octet-stream")}
        )
        self.assertEqual(response.status_code, 400)

    @patch("app.main.get_predictor", return_value=FakePredictor())
    async def test_08_predict_returns_ai_contract(self, _mock):
        response = await self.client.post(
            "/api/predict", files={"file": ("gesture.csv", sensor_csv(), "text/csv")}
        )
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn(payload["label"], FakePredictor.class_names)
        self.assertEqual(len(payload["probabilities"]), 8)
        self.assertAlmostEqual(sum(payload["probabilities"].values()), 1.0)

    @patch("app.main.get_predictor", return_value=FakePredictor())
    async def test_09_predict_rejects_short_recording(self, _mock):
        response = await self.client.post(
            "/api/predict", files={"file": ("short.csv", sensor_csv(205), "text/csv")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("206", response.json()["detail"])

    @patch("app.main.get_predictor", return_value=None)
    async def test_10_predict_returns_503_without_model(self, _mock):
        response = await self.client.post(
            "/api/predict", files={"file": ("gesture.csv", sensor_csv(), "text/csv")}
        )
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
