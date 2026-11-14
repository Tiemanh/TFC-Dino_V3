"""Ten defensive attack simulations against the owned inference API."""

from __future__ import annotations

import asyncio
import io
import sys
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from app.main import app
from tests.test_api_system_cases import FakePredictor, sensor_csv

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def custom_csv(x_values, y_values=None, z_values=None) -> bytes:
    y_values = x_values if y_values is None else y_values
    z_values = x_values if z_values is None else z_values
    output = io.StringIO()
    output.write("timestamp,x,y,z\n")
    for index, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
        output.write(f"{index},{x},{y},{z}\n")
    return output.getvalue().encode("utf-8")


class RecordingPredictor(FakePredictor):
    def __init__(self):
        self.last_input = None

    def predict(self, xyz):
        self.last_input = np.array(xyz, copy=True)
        return super().predict(xyz)


class AiAttackCases(unittest.IsolatedAsyncioTestCase):
    """Exactly ten distinct input and model attack categories."""

    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_attack_01_missing_multipart_file(self):
        response = await self.client.post("/api/predict")
        self.assertEqual(response.status_code, 400)

    async def test_attack_02_extension_spoofing(self):
        response = await self.client.post(
            "/api/preview", files={"file": ("payload.exe", sensor_csv(), "text/csv")}
        )
        self.assertEqual(response.status_code, 400)

    async def test_attack_03_oversized_upload_dos(self):
        payload = b"x" * (5 * 1024 * 1024 + 1)
        response = await self.client.post(
            "/api/preview", files={"file": ("large.csv", payload, "text/csv")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("5 MB", response.json()["detail"])

    async def test_attack_04_binary_content_disguised_as_csv(self):
        response = await self.client.post(
            "/api/preview", files={"file": ("fake.csv", b"\x00\xff\x00\xff", "text/csv")}
        )
        self.assertEqual(response.status_code, 400)

    async def test_attack_05_schema_manipulation_missing_axis(self):
        response = await self.client.post(
            "/api/preview", files={"file": ("missing.csv", b"x,y\n1,2\n", "text/csv")}
        )
        self.assertEqual(response.status_code, 400)

    async def test_attack_06_csv_formula_injection_in_sensor_axis(self):
        formulas = ["=1+1"] * 206
        response = await self.client.post(
            "/api/preview", files={"file": ("formula.csv", custom_csv(formulas), "text/csv")}
        )
        self.assertEqual(response.status_code, 400)

    async def test_attack_07_numeric_overflow_and_infinity(self):
        overflow = ["1e309"] * 206
        response = await self.client.post(
            "/api/preview", files={"file": ("overflow.csv", custom_csv(overflow), "text/csv")}
        )
        self.assertEqual(response.status_code, 400)

    @patch("app.main.get_predictor", return_value=FakePredictor())
    async def test_attack_08_truncated_window(self, _mock):
        response = await self.client.post(
            "/api/predict", files={"file": ("short.csv", sensor_csv(20), "text/csv")}
        )
        self.assertEqual(response.status_code, 400)

    async def test_attack_09_adversarial_noise_preserves_api_contract(self):
        rng = np.random.default_rng(42)
        noisy = rng.normal(0, 1000, size=206)
        predictor = RecordingPredictor()
        with patch("app.main.get_predictor", return_value=predictor):
            response = await self.client.post(
                "/api/predict", files={"file": ("noise.csv", custom_csv(noisy), "text/csv")}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(predictor.last_input.shape, (206, 3))
        self.assertEqual(len(response.json()["probabilities"]), 8)

    async def test_attack_10_axis_permutation_reaches_ai_monitoring(self):
        x = np.linspace(-1, 1, 206)
        y = np.linspace(10, 20, 206)
        z = np.linspace(100, 200, 206)
        predictor = RecordingPredictor()
        with patch("app.main.get_predictor", return_value=predictor):
            response = await self.client.post(
                "/api/predict",
                files={"file": ("swapped.csv", custom_csv(z, x, y), "text/csv")},
            )
        self.assertEqual(response.status_code, 200)
        np.testing.assert_allclose(predictor.last_input[:, 0], z, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
