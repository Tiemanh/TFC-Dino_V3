"""Run the 20 practice cases and write reproducible JSON/Markdown evidence.

The ten attack cases are defensive simulations against this owned project.
Heavy load and destructive tests are intentionally excluded from production.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "testing"


CASE_CATALOG = {
    "test_01_frontend_is_served": ("TC01", "Automation/System", "Giao diện chính hoạt động"),
    "test_02_frontend_assets_are_served": ("TC02", "Automation", "CSS và JavaScript được phục vụ"),
    "test_03_health_reports_ready_model": ("TC03", "API/Mock", "API báo model sẵn sàng"),
    "test_04_health_reports_unavailable_model": ("TC04", "API/Mock", "API xử lý khi thiếu model"),
    "test_05_sample_csv_is_downloadable": ("TC05", "System", "Tải được CSV mẫu"),
    "test_06_preview_returns_time_and_frequency_data": ("TC06", "API/AI data", "Preview time và frequency views"),
    "test_07_preview_rejects_wrong_extension": ("TC07", "API negative", "Từ chối phần mở rộng sai"),
    "test_08_predict_returns_ai_contract": ("TC08", "AI/Mock", "Đầu ra AI đủ tám lớp"),
    "test_09_predict_rejects_short_recording": ("TC09", "AI boundary", "Từ chối chuỗi dưới 206 mẫu"),
    "test_10_inference_failure_is_sanitized": ("TC10", "Mock/Security", "Ẩn lỗi nội bộ khi suy luận thất bại"),
    "test_attack_01_missing_multipart_file": ("AT01", "Malformed request", "Request không có file"),
    "test_attack_02_extension_spoofing": ("AT02", "Extension spoofing", "Giả mạo phần mở rộng"),
    "test_attack_03_oversized_upload_dos": ("AT03", "Bounded DoS", "Upload vượt giới hạn 5 MB"),
    "test_attack_04_binary_content_disguised_as_csv": ("AT04", "Content spoofing", "File nhị phân giả CSV"),
    "test_attack_05_schema_manipulation_missing_axis": ("AT05", "Schema manipulation", "Thiếu trục cảm biến"),
    "test_attack_06_csv_formula_injection_in_sensor_axis": ("AT06", "Formula injection", "Công thức trong dữ liệu cảm biến"),
    "test_attack_07_numeric_overflow_and_infinity": ("AT07", "Numeric overflow", "Giá trị vô hạn"),
    "test_attack_08_truncated_window": ("AT08", "Truncation", "Cắt ngắn cửa sổ tín hiệu"),
    "test_attack_09_adversarial_noise_preserves_api_contract": ("AT09", "Adversarial noise", "Nhiễu biên độ lớn"),
    "test_attack_10_axis_permutation_reaches_ai_monitoring": ("AT10", "Sensor manipulation", "Hoán đổi trục X/Y/Z"),
}


class EvidenceResult(unittest.TextTestResult):
    """Capture per-case duration and status while retaining normal test output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self._started = {}

    def startTest(self, test):
        self._started[test.id()] = time.perf_counter()
        super().startTest(test)

    def _record(self, test, status, detail=""):
        method = test.id().rsplit(".", 1)[-1]
        case_id, category, title = CASE_CATALOG.get(method, (method, "Other", method))
        duration = time.perf_counter() - self._started.get(test.id(), time.perf_counter())
        security_result = "RISK" if case_id in {"AT09", "AT10"} and status == "PASS" else status
        self.records.append(
            {
                "id": case_id,
                "method": method,
                "group": "attack" if case_id.startswith("AT") else "system_ai",
                "category": category,
                "title": title,
                "execution_status": status,
                "assessment": security_result,
                "duration_ms": round(duration * 1000, 2),
                "detail": detail,
            }
        )

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "PASS")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, "FAIL", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, "ERROR", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "BLOCKED", reason)


def build_suite():
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromName("tests.test_api_system_cases.ApiSystemCases"))
    suite.addTests(loader.loadTestsFromName("tests.test_ai_attack_cases.AiAttackCases"))
    return suite


def write_outputs(records, elapsed_seconds, successful):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: (item["group"], item["id"]))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "total": len(ordered),
        "execution_pass": sum(item["execution_status"] == "PASS" for item in ordered),
        "execution_fail": sum(item["execution_status"] in {"FAIL", "ERROR"} for item in ordered),
        "blocked": sum(item["execution_status"] == "BLOCKED" for item in ordered),
        "risks_observed": sum(item["assessment"] == "RISK" for item in ordered),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "successful": successful and len(ordered) == 20,
        "cases": ordered,
    }
    (OUTPUT_DIR / "test-results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Kết quả 20 test case TF-C + DINOv3",
        "",
        f"- Tổng số: **{summary['total']}**",
        f"- Test tự động đạt: **{summary['execution_pass']}**",
        f"- Test lỗi: **{summary['execution_fail']}**",
        f"- Bị chặn: **{summary['blocked']}**",
        f"- Rủi ro AI cần theo dõi: **{summary['risks_observed']}**",
        f"- Tổng thời gian: **{summary['elapsed_seconds']} giây**",
        "",
        "| Mã | Nhóm | Tên kiểm thử | Thực thi | Đánh giá | Thời gian (ms) |",
        "|---|---|---|---|---|---:|",
    ]
    for item in ordered:
        lines.append(
            f"| {item['id']} | {item['category']} | {item['title']} | "
            f"{item['execution_status']} | {item['assessment']} | {item['duration_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Diễn giải rủi ro",
            "",
            "AT09 và AT10 có thể thực thi ổn định nhưng vẫn được đánh dấu RISK: API hiện "
            "chấp nhận nhiễu biên độ lớn và tín hiệu bị hoán đổi trục mà chưa có bộ phát hiện "
            "dữ liệu ngoài phân phối. Đây là hạn chế của model, không phải lỗi chạy test.",
            "",
            "Các thử nghiệm tải nặng chỉ được phép chạy ở local hoặc môi trường kiểm thử riêng.",
        ]
    )
    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    started = time.perf_counter()
    runner = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult)
    result = runner.run(build_suite())
    elapsed = time.perf_counter() - started
    write_outputs(result.records, elapsed, result.wasSuccessful())
    if len(result.records) != 20:
        print(f"Expected 20 cases but captured {len(result.records)}", file=sys.stderr)
        return 2
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
