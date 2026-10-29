"""[v3.23.657] Kịch bản "Bộ dò chữ v6 small" trong Đo tốc độ OCR + cài đặt bộ dò chữ riêng."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from subtitles_extractor.application.services.ocr_benchmark import (
    DEFAULT_KEYS,
    DEFAULT_SCENARIOS,
    apply_scenario,
    current_settings_values,
    is_redundant_scenario,
    recommend,
)
from subtitles_extractor.application.services.ocr_model_resolver import resolve_ocr_model_names
from subtitles_extractor.application.services.ocr_probe_window import compute_probe_window
from subtitles_extractor.infrastructure.settings.application_settings import OcrSettings
from tests.unit.application.test_ocr_benchmark_v323645 import EVENTS, _request, _result

SC = {s.key: s for s in DEFAULT_SCENARIOS}
WINDOW = compute_probe_window(600.0, probe_seconds=4.0)


def _base(tmp_path: Path, det: str = "PP-OCRv6_medium_det"):
    base = _request(tmp_path)
    return dataclasses.replace(base, ocr=dataclasses.replace(base.ocr, detection_model_name=det,
                                                             recognition_model_name="PP-OCRv6_medium_rec"))


def test_scenario_swaps_only_the_detector(tmp_path: Path) -> None:
    assert "det_small" in DEFAULT_KEYS
    request = apply_scenario(_base(tmp_path), SC["det_small"], WINDOW)
    assert request.ocr.detection_model_name == "PP-OCRv6_small_det"
    assert request.ocr.recognition_model_name == "PP-OCRv6_medium_rec"
    assert not is_redundant_scenario(_base(tmp_path), SC["det_small"], WINDOW)
    # Đang dùng sẵn bộ dò small (+ tắt hướng chữ) → đo lại vô nghĩa.
    small = _base(tmp_path, "PP-OCRv6_small_det")
    small = dataclasses.replace(small, ocr=dataclasses.replace(small.ocr, use_textline_orientation=False))
    assert is_redundant_scenario(small, SC["det_small"], WINDOW)


def test_recommendation_writes_detector_override(tmp_path: Path) -> None:
    base_values = current_settings_values(_base(tmp_path))
    rec = recommend([_result("current", 100, EVENTS), _result("det_small", 60, EVENTS)], base_values)
    assert rec is not None and rec.key == "det_small"
    assert rec.settings_patch["ocr"]["detection_model_override"] == "PP-OCRv6_small_det"
    # Bộ dò small cho kết quả khác → không đề xuất.
    other = [*EVENTS[:2], (5.0, 6.0, "bà")]
    rec = recommend([_result("current", 100, EVENTS), _result("det_small", 60, other)], base_values)
    assert rec is None or rec.key != "det_small"


def test_resolver_honours_detector_override() -> None:
    # [v3.23.661] Mặc định nay là small → kiểm cơ chế ghi đè trên phiên bản medium chọn tường minh.
    medium = dict(version="PP-OCRv6_medium", detection_model="PP-OCRv6_medium_det", recognition_model="PP-OCRv6_medium_rec")
    assert resolve_ocr_model_names(OcrSettings(**medium)) == ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec")
    mixed = OcrSettings(**medium, detection_model_override="PP-OCRv6_small_det")
    assert resolve_ocr_model_names(mixed) == ("PP-OCRv6_small_det", "PP-OCRv6_medium_rec")
    assert resolve_ocr_model_names(OcrSettings(**medium, detection_model_override="  ")) == (
        "PP-OCRv6_medium_det", "PP-OCRv6_medium_rec",
    )


def test_reocr_same_version_keeps_detector_override(monkeypatch) -> None:
    """Rà soát độc lập: Re-OCR (Nâng cao) nhớ phiên bản = phiên bản Cài đặt → vẫn phải dùng bộ dò chữ đã chọn,
    không thì engine nạp lại model mỗi lần đổi qua lại giữa trích xuất và Re-OCR."""
    from subtitles_extractor.application.services.ocr_model_resolver import models_for_reocr

    settings = OcrSettings(version="PP-OCRv6_medium", detection_model_override="PP-OCRv6_small_det")
    assert models_for_reocr(settings, "PP-OCRv6_medium") == ("PP-OCRv6_small_det", "PP-OCRv6_medium_rec")
    assert models_for_reocr(settings, None) == ("PP-OCRv6_small_det", "PP-OCRv6_medium_rec")
    # Chọn hẳn phiên bản khác → bộ mặc định của phiên bản đó.
    assert models_for_reocr(settings, "PP-OCRv5_server") == ("PP-OCRv5_server_det", "PP-OCRv5_server_rec")
