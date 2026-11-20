"""[v3.23.776] Cỡ lô engine TensorRT trong Cài đặt + gợi ý theo VRAM.

Chủ dự án: "TensorRT nhận lô sao không đưa vô cài đặt để người dùng chọn và có kèm gợi ý cho người dùng". Bản 3.23.775
đặt cứng bộ dò 16 / bộ nhận dạng 32. Nay: hai ô trong Cài đặt → Phần cứng (chỉ bấm được khi bật TensorRT), nút "Gợi ý
theo GPU máy này" (đọc VRAM bằng ``nvidia-smi``, điền số, CHƯA lưu), và số đã lưu đi tới tận hồ sơ engine — kể cả tiến
trình tự kiểm (biến môi trường) và khoá kết quả tự kiểm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from subtitles_extractor.infrastructure.ocr import tensorrt_runtime_plan as plan
from subtitles_extractor.infrastructure.ocr import tensorrt_selftest as selftest

DATA = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data"
_ENV = (plan.TRT_DET_BATCH_ENV, plan.TRT_REC_BATCH_ENV, plan.TRT_REC_OPT_ENV)   # tests/conftest.py dọn sau mỗi test


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


# ── gợi ý + hồ sơ ───────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(("vram", "expected"), [
    (None, (16, 32)), (4.0, (4, 8)), (6.0, (8, 16)), (8.0, (16, 32)), (7.6, (16, 32)), (12.0, (32, 64)),
    (16.0, (64, 128)), (24.0, (64, 128)),
])
def test_suggestion_by_vram(vram: float | None, expected: tuple[int, int]) -> None:
    det, rec = plan.suggest_trt_batches(vram)
    assert (det, rec) == expected
    assert plan.TRT_DET_BATCH_RANGE[0] <= det <= plan.TRT_DET_BATCH_RANGE[1]
    assert plan.TRT_REC_BATCH_RANGE[0] <= rec <= plan.TRT_REC_BATCH_RANGE[1]


def test_defaults_match_v323775_engines() -> None:
    import os

    assert not any(name in os.environ for name in _ENV)               # test trước không để lại cỡ lô
    assert plan.trt_batches() == (16, 32, 16)
    assert plan.det_profile() == {"x": [[1, 3, 32, 32], [16, 3, 288, 1600], [16, 3, 512, 2048]]}


def test_chosen_batches_reach_both_profiles_and_are_clamped() -> None:
    assert plan.set_trt_batches(8, 64, 16) == (8, 64, 16)
    assert plan.det_profile()["x"][1:] == [[8, 3, 288, 1600], [8, 3, 512, 2048]]
    rec = plan.rec_profile({"x": [[1, 3, 48, 160], [1, 3, 48, 320], [8, 3, 48, 3200]]})
    assert rec == {"x": [[1, 3, 48, 160], [16, 3, 48, 320], [64, 3, 48, 3200]]}
    assert plan.set_trt_batches(500, 1, 999) == (64, 8, 8)          # kẹp vào giới hạn; tối ưu ≤ tối đa


def test_selftest_result_is_keyed_by_engine_batches() -> None:
    kwargs = {"device": "gpu", "use_tensorrt": True, "precision": "fp32"}
    plan.set_trt_batches(16, 32, 16)
    first = selftest.selftest_signature("v", kwargs, 64)
    plan.set_trt_batches(32, 64, 16)
    assert selftest.selftest_signature("v", kwargs, 64) != first   # đổi cỡ lô → tự kiểm lại engine mới


# ── cài đặt → cấu hình OCR → bộ chuyển đổi ─────────────────────────────────────────────────────────────────────────
def test_settings_defaults_bounds_and_strings() -> None:
    from pydantic import ValidationError

    from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings

    hardware = ApplicationSettings().hardware
    assert (hardware.trt_det_batch, hardware.trt_rec_batch) == (16, 32)
    for bad in ({"trt_det_batch": 0}, {"trt_det_batch": 65}, {"trt_rec_batch": 4}, {"trt_rec_batch": 129}):
        with pytest.raises(ValidationError):
            ApplicationSettings.model_validate({"hardware": bad})
    for lang in ("vi", "en"):
        strings = json.loads((DATA / f"strings_{lang}.json").read_text(encoding="utf-8"))["settings"]
        for key in ("rl_trt_det_batch", "rl_trt_rec_batch", "tip_trt_det_batch", "tip_trt_rec_batch",
                    "trt_batch_hint", "trt_batch_suggest_btn", "trt_batch_suggested", "trt_batch_no_gpu"):
            assert strings[key]
        strings["trt_batch_suggested"].format(vram=8.0, det=16, rec=32)
        strings["trt_batch_no_gpu"].format(det=16, rec=32)


def test_every_ocr_config_builder_passes_the_batches() -> None:
    root = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor"
    for relative in ("composition/container.py", "presentation/view_models/extract_page_view_model.py",
                     "presentation/view_models/editor_page_view_model.py",
                     "presentation/view_models/debug_page_view_model.py"):
        source = (root / relative).read_text(encoding="utf-8")
        assert source.count("use_tensorrt=") == source.count("trt_det_batch=") == source.count("trt_rec_batch="), relative


def test_adapter_applies_batches_before_selftest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.infrastructure.ocr import paddle_ocr_adapter as module

    seen: list[tuple[int, int, int]] = []
    monkeypatch.setattr(module, "_prepare_paddle_tensorrt", lambda: tmp_path)
    monkeypatch.setattr(module, "_tensorrt_thread_ok", lambda: True)
    monkeypatch.setattr(module, "_tensorrt_crashed_last_time", lambda trt_dir: False)
    monkeypatch.setattr(module, "_tensorrt_selftest", lambda *a, **k: seen.append(plan.trt_batches())
                        or selftest.SelfTestOutcome(False, "giả lập"))
    adapter = module.PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=True, trt_det_batch=8, trt_rec_batch=64,
                                                      batch_size=24))
    adapter._try_create_model_with_retry = lambda cls, kwargs: "model"  # type: ignore[method-assign]
    adapter._create_model_with_device_fallback(object, use_gpu=True)
    assert seen == [(8, 64, 24)]                                    # tối ưu bộ nhận dạng = lô ứng dụng đưa vào


def test_changing_engine_batches_reloads_only_with_tensorrt() -> None:
    from dataclasses import replace

    from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    base = OcrEngineConfig(use_tensorrt=True)
    adapter = PaddleOcrAdapter(base)
    assert adapter._needs_model_reload(replace(base, trt_det_batch=32))
    assert not adapter._needs_model_reload(replace(base))
    plain = PaddleOcrAdapter(OcrEngineConfig(use_tensorrt=False))
    assert not plain._needs_model_reload(OcrEngineConfig(use_tensorrt=False, trt_det_batch=32))


def test_engine_batches_do_not_invalidate_ocr_checkpoints() -> None:
    from subtitles_extractor.infrastructure.serializers import ocr_checkpoint

    assert {"trt_det_batch", "trt_rec_batch"} <= set(ocr_checkpoint._SPEED_ONLY_OCR_FIELDS)


# ── trang Cài đặt ───────────────────────────────────────────────────────────────────────────────────────────────────
def _container(tmp_path: Path):  # noqa: ANN202
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository

    return ApplicationContainer(settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
                                i18n_data_dir=DATA, user_data_dir=tmp_path)


def test_settings_page_suggests_without_saving_then_saves(app: QApplication, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.infrastructure.tts import vieneu_batch_tuning
    from subtitles_extractor.presentation.pages.settings_page import SettingsPage

    container = _container(tmp_path)
    page = SettingsPage(container)
    try:
        assert (page._sp_trt_det_batch.value(), page._sp_trt_rec_batch.value()) == (16, 32)
        assert not page._sp_trt_det_batch.isEnabled() and not page._btn_trt_batch_suggest.isEnabled()  # TensorRT tắt
        assert "8 GB" in page._lbl_trt_batch_hint.text()
        page._chk_tensorrt.setChecked(True)
        assert page._sp_trt_det_batch.isEnabled() and page._btn_trt_batch_suggest.isEnabled()
        monkeypatch.setattr(vieneu_batch_tuning, "detect_total_vram_gb", lambda: 12.0)
        page._btn_trt_batch_suggest.click()
        assert (page._sp_trt_det_batch.value(), page._sp_trt_rec_batch.value()) == (32, 64)
        assert "12 GB" in page._lbl_trt_batch_hint.text()
        assert container.settings_service.current.hardware.trt_det_batch == 16            # gợi ý chưa lưu
        assert page._save_button.isEnabled()
        page._on_save_clicked()
        hardware = container.settings_service.current.hardware
        assert (hardware.trt_det_batch, hardware.trt_rec_batch, hardware.use_tensorrt) == (32, 64, True)
        monkeypatch.setattr(vieneu_batch_tuning, "detect_total_vram_gb", lambda: None)
        page._btn_trt_batch_suggest.click()
        assert (page._sp_trt_det_batch.value(), page._sp_trt_rec_batch.value()) == (16, 32)
        assert "nvidia-smi" in page._lbl_trt_batch_hint.text()
    finally:
        page.close()
