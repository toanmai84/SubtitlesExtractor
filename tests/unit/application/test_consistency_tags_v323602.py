# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.602] "consistency" bỏ qua tag [..]; trang Dịch truyền model hỏi lại.

Chuẩn đoán thật (v3.23.601, dòng 454/455 "酒逢知己"): "Rượu gặp tri kỷ..." và "[cười]
Rượu gặp tri kỷ" bị báo dịch không nhất quán — tag cảm xúc chỉ vào giọng đọc, bộ xuất
SRT/ASS đã lọc khỏi phụ đề. Đo lại file chuẩn đoán đó: 11 → 10 cờ, chỉ cặp này mất.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
)


def _event(index: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(index=index, text=text, start_sec=index, end_sec=index + 1.0)


def test_tag_cam_xuc_khong_tinh_la_khac() -> None:
    source = [_event(454, "酒逢知己"), _event(455, "酒逢知己")]
    translated = [
        _event(454, "Rượu gặp tri kỷ..."), _event(455, "[cười] Rượu gặp tri kỷ")
    ]
    flags = detect_quality_flags(source, translated)
    assert flags["consistency"] == []


def test_khac_tu_ngu_van_bao() -> None:
    source = [_event(918, "老婆，還是你棒"), _event(1136, "老婆，還是你棒")]
    translated = [
        _event(918, "Bà xã à, vẫn là em cừ nhất."),
        _event(1136, "Bà xã, vẫn là bà giỏi nhất."),
    ]
    flags = detect_quality_flags(source, translated)
    assert [item["indices"] for item in flags["consistency"]] == [[918, 1136]]


def test_trang_dich_truyen_model_phan_tich_cho_buoc_hoi_lai(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        TranslationContext,
        TranslationStageConfig,
        TranslationStageKind,
    )
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.view_models import translate_page_view_model

    QApplication.instance() or QApplication([])
    captured: dict[str, object] = {}

    class _Worker:
        def __init__(self, use_case, request, **_kwargs) -> None:
            captured["request"] = request
            signal = SimpleNamespace(connect=lambda *_a: None)
            self.progress_changed = self.finished_ok = self.cancelled = signal
            self.failed = self.finished = self.video_refs_ready = self.progress_changed

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(translate_page_view_model, "TranslateWorker", _Worker)
    # Không dùng bootstrap_for_gui: nó cài bộ chặn import torch cho CẢ tiến trình,
    # làm hỏng scipy ở các test chạy sau ('NoneType' has no attribute 'Tensor').
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=translate_page_view_model.Path(
            translate_page_view_model.__file__
        ).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    monkeypatch.setattr(
        container, "make_subtitle_translator",
        lambda *_a, **_k: SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setattr(
        container, "make_translate_subtitles_use_case", lambda *_a: object()
    )
    view_model = translate_page_view_model.TranslatePageViewModel(container)
    view_model._source_events = [
        SubtitleEvent(index=1, text="不要！", interval=TimeInterval(1.0, 2.0))
    ]
    stage = TranslationStageConfig(
        kind=TranslationStageKind.LITERAL, model_name="gemini-3.5-flash-lite"
    )
    assert view_model.start_translation(
        "key", 3, [stage], TranslationContext(target_lang="Vietnamese"),
        speaker_model=" gemini-3.8-flash ",
    )
    request = captured["request"]
    assert request.speaker_model == "gemini-3.8-flash"
    assert request.visual_cues_model == "gemini-3.5-flash-lite"


def test_trang_dich_lay_model_tu_o_phan_tich() -> None:
    from pathlib import Path

    source = Path(
        "src/subtitles_extractor/presentation/pages/translate_page.py"
    ).read_text(encoding="utf-8")
    assert "speaker_model=self._analyze_model_combo.currentText().strip()" in source
