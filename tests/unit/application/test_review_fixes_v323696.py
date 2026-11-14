"""[v3.23.696] Lỗi tìm được khi rà độc lập các bản 3.23.683–694 (mỗi ca đều thất bại trên mã 3.23.695)."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.dubbing_cast import match_saved_casting
from subtitles_extractor.application.services.subtitle_pipeline.box_geometry import drift_ratio, drop_drifting_events
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import restore_bound_words
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.video.bbox_analyzer import RawBBox
from subtitles_extractor.infrastructure.video.roi_selection import refit_series_roi

STEP = 0.12


# ── 1. Sửa chữ "dính liền" không được đụng tên riêng / chữ thường gặp ─────────────────────────────────────────────


@pytest.mark.parametrize("text", ["沈清漪", "婉漪姑娘", "清漪园", "麒儿快走", "琥儿", "玫姐", "他很尴了"])
def test_bound_words_leave_names_and_common_chars(text: str) -> None:
    assert restore_bound_words(text) == text


def test_bound_words_still_fix_real_misreads() -> None:
    assert restore_bound_words("凯觎我的女人") == "觊觎我的女人"
    assert restore_bound_words("忐志不安") == "忐忑不安"


# ── 2. Bộ lọc chữ trôi: hộp phình một phía / lệch một khung không phải "trôi" ─────────────────────────────────────


def _frame(i: int, x0: int, x1: int, text: str = "哦") -> OcrFrameResult:
    box = OcrTextBox(text=text, confidence=Confidence(0.98), polygon=[(x0, 20), (x1, 20), (x1, 88), (x0, 88)])
    return OcrFrameResult(frame_index=i, timestamp_sec=10 + i * STEP, text_boxes=[box])


def _event(n: int) -> SubtitleEvent:
    return SubtitleEvent(index=1, text="哦", interval=TimeInterval(10.0, 10 + (n - 1) * STEP),
                         confidence=Confidence(1.0), frame_count=n)


@pytest.mark.parametrize("n", [3, 4, 6])
def test_one_box_swelling_at_the_end_is_not_drift(n: int) -> None:
    frames = [_frame(i, 268, 336) for i in range(n - 1)] + [_frame(n - 1, 268, 477)]   # hộp phình sang phải 1 khung
    assert drop_drifting_events([_event(n)], frames, STEP)[1] == 0


def test_single_frame_jump_is_not_drift() -> None:
    frames = [_frame(i, 268, 336) for i in range(4)] + [_frame(4, 296, 364)]          # lệch 28 px đúng một khung
    assert drop_drifting_events([_event(5)], frames, STEP)[1] == 0


def test_steady_drift_of_both_edges_is_still_dropped() -> None:
    frames = [_frame(i, 293 + 8 * i, 333 + 8 * i, "00") for i in range(5)]           # cả hai mép trôi đều 8 px/khung
    event = SubtitleEvent(index=1, text="00", interval=TimeInterval(10.0, 10.48), confidence=Confidence(1.0),
                          frame_count=5)
    assert drop_drifting_events([event], frames, STEP)[1] == 1
    assert drift_ratio([(10 + i * STEP, 313.0 + 8 * i, 40.0) for i in range(5)]) > 0.4


# ── 5. Giọng đã lưu không khớp chéo hai vai khác chú thích ────────────────────────────────────────────────────────


def test_saved_voice_does_not_cross_different_annotations() -> None:
    assert match_saved_casting(["师父(张三)"], {"师父(李四)": "voiceA"}) == {}
    # Một phía không có chú thích → vẫn khớp như 3.23.687.
    assert match_saved_casting(["叶然（师兄）"], {"叶然": "nam_1"}) == {"叶然（师兄）": "nam_1"}
    assert match_saved_casting(["楚瑶 / 瑶瑶"], {"瑶瑶(女主)": "nu_2"}) == {"楚瑶 / 瑶瑶": "nu_2"}


# ── 6. ROI phim bộ: không nới theo chữ nằm ngoài vùng (logo, chữ cảnh) ────────────────────────────────────────────


def _bbox(frame: int, x0: float, x1: float) -> RawBBox:
    return RawBBox(coord_x_min=x0, coord_y_min=838, coord_x_max=x1, coord_y_max=906, confidence=0.97,
                   frame_idx=frame, timestamp_sec=frame * 5.0)


def test_series_roi_ignores_text_beside_the_roi() -> None:
    roi = Roi(x=110, y=818, width=500, height=121)
    boxes = [_bbox(i, 300, 420) for i in range(20)] + [_bbox(i, 640, 715) for i in range(20)]   # logo góc phải
    assert refit_series_roi(roi, boxes, 720, 1280) == roi
    longer = [_bbox(i, 300, 420) for i in range(20)] + [_bbox(3, 70, 650)]                       # câu dài hơn ROI
    out = refit_series_roi(roi, longer, 720, 1280)
    assert out is not None and (out.x, out.x + out.width) == (70, 650)


# ── 3. Dịch có đính video: model chính hết hạn mức ngày mà còn model dự phòng → không dừng ─────────────────────────


def test_translate_worker_uses_daily_fallback_instead_of_failing(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from tests.unit.presentation import test_cached_rerun_skips_video_v323609 as harness

    QApplication.instance() or QApplication([])
    calls: list[int] = []
    use_case = harness._use_case(tmp_path, calls)
    use_case.has_any_daily_quota = lambda model: model == "gemini-3.1-flash-lite"   # type: ignore[method-assign]
    use_case.daily_quota_fallback = lambda model: "gemini-3.1-flash-lite"            # type: ignore[attr-defined]
    keys_for: list[str] = []
    use_case.ensure_viable_key = lambda model, **_k: keys_for.append(model)          # type: ignore[method-assign]
    provider = harness._VideoProvider()
    finished, failed = harness._run_worker(use_case, provider, harness._request())
    assert failed == []
    assert provider.prepared == 1 and keys_for == ["gemini-3.1-flash-lite"]          # video tải dưới key của dự phòng


def test_translate_worker_without_fallback_still_fails_fast(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from tests.unit.presentation import test_cached_rerun_skips_video_v323609 as harness

    QApplication.instance() or QApplication([])
    use_case = harness._use_case(tmp_path, [], quota=False)
    use_case.daily_quota_fallback = lambda model: None                               # type: ignore[attr-defined]
    provider = harness._VideoProvider()
    _finished, failed = harness._run_worker(use_case, provider, harness._request())
    assert failed and "hết quota ngày" in failed[0] and provider.prepared == 0


def test_use_case_asks_adapter_for_daily_fallback() -> None:
    from subtitles_extractor.application.use_cases.translate_subtitles import TranslateSubtitlesUseCase

    class _Adapter:
        def daily_quota_fallback(self, model: str) -> str | None:
            return "gemini-3.1-flash-lite" if model == "gemini-3.8-flash" else None

    use_case = TranslateSubtitlesUseCase.__new__(TranslateSubtitlesUseCase)
    use_case._translator = _Adapter()
    assert use_case.daily_quota_fallback("gemini-3.8-flash") == "gemini-3.1-flash-lite"
    use_case._translator = object()                                                  # adapter không hỗ trợ
    assert use_case.daily_quota_fallback("gemini-3.8-flash") is None


# ── 3b/4. Adapter thật: hỏi trước model dự phòng; số lượt thử của model dự phòng không dư ─────────────────────────


def _real_translator(monkeypatch, tmp_path, server, keys="KEY_A\nKEY_B"):
    from subtitles_extractor.infrastructure.translation.gemini_quota_manager import GeminiQuotaManager
    from subtitles_extractor.infrastructure.translation.model_health import ModelHealth
    from tests.unit.infrastructure.gemini_call_harness import (
        SCARCE_MODEL,
        build_translator,
        confirmed_scarce_manager,
        install_fake_clock,
    )

    clock = install_fake_clock(monkeypatch)
    manager = confirmed_scarce_manager(tmp_path)
    for key in keys.split("\n"):
        manager.mark_daily_exhausted(SCARCE_MODEL, key_id=GeminiQuotaManager.key_fingerprint(key))
    translator, _slept = build_translator(monkeypatch, clock, manager, server, keys=keys)
    translator._model_health = ModelHealth(seed_models=(SCARCE_MODEL, "gemini-3.1-flash-lite"))
    return translator


def test_adapter_reports_daily_fallback(monkeypatch, tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.translation.model_health import NO_FALLBACK
    from tests.unit.infrastructure.gemini_call_harness import OK, SCARCE_MODEL, ScriptedGemini

    translator = _real_translator(monkeypatch, tmp_path, ScriptedGemini([OK]))
    assert translator.has_any_daily_quota(SCARCE_MODEL) is False
    assert translator.daily_quota_fallback(SCARCE_MODEL) == "gemini-3.1-flash-lite"
    translator._model_health.set_preferred_fallback(NO_FALLBACK)
    assert translator.daily_quota_fallback(SCARCE_MODEL) is None


def test_fallback_after_reservation_switch_gets_the_normal_number_of_tries(monkeypatch, tmp_path: Path) -> None:
    """Model chính cạn theo bộ đếm → chuyển dự phòng ngay ở bước đặt chỗ; dự phòng phải được đúng ``retry_count`` lượt
    như khi gọi thẳng nó (3.23.690 cho dư một lượt: đếm lượt bắt đầu từ -1)."""
    from tests.unit.infrastructure.gemini_call_harness import (
        SCARCE_MODEL,
        ScriptedGemini,
        build_translator,
        call_gemini,
        install_fake_clock,
        quota_manager,
    )

    broken = "500 INTERNAL. {'error': {'code': 500, 'message': 'Internal error encountered.', 'status': 'INTERNAL'}}"
    direct = ScriptedGemini([broken])
    clock = install_fake_clock(monkeypatch)
    plain, _slept = build_translator(monkeypatch, clock, quota_manager(tmp_path / "d"), direct, keys="KEY_A")
    with pytest.raises(Exception):
        call_gemini(plain, "gemini-3.1-flash-lite")
    via_fallback = ScriptedGemini([broken])
    translator = _real_translator(monkeypatch, tmp_path, via_fallback, keys="KEY_A")
    with pytest.raises(Exception):
        call_gemini(translator, SCARCE_MODEL)
    assert via_fallback.calls == direct.calls


# ── Dải ranh giới: chỉ cộng khung trống LIỀN với chuỗi trống ──────────────────────────────────────────────────────


def _track(probe_pixels: list[int]):
    from subtitles_extractor.domain.value_objects.boundary_track import BoundaryProbe, BoundaryTrack

    probes = tuple(BoundaryProbe(timestamp_sec=t, text_pixels=px, sim_prev=0.0, sim_next=0.0)
                   for t, px in zip((0.04, 0.08, 0.16, 0.20), probe_pixels, strict=True))
    return BoundaryTrack(sample_times=(0.0, 0.12, 0.24), sample_pixels=(300, 0, 300), probes=probes,
                         frame_period_sec=0.04)


def test_pixel_blanks_need_contiguous_blank_frames() -> None:
    from subtitles_extractor.application.services.subtitle_pipeline.boundary_refinement import pixel_confirmed_blanks

    assert pixel_confirmed_blanks(_track([300, 0, 300, 300])) == [0.12]          # khung bỏ qua 0,08 trống, liền chuỗi
    assert pixel_confirmed_blanks(_track([0, 300, 300, 300])) == []              # 0,04 trống nhưng 0,08 có chữ ở giữa
    assert pixel_confirmed_blanks(_track([300, 300, 300, 0])) == []              # tương tự phía sau


def test_minority_traditional_keeps_chars_valid_in_simplified() -> None:
    from subtitles_extractor.application.services.subtitle_pipeline.text_correction import unify_minority_traditional

    texts = ["这是我们的师父"] * 60 + ["於谦大人到了", "宫商角徵羽", "楚瑶解決了"]
    assert unify_minority_traditional(texts) == {62: "楚瑶解决了"}
    assert unify_minority_traditional(texts[:60] + ["於谦說了"]) == {60: "於谦说了"}


def test_line_emotion_only_attached_for_models_that_read_it() -> None:
    from subtitles_extractor.infrastructure.tts.gemini_tts_schema import uses_verbatim_tts_schema
    from tests.unit.page_sources import TTS_PAGE_SOURCE

    source = TTS_PAGE_SOURCE.read_text(encoding="utf-8")
    assert ("self._gemini_line_emotion.isChecked() and uses_verbatim_tts_schema(self._gemini_model.currentText())"
            in source)
    assert uses_verbatim_tts_schema("gemini-3.8-flash-tts") and not uses_verbatim_tts_schema("gemini-2.5-flash-tts")
