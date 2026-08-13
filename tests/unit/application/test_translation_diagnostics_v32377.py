"""[v3.23.77] Test hàm thuần ``build_diagnostics_bundle`` — gói chẩn đoán dịch.

Dùng SimpleNamespace giả lập event/session/stage/series để kiểm cấu trúc, đếm số lượng,
giải mã ``lines_json`` và tính JSON-serializable — không phụ thuộc Qt hay DB.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from subtitles_extractor.application.services.translation_diagnostics import (
    build_diagnostics_bundle,
    detect_quality_flags,
)


def _event(index: int, start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(index=index, start_sec=start, end_sec=end, text=text)


def _build(**overrides):
    kwargs = dict(
        app_version="3.23.77",
        exported_at="2026-01-01T00:00:00+00:00",
        video_path="/movies/NOVA.S51E16.mkv",
        source_events=[_event(1, 0.0, 2.0, "Hello"), _event(2, 2.0, 4.0, "World")],
        translated_events=[
            _event(1, 0.0, 2.0, "Xin chào"), _event(2, 2.0, 4.0, "Thế giới")
        ],
        session=None,
        series_context=None,
    )
    kwargs.update(overrides)
    return build_diagnostics_bundle(**kwargs)


def test_minimal_bundle_structure_and_counts() -> None:
    bundle = _build()
    assert bundle["schema"] == "subtitles_extractor.translation_diagnostics"
    assert bundle["schema_version"] == 3
    assert bundle["app_version"] == "3.23.77"
    assert bundle["video"]["filename"] == "NOVA.S51E16.mkv"
    assert bundle["counts"]["source_events"] == 2
    assert bundle["counts"]["translated_events"] == 2
    assert bundle["counts"]["stages"] == 0
    assert bundle["source_events"][0]["text"] == "Hello"
    assert bundle["final_translation"][0]["text"] == "Xin chào"


def test_bundle_is_json_serializable() -> None:
    bundle = _build()
    # Không được ném lỗi; ensure_ascii=False để giữ tiếng Việt.
    dumped = json.dumps(bundle, ensure_ascii=False)
    assert "Xin chào" in dumped


def test_session_analysis_and_stage_lines_parsed() -> None:
    stage = SimpleNamespace(
        stage_id="literal",
        input_hash="abc",
        completed_at="2026-01-01T00:00:00+00:00",
        lines_json='[{"index": 1, "text": "Xin chào"}]',
    )
    session = SimpleNamespace(
        analysis_source_lang="en",
        analysis_target_lang="vi",
        analysis_characters="Patrick: thợ lặn",
        analysis_overview="Phim tài liệu.",
        analysis_glossary="submarine = tàu ngầm",
        analysis_visual_cues="Cảnh biển sâu.",
        analysis_input_hash="hash123",
        stages=(stage,),
        cloud_files=(),
    )
    bundle = _build(session=session)
    assert bundle["languages"] == {"source": "en", "target": "vi"}
    assert bundle["analysis"]["glossary"] == "submarine = tàu ngầm"
    assert bundle["counts"]["stages"] == 1
    # lines_json phải được GIẢI MÃ thành list, không để nguyên chuỗi.
    assert bundle["stages"][0]["lines"] == [{"index": 1, "text": "Xin chào"}]


def test_malformed_stage_lines_json_does_not_crash() -> None:
    stage = SimpleNamespace(
        stage_id="style", input_hash="x", completed_at="",
        lines_json="{khong-phai-json",
    )
    session = SimpleNamespace(
        analysis_source_lang="", analysis_target_lang="", analysis_characters="",
        analysis_overview="", analysis_glossary="", analysis_visual_cues="",
        analysis_input_hash="", stages=(stage,), cloud_files=(),
    )
    bundle = _build(session=session)
    parsed = bundle["stages"][0]["lines"]
    assert isinstance(parsed, dict) and parsed.get("_parse_error") is True


def test_series_context_included_when_present() -> None:
    series = SimpleNamespace(
        glossary="g", characters="c", overview="o"
    )
    bundle = _build(series_context=series)
    assert bundle["series_context"] == {
        "glossary": "g", "characters": "c", "overview": "o"
    }


class TestDetectQualityFlags:
    def test_identical_to_source_flagged(self) -> None:
        # [v3.23.156] Dùng CÂU THẬT chưa dịch (tên riêng như "Patrick Lahey" nay được
        # miễn cờ có chủ đích — dịch giữ nguyên tên riêng là ĐÚNG, không phải lỗi).
        source = [
            _event(1, 0.0, 1.0, "he said nothing would change"),
            _event(2, 1.0, 2.0, "Hello"),
        ]
        translated = [
            _event(1, 0.0, 1.0, "he said nothing would change"),
            _event(2, 1.0, 2.0, "Xin chào"),
        ]
        flags = detect_quality_flags(source, translated)
        assert flags["identical_to_source_indices"] == [1]
        assert flags["counts"]["identical_to_source"] == 1

    def test_empty_translation_flagged(self) -> None:
        source = [_event(1, 0.0, 1.0, "Something")]
        translated = [_event(1, 0.0, 1.0, "   ")]
        flags = detect_quality_flags(source, translated)
        assert flags["empty_translation_indices"] == [1]

    def test_length_anomaly_flagged(self) -> None:
        # Nguồn rất dài → dịch một từ (tỉ lệ < 0.2).
        source = [_event(1, 0.0, 1.0, "x" * 100)]
        translated = [_event(1, 0.0, 1.0, "ngắn")]
        flags = detect_quality_flags(source, translated)
        assert flags["length_anomaly"] and flags["length_anomaly"][0]["index"] == 1

    def test_clean_translation_has_no_flags(self) -> None:
        source = [_event(1, 0.0, 1.0, "Good morning")]
        translated = [_event(1, 0.0, 1.0, "Chào buổi sáng")]
        flags = detect_quality_flags(source, translated)
        assert flags["counts"]["identical_to_source"] == 0
        assert flags["counts"]["empty_translation"] == 0
        assert flags["counts"]["length_anomaly"] == 0

    def test_bundle_includes_quality_flags(self) -> None:
        bundle = _build()
        assert "quality_flags" in bundle
        assert "counts" in bundle["quality_flags"]


class TestLengthAnomalyCjkAware:
    """[v3.23.83] Calibrate theo dữ liệu thật zh->vi: bỏ tag, CJK-aware, bỏ nguồn ngắn."""

    def test_short_cjk_source_not_flagged(self) -> None:
        # "滚" (1 Hán tự) -> "[Tần Chính:] Cút!": trước đây bị cờ (ratio thô ~17), nay
        # nguồn hiệu dụng = 3 < 8 -> BỎ QUA, không cờ.
        source = [_event(1, 0.0, 1.0, "滚")]
        translated = [_event(1, 0.0, 1.0, "[Tần Chính:] Cút!")]
        flags = detect_quality_flags(source, translated)
        assert flags["counts"]["length_anomaly"] == 0

    def test_normal_cjk_translation_not_flagged(self) -> None:
        # Câu CJK đủ dài, dịch hợp lý (sau khi bỏ tag) -> không cờ.
        source = [_event(1, 0.0, 1.0, "你好世界我的朋友们")]  # 9 Hán tự, eff=27
        translated = [_event(1, 0.0, 1.0, "[An:] Xin chào thế giới các bạn của tôi")]
        flags = detect_quality_flags(source, translated)
        assert flags["counts"]["length_anomaly"] == 0

    def test_speaker_tag_stripped_before_measuring(self) -> None:
        # Tag dài không được tính vào độ dài bản dịch.
        source = [_event(1, 0.0, 1.0, "我们一起去看电影吧")]  # 9 Hán tự, eff=27
        translated = [
            _event(1, 0.0, 1.0, "[Nhân vật có tên rất dài:] Đi xem phim nào")
        ]
        flags = detect_quality_flags(source, translated)
        assert flags["counts"]["length_anomaly"] == 0

    def test_genuine_over_translation_still_flagged(self) -> None:
        # Nguồn đủ dài nhưng bản dịch phồng bất thường (>2.5x hiệu dụng) -> CÒN cờ.
        source = [_event(1, 0.0, 1.0, "你好朋友")]  # 4 Hán tự, eff=12
        translated = [_event(1, 0.0, 1.0, "x" * 40)]  # 40/12 = 3.33 > 2.5
        flags = detect_quality_flags(source, translated)
        assert flags["counts"]["length_anomaly"] == 1

    def test_identical_source_detected_after_tag_strip(self) -> None:
        # Model lặp lại nguyên văn nguồn (kèm tag) -> vẫn bắt được sau khi bỏ tag.
        source = [_event(1, 0.0, 1.0, "Hello world friend")]
        translated = [_event(1, 0.0, 1.0, "[Bob:] Hello world friend")]
        flags = detect_quality_flags(source, translated)
        assert flags["identical_to_source_indices"] == [1]
