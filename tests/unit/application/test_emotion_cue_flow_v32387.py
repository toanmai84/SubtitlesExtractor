"""[v3.23.87] Test luồng cue cảm xúc: phân tích -> bơm dòng -> payload -> dịch.

``scene`` (bối cảnh + thái độ/cảm xúc do AI nghe-nhìn) trước bị BỎ khi áp cue;
đường active còn không yêu cầu ``cue``. Nay cue được yêu cầu, bơm vào dòng, gửi
kèm trong dual payload và checkpoint round-trip giữ nguyên.
"""

from __future__ import annotations

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    VisualCue,
    apply_visual_cues_to_lines,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _CONTEXT_WITH_CUES_SCHEMA,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator as G,
)


def test_apply_cue_injects_scene_into_line() -> None:
    lines = [TranslationLine(index=1, start_ms=0, end_ms=1000, text="滚")]
    cues = [VisualCue(line_no=1, speaker="Tần Chính", addressee="Sở Nhan",
                      scene="giận dữ quát trong đại điện")]
    enriched = apply_visual_cues_to_lines(lines, cues)
    assert enriched[0].scene == "giận dữ quát trong đại điện"
    assert enriched[0].speaker == "Tần Chính"
    assert enriched[0].addressee == "Sở Nhan"


def test_dual_payload_includes_cue() -> None:
    line = TranslationLine(
        index=1, start_ms=0, end_ms=1000, text="Cút!",
        original_text="滚", scene="giận dữ", speaker="Tần Chính",
    )
    payload = G._line_to_dual_payload(line)
    assert payload["cue"] == "giận dữ"
    assert payload["original"] == "滚"


def test_dual_payload_omits_cue_when_absent() -> None:
    line = TranslationLine(index=1, start_ms=0, end_ms=1000, text="Cút!",
                           original_text="滚")
    payload = G._line_to_dual_payload(line)
    assert "cue" not in payload


def test_active_schema_requests_cue_field() -> None:
    props = _CONTEXT_WITH_CUES_SCHEMA["properties"]["cues"]["items"]["properties"]
    assert "cue" in props


def test_checkpoint_round_trips_scene() -> None:
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        _json_to_lines,
        _lines_to_json,
    )

    lines = [TranslationLine(index=1, start_ms=0, end_ms=1000, text="Cút!",
                             scene="giận dữ", original_text="滚", addressee="Sở Nhan")]
    restored = _json_to_lines(_lines_to_json(lines))
    assert restored[0].scene == "giận dữ"
    assert restored[0].original_text == "滚"
    assert restored[0].addressee == "Sở Nhan"


def test_literal_payload_includes_addressee_and_cue() -> None:
    # [v3.23.88] Khâu LITERAL cũng nhận addressee + cue nghe-nhìn (trước đây bị bỏ).
    line = TranslationLine(
        index=1, start_ms=0, end_ms=1000, text="滚",
        speaker="Tần Chính", addressee="Sở Nhan", scene="giận dữ",
    )
    payload = G._line_to_payload(line)
    assert payload["addressee"] == "Sở Nhan"
    assert payload["cue"] == "giận dữ"
    assert payload["speaker"] == "Tần Chính"
    # original chỉ thuộc dual payload, KHÔNG có ở _line_to_payload.
    assert "original" not in payload
