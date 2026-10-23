"""[v3.23.622] Phân tích Visual Cues một-đoạn gửi mốc [MM:SS] + chỉ thị cuối prompt.

Mọi tập ngắn (≤ 18 phút) là ĐÚNG MỘT đoạn video nên đi đường "gửi thẳng": prompt chỉ có
``số. chữ``, không mốc giờ (chỉ đường nhiều đoạn có). Đo thật (gemini-3.1-flash-lite): lượt đầu
điền người nói 6/95 và 5/143 dòng → thêm một lời gọi mang lại cả video để vá. Tài liệu
video-understanding: dùng ``MM:SS`` để trỏ khung hình; prompting-strategies: đặt chỉ thị ở CUỐI.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _CONTEXT_ANALYSIS_SCHEMA,
    _CONTEXT_WITH_CUES_SCHEMA,
)
from tests.unit.infrastructure.gemini_call_harness import bare_analysis_adapter

LINES = [
    TranslationLine(index=1, start_ms=1_200, end_ms=2_500, text="修炼一途"),
    TranslationLine(index=2, start_ms=65_400, end_ms=66_900, text="自炼气始"),
    TranslationLine(index=3, start_ms=65_900, end_ms=67_000, text="自炼气始"),  # lặp liền nhau
]


def _ref() -> MagicMock:
    ref = MagicMock()
    ref.start_sec, ref.end_sec, ref.remote_name = 0.0, 177.0, "files/a"
    return ref


def _prompt(*, with_cues: bool, refs: list | None) -> str:
    adapter = bare_analysis_adapter()
    seen: list[str] = []

    def fake_call(_model, prompt, _config, _validator, **_kw):
        seen.append(prompt)
        return {"source_lang": "zh", "characters": [], "overview": "x", "cues": []}

    with (
        patch.object(adapter, "_resolve_handles_with_heal", lambda r, h: ["h"] if r else []),
        patch.object(adapter, "_build_config", lambda *a, **k: MagicMock()),
        patch.object(adapter, "_call_gemini", fake_call),
        patch.object(adapter, "_ensure_available", lambda: None),
    ):
        adapter.analyze_global_context(
            LINES, "Vietnamese", video_refs=refs, with_visual_cues=with_cues)
    return seen[0]


def test_single_clip_with_cues_has_timestamps_and_closing_task() -> None:
    prompt = _prompt(with_cues=True, refs=[_ref()])
    assert "1. [00:01] 修炼一途" in prompt
    assert "2. [01:05] 自炼气始" in prompt and "3. [01:05] 自炼气始" in prompt  # giữ dòng lặp
    assert prompt.rstrip().endswith("không để trống spk.")  # chỉ thị đứng CUỐI
    assert "ĐÚNG một phần tử cho MỖI id trong 3 dòng" in prompt


def test_prompt_unchanged_without_video_or_cues() -> None:
    for with_cues, refs in ((True, None), (False, [_ref()]), (False, None)):
        prompt = _prompt(with_cues=with_cues, refs=refs)
        assert "[00:01]" not in prompt and "Dựa trên video" not in prompt


def test_cues_are_required_only_in_the_cue_schema() -> None:
    assert "cues" in _CONTEXT_WITH_CUES_SCHEMA["required"]
    item = _CONTEXT_WITH_CUES_SCHEMA["properties"]["cues"]["items"]
    assert item["required"] == ["id", "spk", "to", "cue"]
    assert "cues" not in _CONTEXT_ANALYSIS_SCHEMA["required"]
