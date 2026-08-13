"""Unit test cho Visual Cues Studio backend (v3.14.1, Nhóm 3).

Phủ: giải nén JSON minified (id/spk/to/cue), bơm cue vào dòng (Silent Context
Injection), và micro-batching + nghỉ chống rate-limit.
"""

from __future__ import annotations

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    VisualCue,
    apply_visual_cues_to_lines,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


class TestParseMinifiedCues:
    def test_parse_minified_keys(self) -> None:
        items = [{"id": 3, "spk": "Lâm Côn", "to": "Nữ tỳ", "cue": "tức giận"}]
        cues = GeminiSubtitleTranslator._parse_visual_cue_items(items)
        assert cues[0].line_no == 3
        assert cues[0].speaker == "Lâm Côn"
        assert cues[0].addressee == "Nữ tỳ"
        assert cues[0].scene == "tức giận"

    def test_parse_fallback_full_keys(self) -> None:
        items = [{"line_no": 5, "speaker": "A", "addressee": "B", "scene": "C"}]
        cues = GeminiSubtitleTranslator._parse_visual_cue_items(items)
        assert cues[0].line_no == 5 and cues[0].speaker == "A"

    def test_parse_skips_invalid_id(self) -> None:
        items = [{"id": 0, "spk": "x"}, {"spk": "y"}, {"id": 2, "spk": "ok"}]
        cues = GeminiSubtitleTranslator._parse_visual_cue_items(items)
        assert len(cues) == 1 and cues[0].line_no == 2


class TestSilentContextInjection:
    def test_apply_cues_sets_speaker_addressee(self) -> None:
        lines = [
            TranslationLine(index=1, start_ms=0, end_ms=1, text="a"),
            TranslationLine(index=2, start_ms=0, end_ms=1, text="b"),
        ]
        cues = [VisualCue(line_no=1, speaker="Sư phụ", addressee="Đệ tử")]
        out = apply_visual_cues_to_lines(lines, cues)
        assert out[0].speaker == "Sư phụ" and out[0].addressee == "Đệ tử"
        assert out[1].speaker == ""  # dòng không có cue giữ nguyên

    def test_apply_cues_is_pure(self) -> None:
        lines = [TranslationLine(index=1, start_ms=0, end_ms=1, text="a")]
        apply_visual_cues_to_lines(lines, [VisualCue(line_no=1, speaker="X")])
        assert lines[0].speaker == ""  # bản gốc không bị đột biến


class TestMicroBatching:
    def test_batches_and_sleeps(self, monkeypatch) -> None:
        translator = GeminiSubtitleTranslator(api_key="dummy")
        lines = [
            TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=f"l{i}")
            for i in range(1, 6)
        ]
        call_log: dict[str, int] = {"calls": 0, "sleeps": 0}

        monkeypatch.setattr(translator, "_ensure_available", lambda: None)
        monkeypatch.setattr(translator, "_build_config", lambda *a, **k: None)
        monkeypatch.setattr(translator, "_resolve_video_handles", lambda refs: [])

        def fake_call(model, prompt, config, validator, **kwargs):
            call_log["calls"] += 1
            # Trả cue cho mọi id trong prompt (giả lập AI).
            import re
            ids = [int(m) for m in re.findall(r'"id":\s*(\d+)', prompt)]
            return {"cues": [{"id": i, "spk": f"spk{i}"} for i in ids]}

        def fake_sleep(seconds, cancel_cb):
            call_log["sleeps"] += 1
            return True

        monkeypatch.setattr(translator, "_call_gemini", fake_call)
        monkeypatch.setattr(translator, "_interruptible_sleep", fake_sleep)

        cues = translator.analyze_visual_cues(
            lines, target_lang="Vietnamese", batch_size=2, sleep_between_s=4.0
        )
        assert len(cues) == 5
        assert call_log["calls"] == 3  # 5 dòng / lô 2 = 3 lô
        assert call_log["sleeps"] == 2  # nghỉ giữa lô, không nghỉ sau lô cuối
        assert cues[0].speaker == "spk1"

    def test_empty_returns_empty(self, monkeypatch) -> None:
        translator = GeminiSubtitleTranslator(api_key="dummy")
        monkeypatch.setattr(translator, "_ensure_available", lambda: None)
        assert translator.analyze_visual_cues([], target_lang="Vietnamese") == []
