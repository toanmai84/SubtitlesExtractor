"""[v3.23.623] Tách giọng bằng gemini-3.5-transcribe → nhãn giọng cho từng dòng phụ đề.

Dữ liệu đo thật (tập mẫu 2): cụm spk:0 = nhân vật chính 77/77 dòng, spk:2 = sư thúc 28/28,
spk:3 = hệ thống 24/24 — kể cả những dòng mà đọc chữ cứ gán nhầm.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.infrastructure.translation import gemini_diarization as diar
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _text_cues_system_instruction,
)


def _part(label: str, words: list[tuple[str, str, str]]) -> SimpleNamespace:
    return SimpleNamespace(audio_transcription=SimpleNamespace(
        speaker_label=label,
        words=[SimpleNamespace(word=w, start_offset=s, end_offset=e) for w, s, e in words]))


def test_words_from_response_parses_offsets_and_short_labels() -> None:
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
        _part("spk:2", [("你", "1.700s", "2.000s"), ("好", "2.000s", "2.400s")]),
        _part("spk_0", [("嗯", "0.500s", "0.900s")]),
        SimpleNamespace(),                       # phần không có audio_transcription
    ]))])
    words = diar.words_from_response(response)
    assert [(w.speaker, w.start, w.end) for w in words] == [
        ("0", 0.5, 0.9), ("2", 1.7, 2.0), ("2", 2.0, 2.4)]


@pytest.mark.parametrize(("value", "expected"), [
    ("1.700s", 1.7), (2, 2.0), ("", None), (None, None), ("abc", None)])
def test_parse_offset(value: object, expected: float | None) -> None:
    assert diar.parse_offset(value) == expected


def _line(index: int, start: float, end: float) -> SimpleNamespace:
    return SimpleNamespace(index=index, start_ms=int(start * 1000), end_ms=int(end * 1000))


def test_labels_take_the_cluster_with_most_overlap() -> None:
    words = [diar.VoiceWord("0", 0.0, 1.0), diar.VoiceWord("0", 1.0, 2.0),
             diar.VoiceWord("1", 2.0, 2.2), diar.VoiceWord("1", 4.0, 5.0)]
    lines = [_line(1, 0.2, 1.8), _line(2, 1.9, 2.5), _line(3, 4.1, 4.9), _line(4, 9.0, 9.5)]
    labels = diar.voice_labels_from_words(words, lines, pad_s=0.0)
    assert labels == {1: "0", 2: "1", 3: "1"}     # dòng 4 không có từ nào → vắng mặt


def test_labels_empty_when_no_words() -> None:
    assert diar.voice_labels_from_words([], [_line(1, 0, 1)]) == {}


def test_language_codes() -> None:
    assert diar.language_codes_for("zh") == ["cmn-Hans-CN"]
    assert diar.language_codes_for("xx") == [] and diar.language_codes_for("") == []


def test_word_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diar.tempfile, "gettempdir", lambda: str(tmp_path))
    video = tmp_path / "v.mp4"
    video.write_bytes(b"abc")
    assert diar.load_cached_words(video) is None
    diar.save_cached_words(video, [diar.VoiceWord("1", 0.5, 0.9)])
    assert diar.load_cached_words(video) == [diar.VoiceWord("1", 0.5, 0.9)]
    video.write_bytes(b"abcdef")                      # đổi kích thước → bộ nhớ đệm không còn đúng
    assert diar.load_cached_words(video) is None


def test_system_instruction_mentions_voices_only_when_present() -> None:
    with_voices = _text_cues_system_instruction("Vietnamese", "叶然 (A)", True)
    without = _text_cues_system_instruction("Vietnamese", "叶然 (A)", False)
    # [v3.23.633] Nhãn giọng vẫn là mặc định "cùng nhãn = cùng một người" nhưng phải kiểm nội dung.
    assert "[v:N]" in with_voices and "cùng một người" in with_voices
    assert "kiểm nội dung" in with_voices
    assert "[v:N]" not in without and "chưa có video" in without


def test_label_voices_is_safe_without_video(tmp_path: Path) -> None:
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    assert adapter.label_voices("", [_line(1, 0, 1)]) == {}
    assert adapter.label_voices(str(tmp_path / "nope.mp4"), [_line(1, 0, 1)]) == {}


def test_label_voices_uses_cache_without_calling_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diar.tempfile, "gettempdir", lambda: str(tmp_path))
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 10)
    diar.save_cached_words(video, [diar.VoiceWord("3", 0.0, 2.0)])
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    assert adapter.label_voices(str(video), [_line(1, 0.2, 1.5)]) == {1: "3"}


def test_analyze_text_cues_sends_voice_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    seen: list[str] = []
    monkeypatch.setattr(adapter, "_ensure_available", lambda: None)
    monkeypatch.setattr(adapter, "_build_config", lambda *a, **k: SimpleNamespace(system=a[2]))

    def fake_call(model, prompt, config, validator, **_k):
        seen.append(prompt + "||" + config.system)
        return {"cues": [{"id": 1, "spk": "A", "to": "B", "cue": ""}]}

    monkeypatch.setattr(adapter, "_call_gemini", fake_call)
    lines = [SimpleNamespace(index=1, start_ms=0, end_ms=1000, text="你好", speaker="",
                             addressee="", scene="", description="", original_text="")]
    cues = adapter.analyze_text_cues(lines, "Vietnamese", voice_labels={1: "2"})
    assert cues[0].speaker == "A"
    assert "1|[v:2]|你好" in seen[0] and '"voice": "2"' in seen[0] and "[v:N]" in seen[0]
    seen.clear()
    adapter.analyze_text_cues(lines, "Vietnamese")
    assert "[v:" not in seen[0].split("||")[0]
