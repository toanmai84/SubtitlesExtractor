"""[v3.23.662] Live TTS: chỉ dẫn đọc theo khung từng dòng + thêm model Live đời mới.

Đo thật 238 dòng (tập 1–2, 3 key): nhịp thích ứng giảm dòng phải nén >1,5x từ 31–35% xuống 14–15%, CER 0,040.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("google.genai")

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.tts import gemini_tts_adapter as ga

from .test_gemini_live_tts_v323661 import _Server


def test_style_follows_line_window() -> None:
    text = "Sư huynh đừng đi nữa"  # 5 âm tiết
    assert ga._live_style_for(text, 3.0) is ga._LIVE_VERBATIM_STYLE   # 1,7 âm tiết/giây — đủ chỗ
    assert ga._live_style_for(text, 1.0) is ga._LIVE_FAST_STYLE       # 5 âm tiết/giây — chật
    assert ga._live_style_for(text, 0.0) is ga._LIVE_VERBATIM_STYLE   # không rõ khung
    assert "NGUYÊN VĂN" in ga._LIVE_FAST_STYLE


def test_new_live_models_listed_and_verbatim() -> None:
    models = ga.GEMINI_TTS_MODELS
    assert models[0] == "gemini-3.8-live"
    for name in ("gemini-3.8-live-extended-thinking", "gemini-3.1-flash-live-preview"):
        assert name in models and name in ga._LIVE_VERBATIM_MODELS
    # Đời 2.5 phát âm sai nhiều → xếp sau các model Live mới.
    assert models.index("gemini-3.1-flash-live-preview") < models.index("gemini-2.5-flash-native-audio-latest")


def _run(monkeypatch, tmp_path: Path, model: str, events: list[SubtitleEvent]) -> _Server:
    server = _Server()
    adapter = ga.GeminiTTSAdapter(api_key="KA")
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "_make_client", lambda use_v1alpha=False, api_key="": server.client(api_key or "KA"))
    request = TTSRequest(events=events, language=model, speaker="Kore", normalize=False)
    results = adapter.generate(request, tmp_path / "out.wav")
    assert all(not r.was_skipped for r in results)
    return server


@pytest.mark.parametrize(("text", "end", "expected"), [
    ("Được.", 2.0, "_LIVE_VERBATIM_STYLE"),
    ("Ngươi tưởng ta không biết ngươi đã làm gì sao?", 1.2, "_LIVE_FAST_STYLE"),  # 11 âm tiết / 1,2 s
])
def test_tight_line_gets_fast_style(monkeypatch, tmp_path: Path, text: str, end: float, expected: str) -> None:
    events = [
        SubtitleEvent(index=1, text=text, interval=TimeInterval(0.0, end)),
        SubtitleEvent(index=2, text="Ừ.", interval=TimeInterval(end + 0.05, end + 2.0)),  # câu sau sát → không mượn gap
    ]
    server = _run(monkeypatch, tmp_path, "gemini-3.8-live", events[:1] + events[1:])
    used = {cfg.system_instruction.parts[0].text for cfg in server.configs}
    assert getattr(ga, expected) in used
    if expected == "_LIVE_VERBATIM_STYLE":
        assert ga._LIVE_FAST_STYLE not in used


def test_extended_thinking_sends_thinking_level(monkeypatch, tmp_path: Path) -> None:
    events = [SubtitleEvent(index=1, text="Đi thôi.", interval=TimeInterval(0.0, 2.0))]
    server = _run(monkeypatch, tmp_path, "gemini-3.8-live-extended-thinking", events)
    cfg = server.configs[0]
    assert cfg.thinking_config is not None and str(cfg.thinking_config.thinking_level).lower().endswith("low")
    plain = _run(monkeypatch, tmp_path, "gemini-3.8-live", events)
    assert plain.configs[0].thinking_config is None
