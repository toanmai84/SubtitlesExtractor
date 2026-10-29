"""[v3.23.655] Edge/Gemini giữ audio RIÊNG từng dòng → "TTS lại dòng chọn" dùng được.

Trước đây chỉ VieNeu ghi ``seg_<vị trí>.wav``; với Edge/Gemini nút TTS lại bị khoá
(``can_resynthesize`` = False), sửa một chữ phải chạy lại cả phim.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from subtitles_extractor.application.use_cases.resynthesize_lines import ResynthesizeLinesUseCase
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts import edge_tts_adapter, gemini_tts_adapter

_SR = 24000


def _tone(text: str, sr: int = _SR) -> np.ndarray:
    n = int(sr * (0.25 + 0.04 * len(text)))
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * (200 + 10 * len(text)) * t)).astype(np.float32)


def _events() -> list[SimpleNamespace]:
    texts = ["Xin chào", "Đi đâu đấy", "Về nhà thôi"]
    return [
        SimpleNamespace(index=i + 1, text=text, start_sec=1.0 + 3.0 * i, end_sec=3.0 + 3.0 * i)
        for i, text in enumerate(texts)
    ]


def _request(tmp_path: Path, **kw) -> TTSRequest:
    return TTSRequest(
        events=_events(), keep_line_audio=True, normalize=False, double_pass=False,
        media_duration_s=12.0, **kw,
    )


def _patch_edge(monkeypatch, calls: list[str]) -> edge_tts_adapter.EdgeTTSAdapter:
    adapter = edge_tts_adapter.EdgeTTSAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)

    async def fake_generate(text, voice, rate):
        calls.append(text)
        return _tone(text), _SR

    monkeypatch.setattr(adapter, "_async_generate", staticmethod(fake_generate))
    return adapter


def _assert_segments(results, n: int) -> None:
    assert len(results) == n
    for result in results:
        path = Path(result.segment_audio_path)
        assert result.segment_audio_path and path.is_file(), result
        audio, sr = sf.read(str(path), dtype="float32")
        assert abs(len(audio) / sr - result.audio_duration_s) < 0.002


def test_edge_keeps_line_audio_and_resynth_selected(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    adapter = _patch_edge(monkeypatch, calls)
    out = tmp_path / "ep.tts.wav"
    request = _request(tmp_path, language="vi-VN", speaker="vi-VN-HoaiMyNeural")
    results = adapter.generate(request, out)
    _assert_segments(results, 3)
    assert Path(results[1].segment_audio_path).name == "seg_000001.wav"

    calls.clear()
    edited = list(request.events)
    edited[1] = SimpleNamespace(**{**vars(edited[1]), "text": "Đi đâu đấy hả em"})
    merged = ResynthesizeLinesUseCase(adapter).execute(
        full_request=_replace(request, edited),
        selected_indices={1}, prior_results=results, output_path=out,
        sample_rate=_SR, total_length_samples=int(12.0 * _SR),
    )
    assert any("hả em" in text for text in calls)        # chỉ dòng chọn được đọc lại
    assert not any("Xin chào" in text for text in calls)
    assert merged[0].segment_audio_path == results[0].segment_audio_path
    assert Path(merged[1].segment_audio_path).is_file()


def _replace(request: TTSRequest, events) -> TTSRequest:
    from dataclasses import replace

    return replace(request, events=events)


def test_gemini_standard_keeps_line_audio(tmp_path, monkeypatch) -> None:
    adapter = gemini_tts_adapter.GeminiTTSAdapter(api_key="k")
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "_make_client", lambda **_: object())
    monkeypatch.setattr(adapter, "_standard_speech_config", lambda *a, **k: None)
    monkeypatch.setattr(
        adapter, "_standard_with_retry",
        lambda client, model, text, config, request, cancel_cb, available_s=0.0: _tone(text),
    )
    out = tmp_path / "ep.tts.wav"
    results = adapter.generate(_request(tmp_path, speaker="Aoede"), out)
    _assert_segments(results, 3)


def test_line_audio_off_writes_nothing(tmp_path, monkeypatch) -> None:
    adapter = _patch_edge(monkeypatch, [])
    out = tmp_path / "ep.tts.wav"
    request = _replace(_request(tmp_path, language="vi-VN"), _events())
    from dataclasses import replace

    results = adapter.generate(replace(request, keep_line_audio=False), out)
    assert all(not r.segment_audio_path for r in results)
    assert not (tmp_path / "ep.tts_segments").exists()
