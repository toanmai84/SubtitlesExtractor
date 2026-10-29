"""[v3.23.661] Gemini 3.8 Live làm TTS: chia phiên theo key (≤3 phiên/key), gửi NGUYÊN VĂN câu.

Đo thật 28/10: 10 phiên trên MỘT key → 76 lỗi "1011 … exceeded your current quota", 9/95 dòng câm.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("google.genai")

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.tts import gemini_tts_adapter as ga


def _pcm(seconds: float) -> bytes:
    t = np.arange(int(24000 * seconds)) / 24000
    return (np.sin(2 * np.pi * 220 * t) * 0.3 * 32767).astype("<i2").tobytes()


class _Server:
    def __init__(self) -> None:
        self.active: Counter[str] = Counter()
        self.peak: Counter[str] = Counter()
        self.texts: list[str] = []
        self.configs: list = []

    def client(self, key: str):
        server = self

        class _Session:
            async def send_client_content(self, turns, turn_complete):
                server.texts.append(turns.parts[0].text)

            async def receive(self):
                await asyncio.sleep(0.01)
                part = SimpleNamespace(inline_data=SimpleNamespace(data=_pcm(0.8)))
                yield SimpleNamespace(server_content=SimpleNamespace(model_turn=SimpleNamespace(parts=[part]),
                                                                     turn_complete=True))

        class _Conn:
            async def __aenter__(self):
                server.active[key] += 1
                server.peak[key] = max(server.peak[key], server.active[key])
                if server.active[key] > 4:
                    raise RuntimeError("1011 None. You exceeded your current quota")
                return _Session()

            async def __aexit__(self, *exc):
                server.active[key] -= 1

        def connect(model, config):
            server.configs.append(config)
            return _Conn()

        return SimpleNamespace(aio=SimpleNamespace(live=SimpleNamespace(connect=connect)))


def test_live_sessions_spread_over_keys_and_read_verbatim(monkeypatch, tmp_path: Path) -> None:
    server = _Server()
    adapter = ga.GeminiTTSAdapter(api_key="KA,KB,KC")
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "_make_client", lambda use_v1alpha=False, api_key="": server.client(api_key or "KA"))
    events = [SubtitleEvent(index=i + 1, text=f"Sư huynh, câu số {i + 1}.", interval=TimeInterval(i * 3.0, i * 3.0 + 2.5))
              for i in range(30)]
    request = TTSRequest(events=events, language="gemini-3.8-live", speaker="Kore", normalize=False)
    results = adapter.generate(request, tmp_path / "out.wav")
    assert sum(not r.was_skipped for r in results) == 30
    assert set(server.peak) == {"KA", "KB", "KC"}, server.peak
    assert max(server.peak.values()) <= ga._LIVE_SESSIONS_PER_KEY
    assert server.texts[0].startswith("Sư huynh") and all(t.endswith(".") for t in server.texts)
    cfg = server.configs[0]
    assert "NGUYÊN VĂN" in cfg.system_instruction.parts[0].text
    assert not getattr(cfg, "enable_affective_dialog", None)


def test_model_list_puts_live_first() -> None:
    assert ga.GEMINI_TTS_MODELS[0] == "gemini-3.8-live"
    assert "gemini-3.8-live" in ga._NATIVE_AUDIO_MODELS
