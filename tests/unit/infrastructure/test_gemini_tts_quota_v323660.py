"""[v3.23.660] Gemini TTS hạn mức miễn phí: 3 lượt/phút/model — lỗi 429 từng làm mất tiếng hàng loạt dòng.

Đo thật 28/10 (gemini-3.8-flash-lite-tts, 20 dòng): 11/20 dòng bị bỏ trống vì adapter chỉ dùng key đầu,
thử lại 3 lần cách 1–2 s rồi bỏ dòng, không theo ``retryDelay`` máy chủ báo.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("soundfile")

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSGenerationError, TTSRequest
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.tts import gemini_tts_adapter as ga

_SR = 24000


def _pcm(seconds: float) -> bytes:
    t = np.arange(int(_SR * seconds)) / _SR
    return (np.sin(2 * np.pi * 220 * t) * 0.3 * 32767).astype("<i2").tobytes()


def _quota_error(per_day: bool = False, delay: str = "2s") -> Exception:
    quota = "GenerateRequestsPerDayPerProjectPerModel-FreeTier" if per_day else "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    return RuntimeError(
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota. Please retry in 1.5s.', "
        f"'status': 'RESOURCE_EXHAUSTED', 'details': [{{'violations': [{{'quotaId': '{quota}', 'quotaValue': '3'}}]}}, "
        f"{{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '{delay}'}}]}}}}"
    )


class _Server:
    """Máy chủ giả: mỗi key chỉ cho ``per_key`` lượt trước khi trả 429 (theo phút hoặc theo ngày)."""

    def __init__(self, per_key: int, per_day: bool = False) -> None:
        self.per_key, self.per_day = per_key, per_day
        self.calls: dict[str, int] = {}
        self.sleeps: list[float] = []

    def client(self, key: str):
        server = self

        class _Models:
            def generate_content(self, model, contents, config):
                n = server.calls.get(key, 0)
                server.calls[key] = n + 1
                if n >= server.per_key:
                    raise _quota_error(per_day=server.per_day)
                part = SimpleNamespace(inline_data=SimpleNamespace(data=_pcm(0.8)))
                return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])

        return SimpleNamespace(models=_Models())


def _adapter(monkeypatch, server: _Server, keys: str) -> ga.GeminiTTSAdapter:
    adapter = ga.GeminiTTSAdapter(api_key=keys)
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(adapter, "_make_client", lambda use_v1alpha=False: server.client(adapter._api_key))

    def fake_sleep(seconds: float) -> None:
        server.sleeps.append(seconds)
        server.calls.clear()  # hết cửa sổ phút → mọi key được gọi lại
    monkeypatch.setattr(ga, "_quota_sleep", fake_sleep, raising=False)
    monkeypatch.setattr(ga.time, "sleep", lambda s: None)
    return adapter


def _request(n: int) -> TTSRequest:
    events = [SubtitleEvent(index=i + 1, text=f"xin chào số {i + 1}", interval=TimeInterval(i * 3.0, i * 3.0 + 2.5))
              for i in range(n)]
    return TTSRequest(events=events, language="gemini-3.8-flash-lite-tts", speaker="Kore", normalize=False)


def test_rate_limited_lines_are_not_dropped(monkeypatch, tmp_path: Path) -> None:
    server = _Server(per_key=3)
    adapter = _adapter(monkeypatch, server, "KEY_A\nKEY_B\nKEY_C")
    results = adapter.generate(_request(12), tmp_path / "out.wav")
    assert sum(not r.was_skipped for r in results) == 12, [r.error_msg for r in results if r.was_skipped]


def test_uses_every_key_before_waiting(monkeypatch, tmp_path: Path) -> None:
    server = _Server(per_key=3)
    adapter = _adapter(monkeypatch, server, "KEY_A,KEY_B,KEY_C")
    adapter.generate(_request(9), tmp_path / "out.wav")
    assert server.sleeps == [], "9 dòng vừa đủ 3 key × 3 lượt — không phải chờ"


def test_waits_server_retry_delay_when_all_keys_limited(monkeypatch, tmp_path: Path) -> None:
    server = _Server(per_key=2)
    adapter = _adapter(monkeypatch, server, "KEY_A")
    results = adapter.generate(_request(5), tmp_path / "out.wav")
    assert all(not r.was_skipped for r in results)
    assert sum(server.sleeps) >= 2.5  # chờ đúng retryDelay (2 s) + biên an toàn, không phải 1–2 s cố định


def test_daily_quota_on_all_keys_stops_with_clear_error(monkeypatch, tmp_path: Path) -> None:
    server = _Server(per_key=2, per_day=True)
    adapter = _adapter(monkeypatch, server, "KEY_A,KEY_B")
    with pytest.raises(TTSGenerationError, match="hạn mức"):
        adapter.generate(_request(8), tmp_path / "out.wav")


def test_parse_quota_error() -> None:
    info = ga._parse_quota_error(_quota_error(delay="7s"))
    assert info == (False, 7.0)
    assert ga._parse_quota_error(_quota_error(per_day=True))[0] is True
    assert ga._parse_quota_error(RuntimeError("500 INTERNAL")) is None
