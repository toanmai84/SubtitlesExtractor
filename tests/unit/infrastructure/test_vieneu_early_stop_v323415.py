"""[v3.23.415] Test dừng sớm retry 'dài bất thường' khi không cải thiện."""

from __future__ import annotations

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as mod
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter


class _Engine:
    sample_rate = 48000


class _Request:
    retry_count = 10
    retry_delay_s = 0.0


def _make_adapter() -> VieNeuTtsAdapter:
    adapter = object.__new__(VieNeuTtsAdapter)
    adapter._batch_size = 0
    adapter._batch_texts = []
    adapter._batch_cache = {}
    adapter._force_cpu = True
    adapter._gpu_worker_disabled = True
    return adapter


def test_early_stop_when_overlong_not_improving(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter()
    calls = {"n": 0}
    # Mọi lần infer trả audio DÀI BẰNG NHAU (không cải thiện).
    fixed = np.ones(48000 * 2, dtype=np.float32)  # 2s

    def fake_infer_once(engine, text, voice_data, idx=-1, **_kw):
        calls["n"] += 1
        return fixed

    monkeypatch.setattr(adapter, "_infer_once", fake_infer_once)
    monkeypatch.setattr(adapter, "_to_mono_pipeline_rate", lambda raw, sr=None: raw)
    monkeypatch.setattr(mod, "is_effectively_silent", lambda a: False)
    monkeypatch.setattr(mod, "is_abnormally_long_vs_floor", lambda *a, **k: True)
    monkeypatch.setattr(mod, "dem_am_tiet", lambda t: 1)

    result = adapter._synthesize_with_retry(
        engine=_Engine(), text="Được.", request=_Request(),
        voice_data={}, cancel_cb=None, idx=0,
    )
    assert result is not None  # trả bản ngắn nhất
    # Dừng sớm: MIN_ATTEMPTS(3) + STALL_LIMIT(3) → ~4 lần, KHÔNG chạy đủ 10.
    assert calls["n"] < 10
    assert calls["n"] <= mod._OVERLONG_MIN_ATTEMPTS + mod._OVERLONG_STALL_LIMIT


def test_normal_audio_returns_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter()
    calls = {"n": 0}
    good = np.ones(48000, dtype=np.float32)

    def fake_infer_once(engine, text, voice_data, idx=-1, **_kw):
        calls["n"] += 1
        return good

    monkeypatch.setattr(adapter, "_infer_once", fake_infer_once)
    monkeypatch.setattr(adapter, "_to_mono_pipeline_rate", lambda raw, sr=None: raw)
    monkeypatch.setattr(mod, "is_effectively_silent", lambda a: False)
    monkeypatch.setattr(mod, "is_abnormally_long_vs_floor", lambda *a, **k: False)
    monkeypatch.setattr(mod, "dem_am_tiet", lambda t: 3)

    result = adapter._synthesize_with_retry(
        engine=_Engine(), text="Xin chào bạn", request=_Request(),
        voice_data={}, cancel_cb=None, idx=0,
    )
    assert result is not None
    assert calls["n"] == 1  # audio bình thường → trả ngay, không retry
