"""[v3.23.193] Test VieNeu set adjusted timing (để xuất SRT) + hủy gỡ blocker sạch.

Hai lỗi từ người dùng:
1. VieNeu xong KHÔNG xuất kèm SRT: use case chỉ xuất SRT khi có ``adjusted_start_sec>=0``
   (has_adjusted). VieNeu để trống -> mặc định -1.0 -> has_adjusted=False -> không xuất.
   Fix: set adjusted = mốc gốc cho mọi result (VieNeu đặt audio đúng start gốc).
2. Hủy TTS báo lỗi: thực ra worker bắt TTSCancelledError đúng; kiểm tra blocker torch
   được gỡ SẠCH khi hủy giữa isolation (không rò rỉ sys.modules).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSCancelledError,
    TTSRequest,
)
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as mod
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_cache():
    mod._ENGINE_CACHE.clear()
    yield
    mod._ENGINE_CACHE.clear()


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class _FakeEngine:
    def list_preset_voices(self):
        return [("Giọng A", "a")]

    def get_preset_voice(self, vid):
        return {"id": vid}

    def infer(self, text, voice=None):
        return (0.3 * np.sin(2 * np.pi * 200 * np.arange(int(_SR * 0.4)) / _SR)).astype(np.float32)


def _prime(adapter, engine, monkeypatch):
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = engine
    adapter._force_cpu = False  # né torch_isolation trong test (không có torch thật)


# ── Vấn đề 2: set adjusted để xuất SRT ───────────────────────────────────


def test_result_has_adjusted_timing(monkeypatch, tmp_path) -> None:
    adapter = VieNeuTtsAdapter()
    _prime(adapter, _FakeEngine(), monkeypatch)
    request = TTSRequest(
        events=[_Event(1.0, 3.0, "Xin chào")], speaker="a", normalize=False,
    )
    results = adapter.generate(request, tmp_path / "out")
    # adjusted phải >= 0 để use case nhận diện "có điều chỉnh" -> xuất SRT.
    assert results[0].adjusted_start_sec >= 0.0
    assert results[0].adjusted_end_sec >= 0.0
    # [v3.23.218] adjusted = mốc THẬT của tiếng; có thể SỚM hơn mốc phụ đề tối đa 0.25s
    # khi câu trước đã đọc xong ("ăn gian đầu" -> nén nhẹ hơn -> giọng rõ hơn).
    assert 0.6 <= results[0].adjusted_start_sec <= 1.0
    assert results[0].adjusted_end_sec >= 3.0


def test_skipped_result_also_has_adjusted(monkeypatch, tmp_path) -> None:
    # Câu bị skip (cửa sổ ngắn) cũng set adjusted -> phụ đề vẫn hiển thị dòng đó.
    # [v3.23.202] Cửa sổ ngắn chỉ còn bị skip khi KHÔNG cho chồng tiếng (cho chồng ->
    # đọc + tràn tự nhiên, giữ trọn thoại) -> tắt chồng tiếng để giữ mục đích test này.
    adapter = VieNeuTtsAdapter()
    _prime(adapter, _FakeEngine(), monkeypatch)
    request = TTSRequest(
        events=[_Event(1.0, 1.005, "x")], speaker="a", normalize=False,
        gap_threshold_s=0.5, allow_audio_overlap=False,
    )
    results = adapter.generate(request, tmp_path / "out")
    assert results[0].was_skipped
    assert results[0].adjusted_start_sec == 1.0


def test_has_adjusted_true_enables_srt(monkeypatch, tmp_path) -> None:
    # Kiểm điều kiện use case: any(adjusted>=0 and not skipped) phải True.
    adapter = VieNeuTtsAdapter()
    _prime(adapter, _FakeEngine(), monkeypatch)
    request = TTSRequest(
        events=[_Event(0.0, 2.0, "Câu một"), _Event(2.0, 4.0, "Câu hai")],
        speaker="a", normalize=False,
    )
    results = adapter.generate(request, tmp_path / "out")
    has_adjusted = any(
        r.adjusted_start_sec >= 0.0 and not r.was_skipped for r in results
    )
    assert has_adjusted is True  # trước fix: False -> không xuất SRT


# ── Vấn đề 1: hủy gỡ blocker sạch ────────────────────────────────────────


def test_cancel_cleans_torch_blocker() -> None:
    from subtitles_extractor.infrastructure.torch_import_blocker import (
        is_torch_import_blocked,
        torch_isolation,
    )
    if sys.modules.get("torch", "absent") not in ("absent", None):
        return
    with pytest.raises(TTSCancelledError), torch_isolation():
        assert is_torch_import_blocked()
        raise TTSCancelledError("Người dùng đã huỷ TTS.")
    assert not is_torch_import_blocked()  # gỡ sạch sau hủy
    assert "torch" not in sys.modules      # không rò rỉ torch=None
