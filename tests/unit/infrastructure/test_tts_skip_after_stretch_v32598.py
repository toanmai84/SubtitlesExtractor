"""[v3.23.198] Test skip-overlap SAU nén (VieNeu) + đồng bộ 3 fix sang Gemini.

Rà soát tiếp trang TTS phát hiện:
1. VieNeu đánh giá "Bỏ qua lấn" TRƯỚC khi nén, so khung GỐC -> câu audio 4s/khung 2s bị
   BỎ HẲN (mất thoại) dù sau nén 2x vừa khít. Fix: đánh giá SAU nén, so khung HIỆU DỤNG.
2. Gemini dính 3 bug cùng loại VieNeu đã sửa: không set adjusted (KHÔNG xuất SRT — bug
   v193), cắt cứng bỏ qua "Cho phép chồng tiếng" (bug v197), thiếu trần chất lượng nén
   2.0 (bug v196). Đồng bộ cả 3 fix, tái dùng hàm thuần (DRY).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter

_SR = 24000


@pytest.fixture(autouse=True)
def _clear_cache():
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class _LongAudioEngine:
    """Engine giả trả audio DÀI (4s) để kiểm nén + skip."""

    def __init__(self, duration_s: float = 4.0) -> None:
        self._duration_s = duration_s

    def list_preset_voices(self):
        return [("Giọng A", "a")]

    def get_preset_voice(self, vid):
        return {"id": vid}

    def infer(self, text, voice=None):
        n = int(_SR * self._duration_s)
        return (0.3 * np.sin(2 * np.pi * 200 * np.arange(n) / _SR)).astype(np.float32)


def _prime(adapter, engine, monkeypatch):
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = engine
    adapter._force_cpu = False


def test_skip_overlap_evaluated_after_stretch(monkeypatch, tmp_path) -> None:
    # Audio 4s / khung 2s, skip 500ms: TRƯỚC fix bị bỏ (overlap thô 2000ms > 500ms);
    # SAU fix: nén 2x -> 2s vừa khít hiệu dụng -> KHÔNG bỏ, giữ trọn thoại.
    adapter = VieNeuTtsAdapter()
    _prime(adapter, _LongAudioEngine(4.0), monkeypatch)
    request = TTSRequest(
        events=[_Event(0.0, 2.0, "Câu thoại dài")], speaker="a",
        normalize=False, skip_overlap_ms=500, base_speed=1.0, max_speed=3.0,
    )
    results = adapter.generate(request, tmp_path / "out")
    assert not results[0].was_skipped  # trước fix: bị bỏ -> mất thoại


def test_skip_overlap_still_skips_when_residual_exceeds(monkeypatch, tmp_path) -> None:
    """Nén hết cỡ mà vẫn lấn quá ngưỡng -> vẫn BỎ, đúng thiết kế.

    [v3.23.547] Cập nhật số liệu theo v3.23.464: trần nén nay là ĐÚNG ``max_speed``
    người dùng đặt, không còn bị kẹp cứng về 2.0. Với ``max_speed=3.0`` cũ, audio 10s
    nén còn 3.33s — chỉ lấn 0.33s so với khung hiệu dụng 3s, DƯỚI ngưỡng 500ms nên
    không bỏ nữa. Đặt ``max_speed=2.0`` để giữ đúng tình huống mà bài kiểm muốn xét:
    audio 10s -> 5s, khung hiệu dụng 3s (1s khung + 2s nới câu cuối) -> lấn dư 2s.
    """
    adapter = VieNeuTtsAdapter()
    _prime(adapter, _LongAudioEngine(10.0), monkeypatch)
    request = TTSRequest(
        events=[_Event(0.0, 1.0, "Quá dài")], speaker="a",
        normalize=False, skip_overlap_ms=500, base_speed=1.0, max_speed=2.0,
    )
    results = adapter.generate(request, tmp_path / "out")
    assert results[0].was_skipped
    assert "sau nén" in (results[0].error_msg or "")


# ── Gemini: đồng bộ fix ──────────────────────────────────────────────────


def test_gemini_source_sets_adjusted_everywhere() -> None:
    # Khoá hành vi: MỌI chỗ tạo TTSSegmentResult trong Gemini đều set adjusted
    # (điều kiện xuất SRT của use case).
    import pathlib

    source = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    n_results = source.count("TTSSegmentResult(")
    n_adjusted = source.count("adjusted_start_sec=")
    assert n_adjusted >= n_results, (
        f"Gemini: {n_results} chỗ tạo result nhưng chỉ {n_adjusted} set adjusted"
    )


def test_gemini_source_respects_allow_overlap() -> None:
    import pathlib

    source = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert "fit_limit_samples" in source          # tôn trọng chồng tiếng
    # [v3.23.215] Trần chất lượng nén nay nằm trong ``total_speed_ratio`` (hàm thuần
    # dùng chung — đồng thời áp base_speed đúng ngữ nghĩa). Tên cũ ``stretch_ratio_cap``
    # không còn được Gemini gọi trực tiếp.
    assert "total_speed_ratio" in source
