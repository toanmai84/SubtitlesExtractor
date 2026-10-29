# ruff: noqa: RUF002, RUF003 — dấu × (nhân) trong số đo là CHỦ ĐÍCH
"""[v3.23.601] Lưới "dài bất thường" đo PHẦN TIẾNG; tóm tắt TTS chỉ đếm chồng tiếng thật.

Số liệu thật (chuẩn đoán 灭门惨案II, VieNeu v3 Turbo, 2 giọng, 18:05 ngày 10/10):

* Nhật ký: 27 lần lấy mẫu lại "audio DÀI BẤT THƯỜNG 2.16-2.88s" cho 11 dòng 1-3 âm tiết
  (đều giọng Ngọc Huyền: 'Cảm ơn.', 'Đừng!', 'Đúng thế.'…). Lưới đo độ dài THÔ, trong khi
  khâu đặt câu cắt lặng biên ngay sau đó; bản được giữ của chính các dòng ấy có 0.57-2.08s
  tiếng (cột "Thời lượng audio" × tốc độ trong CSV), dưới ngưỡng 1.8-2.1s.
* Dòng 991 "Đúng thế.": cả 5 lần thử đều 2.16s TIẾNG (3.4 lần mức đọc gọn) → nén 2.5x còn
  0.864s; cờ "dài bất thường" của bảng đo audio SAU nén nên không bật.
* Tóm tắt "Có chồng tiếng (lấn): 113" nhưng CSV chỉ có 7 dòng lấn ≥ 1 ms (tổng 1.16s);
  lượt trước 125 so với 11. Phần còn lại là dư vài mẫu do làm tròn sau nén.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSRequest,
    TTSSegmentResult,
)
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.audio_utils import (
    shorter_take,
    speech_seconds,
)
from subtitles_extractor.infrastructure.tts.timing_math import audible_overlap_seconds
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter

_ENGINE_SR = 48_000
_PIPELINE_SR = 24_000

#: Độ dài thô của lần thử "Đúng thế." trong nhật ký thật.
_LOGGED_RAW_S = 2.16
#: Phần tiếng của bản được giữ cho "Đúng thế." dòng 718: 0.569s × 1.3.
_KEPT_SPEECH_S = 0.74
#: Im lặng đầu câu VieNeu dài nhất đã đo (v3.23.204).
_MAX_LEADING_SILENCE_S = 1.42


def _voice(seconds: float, sr: int) -> np.ndarray:
    """Tín hiệu có tiếng (biên độ đổi theo âm tiết) dài ``seconds``."""
    t = np.arange(int(sr * seconds)) / sr
    envelope = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3.0 * t))
    return (0.3 * envelope * np.sin(2 * np.pi * 190.0 * t)).astype(np.float32)


def _take(speech_s: float, raw_s: float, sr: int = _ENGINE_SR) -> np.ndarray:
    """Bản thu thô ``raw_s`` giây: im lặng đầu (dài nhất đã đo), tiếng, lặng cuối."""
    lead = min(_MAX_LEADING_SILENCE_S, raw_s - speech_s)
    tail = raw_s - speech_s - lead
    rng = np.random.default_rng(7)
    parts = [
        (0.0005 * rng.standard_normal(int(sr * lead))).astype(np.float32),
        _voice(speech_s, sr),
        (0.0005 * rng.standard_normal(int(sr * tail))).astype(np.float32),
    ]
    return np.concatenate(parts)


class _Engine:
    sample_rate = _ENGINE_SR


class _Request:
    retry_count = 10
    retry_delay_s = 0.0


def _adapter() -> VieNeuTtsAdapter:
    adapter = object.__new__(VieNeuTtsAdapter)
    adapter._batch_size = 0
    adapter._batch_texts = []
    adapter._batch_cache = {}
    adapter._force_cpu = True
    adapter._gpu_worker_disabled = True
    return adapter


def _retry(
    monkeypatch: pytest.MonkeyPatch, takes: list[np.ndarray], text: str
) -> tuple[np.ndarray | None, int]:
    adapter = _adapter()
    calls = {"n": 0}

    def fake_infer_once(engine, text, voice_data, idx=-1, **_kw):
        take = takes[min(calls["n"], len(takes) - 1)]
        calls["n"] += 1
        return take

    monkeypatch.setattr(adapter, "_infer_once", fake_infer_once)
    result = adapter._synthesize_with_retry(
        engine=_Engine(), text=text, request=_Request(),
        voice_data={}, cancel_cb=None, idx=0,
    )
    return result, calls["n"]


# ── Đo phần tiếng ────────────────────────────────────────────────────────────


def test_speech_seconds_bo_im_lang_bien() -> None:
    take = _take(_KEPT_SPEECH_S, _LOGGED_RAW_S)
    assert take.size / _ENGINE_SR == pytest.approx(_LOGGED_RAW_S, abs=0.01)
    # Toàn bộ 1.42s im lặng đầu bị cắt, chỉ còn tiếng + đệm 50 ms vào câu.
    assert speech_seconds(take, _ENGINE_SR) == pytest.approx(
        _KEPT_SPEECH_S + 0.05, abs=0.03
    )


def test_speech_seconds_sr_khong_hop_le() -> None:
    assert speech_seconds(_voice(1.0, _ENGINE_SR), 0) == 0.0


def test_shorter_take_theo_phan_tieng() -> None:
    # Bản thô DÀI hơn nhưng ít tiếng hơn mới là bản gọn.
    long_raw_short_voice = _take(1.9, 2.88)
    short_raw_long_voice = _take(2.3, 2.40)
    assert shorter_take(
        short_raw_long_voice, long_raw_short_voice, sr=_ENGINE_SR
    ) is long_raw_short_voice
    # Không truyền sr: giữ cách so độ dài thô như cũ.
    kept = shorter_take(short_raw_long_voice, long_raw_short_voice)
    assert kept is short_raw_long_voice


# ── Lấy mẫu lại ──────────────────────────────────────────────────────────────


def test_ban_tho_dai_vi_im_lang_khong_bi_lay_mau_lai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    take = _take(_KEPT_SPEECH_S, _LOGGED_RAW_S)
    result, calls = _retry(monkeypatch, [take], "Đúng thế.")
    assert calls == 1  # v3.23.600: 5 lần (lưới đo 2.16s thô > ngưỡng 1.8s)
    assert result is take


def test_ngan_dai_that_van_bi_lay_mau_lai(monkeypatch: pytest.MonkeyPatch) -> None:
    # Dòng 991: 2.16s là TIẾNG (không có im lặng) → vẫn phải lấy mẫu lại.
    ngan = _voice(_LOGGED_RAW_S, _ENGINE_SR)
    gon = _take(_KEPT_SPEECH_S, 1.2)
    result, calls = _retry(monkeypatch, [ngan, gon], "Đúng thế.")
    assert calls == 2
    assert result is gon


def test_moi_lan_deu_ngan_thi_giu_ban_it_tieng_nhat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    it_tieng = _take(1.9, 2.88)  # thô dài nhất, tiếng ngắn nhất
    takes = [_voice(2.32, _ENGINE_SR), it_tieng, _voice(2.24, _ENGINE_SR)]
    result, _calls = _retry(monkeypatch, takes, "Đúng thế.")
    assert result is it_tieng


# ── Cờ "dài bất thường" của bảng ─────────────────────────────────────────────


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class _FixedEngine:
    """Engine giả: mỗi câu trả một bản thu cố định."""

    def __init__(self, takes: dict[str, np.ndarray]) -> None:
        self._takes = takes

    def list_preset_voices(self) -> list[tuple[str, str]]:
        return [("Giọng A", "a")]

    def get_preset_voice(self, voice_id: str) -> dict[str, str]:
        return {"id": voice_id}

    def infer(self, text: str, voice: object = None) -> np.ndarray:
        return self._takes[text]


@pytest.fixture()
def _no_engine_cache() -> None:
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


def _generate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    takes: dict[str, np.ndarray],
    events: list[_Event],
) -> list[TTSSegmentResult]:
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = _FixedEngine(takes)
    adapter._force_cpu = False
    request = TTSRequest(
        events=events, speaker="a", normalize=False, base_speed=1.3, max_speed=2.5,
        retry_count=3,
    )
    return adapter.generate(request, tmp_path / "out")


@pytest.mark.usefixtures("_no_engine_cache")
def test_bang_bao_dong_dong_991_bi_nen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    # Khung thật dòng 991 (4310.933-4311.600) và câu kế ngay sau.
    takes = {
        "Đúng thế.": _voice(_LOGGED_RAW_S, _PIPELINE_SR),
        "Cậu nói tiếp đi.": _voice(1.0, _PIPELINE_SR),
    }
    events = [_Event(0.0, 0.667, "Đúng thế."), _Event(0.70, 2.5, "Cậu nói tiếp đi.")]
    results = _generate(monkeypatch, tmp_path, takes, events)
    assert results[0].speed_used > 1.3  # bị nén để vừa khung
    assert results[0].audio_duration_s < 1.8  # audio cuối đã nén lọt dưới ngưỡng
    assert results[0].abnormal_length is True  # v3.23.600: False
    assert results[1].abnormal_length is False


# ── Chồng tiếng thật ─────────────────────────────────────────────────────────


def test_audible_overlap_bo_du_lam_tron() -> None:
    assert audible_overlap_seconds(0.864, 0.683) == pytest.approx(0.181)
    one_sample = 1 / _PIPELINE_SR
    assert audible_overlap_seconds(1.0 + one_sample, 1.0) == 0.0
    assert audible_overlap_seconds(1.001, 1.0) == pytest.approx(0.001)
    assert audible_overlap_seconds(0.9, 1.0) == 0.0


def test_result_chi_dem_chong_tieng_tu_1ms() -> None:
    def result(overlap_s: float) -> TTSSegmentResult:
        return TTSSegmentResult(
            event_index=0, start_sec=0.0, end_sec=1.0, text="x", overlap_s=overlap_s
        )

    # Bảy giá trị lấn thật trong CSV + phần dư làm tròn (hiện 0.000 trong CSV).
    real = [0.446, 0.215, 0.181, 0.102, 0.097, 0.07, 0.049]
    noise = [3.3e-15, 5.4e-07, 4.6e-06, 9.8e-06, 4.2e-05]
    counted = [r for r in map(result, real + noise) if r.has_audible_overlap]
    assert len(counted) == 7
    assert sum(r.overlap_s for r in counted) == pytest.approx(1.16)


@pytest.mark.usefixtures("_no_engine_cache")
def test_vieneu_khong_bao_lan_vi_du_lam_tron(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    logging.disable(logging.WARNING)
    try:
        rng = np.random.default_rng(1)
        texts = [f"Câu số {i} ở đây" for i in range(60)]
        takes = {
            text: _voice(float(rng.uniform(0.6, 2.6)), _PIPELINE_SR) for text in texts
        }
        events: list[_Event] = []
        start = 0.0
        for text in texts:
            width = float(rng.uniform(0.5, 1.6))
            events.append(_Event(start, start + width, text))
            start += width + float(rng.choice([0.0, 0.03, 0.1, 0.4]))
        results = _generate(monkeypatch, tmp_path, takes, events)
    finally:
        logging.disable(logging.NOTSET)
    positive = [r.overlap_s for r in results if r.overlap_s > 0]
    # v3.23.600: 14 dòng "lấn" > 0, trong đó 13 dòng dưới 1 ms.
    assert positive and all(value >= 0.001 for value in positive)
