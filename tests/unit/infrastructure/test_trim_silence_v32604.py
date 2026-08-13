"""[v3.23.204] Test cắt im lặng đầu/cuối audio VieNeu vừa sinh.

Phát hiện từ đối chiếu FLAC thực với phụ đề (95 câu): VieNeu sinh im lặng ĐẦU câu rất
dài (median 160ms, 28 câu >300ms, max 1420ms — câu #3 phụ đề hiện 19.92s nhưng tiếng
vang ~21.3s). Hệ quả kép: tiếng TRỄ so với phụ đề/khẩu hình; im lặng chiếm khung đẩy
phần tiếng TRÀN sang câu sau (14.8s lãng phí/video 155s); stretch nén mạnh hơn cần.
Fix: ``trim_edge_silence`` cắt biên ngay sau tổng hợp (giữ đệm nhỏ), pause hội thoại
có kiểm soát chèn SAU trim.
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    trim_edge_silence,
)

_SR = 24000


def _voice(duration_s: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(_SR * duration_s)) / _SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(_SR * duration_s), dtype=np.float32)


def test_leading_silence_trimmed() -> None:
    # 1.4s im lặng + 1.0s tiếng (ca thực tế câu #3) -> đầu ra tiếng vang gần như ngay.
    audio = np.concatenate([_silence(1.4), _voice(1.0)])
    out = trim_edge_silence(audio, _SR)
    assert len(out) / _SR < 1.0 + 0.15  # còn ~tiếng + đệm nhỏ
    head_rms = float(np.sqrt((out[: int(0.15 * _SR)] ** 2).mean()))
    assert head_rms > 0.05  # tiếng ngay đầu (trước fix: im lặng)


def test_trailing_silence_trimmed() -> None:
    audio = np.concatenate([_voice(1.0), _silence(1.0)])
    out = trim_edge_silence(audio, _SR)
    assert len(out) / _SR < 1.0 + 0.15


def test_keeps_small_head_padding() -> None:
    # Đệm đầu ~50ms được giữ cho vào nhịp tự nhiên (không cắt sát rạt).
    audio = np.concatenate([_silence(0.5), _voice(1.0)])
    out = trim_edge_silence(audio, _SR, keep_head_s=0.05)
    lead = 0
    win = int(0.01 * _SR)
    while lead + win < len(out):
        if float(np.sqrt((out[lead : lead + win] ** 2).mean())) > 0.008:
            break
        lead += win
    assert 0.0 <= lead / _SR <= 0.08  # còn đệm nhỏ, không quá 80ms


def test_all_silence_returned_unchanged() -> None:
    sil = _silence(1.0)
    assert trim_edge_silence(sil, _SR) is sil  # để retry/skip quyết định


def test_clean_audio_nearly_untouched() -> None:
    clean = _voice(1.0)
    out = trim_edge_silence(clean, _SR)
    assert abs(len(out) - len(clean)) / _SR < 0.1


def test_empty_audio_safe() -> None:
    empty = np.zeros(0, dtype=np.float32)
    assert trim_edge_silence(empty, _SR) is empty


def test_process_event_trims_before_pause() -> None:
    # Khoá thứ tự: trim NGAY sau synth, TRƯỚC dialog pause (pause có kiểm soát không
    # bị trim ngược).
    import pathlib

    source = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/vieneu_tts_adapter.py"
    ).read_text(encoding="utf-8")
    # [v3.23.217] Pause nay được chèn SAU khi nén (không bị bóp), nhưng trim vẫn phải
    # chạy TRƯỚC để không cắt nhầm khoảng lặng có chủ đích đó.
    # [v3.23.241] trim nay bật adaptive=True (ngưỡng tự dò) — thứ tự vẫn phải là
    # trim TRƯỚC pause.
    trim_pos = source.index("trim_edge_silence(audio, sr, adaptive=True)")
    pause_pos = source.index("if pause_s > 0.0:")
    assert trim_pos < pause_pos
