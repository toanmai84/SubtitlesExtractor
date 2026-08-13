"""[v3.23.173] Test MAX SQUEEZE stretch RIÊNG phần giọng, giữ nguyên nghỉ hội thoại.

Bug: MAX SQUEEZE stretch CẢ khối ``[nghỉ im lặng] + [giọng]`` cùng một tỉ lệ khi câu
vượt khung. Hệ quả: (1) khoảng nghỉ hội thoại bị CO lại sai độ dài; (2) stretch một
đoạn IM LẶNG là vô ích và có thể tạo artifact ở ranh giới nghỉ->giọng; (3) công thức
``new_speech = audio_dur - pause_dur`` dùng pause GỐC (chưa co) -> speed_used báo cáo
sai. Fix: hàm thuần ``_speech_only_stretch_ratio`` tính tỉ lệ riêng cho phần giọng để
tổng (nghỉ giữ nguyên + giọng nén) khớp đích.
"""

from __future__ import annotations

import math

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _speech_only_stretch_ratio,
)


def test_ratio_keeps_pause_and_hits_target() -> None:
    """Nghỉ giữ nguyên, giọng nén sao cho TỔNG khớp đích."""
    ratio = _speech_only_stretch_ratio(
        total_dur=1.5, pause_dur=0.1, target_total_dur=1.2
    )
    speech_dur = 1.5 - 0.1
    resulting_total = speech_dur / ratio + 0.1
    assert math.isclose(resulting_total, 1.2, abs_tol=1e-6)


def test_no_pause_equals_whole_block_ratio() -> None:
    """Không có nghỉ -> tỉ lệ đúng bằng tổng/đích như khối đơn."""
    ratio = _speech_only_stretch_ratio(1.5, 0.0, 1.2)
    assert math.isclose(ratio, 1.25, abs_tol=1e-9)


def test_no_compression_needed_returns_one() -> None:
    """Đích rộng hơn hiện tại -> không nén (tỉ lệ 1.0)."""
    assert _speech_only_stretch_ratio(1.0, 0.1, 1.5) == 1.0


def test_degenerate_speech_returns_one() -> None:
    """Phần giọng ~0 (toàn nghỉ) -> không stretch."""
    assert _speech_only_stretch_ratio(0.1, 0.1, 0.05) == 1.0


def test_degenerate_target_returns_one() -> None:
    """Đích nhỏ hơn cả nghỉ -> không thể nén hợp lệ -> 1.0."""
    assert _speech_only_stretch_ratio(1.5, 0.2, 0.15) == 1.0


def test_ratio_larger_than_whole_block_ratio() -> None:
    """Vì nghỉ chiếm chỗ cố định, giọng phải nén MẠNH hơn tỉ lệ khối -> ratio lớn hơn."""
    whole_block_ratio = 1.5 / 1.2
    speech_ratio = _speech_only_stretch_ratio(1.5, 0.1, 1.2)
    assert speech_ratio > whole_block_ratio
