"""[v3.23.164] Test độ dài master track đủ chỗ đuôi câu cuối (chống cắt cụt chữ cuối phim).

Bug: master cấp phát ``last_end + 1.0s`` (khi không elastic) -> câu cuối sau khi cộng
dialog_pause + phần nới cuối + audio dài có thể vượt biên -> slice ``min(es, len(master))``
CẮT CỤT ÂM THẦM (mất chữ cuối phim) mà không đánh dấu was_truncated. Fix: tính đệm đuôi
theo đúng các thành phần có thể kéo dài câu cuối.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _master_track_length_samples,
)

_SR = 24000


def test_includes_dialog_pause_and_extend() -> None:
    # last_end 10s, pause 0.3s, nới cuối 3.0s -> đệm = 1.0 + 0.3 + 3.0 = 4.3s.
    length = _master_track_length_samples(
        last_end_s=10.0, sample_rate=_SR, dialog_pause_s=0.3,
        last_line_extend_s=3.0, drift_s=0.0, extra_tail_s=0.0,
    )
    assert length == int((10.0 + 4.3) * _SR)


def test_elastic_adds_drift_and_extra_tail() -> None:
    length = _master_track_length_samples(
        last_end_s=10.0, sample_rate=_SR, dialog_pause_s=0.0,
        last_line_extend_s=0.0, drift_s=1.5, extra_tail_s=3.0,
    )
    assert length == int((10.0 + 1.0 + 1.5 + 3.0) * _SR)


def test_longer_than_naive_buffer_when_pause_present() -> None:
    # Bằng chứng bug: đệm cũ chỉ 1.0s bỏ qua pause + nới cuối.
    naive = int((10.0 + 1.0) * _SR)
    fixed = _master_track_length_samples(
        last_end_s=10.0, sample_rate=_SR, dialog_pause_s=0.3,
        last_line_extend_s=3.0, drift_s=0.0, extra_tail_s=0.0,
    )
    assert fixed > naive


def test_negative_values_clamped() -> None:
    length = _master_track_length_samples(
        last_end_s=5.0, sample_rate=_SR, dialog_pause_s=0.0,
        last_line_extend_s=-2.0, drift_s=-1.0, extra_tail_s=-1.0,
    )
    assert length == int((5.0 + 1.0) * _SR)  # phần âm bị kẹp về 0


def test_minimum_one_sample() -> None:
    length = _master_track_length_samples(
        last_end_s=0.0, sample_rate=_SR, dialog_pause_s=0.0,
        last_line_extend_s=0.0, drift_s=0.0, extra_tail_s=0.0,
        base_tail_s=0.0,
    )
    assert length >= 1
