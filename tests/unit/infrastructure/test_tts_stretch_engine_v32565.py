"""[v3.23.165] Test time-stretch chọn engine đối xứng + đúng độ dài mọi ratio.

Bug: guard dùng librosa (phase-vocoder, chất lượng cao) CHỈ cho chiều nén mạnh
(ratio > 2.0). Từ v162 Pass 2.5 sinh cả yêu cầu GIÃN mạnh (ratio < 0.5, câu ngắn hơn
khung) -> chiều giãn mạnh rơi vào WSOLA (chất lượng kém khi biến đổi lớn). Fix: guard
đối xứng ratio > 2.0 HOẶC ratio < 0.5. Test kiểm cả bất biến độ dài (fallback WSOLA khi
môi trường không có librosa vẫn phải cho đúng thời lượng mục tiêu).
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter

_SR = 24000


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(_SR * seconds))
    return np.sin(2 * np.pi * 220 * t / _SR).astype(np.float32)


def test_no_op_for_near_unity_ratio() -> None:
    audio = _tone(1.0)
    result = EdgeTTSAdapter._time_stretch_vocal(audio, _SR, 1.005)
    assert result is audio  # bỏ qua biến đổi < 2%


def test_compress_length_correct() -> None:
    audio = _tone(1.0)
    for ratio in (1.5, 2.5, 3.0):
        out = EdgeTTSAdapter._time_stretch_vocal(audio, _SR, ratio)
        assert abs(len(out) - len(audio) / ratio) <= _SR * 0.02


def test_stretch_length_correct_both_mild_and_strong() -> None:
    # Chiều GIÃN: vừa (0.8) và MẠNH (0.4, 0.05) — độ dài phải đúng target = len/ratio.
    audio = _tone(1.0)
    for ratio in (0.8, 0.4, 0.05):
        out = EdgeTTSAdapter._time_stretch_vocal(audio, _SR, ratio)
        assert abs(len(out) - len(audio) / ratio) <= _SR * 0.02


def test_strong_stretch_selects_librosa_branch(monkeypatch) -> None:
    # [v3.23.197] pedalboard được ƯU TIÊN trước librosa; test này kiểm FALLBACK: khi
    # pedalboard vắng mặt, ratio mạnh (<0.5 hoặc >2.0) phải ĐI VÀO nhánh librosa.
    # [v3.23.260] Sau refactor v220, _time_stretch_vocal uỷ cho time_stretch.
    # vocal_time_stretch -> phải mock stretch_with_pedalboard TRONG module time_stretch
    # (không phải symbol cũ trong edge_tts_adapter). Test cũ mock sai chỗ nên lỗi thời khi
    # pedalboard thật được cài.
    from subtitles_extractor.infrastructure.tts import time_stretch as ts_mod

    monkeypatch.setattr(ts_mod, "stretch_with_pedalboard", lambda *a, **k: None)
    calls: list[float] = []

    class _FakeLibrosa:
        class effects:  # noqa: N801 — nhái namespace librosa.effects
            @staticmethod
            def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
                calls.append(rate)
                new_len = max(1, int(len(audio) / rate))
                return np.zeros(new_len, dtype=np.float32)

    import sys

    monkeypatch.setitem(sys.modules, "librosa", _FakeLibrosa)
    audio = _tone(1.0)
    EdgeTTSAdapter._time_stretch_vocal(audio, _SR, 0.4)
    EdgeTTSAdapter._time_stretch_vocal(audio, _SR, 3.0)
    assert 0.4 in calls  # chiều giãn mạnh nay dùng librosa (trước đây KHÔNG)
    assert 3.0 in calls  # chiều nén mạnh vẫn dùng librosa


def test_mild_ratio_uses_wsola_not_librosa(monkeypatch) -> None:
    calls: list[float] = []

    class _FakeLibrosa:
        class effects:  # noqa: N801
            @staticmethod
            def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
                calls.append(rate)
                return audio

    import sys

    monkeypatch.setitem(sys.modules, "librosa", _FakeLibrosa)
    audio = _tone(1.0)
    EdgeTTSAdapter._time_stretch_vocal(audio, _SR, 1.5)  # nén vừa
    EdgeTTSAdapter._time_stretch_vocal(audio, _SR, 0.8)  # giãn vừa
    assert calls == []  # biến đổi vừa KHÔNG gọi librosa (giữ pitch bằng WSOLA)
