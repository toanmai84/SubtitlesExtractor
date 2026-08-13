"""[v3.23.261] Khóa hành vi Edge TTS + librosa sau điều tra với thư viện thật.

Cài edge-tts 7.2.8, librosa 0.11.0, pydub, soundfile 0.14 trong sandbox và nội soi API.
KHÔNG tìm thấy bug — app phòng thủ tốt. Các test dưới KHÓA những bất biến đã xác minh để
tránh hồi quy khi nâng cấp thư viện:

- ``_rate_from_speed`` tạo đúng định dạng edge-tts yêu cầu ("+X%" / "-X%").
- ``vocal_time_stretch`` cho độ dài chính xác qua cả 3 nhánh (pedalboard/librosa/WSOLA).
- librosa import LAZY (trong hàm) — tránh segfault numba+Qt và tải nặng khi không cần.
"""

from __future__ import annotations

import pathlib

import numpy as np

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import _rate_from_speed
from subtitles_extractor.infrastructure.tts.time_stretch import vocal_time_stretch


def test_rate_string_định_dạng_edge_tts() -> None:
    # edge-tts yêu cầu rate là "+X%" hoặc "-X%". Xác minh mọi tốc độ ra đúng định dạng.
    assert _rate_from_speed(1.0) == "+0%"
    assert _rate_from_speed(1.3).startswith("+") and _rate_from_speed(1.3).endswith("%")
    assert _rate_from_speed(0.8).startswith("-") and _rate_from_speed(0.8).endswith("%")


def test_rate_string_luôn_hợp_lệ() -> None:
    # Mọi tốc độ trong dải thực tế -> chuỗi parse được (không ký tự lạ).
    for speed in (0.1, 0.5, 1.0, 1.5, 2.0, 3.0):
        rate = _rate_from_speed(speed)
        assert rate[-1] == "%"
        assert rate[0] in "+-"
        # phần số hợp lệ
        int(rate[:-1])  # không ném ValueError


def test_vocal_stretch_nén_nhẹ_độ_dài_đúng() -> None:
    sr = 24_000
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    out = vocal_time_stretch(audio, sr, 1.3)
    expected = int(len(audio) / 1.3)
    assert abs(len(out) - expected) / expected < 0.05


def test_vocal_stretch_nén_mạnh_độ_dài_đúng() -> None:
    # ratio > 2.0 -> nhánh librosa (phase-vocoder).
    sr = 24_000
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    out = vocal_time_stretch(audio, sr, 2.5)
    expected = int(len(audio) / 2.5)
    assert abs(len(out) - expected) / expected < 0.05


def test_vocal_stretch_giãn_mạnh_độ_dài_đúng() -> None:
    # ratio < 0.5 -> nhánh librosa (giãn dài).
    sr = 24_000
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    out = vocal_time_stretch(audio, sr, 0.4)
    expected = int(len(audio) / 0.4)
    assert abs(len(out) - expected) / expected < 0.05


def test_librosa_không_import_top_level() -> None:
    # librosa phải import LAZY (trong hàm) — tránh segfault numba+Qt + tải nặng vô ích.
    for mod in ("edge_tts_adapter.py", "time_stretch.py"):
        src = pathlib.Path(
            f"src/subtitles_extractor/infrastructure/tts/{mod}"
        ).read_text(encoding="utf-8")
        for line in src.splitlines():
            stripped = line.lstrip()
            # import librosa ở đầu dòng (không thụt) = top-level -> cấm.
            if stripped.startswith("import librosa") and line == stripped:
                raise AssertionError(f"{mod} import librosa top-level (rủi ro segfault)")
