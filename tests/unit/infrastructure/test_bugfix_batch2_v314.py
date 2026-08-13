"""Unit test cho các mục đợt 2 kiểm thử được (v3.14.7): #3 clamp, #21 temp-file."""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.presentation.utils.overlay_geometry import (
    SAFE_RENDER_MARGIN_PX,
    clamp_box_coords,
)


class TestClampBoxCoords:
    def test_normal_box_unchanged(self) -> None:
        assert clamp_box_coords(10, 20, 100, 50, 1920, 1080) == (10, 20, 100, 50)

    def test_giant_box_clamped(self) -> None:
        x, y, w, h = clamp_box_coords(0, 0, 5_000_000, 5_000_000, 1920, 1080)
        assert w <= 1920 + 2 * SAFE_RENDER_MARGIN_PX
        assert h <= 1080 + 2 * SAFE_RENDER_MARGIN_PX

    def test_negative_origin_clamped(self) -> None:
        x, y, w, h = clamp_box_coords(-9_000_000, -9_000_000, 100, 100, 1920, 1080)
        assert x >= -SAFE_RENDER_MARGIN_PX and y >= -SAFE_RENDER_MARGIN_PX
        assert w >= 0 and h >= 0

    def test_width_never_negative(self) -> None:
        _, _, w, h = clamp_box_coords(10_000_000, 10_000_000, -50, -50, 1920, 1080)
        assert w >= 0 and h >= 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="Cần ffmpeg để test encode.")
class TestFfmpegTempFileEncode:
    def test_encode_produces_nonempty_file(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.tts.audio_mastering import (
            _encode_with_ffmpeg,
        )

        sr = 22050
        tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        target = tmp_path / "out.mp3"
        ok = _encode_with_ffmpeg(tone, sr, target, "mp3", 128)
        if not ok:
            pytest.skip("ffmpeg build thiếu libmp3lame trong môi trường test.")
        assert target.exists() and target.stat().st_size > 0

    def test_temp_wav_cleaned_up(self, tmp_path: Path, monkeypatch) -> None:
        # Xác nhận không để lại file .wav tạm sau khi encode (dù thành/bại).
        import tempfile

        created: list[str] = []
        real_mkstemp = tempfile.mkstemp

        def tracking_mkstemp(*args, **kwargs):
            fd, path = real_mkstemp(*args, **kwargs)
            created.append(path)
            return fd, path

        monkeypatch.setattr(tempfile, "mkstemp", tracking_mkstemp)
        from subtitles_extractor.infrastructure.tts.audio_mastering import (
            _encode_with_ffmpeg,
        )

        sr = 16000
        tone = (0.1 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
        _encode_with_ffmpeg(tone, sr, tmp_path / "x.mp3", "mp3", 96)
        for tmp_file in created:
            assert not Path(tmp_file).exists(), "File WAV tạm phải được dọn sạch"
