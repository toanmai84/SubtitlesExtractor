"""[v3.23.737] Tách tiếng của phim thành tệp nhỏ (mono 16 kHz Opus 24 kbps ≈ 10 MB/giờ) cho gói kiểm định chất lượng.

Đủ cho nhận dạng giọng nói (mô hình STT dùng 16 kHz mono). ffmpeg không có ``libopus`` → AAC 32 kbps (``.m4a``).
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

from subtitles_extractor.domain.ports.quality_audit_port import AudioExporterPort

logger = logging.getLogger(__name__)

__all__ = ["FfmpegAudioExporter", "audio_command"]


def audio_command(ffmpeg: str, video_path: Path, output_path: Path, codec: str) -> list[str]:
    """Lệnh ffmpeg tách tiếng (hàm thuần): ``codec`` = ``libopus`` hoặc ``aac``."""
    bitrate = "24k" if codec == "libopus" else "32k"
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-i", str(video_path),
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-c:a", codec, "-b:a", bitrate,
        str(output_path),
    ]


class FfmpegAudioExporter(AudioExporterPort):
    """Tách tiếng bằng ffmpeg (CPU — giải mã tiếng nhẹ, không cần GPU)."""

    def __init__(self, ffmpeg_finder: Callable[[], str | None] | None = None) -> None:
        if ffmpeg_finder is None:
            from subtitles_extractor.infrastructure.media.ffmpeg_locator import (
                find_ffmpeg,
            )

            ffmpeg_finder = find_ffmpeg
        self._find_ffmpeg = ffmpeg_finder

    def export(self, video_path: Path, output_path: Path) -> Path | None:
        from subtitles_extractor.infrastructure.process.hidden_process import run_hidden

        ffmpeg = self._find_ffmpeg()
        if not ffmpeg or not video_path.is_file():
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for codec, target in (("libopus", output_path), ("aac", output_path.with_suffix(".m4a"))):
            try:
                # communicate() đọc hết stdout/stderr — không thể kẹt vì ống đầy (bài học 3.23.735).
                result = run_hidden(
                    audio_command(ffmpeg, video_path, target, codec), capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=3600, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.warning("Không tách được tiếng phim (%s): %s", codec, exc)
                target.unlink(missing_ok=True)
                continue
            if result.returncode == 0 and target.is_file() and target.stat().st_size > 0:
                logger.info("Đã tách tiếng phim → %s (%.1f MB, %s).", target.name, target.stat().st_size / 1e6, codec)
                return target
            target.unlink(missing_ok=True)
            tail = (result.stderr or "").strip()[-300:]
            if "matches no streams" in tail or "does not contain any stream" in tail:
                logger.info("Phim không có luồng tiếng — bỏ qua tách tiếng.")
                return None
            logger.debug("ffmpeg tách tiếng (%s) lỗi %s: %s", codec, result.returncode, tail)
        logger.warning("Không tách được tiếng phim.")
        return None
