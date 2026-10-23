"""[v3.23.618] Chọn bộ mã hoá hình (GPU nếu máy hỗ trợ) + tham số chất lượng tương ứng.

Dùng chung cho khâu Nối cả bộ (chuẩn hoá các tập lệch thông số) và khâu Xuất bản (lựa chọn
"Tự động (GPU nếu có)").

Vì sao phải THỬ MÃ HOÁ thật thay vì đọc ``ffmpeg -encoders``: bản ffmpeg đóng gói (BtbN
gpl-shared) luôn liệt kê ``h264_nvenc``/``hevc_qsv``/``hevc_amf``… kể cả trên máy không có
GPU tương ứng — chỉ khi mở phiên mã hoá mới biết có dùng được không. Trước đây lựa chọn
"Tự động (GPU nếu có)" ở trang Xuất bản thực chất luôn là ``h264_nvenc``: máy không có GPU
NVIDIA xuất phụ đề cháy là lỗi ngay.

Thứ tự ưu tiên: NVIDIA (NVENC) → Intel (QSV) → AMD (AMF) → Apple (VideoToolbox) → CPU.
Kết quả dò được nhớ theo (ffmpeg, codec) trong suốt phiên chạy.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Final

from subtitles_extractor.infrastructure.process.hidden_process import no_window_kwargs

logger = logging.getLogger(__name__)

#: Bộ mã hoá phần cứng theo thứ tự ưu tiên, kết thúc bằng bộ mã hoá CPU.
ENCODER_CANDIDATES: Final[dict[str, tuple[str, ...]]] = {
    "hevc": ("hevc_nvenc", "hevc_qsv", "hevc_amf", "hevc_videotoolbox", "libx265"),
    "h264": ("h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox", "libx264"),
}

#: Bộ mã hoá CPU cuối cùng của mỗi codec (luôn có trong bản ffmpeg GPL).
SOFTWARE_ENCODERS: Final[dict[str, str]] = {"hevc": "libx265", "h264": "libx264"}

#: Chất lượng mặc định trên thang CRF của x264 (thấp hơn = nét hơn).
DEFAULT_QUALITY: Final[int] = 20

#: HEVC cùng chất lượng hình với H.264 ở CRF cao hơn ~2 (đo trên phim mẫu 720x1280:
#: x264 crf 20 SSIM 0.9918 / 14.4 MB; x265 fast crf 22 SSIM 0.9896 / 8.0 MB).
_HEVC_QUALITY_OFFSET: Final[int] = 2

_PROBE_TIMEOUT_SEC: Final[float] = 20.0

_cache: dict[tuple[str, str], str] = {}
_broken: set[tuple[str, str]] = set()
_lock = threading.Lock()


def is_hardware_encoder(encoder: str) -> bool:
    """``True`` với bộ mã hoá GPU (nvenc/qsv/amf/videotoolbox)."""
    return encoder.endswith(("_nvenc", "_qsv", "_amf", "_videotoolbox"))


def encoder_codec(encoder: str) -> str:
    """Codec của bộ mã hoá: ``"hevc"`` hoặc ``"h264"``."""
    return "hevc" if encoder.startswith("hevc") or encoder == "libx265" else "h264"


def _platform_candidates(codec: str) -> tuple[str, ...]:
    """Bỏ ứng viên không thể có trên hệ điều hành hiện tại (đỡ tốn thời gian thử)."""
    candidates = ENCODER_CANDIDATES[codec]
    if sys.platform == "darwin":
        return tuple(c for c in candidates if c.endswith("_videotoolbox") or c.startswith("lib"))
    return tuple(c for c in candidates if not c.endswith("_videotoolbox"))


def probe_encoder(ffmpeg: str, encoder: str) -> bool:
    """Thử mã hoá vài khung đen bằng ``encoder``; ``True`` nếu chạy được thật."""
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi", "-i", "color=c=black:s=320x320:r=25:d=0.2",
        "-frames:v", "3", "-pix_fmt", "yuv420p", "-c:v", encoder, "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_PROBE_TIMEOUT_SEC, stdin=subprocess.DEVNULL, **no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Thử bộ mã hoá %s lỗi: %s", encoder, exc)
        return False
    if result.returncode != 0:
        logger.debug(
            "Bộ mã hoá %s không dùng được: %s", encoder, (result.stderr or "").strip()[-200:]
        )
        return False
    return True


def pick_video_encoder(ffmpeg: str, codec: str = "hevc", *, allow_hardware: bool = True) -> str:
    """Bộ mã hoá tốt nhất dùng được cho ``codec`` (``"hevc"``/``"h264"``).

    GPU trước, rồi CPU. Nếu không có cả ``libx265`` (ffmpeg bản LGPL) thì lùi về chuỗi
    H.264. Luôn trả về một tên bộ mã hoá — cuối cùng là ``libx264``.
    """
    if codec not in ENCODER_CANDIDATES:
        raise ValueError(f"Codec không hỗ trợ: {codec}")
    key = (ffmpeg, f"{codec}:{'hw' if allow_hardware else 'sw'}")
    with _lock:
        cached = _cache.get(key)
    if cached is not None and (ffmpeg, cached) not in _broken:
        return cached

    chosen = ""
    for encoder in _platform_candidates(codec):
        if not allow_hardware and is_hardware_encoder(encoder):
            continue
        with _lock:
            if (ffmpeg, encoder) in _broken:
                continue
        if probe_encoder(ffmpeg, encoder):
            chosen = encoder
            break
    if not chosen:
        chosen = (
            pick_video_encoder(ffmpeg, "h264", allow_hardware=allow_hardware)
            if codec == "hevc" else "libx264"
        )
    logger.info("Bộ mã hoá hình cho %s: %s", codec.upper(), chosen)
    with _lock:
        _cache[key] = chosen
    return chosen


def mark_encoder_broken(ffmpeg: str, encoder: str) -> None:
    """Ghi nhận bộ mã hoá vừa lỗi khi chạy thật (vd hết phiên NVENC) để lần chọn sau bỏ qua."""
    with _lock:
        _broken.add((ffmpeg, encoder))
        for key in [k for k, v in _cache.items() if k[0] == ffmpeg and v == encoder]:
            del _cache[key]


def software_fallback(encoder: str) -> str:
    """Bộ mã hoá CPU cùng codec để chạy lại khi bộ mã hoá GPU lỗi giữa chừng."""
    return SOFTWARE_ENCODERS[encoder_codec(encoder)]


def encoder_quality_args(
    encoder: str, quality: int = DEFAULT_QUALITY, *, output_path: Path | None = None
) -> list[str]:
    """Tham số chất lượng cho ``encoder`` từ một mức chất lượng trên thang CRF của x264.

    Mỗi họ bộ mã hoá dùng thang riêng (``-crf``, ``-cq``, ``-global_quality``, ``-qp_*``,
    ``-q:v``); hàm này quy đổi để cùng một mức cho chất lượng tương đương. HEVC trong
    MP4/MOV được gắn thẻ ``hvc1`` để QuickTime/iPhone phát được.
    """
    is_hevc = encoder_codec(encoder) == "hevc"
    level = quality + (_HEVC_QUALITY_OFFSET if is_hevc else 0)
    if encoder.endswith("_nvenc"):
        args = ["-preset", "p5", "-rc", "vbr", "-cq", str(level), "-b:v", "0"]
    elif encoder.endswith("_qsv"):
        args = ["-preset", "medium", "-global_quality", str(level)]
    elif encoder.endswith("_amf"):
        args = ["-quality", "quality", "-rc", "cqp",
                "-qp_i", str(level), "-qp_p", str(level + 2)]
    elif encoder.endswith("_videotoolbox"):
        args = ["-q:v", str(max(1, min(100, 105 - 2 * level)))]
    elif encoder == "libx265":
        args = ["-preset", "fast", "-crf", str(level), "-x265-params", "log-level=error"]
    else:
        args = ["-preset", "medium", "-crf", str(level)]
    if is_hevc and output_path is not None and output_path.suffix.lower() in {
        ".mp4", ".m4v", ".mov"
    }:
        args += ["-tag:v", "hvc1"]
    return args


def reset_encoder_cache() -> None:
    """Xoá kết quả dò (dùng trong test hoặc khi đổi bản ffmpeg)."""
    with _lock:
        _cache.clear()
        _broken.clear()


__all__ = [
    "DEFAULT_QUALITY",
    "ENCODER_CANDIDATES",
    "encoder_codec",
    "encoder_quality_args",
    "is_hardware_encoder",
    "mark_encoder_broken",
    "pick_video_encoder",
    "probe_encoder",
    "reset_encoder_cache",
    "software_fallback",
]
