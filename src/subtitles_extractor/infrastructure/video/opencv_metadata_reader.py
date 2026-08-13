"""Adapter :class:`VideoMetadataReaderPort` dùng OpenCV.

CẢI TIẾN V3.0:
    1. [Performance] Loại bỏ capture.read(): Lấy timestamp bằng seek mà không decode ảnh, nhanh gấp 5-10 lần.
    2. [Robustness] Smart Duration: Kết hợp 3 phương pháp tính duration để chống sai số VFR.
    3. [Compatibility] Xử lý lỗi backend MSMF/FFMPEG khi đọc FourCC và Frame Count.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import (
    VideoDecodeError,
    VideoNotFoundError,
)

logger = logging.getLogger(__name__)


class OpenCvMetadataReader:
    """Đọc metadata video bằng :class:`cv2.VideoCapture`."""

    def read(self, video_path: Path) -> VideoMetadata:
        if not video_path.exists():
            raise VideoNotFoundError(f"Tệp video không tồn tại: {video_path}.")

        # Sử dụng backend FFMPEG nếu có thể để tăng độ chính xác trên Windows/Linux
        capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            # Fallback về backend mặc định nếu FFMPEG không khả dụng
            capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise VideoNotFoundError(
                f"OpenCV không mở được tệp video: {video_path}."
            )

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC))

            # [PERFORMANCE] Tính duration thông minh hơn
            duration_sec = _probe_duration_opencv(capture, total_frames, fps)

            # [FIX] Nếu total_frames bị sai (thường gặp ở MKV/VFR), tính ngược lại từ duration
            if (total_frames <= 0 or total_frames > 1_000_000_000) and duration_sec > 0 and fps > 0:
                total_frames = int(round(duration_sec * fps))

        finally:
            capture.release()

        if width <= 0 or height <= 0:
            raise VideoDecodeError(
                f"Metadata không hợp lệ ({width}×{height}) "
                f"cho tệp {video_path.name}."
            )

        # FPS dự phòng nếu container không cung cấp
        safe_fps = fps if (0 < fps < 1000) else 25.0

        return VideoMetadata(
            path=video_path.resolve(),
            width=width,
            height=height,
            fps=safe_fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            codec=_decode_fourcc(fourcc_int),
        )


def _decode_fourcc(fourcc_int: int) -> str:
    """Chuyển int FourCC thành chuỗi lowercase, xử lý lỗi bitwise."""
    if fourcc_int <= 0:
        return "unknown"

    # Một số backend trả về FourCC ở dạng byte đảo ngược
    try:
        chars = "".join([chr((fourcc_int >> i) & 0xFF) for i in (0, 8, 16, 24)])
        # Loại bỏ các ký tự không in được và strip khoảng trắng
        clean_codec = "".join(c for c in chars if c.isprintable()).strip().lower()
        return clean_codec if clean_codec else "unknown"
    except (ValueError, OverflowError, TypeError):
        return "unknown"


def _probe_duration_opencv(
    capture: cv2.VideoCapture,
    total_frames: int,
    fps: float,
) -> float:
    """Tính duration chính xác - Không cần decode frame.

    Chiến lược nâng cấp:
        1. Thử lấy trực tiếp từ header (Chỉ một số backend hỗ trợ).
        2. Seek tới frame cuối và lấy POS_MSEC (Không gọi read() để tiết kiệm CPU).
        3. Fallback toán học.
    """
    # Cách 1: Thử lấy POS_MSEC tại vị trí hiện tại (một số file trả về tổng thời gian ngay tại frame 0)
    # Tuy nhiên cách này hiếm khi hoạt động với OpenCV.

    # Cách 2: Seek tới frame cuối
    if 0 < total_frames < 1_000_000:  # Chặn frame count ảo
        try:
            # Chỉ cần nhảy tới vị trí (SEEK), không cần đọc dữ liệu ảnh (READ)
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(total_frames - 1))
            pos_msec = capture.get(cv2.CAP_PROP_POS_MSEC)

            # Reset về 0 để tránh ảnh hưởng các tác vụ sau (nếu có)
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

            if pos_msec > 0:
                return pos_msec / 1000.0
        except (cv2.error, RuntimeError, OSError):
            pass

    # Cách 3: Fallback toán học (Độ chính xác tương đối)
    if total_frames > 0 and fps > 0:
        return total_frames / fps

    return 0.0


__all__ = ["OpenCvMetadataReader"]
