"""Đọc metadata video bằng PyAV — nhanh hơn OpenCV, hỗ trợ mọi container.

CẢI TIẾN ĐỘT PHÁ (V3.1 - Bất khả chiến bại):
    1. [Smart Stream Selection]: Khắc phục lỗi Video bị đánh cờ nhầm thành Ảnh bìa (Attached Pic) khiến danh sách Stream bị rỗng.
    2. [1-Frame Probe]: Cứu cánh tuyệt đối cho Fragmented MP4 (Video có Width/Height = 0 ở Header).
       Tự động giải mã 1 khung hình đầu tiên để lấy kích thước thực mà không cần ném sang OpenCV.
    3.[Zero FPS Guard]: Tự động cứu vãn video bị mất thông tin FPS.
    4. [Rotation Awareness]: Đảo Width/Height chính xác khi video bị xoay dọc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import VideoDecodeError, VideoNotFoundError

logger = logging.getLogger(__name__)


class PyAvMetadataReader:
    """Đọc metadata video bằng PyAV (libav)."""

    def read(self, video_path: Path) -> VideoMetadata:
        if not video_path.exists():
            raise VideoNotFoundError(f"Không tìm thấy tệp video: {video_path}")

        try:
            import av  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VideoDecodeError("PyAV chưa cài đặt. Chạy: pip install av") from exc

        try:
            # Luôn dùng str(path) để tương thích tối đa với libav C-bindings
            with av.open(str(video_path)) as container:
                return self._extract(container, video_path)
        except (RuntimeError, OSError, ValueError, Exception) as exc:
            # Bắt tất cả lỗi định dạng lạ hoặc file bị hỏng nghiêm trọng
            logger.warning(
                "PyAV metadata failed for %s (%s) — fallback OpenCV.",
                video_path.name, exc,
            )
            return self._fallback_opencv(video_path)

    def _extract(self, container: Any, video_path: Path) -> VideoMetadata:
        import av

        # 1. Lấy toàn bộ luồng video
        video_streams = list(container.streams.video)
        if not video_streams:
            raise VideoDecodeError(f"{video_path.name} hoàn toàn không chứa luồng video.")

        # 2. Lọc thông minh: Cố gắng bỏ qua luồng ảnh bìa (Cover Art)
        real_video_streams = []
        for stream in video_streams:
            is_attached_pic = False
            if hasattr(stream, 'disposition'):
                # Xử lý an toàn vì disposition có thể là Dict hoặc Object tùy phiên bản PyAV
                disp = stream.disposition
                if isinstance(disp, dict):
                    is_attached_pic = bool(disp.get('attached_pic', 0))
                else:
                    is_attached_pic = getattr(disp, 'attached_pic', False)

            if not is_attached_pic:
                real_video_streams.append(stream)

        # [CRITICAL FIX]: Nếu lọc xong mà rỗng (Tức là luồng video duy nhất bị lỗi đánh nhầm cờ ảnh bìa),
        # thì hủy bỏ bộ lọc và sử dụng lại danh sách gốc!
        candidates = real_video_streams if real_video_streams else video_streams

        # Lấy luồng có độ phân giải cao nhất (luồng chính) thay vì lấy bừa luồng đầu tiên
        def get_resolution(s: Any) -> int:
            return (s.width or 0) * (s.height or 0)

        main_stream = max(candidates, key=get_resolution)

        width = main_stream.width or 0
        height = main_stream.height or 0

        # [CỨU CÁNH TUYỆT ĐỐI]: Xử lý Fragmented MP4 (Header bị rỗng Width/Height)
        # Ép PyAV phải giải mã đúng 1 khung hình đầu tiên để nhè ra độ phân giải thực sự!
        if width <= 0 or height <= 0:
            try:
                for packet in container.demux(main_stream):
                    for frame in packet.decode():
                        width = frame.width
                        height = frame.height
                        break  # Chỉ cần 1 frame là đủ
                    if width > 0 and height > 0:
                        break
            except (RuntimeError, OSError, ValueError, AttributeError) as exc:
                logger.debug("1-Frame Probe thất bại: %s.", exc)

        if width <= 0 or height <= 0:
            # Đến nước này thì PyAV thực sự bó tay, buộc phải gọi viện binh OpenCV
            return self._fallback_opencv(video_path)

        fps = self._resolve_fps(main_stream)
        # [Zero FPS Guard]: Cứu vãn video bị lỗi khai báo FPS ở Header
        if fps <= 0:
            fps = 25.0

        # [Rotation Awareness]: Xử lý Xoay video (Cho video Smartphone)
        rotation = 0
        try:
            rot_str = main_stream.metadata.get('rotate', '0')
            rotation = int(rot_str)
        except (ValueError, TypeError, AttributeError):
            pass

        if rotation in (90, 270, -90, -270):
            width, height = height, width  # Đảo kích thước để UI vẽ đúng tỷ lệ

        duration = 0.0
        if container.duration is not None:
            duration = float(container.duration) / av.time_base
        if duration <= 0 and main_stream.duration is not None and main_stream.time_base is not None:
            duration = float(main_stream.duration * main_stream.time_base)

        codec = main_stream.codec_context.name or ""

        # Ưu tiên lấy số frame thực tế từ stream header thay vì tính toán phỏng đoán
        total_frames = main_stream.frames
        if total_frames <= 0 and duration > 0:
            total_frames = int(round(duration * fps))

        metadata = VideoMetadata(
            path=video_path.resolve(),
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration,
            codec=codec,
        )

        logger.info(
            "PyAV metadata: %s (%dx%d @ %.2ffps, %s, %.1fs).",
            video_path.name, width, height, fps, codec, duration,
        )
        return metadata

    @staticmethod
    def _resolve_fps(stream: Any) -> float:
        """Giải quyết FPS với độ ưu tiên giảm dần (Bọc Try-Catch an toàn tuyệt đối)."""
        for attr in ("average_rate", "guessed_rate", "base_rate"):
            rate = getattr(stream, attr, None)
            try:
                if rate and rate.denominator > 0:
                    val = float(rate.numerator) / float(rate.denominator)
                    if val > 0:
                        return val
            except (AttributeError, ZeroDivisionError, TypeError):
                continue
        return 0.0

    @staticmethod
    def _fallback_opencv(video_path: Path) -> VideoMetadata:
        """Viện binh cuối cùng khi libav C-core từ chối phân tích."""
        try:
            from subtitles_extractor.infrastructure.video.opencv_metadata_reader import (
                OpenCvMetadataReader,
            )
            logger.info("Fallback sử dụng OpenCV Metadata Reader cho: %s.", video_path.name)
            return OpenCvMetadataReader().read(video_path)
        except (VideoDecodeError, OSError, RuntimeError, ImportError) as exc:
            logger.exception("Cả PyAV và OpenCV đều bó tay: %s.", exc)
            raise VideoDecodeError(
                f"Không thể đọc metadata bằng bất kỳ backend nào: {exc}."
            ) from exc


__all__ = ["PyAvMetadataReader"]
