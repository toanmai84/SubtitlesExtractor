"""Adapter FrameSamplerPort dùng PyAV — hỗ trợ VFR đúng cách.

TỐI ƯU HÓA ĐỘT PHÁ & CẢI TIẾN:
    -[PERFORMANCE] Explicit GC Cleanup: Giải phóng bộ nhớ Frame gốc ngay lập tức
      để ngăn hiện tượng Memory Leak trên Video độ phân giải cao (4K/8K).
    -[CRITICAL FIX] Cập nhật Dynamic Exception Fetcher (_get_av_errors) tương thích
      mọi phiên bản thư viện PyAV, khắc phục lỗi mất attribute 'AVError'.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import VideoDecodeError, VideoNotFoundError
from subtitles_extractor.domain.ports.frame_sampler_port import (
    FrameSamplingConfig,
    SampledFrame,
)
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.infrastructure.video.perceptual_hash import (
    compute_phash,
    hamming_distance,
    pixel_diff_ratio,
)

logger = logging.getLogger(__name__)


def _get_av_errors() -> tuple[type[Exception], ...]:
    """Lấy danh sách lỗi PyAV linh hoạt, tương thích với cả bản cũ lẫn bản mới nhất."""
    errors: list[type[Exception]] = [OSError, ValueError, RuntimeError]
    try:
        import av
        if hasattr(av, "AVError"):
            errors.append(av.AVError)
        if hasattr(av, "error") and hasattr(av.error, "FFmpegError"):
            errors.append(av.error.FFmpegError)
        if hasattr(av, "error") and hasattr(av.error, "InvalidDataError"):
            errors.append(av.error.InvalidDataError)
    except ImportError:
        pass
    return tuple(errors)


class PyAvFrameSampler:
    """Trình lấy mẫu khung hình dựa trên thư viện PyAV (libav/FFmpeg)."""

    def iter_frames(
        self,
        metadata: VideoMetadata,
        roi: Roi | None,
        config: FrameSamplingConfig
    ) -> Iterator[SampledFrame]:

        if not metadata.path.exists():
            raise VideoNotFoundError(f"Không tìm thấy tệp video: {metadata.path}")

        try:
            import av
        except ImportError as exc:
            raise VideoDecodeError("Thư viện PyAV chưa được cài đặt. Hãy chạy: pip install av") from exc

        container: Any = None
        try:
            container = av.open(str(metadata.path))
        except _get_av_errors() as exc:
            raise VideoDecodeError(f"PyAV không mở được {metadata.path.name}: {exc}") from exc

        try:
            yield from self._iterate(container, metadata, roi, config)
        finally:
            if container is not None:
                with contextlib.suppress(*_get_av_errors(), AttributeError):
                    container.close()

    def _iterate(
        self,
        container: Any,
        metadata: VideoMetadata,
        roi: Roi | None,
        config: FrameSamplingConfig
    ) -> Iterator[SampledFrame]:

        video_streams = [s for s in container.streams if s.type == "video"]
        if not video_streams:
            raise VideoDecodeError(f"{metadata.path.name} không có luồng video nào hợp lệ.")

        stream = video_streams[0]

        step_sec = max(0.001, config.sample_step_sec)
        total_sec = metadata.duration_sec
        skip_intro_sec = config.skip_intro_sec
        skip_outro_sec = config.skip_outro_sec

        start_sec = max(0.0, skip_intro_sec)
        end_sec = total_sec - skip_outro_sec if skip_outro_sec > 0 else total_sec
        next_target_sec = start_sec

        last_phash_val: int | None = None
        last_image_arr: np.ndarray | None = None
        frame_counter_idx: int = 0

        if start_sec > 0:
            try:
                # [BUG FIX v2.9+]: Dùng int(start_sec / float(time_base)) thay vì
                # * denominator. Công thức cũ chỉ đúng khi numerator=1 (phần lớn
                # stream thông thường). Với VFR có time_base=1001/30000, denominator
                # =30000 nhưng pts/sec thực tế = 30000/1001 ≈ 29.97, không phải 30000.
                # seek_worker.py đã dùng đúng công thức này — đồng nhất ở đây.
                tb_float = float(stream.time_base) if stream.time_base else 1.0 / 90000.0
                seek_pts = int(start_sec / tb_float)
                container.seek(seek_pts, backward=True, stream=stream)
            except _get_av_errors() as exc:
                logger.warning("Lỗi khi seek bằng PyAV: %s", exc)

        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except MemoryError:
                logger.exception("Hết bộ nhớ (OOM) khi decode PyAV packet.")
                return
            except (RuntimeError, ValueError, OSError, AttributeError) as exc:
                # Chỉ bắt các lỗi decode cụ thể — KHÔNG dùng Exception (quá rộng,
                # nuốt cả AttributeError/NameError từ lỗi lập trình, khó debug).
                logger.debug("Bỏ qua packet bị lỗi giải mã: %s.", exc)
                continue

            for frame in frames:
                if frame.pts is None:
                    if frame.dts is not None:
                        frame_timestamp_sec = float(frame.dts * frame.time_base)
                    else:
                        continue
                else:
                    frame_timestamp_sec = float(frame.pts * frame.time_base)

                if frame_timestamp_sec < start_sec:
                    continue
                if frame_timestamp_sec > end_sec:
                    return
                if frame_timestamp_sec < next_target_sec:
                    continue

                try:
                    frame_rgb_arr = frame.to_ndarray(format="rgb24")
                except ValueError as exc:
                    logger.debug("Lỗi convert to_ndarray: %s", exc)
                    next_target_sec = frame_timestamp_sec + step_sec
                    continue

                cropped_rgb_arr = self._apply_roi(frame_rgb_arr, roi)
                # Giải phóng mảng lớn gốc ngay tức thì (Quan trọng cho Video 4K)
                del frame_rgb_arr

                if cropped_rgb_arr is None or cropped_rgb_arr.size == 0:
                    next_target_sec = frame_timestamp_sec + step_sec
                    continue

                if not self._should_keep_with_hash(cropped_rgb_arr, last_phash_val, last_image_arr, config):
                    yield SampledFrame(
                        frame_index=frame_counter_idx,
                        timestamp_sec=frame_timestamp_sec,
                        image_rgb=np.empty(0),
                        is_duplicate=True
                    )
                    frame_counter_idx += 1
                    next_target_sec = frame_timestamp_sec + step_sec
                    del cropped_rgb_arr
                    continue

                current_phash_val = compute_phash(cropped_rgb_arr)

                yield SampledFrame(
                    frame_index=frame_counter_idx,
                    timestamp_sec=frame_timestamp_sec,
                    image_rgb=cropped_rgb_arr,
                    is_duplicate=False
                )

                last_phash_val = current_phash_val
                last_image_arr = cropped_rgb_arr
                frame_counter_idx += 1
                next_target_sec = frame_timestamp_sec + step_sec

    @staticmethod
    def _apply_roi(frame_rgb: np.ndarray, roi: Roi | None) -> np.ndarray | None:
        """Cắt ảnh theo vùng quan tâm (ROI), đảm bảo ảnh mới có vùng nhớ độc lập."""
        if roi is None:
            return frame_rgb.copy()

        frame_height, frame_width = frame_rgb.shape[:2]
        clipped_roi = roi.clip_to(frame_width, frame_height)
        cropped_arr = frame_rgb[
            clipped_roi.y : clipped_roi.y2,
            clipped_roi.x : clipped_roi.x2
        ].copy()
        return cropped_arr if cropped_arr.size > 0 else None

    @staticmethod
    def _should_keep_with_hash(
        image_rgb: np.ndarray,
        last_hash: int | None,
        last_image: np.ndarray | None,
        config: FrameSamplingConfig
    ) -> bool:
        """Quyết định giữ lại frame hay không dựa vào độ khác biệt hình ảnh."""
        if last_hash is None or last_image is None:
            return True

        current_hash = compute_phash(image_rgb)
        if hamming_distance(current_hash, last_hash) > config.phash_distance_threshold:
            return True

        return pixel_diff_ratio(image_rgb, last_image) > config.pixel_diff_threshold

__all__ = ["PyAvFrameSampler"]
