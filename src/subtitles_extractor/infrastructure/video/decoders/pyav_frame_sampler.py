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
    is_sample_due,
)
from subtitles_extractor.domain.value_objects.boundary_track import BoundaryTrack
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.infrastructure.video.decoders.boundary_probe import (
    BoundaryProber,
    probe_av_frame,
    probing_enabled,
)
from subtitles_extractor.infrastructure.video.frame_dedupe import (
    FrameDeduper,
    FrameVerdict,
)

logger = logging.getLogger(__name__)

#: [v3.23.632] Lùi bao nhiêu giây trước mốc bắt đầu khi seek (phần lùi được giải mã rồi bỏ qua).
#: [v3.23.676] Bước lấy mẫu từ mức này trở lên chỉ giải mã khung khoá.
_KEYFRAME_ONLY_MIN_STEP_SEC = 5.0


def _next_target(target: float, frame_ts: float, step: float, keyframe_only: bool) -> float:
    """Mốc lấy mẫu kế tiếp. Chỉ giải mã khung khoá: tính từ MỐC MỤC TIÊU (không cộng dồn độ trễ tới khung khoá kế —
    trước đây mỗi mẫu trễ tới ~10 s làm hụt 5-10 % số mẫu), nhưng không sát mẫu vừa lấy hơn nửa bước."""
    if not keyframe_only:
        return frame_ts + step
    return max(target + step, frame_ts + step / 2)
_SEEK_MARGIN_SEC = 5.0


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

    #: [v3.23.680] Dải ranh giới của lượt ``iter_frames`` vừa xong (mọi khung, kể cả không OCR) — use case trích xuất đọc
    #: sau khi lặp hết để đặt mốc phụ đề đúng khung. ``None`` khi lượt bị dừng giữa chừng / bước thưa (khung khoá).
    last_boundary_track: BoundaryTrack | None = None

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

        self.last_boundary_track = None
        # [v3.23.680] Chỉ ghi khi có ROI (mặt nạ chữ đo trên vùng phụ đề) và bước không thưa (khung khoá).
        prober = BoundaryProber(roi) if probing_enabled(roi, config.sample_step_sec) else None
        try:
            yield from self._iterate(container, metadata, roi, config, prober)
            if prober is not None:
                self.last_boundary_track = prober.finish()
        finally:
            if container is not None:
                with contextlib.suppress(*_get_av_errors(), AttributeError):
                    container.close()

    def _iterate(
        self,
        container: Any,
        metadata: VideoMetadata,
        roi: Roi | None,
        config: FrameSamplingConfig,
        prober: BoundaryProber | None = None,
    ) -> Iterator[SampledFrame]:

        video_streams = [s for s in container.streams if s.type == "video"]
        if not video_streams:
            raise VideoDecodeError(f"{metadata.path.name} không có luồng video nào hợp lệ.")

        stream = video_streams[0]
        # [v3.23.643] Giải mã đa luồng theo khung (AUTO) thay cho mặc định SLICE: đo trên 4 nhân,
        # 720x1280 HEVC, 190-221 → 415-437 khung/giây. Chỉ ảnh hưởng hiệu năng.
        with contextlib.suppress(AttributeError, ValueError, TypeError):
            stream.thread_type = "AUTO"

        step_sec = max(0.001, config.sample_step_sec)
        # [v3.23.676] Bước thưa (tự dò vùng trên phim dài: ~20 s/khung) → chỉ giải mã KHUNG KHOÁ. Đo phim ghép 139
        # phút: giải mã hết 208 nghìn khung ~404 s, chỉ khung khoá 58 s (khung khoá cách nhau trung vị 1,8 s, xa nhất 10 s).
        # Bước trích xuất (0,04-0,5 s) không bị ảnh hưởng.
        keyframe_only = step_sec >= _KEYFRAME_ONLY_MIN_STEP_SEC
        if keyframe_only:
            with contextlib.suppress(AttributeError, ValueError, TypeError):
                stream.codec_context.skip_frame = "NONKEY"
        total_sec = metadata.duration_sec
        skip_intro_sec = config.skip_intro_sec
        skip_outro_sec = config.skip_outro_sec

        start_sec = max(0.0, skip_intro_sec)
        end_sec = total_sec - skip_outro_sec if skip_outro_sec > 0 else total_sec
        next_target_sec = start_sec

        # [v3.23.644] Quyết định OCR / trùng / trống dùng chung với bộ NVDEC (xem ``frame_dedupe``).
        deduper = FrameDeduper(config)
        frame_counter_idx: int = 0

        if start_sec > 0:
            try:
                # [BUG FIX v2.9+]: Dùng int(start_sec / float(time_base)) thay vì
                # * denominator. Công thức cũ chỉ đúng khi numerator=1 (phần lớn
                # stream thông thường). Với VFR có time_base=1001/30000, denominator
                # =30000 nhưng pts/sec thực tế = 30000/1001 ≈ 29.97, không phải 30000.
                # seek_worker.py đã dùng đúng công thức này — đồng nhất ở đây.
                tb_float = float(stream.time_base) if stream.time_base else 1.0 / 90000.0
                # [v3.23.632] Lùi 5 s như bộ NVDEC: seek tới ĐÚNG mốc rơi vào khung khoá có
                # dts <= mốc nhưng pts > mốc (video có B-frame) → mất các khung từ mốc tới khung
                # khoá kế (đo: skip_intro=5.85 s cho khung đầu 6.00 s thay vì 5.88 s). Phần lùi
                # được bỏ qua bằng phép so ``< start_sec`` bên dưới.
                seek_pts = int(max(0.0, start_sec - _SEEK_MARGIN_SEC) / tb_float)
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
                if not is_sample_due(frame_timestamp_sec, next_target_sec):
                    if prober is not None:
                        probe_av_frame(prober.skipped, frame, frame_timestamp_sec)
                    continue
                if prober is not None:
                    probe_av_frame(prober.sampled, frame, frame_timestamp_sec)

                try:
                    frame_rgb_arr = frame.to_ndarray(format="rgb24")
                except ValueError as exc:
                    logger.debug("Lỗi convert to_ndarray: %s", exc)
                    next_target_sec = _next_target(next_target_sec, frame_timestamp_sec, step_sec, keyframe_only)
                    continue

                cropped_rgb_arr = self._apply_roi(frame_rgb_arr, roi)
                # Giải phóng mảng lớn gốc ngay tức thì (Quan trọng cho Video 4K)
                del frame_rgb_arr

                if cropped_rgb_arr is None or cropped_rgb_arr.size == 0:
                    next_target_sec = _next_target(next_target_sec, frame_timestamp_sec, step_sec, keyframe_only)
                    continue

                verdict = deduper.verdict(cropped_rgb_arr, frame_timestamp_sec)
                if verdict is not FrameVerdict.OCR:
                    yield SampledFrame(
                        frame_index=frame_counter_idx,
                        timestamp_sec=frame_timestamp_sec,
                        image_rgb=np.empty(0),
                        is_duplicate=verdict is FrameVerdict.DUPLICATE,
                        is_blank=verdict is FrameVerdict.BLANK,
                    )
                    frame_counter_idx += 1
                    next_target_sec = _next_target(next_target_sec, frame_timestamp_sec, step_sec, keyframe_only)
                    del cropped_rgb_arr
                    continue

                yield SampledFrame(
                    frame_index=frame_counter_idx,
                    timestamp_sec=frame_timestamp_sec,
                    image_rgb=cropped_rgb_arr,
                    is_duplicate=False
                )

                deduper.accept(cropped_rgb_arr, frame_timestamp_sec)
                frame_counter_idx += 1
                next_target_sec = _next_target(next_target_sec, frame_timestamp_sec, step_sec, keyframe_only)

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


__all__ = ["PyAvFrameSampler"]
