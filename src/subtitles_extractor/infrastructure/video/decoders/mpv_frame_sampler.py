"""Adapter :class:`FrameSamplerPort` dùng mpv với HW decoding."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
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


def _get_mpv_errors() -> tuple[type[Exception], ...]:
    errors: list[type[Exception]] = [RuntimeError, AttributeError, ValueError, TypeError]
    try:
        import mpv
        if hasattr(mpv, "MPVError"):
            errors.append(mpv.MPVError)
        if hasattr(mpv, "ShutdownError"):
            errors.append(mpv.ShutdownError)
    except (ImportError, OSError):
        # [v3.23.261] OSError khi python-mpv cài nhưng thiếu libmpv.
        pass
    return tuple(errors)


class MpvFrameSampler:
    def __init__(self, mpv_options: dict[str, Any] | None = None) -> None:
        self._mpv_options = mpv_options or {}

    def iter_frames(self, metadata: VideoMetadata, roi: Roi | None, config: FrameSamplingConfig) -> Iterator[SampledFrame]:
        try:
            import mpv
        except ImportError as exc:
            raise VideoDecodeError("python-mpv không cài đặt.") from exc
        except OSError as exc:
            # [v3.23.261] python-mpv cài nhưng thiếu libmpv -> import ném OSError.
            raise VideoDecodeError(
                "Không tìm thấy libmpv (mpv). Cài mpv hoặc thêm libmpv vào PATH."
            ) from exc

        if not metadata.path.exists():
            raise VideoNotFoundError(f"Không tìm thấy tệp video: {metadata.path}")

        os.environ.setdefault("HOME", str(Path.home()))

        mpv_kwargs = self._build_mpv_kwargs()
        instance = mpv.MPV(**mpv_kwargs)

        try:
            instance.command("loadfile", str(metadata.path))
            instance.pause = True
            # [v3.23.289] Voi vo=null + pause=True, mpv co the KHONG vao "playing" ->
            # wait_until_playing timeout sai. Cho 'file-loaded' (video da mo). Neu van
            # timeout (thuong do hwdec loi headless), thu lai voi hwdec=no.
            if not self._wait_file_loaded(instance, timeout=15.0):
                instance.terminate()
                # Thu lai voi hwdec=no (an toan nhat cho headless).
                safe_kwargs = dict(mpv_kwargs)
                safe_kwargs["hwdec"] = "no"
                instance = mpv.MPV(**safe_kwargs)
                instance.command("loadfile", str(metadata.path))
                instance.pause = True
                if not self._wait_file_loaded(instance, timeout=15.0):
                    raise VideoDecodeError(
                        f"Mpv không mở được {metadata.path.name}: timeout hoặc lỗi codec."
                    )
            yield from self._iterate(instance, metadata, roi, config)
        except (VideoDecodeError, VideoNotFoundError):
            raise
        except _get_mpv_errors() as exc:
            raise VideoDecodeError(
                f"Mpv frame sampler thất bại trên {metadata.path.name}: {exc}."
            ) from exc
        except OSError as exc:
            raise VideoDecodeError(
                f"Mpv frame sampler thất bại I/O trên {metadata.path.name}: {exc}."
            ) from exc
        finally:
            try:
                instance.terminate()
            except _get_mpv_errors() as exc:
                logger.debug("Lỗi khi terminate mpv instance: %s.", exc)

    def _wait_file_loaded(self, instance: Any, timeout: float) -> bool:
        """Chờ mpv mở xong video (property ``duration`` có giá trị). True nếu mở được.

        Với ``vo=null`` + ``pause=True``, mpv không vào trạng thái "playing" nên
        ``wait_until_playing`` timeout sai. Thay vào đó chờ ``duration`` > 0 — dấu hiệu
        video đã mở và giải mã header thành công.

        Args:
            instance: Đối tượng ``mpv.MPV``.
            timeout: Thời gian chờ tối đa (giây).

        Returns:
            True nếu video mở được trong thời gian chờ, False nếu timeout.
        """
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                duration = instance.duration
                if duration is not None and duration > 0:
                    return True
            except (*_get_mpv_errors(), RuntimeError, AttributeError):
                pass
            time.sleep(0.1)
        return False

    def _build_mpv_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self._mpv_options)
        kwargs["vo"] = "null"
        kwargs["ao"] = "null"
        kwargs["input_default_bindings"] = False
        kwargs["input_vo_keyboard"] = False
        kwargs["osc"] = False
        kwargs["hr_seek"] = "yes"
        kwargs["keep_open"] = "yes"
        kwargs["loop_file"] = "no"
        kwargs.pop("wid", None)
        kwargs.pop("log_handler", None)
        return kwargs

    def _iterate(self, instance: Any, metadata: VideoMetadata, roi: Roi | None, config: FrameSamplingConfig) -> Iterator[SampledFrame]:
        step_sec = max(config.sample_step_sec, 0.001)
        total_sec = metadata.duration_sec
        end_sec = total_sec - config.skip_outro_sec if config.skip_outro_sec > 0 else total_sec
        last_hash: int | None = None
        last_image: np.ndarray | None = None
        frame_counter: int = 0
        timestamp_sec = max(0.0, config.skip_intro_sec)

        while timestamp_sec <= end_sec:
            try:
                instance.command("seek", timestamp_sec, "absolute", "exact")
                screenshot = instance.screenshot_raw()
            except _get_mpv_errors() as exc:
                # Bắt tất cả lỗi libmpv kể cả ShutdownError
                logger.debug(
                    "Bỏ qua mốc %.3fs do mpv lỗi: %s.", timestamp_sec, exc
                )
                timestamp_sec += step_sec
                continue

            image_rgb = _screenshot_to_rgb(screenshot)
            # [Deep Memory Leak] screenshot là dict chứa mảng byte thô do C-core mpv
            # sinh ra — GC Python dọn "lười", phim dài tích tụ hàng chục GB rồi sập.
            # Ép giải phóng NGAY sau khi đã chuyển sang RGB.
            del screenshot
            if image_rgb is None:
                timestamp_sec += step_sec
                continue

            cropped = self._apply_roi(image_rgb, roi)
            # image_rgb đã được sao chép vào `cropped` qua _apply_roi → giải phóng ngay.
            del image_rgb
            current_hash = compute_phash(cropped)

            try:
                pt = instance.playback_time
                actual_sec = float(pt) if pt is not None else timestamp_sec
            except _get_mpv_errors():
                actual_sec = timestamp_sec

            if last_hash is not None and last_image is not None:
                if hamming_distance(current_hash, last_hash) <= config.phash_distance_threshold and pixel_diff_ratio(cropped, last_image) <= config.pixel_diff_threshold:
                    yield SampledFrame(frame_index=frame_counter, timestamp_sec=actual_sec, image_rgb=np.empty(0), is_duplicate=True)
                    frame_counter += 1
                    timestamp_sec += step_sec
                    continue

            yield SampledFrame(frame_index=frame_counter, timestamp_sec=actual_sec, image_rgb=cropped, is_duplicate=False)
            last_hash = current_hash
            last_image = cropped
            frame_counter += 1
            timestamp_sec += step_sec

    @staticmethod
    def _apply_roi(image_rgb: np.ndarray, roi: Roi | None) -> np.ndarray:
        if roi is None:
            return image_rgb.copy()
        height, width = image_rgb.shape[:2]
        clipped = roi.clip_to(width, height)
        return image_rgb[clipped.y : clipped.y2, clipped.x : clipped.x2].copy()

def _screenshot_to_rgb(screenshot: dict[str, Any]) -> np.ndarray | None:
    if not screenshot: return None
    width = int(screenshot.get("w", 0))
    height = int(screenshot.get("h", 0))
    stride = int(screenshot.get("stride", width * 4))
    fmt = str(screenshot.get("format", "bgr0")).lower()
    data = screenshot.get("data")
    if not data or width <= 0 or height <= 0: return None

    raw = np.frombuffer(data, dtype=np.uint8)
    bytes_per_pixel = 4 if fmt in ("bgr0", "rgb0", "bgra", "rgba") else 3
    if bytes_per_pixel == 4:
        try: buffer = raw.reshape(height, stride // 4, 4)
        except ValueError: return None
        bgra_or_rgba = buffer[:, :width, :]
        if fmt.startswith("bgr"): rgb = bgra_or_rgba[:, :,[2, 1, 0]]
        else: rgb = bgra_or_rgba[:, :, :3]
    else:
        try: buffer = raw.reshape(height, stride // 3, 3)
        except ValueError: return None
        bgr_or_rgb = buffer[:, :width, :]
        rgb = bgr_or_rgb[:, :, [2, 1, 0]] if fmt.startswith("bgr") else bgr_or_rgb
    return np.ascontiguousarray(rgb)

__all__ = ["MpvFrameSampler"]
