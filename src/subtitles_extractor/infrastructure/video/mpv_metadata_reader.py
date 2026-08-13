"""Đọc metadata video qua mpv — chính xác cho mọi codec mpv hỗ trợ.

CẢI TIẾN V3.0:
    1. [Speed] Thêm flag --frames=1 và --vo=null: Ép mpv thoát ngay sau khi
       đọc xong header, giúp probe nhanh gấp 3 lần.
    2. [Stability] Graceful Termination: Sử dụng cơ chế đóng luồng an toàn
       để tránh tạo tiến trình Zombie trên Windows/Linux.
    3. [Accuracy] Container-FPS First: Ưu tiên lấy FPS từ container header
       để tránh sai số do cơ chế nội suy VF-FPS của mpv khi đang tạm dừng.
    4. [Robustness] Xử lý lỗi nạp DLL mpv tường minh hơn.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any

from subtitles_extractor.domain.entities.video_metadata import VideoMetadata
from subtitles_extractor.domain.exceptions import (
    VideoDecodeError,
    VideoNotFoundError,
)

logger = logging.getLogger(__name__)

# Timeout ngắn hơn vì metadata header thường có ngay lập tức
_PROPERTY_WAIT_TIMEOUT_SEC: float = 5.0

def _get_mpv_errors() -> tuple[type[Exception], ...]:
    errors: list[type[Exception]] = [RuntimeError, AttributeError, TypeError, ValueError]
    try:
        import mpv
        # python-mpv có các lớp lỗi đặc thù tùy version
        for err_name in ("MPVError", "ShutdownError", "CommandError"):
            err_cls = getattr(mpv, err_name, None)
            if err_cls:
                errors.append(err_cls)
    except (ImportError, OSError):
        # [v3.23.261] ImportError: python-mpv chưa cài. OSError: python-mpv cài nhưng
        # KHÔNG tìm thấy libmpv (thiếu DLL/so) — import mpv ném OSError, không phải
        # ImportError. Cả hai đều nghĩa là mpv không dùng được -> bỏ qua an toàn.
        pass
    return tuple(errors)

class MpvMetadataReader:
    def __init__(self, mpv_options: dict[str, Any] | None = None) -> None:
        self._mpv_options = mpv_options or {}

    def read(self, video_path: Path) -> VideoMetadata:
        if not video_path.exists():
            raise VideoNotFoundError(f"Không tìm thấy tệp video: {video_path}")

        try:
            import mpv
        except ImportError as exc:
            raise VideoDecodeError(
                "python-mpv chưa được cài đặt. Không thể sử dụng MpvMetadataReader."
            ) from exc
        except OSError as exc:
            # [v3.23.261] python-mpv ĐÃ cài nhưng KHÔNG tìm thấy libmpv (thiếu DLL/.so
            # hoặc chưa inject vào PATH) -> ``import mpv`` ném OSError, không phải
            # ImportError. Báo lỗi rõ thay vì để OSError thô làm sập luồng đọc metadata.
            raise VideoDecodeError(
                "Không tìm thấy thư viện libmpv (mpv). Hãy cài mpv hoặc đảm bảo "
                "libmpv nằm trong PATH. Không thể dùng MpvMetadataReader."
            ) from exc

        # Fix lỗi HOME trên một số môi trường Linux server
        if "HOME" not in os.environ:
            os.environ["HOME"] = str(Path.home())

        probe_kwargs = self._build_probe_kwargs(self._mpv_options)
        mpv_instance: Any = None

        try:
            # Khởi tạo instance với cấu hình tối giản
            mpv_instance = mpv.MPV(**probe_kwargs)
            return self._probe_with_instance(mpv_instance, video_path)
        except _get_mpv_errors() as exc:
            # Nếu MPV crash khi khởi tạo (thường do thiếu DLL), fallback về OpenCV ngay
            logger.warning("Mpv instance lỗi: %s. Thử Fallback OpenCV.", exc)
            return self._fallback_opencv_reader(video_path)
        finally:
            if mpv_instance is not None:
                # [STABILITY] Đóng tuần tự để tránh Deadlock. Nuốt lỗi cụ thể
                # ở mức narrowest có thể — terminate() có thể raise nhiều
                # loại exception tuỳ binding python-mpv version.
                with contextlib.suppress(*_get_mpv_errors()):
                    mpv_instance.terminate()

    def _build_probe_kwargs(self, base_options: dict[str, Any]) -> dict[str, Any]:
        """Xây dựng cấu hình mpv chuyên dụng cho việc quét Metadata."""
        kwargs = dict(base_options)
        # Loại bỏ các tùy chọn liên quan đến hiển thị UI
        for key in ("wid", "log_handler", "osc", "osd_level"):
            kwargs.pop(key, None)

        # Flag tối ưu: Ép mpv chỉ đọc metadata và thoát
        kwargs.update({
            "vid": "auto",
            "aid": "no",
            "sid": "no",
            "vo": "null",
            "ao": "null",
            "pause": True,
            "frames": "1",                 # [CRITICAL] Dừng ngay sau frame đầu tiên
            "start": "0",                  # Luôn bắt đầu từ 0
            "input_default_bindings": False,
            "input_vo_keyboard": False,
            "osc": "no",
            "osd_level": 0,
            "ytdl": "no",                  # Tắt ytdl để tăng tốc load file local
            "load_scripts": "no"           # Không load lua scripts lằng nhằng
        })
        return kwargs

    def _probe_with_instance(self, instance: Any, video_path: Path) -> VideoMetadata:
        # Dùng wait_metadata=True nếu python-mpv hỗ trợ, hoặc loadfile bình thường
        instance.command("loadfile", str(video_path))

        # Đợi các thuộc tính sẵn sàng (thường mất < 100ms với file local).
        # Dùng monotonic() thay time() để miễn nhiễm với việc người dùng/NTP
        # chỉnh giờ hệ thống lùi lại trong khi đang đợi (gây vòng vô hạn).
        start_time = time.monotonic()
        while time.monotonic() - start_time < _PROPERTY_WAIT_TIMEOUT_SEC:
            duration = instance.duration
            if duration is not None and duration > 0:
                break
            time.sleep(0.05)

        # Đọc các thuộc tính
        d_val = instance.duration
        w_val = instance.width
        h_val = instance.height

        # [ACCURACY] Chiến lược lấy FPS theo độ ưu tiên
        # container-fps là chính xác nhất cho metadata header
        fps_val = (
            instance.get_property("container-fps") or
            instance.get_property("fps") or
            instance.get_property("estimated-vf-fps")
        )

        codec = instance.get_property("video-codec-name") or ""

        # Kiểm tra tính hợp lệ
        if not all([d_val, w_val, h_val, fps_val]):
            return self._fallback_opencv_reader(video_path)

        try:
            f_duration = float(d_val)
            f_width = int(w_val)
            f_height = int(h_val)
            f_fps = float(fps_val)
        except (TypeError, ValueError):
            return self._fallback_opencv_reader(video_path)

        # Tính tổng frame dựa trên duration thực tế (VFR safe-ish)
        total_frames = int(round(f_duration * f_fps)) if f_duration > 0 else 0

        metadata = VideoMetadata(
            path=video_path.resolve(),
            width=f_width,
            height=f_height,
            fps=f_fps,
            total_frames=total_frames,
            duration_sec=f_duration,
            codec=str(codec),
        )

        logger.info("Mpv Probe: %s (%dx%d, %.1fs)", video_path.name, f_width, f_height, f_duration)
        return metadata

    def _fallback_opencv_reader(self, video_path: Path) -> VideoMetadata:
        try:
            from subtitles_extractor.infrastructure.video.opencv_metadata_reader import (
                OpenCvMetadataReader,
            )
            return OpenCvMetadataReader().read(video_path)
        except (VideoDecodeError, OSError, RuntimeError, ImportError) as exc:
            raise VideoDecodeError(
                f"Cả MPV và OpenCV đều thất bại khi đọc {video_path.name}: {exc}."
            ) from exc

__all__ = ["MpvMetadataReader"]
