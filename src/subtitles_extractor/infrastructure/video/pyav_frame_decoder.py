"""Lõi giải mã video bằng PyAV — độc lập hoàn toàn với Qt (thuần PyAV + numpy).

VÌ SAO tồn tại module này
=========================
``libmpv-2.dll`` (bản Windows dựng sẵn phổ biến) được xác định là **GPL** — nó nhúng
ffmpeg build kèm x264/x265 nên bắt buộc ``--enable-gpl`` (xem v3.23.308). Phân phối
kèm sẽ buộc TOÀN BỘ ứng dụng theo GPL, mâu thuẫn mục tiêu thương mại license-clean.

Module này là lõi cho một trình phát thay thế dựa trên **PyAV** (libav LGPL, ĐÃ được
audit và ĐÃ nằm trong bundle) — không thêm bất kỳ phụ thuộc mới nào.

Lợi ích phụ quan trọng: preview và OCR dùng CHUNG một bộ giải mã, nên timestamp khớp
tuyệt đối. Trước đây mpv và PyAV là hai decoder khác nhau, về nguyên tắc có thể lệch.

Nguyên tắc thiết kế
-------------------
* **KHÔNG phụ thuộc Qt** — nhờ vậy kiểm thử được đầy đủ, không cần màn hình.
* **Timestamp luôn dùng ``PTS × time_base``**, TUYỆT ĐỐI không dùng ``frame_idx / fps``
  (nguyên tắc xuyên suốt dự án; bắt buộc để đúng với video VFR).
* Seek chính xác tới frame: nhảy tới keyframe TRƯỚC mốc đích rồi giải mã tiến tới.

Module này chỉ lo GIẢI MÃ. Việc hiển thị (Qt) và đồng bộ âm thanh nằm ở tầng trên.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import numpy as np

from subtitles_extractor.domain.exceptions import VideoDecodeError, VideoNotFoundError

logger = logging.getLogger(__name__)

# Khi lùi frame, seek về trước mốc hiện tại một khoảng an toàn (giây) rồi giải mã tiến.
# Cần > 1 khoảng frame để chắc chắn bắt được frame liền trước, nhưng đủ nhỏ để nhanh.
_BACKWARD_SEEK_MARGIN_SEC: Final[float] = 0.5

# FPS mặc định khi không đọc được từ metadata (chỉ dùng cho ước lượng bước frame).
_FALLBACK_FPS: Final[float] = 25.0


def _av_error_types() -> tuple[type[BaseException], ...]:
    """Trả về tuple lớp lỗi PyAV, tương thích nhiều phiên bản.

    PyAV đổi tên lớp lỗi giữa các bản (``AVError`` cũ → ``error.FFmpegError`` mới),
    nên KHÔNG tham chiếu trực tiếp mà dựng tuple phòng thủ.

    Returns:
        Tuple các lớp ngoại lệ nên bắt khi thao tác với PyAV.
    """
    import av

    errors: list[type[BaseException]] = [OSError, ValueError, RuntimeError, StopIteration]
    ffmpeg_error = getattr(getattr(av, "error", None), "FFmpegError", None)
    if ffmpeg_error is not None:
        errors.append(ffmpeg_error)
    legacy_error = getattr(av, "AVError", None)
    if legacy_error is not None:
        errors.append(legacy_error)
    return tuple(errors)


class PyAvFrameDecoder:
    """Giải mã video theo frame bằng PyAV, hỗ trợ seek chính xác và bước frame.

    Đối tượng này KHÔNG an toàn đa luồng: mỗi luồng cần một thực thể riêng, hoặc
    tầng gọi phải tự khoá. (Cùng quy ước với ``PersistentVideoReader`` hiện có.)

    Attributes:
        _container: Container PyAV đang mở (``None`` khi chưa nạp).
        _stream: Luồng video đang giải mã.
        _current_frame: Frame vừa giải mã gần nhất.
        _current_pts_sec: Mốc thời gian của ``_current_frame`` (giây), tính từ PTS.
    """

    def __init__(self) -> None:
        self._container: Any | None = None
        self._stream: Any | None = None
        self._current_frame: Any | None = None
        self._current_pts_sec: float = 0.0
        self._duration_sec: float = 0.0
        self._fps: float = _FALLBACK_FPS
        self._video_path: Path | None = None

    # ── Thuộc tính chỉ đọc ───────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        """``True`` khi đang có video được nạp."""
        return self._container is not None

    @property
    def duration_sec(self) -> float:
        """Tổng thời lượng video (giây); ``0.0`` nếu chưa nạp."""
        return self._duration_sec

    @property
    def fps(self) -> float:
        """Khung hình/giây trung bình (chỉ dùng để ước lượng, không dùng tính mốc)."""
        return self._fps

    @property
    def position_sec(self) -> float:
        """Mốc thời gian của frame hiện tại (giây), tính từ ``PTS × time_base``."""
        return self._current_pts_sec

    @property
    def frame_duration_sec(self) -> float:
        """Độ dài danh nghĩa của một frame (giây)."""
        return 1.0 / self._fps if self._fps > 0 else 1.0 / _FALLBACK_FPS

    @property
    def video_size(self) -> tuple[int, int]:
        """Kích thước video ``(rộng, cao)``; ``(0, 0)`` nếu chưa nạp."""
        if self._stream is None:
            return (0, 0)
        return (int(self._stream.width), int(self._stream.height))

    # ── Vòng đời ─────────────────────────────────────────────────────────────
    def open(self, video_path: Path) -> None:
        """Mở video và giải mã frame đầu tiên.

        Args:
            video_path: Đường dẫn tệp video.

        Raises:
            VideoNotFoundError: Khi tệp không tồn tại.
            VideoDecodeError: Khi không mở/giải mã được, hoặc không có luồng video.
        """
        if not video_path.is_file():
            raise VideoNotFoundError(f"Không tìm thấy video: {video_path}")

        self.release()

        import av

        try:
            container = av.open(str(video_path))
        except _av_error_types() as exc:
            raise VideoDecodeError(f"Không mở được video: {exc}") from exc

        if not container.streams.video:
            container.close()
            raise VideoDecodeError(f"Video không có luồng hình: {video_path.name}")

        stream = container.streams.video[0]
        # Bật giải mã đa luồng để phát mượt hơn (an toàn, chỉ ảnh hưởng hiệu năng).
        stream.thread_type = "AUTO"

        self._container = container
        self._stream = stream
        self._video_path = video_path
        self._duration_sec = self._read_duration_sec(container, stream)
        self._fps = self._read_fps(stream)

        # Giải mã frame đầu để có trạng thái hợp lệ ngay sau khi mở.
        if self._decode_next_frame() is None:
            self.release()
            raise VideoDecodeError(f"Không giải mã được frame nào: {video_path.name}")

        logger.info(
            "PyAvFrameDecoder mở %s (%dx%d, %.3f fps, %.2fs).",
            video_path.name,
            self._stream.width,
            self._stream.height,
            self._fps,
            self._duration_sec,
        )

    def release(self) -> None:
        """Giải phóng tài nguyên. An toàn khi gọi nhiều lần (idempotent)."""
        if self._container is not None:
            try:
                self._container.close()
            except Exception as exc:  # noqa: BLE001 — dọn dẹp: log rồi bỏ qua
                logger.debug("Bỏ qua lỗi khi đóng container: %s.", exc)
        self._container = None
        self._stream = None
        self._current_frame = None
        self._current_pts_sec = 0.0
        self._duration_sec = 0.0
        self._video_path = None

    # ── Đọc metadata (dùng PTS, không dùng frame_idx/fps) ────────────────────
    @staticmethod
    def _read_duration_sec(container: Any, stream: Any) -> float:
        """Đọc thời lượng theo quy ước PTS của dự án.

        Ưu tiên ``container.duration / av.time_base``; nếu không có thì dùng
        ``stream.duration * stream.time_base``.
        """
        import av

        if container.duration is not None:
            duration = float(container.duration) / av.time_base
            if duration > 0:
                return duration
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        return 0.0

    @staticmethod
    def _read_fps(stream: Any) -> float:
        """Đọc fps trung bình; trả về giá trị dự phòng nếu metadata thiếu."""
        for attr in ("average_rate", "guessed_rate", "base_rate"):
            rate = getattr(stream, attr, None)
            if rate:
                try:
                    value = float(rate)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value
        return _FALLBACK_FPS

    def _frame_pts_sec(self, frame: Any) -> float:
        """Tính mốc thời gian của frame theo ``PTS × time_base``.

        TUYỆT ĐỐI không dùng ``frame_idx / fps`` — sai với video VFR.

        Args:
            frame: Frame PyAV.

        Returns:
            Mốc thời gian (giây); giữ nguyên vị trí hiện tại nếu frame không có PTS.
        """
        pts = getattr(frame, "pts", None)
        if pts is None or self._stream is None or self._stream.time_base is None:
            # Một số frame (hiếm) thiếu PTS — dùng thời điểm đã biết gần nhất.
            return self._current_pts_sec
        return float(pts * self._stream.time_base)

    # ── Giải mã ──────────────────────────────────────────────────────────────
    def _decode_next_frame(self) -> Any | None:
        """Giải mã frame kế tiếp và cập nhật trạng thái.

        Returns:
            Frame vừa giải mã, hoặc ``None`` khi hết video / lỗi giải mã.
        """
        if self._container is None or self._stream is None:
            return None
        try:
            for frame in self._container.decode(self._stream):
                self._current_frame = frame
                self._current_pts_sec = self._frame_pts_sec(frame)
                return frame
        except _av_error_types() as exc:
            logger.debug("Kết thúc/lỗi giải mã: %s.", exc)
        return None

    def next_frame(self) -> np.ndarray | None:
        """Tiến tới frame kế tiếp.

        Returns:
            Ảnh RGB dạng ``ndarray`` (H, W, 3) của frame mới; ``None`` nếu hết video.
        """
        if self._decode_next_frame() is None:
            return None
        return self.current_image()

    def previous_frame(self) -> np.ndarray | None:
        """Lùi lại một frame.

        Cách làm: nhảy về trước mốc hiện tại một khoảng an toàn rồi giải mã tiến tới
        frame CUỐI CÙNG còn nhỏ hơn mốc hiện tại. Cần vậy vì codec chỉ giải mã xuôi.

        Returns:
            Ảnh RGB của frame trước; ``None`` nếu đã ở đầu video hoặc lỗi.
        """
        if self._container is None or self._stream is None:
            return None

        target_sec = self._current_pts_sec
        if target_sec <= 0.0:
            return None

        rewind_to = max(0.0, target_sec - _BACKWARD_SEEK_MARGIN_SEC)
        self._seek_container(rewind_to)

        # Giải mã tiến, giữ lại frame cuối cùng có PTS < mốc hiện tại.
        best_frame: Any | None = None
        best_pts: float = -1.0
        epsilon = self.frame_duration_sec * 0.25  # tránh sai số dấu phẩy động
        while True:
            frame = self._decode_next_frame()
            if frame is None:
                break
            pts = self._current_pts_sec
            if pts >= target_sec - epsilon:
                break
            best_frame, best_pts = frame, pts

        if best_frame is None:
            # Không tìm được frame trước -> quay lại đúng vị trí cũ để không trôi.
            self.seek(target_sec)
            return None

        self._current_frame = best_frame
        self._current_pts_sec = best_pts
        return self.current_image()

    def _seek_container(self, position_sec: float) -> None:
        """Nhảy tới keyframe gần nhất TRƯỚC ``position_sec``.

        Args:
            position_sec: Mốc thời gian đích (giây).
        """
        if self._container is None or self._stream is None:
            return
        time_base = self._stream.time_base
        if time_base is None:
            return
        target_ts = int(max(0.0, position_sec) / float(time_base))
        try:
            self._container.seek(
                target_ts, stream=self._stream, backward=True, any_frame=False
            )
        except _av_error_types() as exc:
            logger.debug("Seek thất bại tại %.3fs: %s.", position_sec, exc)

    def seek(self, position_sec: float) -> np.ndarray | None:
        """Nhảy tới mốc thời gian, chính xác tới frame.

        Nhảy tới keyframe trước mốc đích rồi giải mã tiến cho tới frame đầu tiên có
        ``PTS >= position_sec``. Đây là cách duy nhất để chính xác vì codec liên
        khung chỉ giải mã được theo chiều xuôi từ keyframe.

        Args:
            position_sec: Mốc thời gian đích (giây); tự kẹp vào ``[0, duration]``.

        Returns:
            Ảnh RGB của frame tại mốc đó; ``None`` nếu không giải mã được.
        """
        if self._container is None or self._stream is None:
            return None

        target = max(0.0, position_sec)
        if self._duration_sec > 0:
            target = min(target, self._duration_sec)

        self._seek_container(target)

        epsilon = self.frame_duration_sec * 0.25
        frame = self._decode_next_frame()
        while frame is not None and self._current_pts_sec < target - epsilon:
            frame = self._decode_next_frame()

        if frame is None:
            return None
        return self.current_image()

    def current_image(self) -> np.ndarray | None:
        """Ảnh RGB của frame hiện tại.

        Returns:
            ``ndarray`` (H, W, 3) dtype ``uint8`` theo thứ tự RGB; ``None`` nếu chưa
            có frame nào.
        """
        if self._current_frame is None:
            return None
        try:
            return self._current_frame.to_ndarray(format="rgb24")
        except _av_error_types() as exc:
            logger.warning("Không chuyển được frame sang RGB: %s.", exc)
            return None


__all__ = ["PyAvFrameDecoder"]
