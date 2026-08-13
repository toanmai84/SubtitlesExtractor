"""Worker seek frame + persistent video reader cho ExtractPage canvas.

CẢI TIẾN TRỌNG TÂM:
    1. [LOGIC FIX] Bỏ cờ skip_frame="NONREF": Đảm bảo bộ giải mã quét đủ B-Frames,
       đạt độ chính xác Frame-perfect khi lấy mẫu.
    2. [CRITICAL BUG FIX] Sửa lỗi trượt I-Frame: Hàm đọc không reader giờ đã quét tiến (Scan forward)
       tới đúng mốc thời gian thực tế thay vì trả về Keyframe ngẫu nhiên.
    3. [LOGIC FIX] Sửa lỗi khởi tạo lười (Lazy Init) OpenCV khi gọi next_frame() đầu tiên.
    4. [MEMORY FIX] Đóng Generator đúng cách để giải phóng tài nguyên C-level của libav.
    5. [CRITICAL FIX] Cập nhật Dynamic Exception Fetcher (_get_av_errors) tương thích
       mọi phiên bản thư viện PyAV, khắc phục lỗi mất attribute 'AVError'.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

logger = logging.getLogger(__name__)

_AV_TIME_BASE_MICROSECONDS: int = 1_000_000


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


class PersistentVideoReader:
    """Giữ PyAV container (hoặc OpenCV capture) + decoder mở liên tục.

    Thread-safe: dùng RLock, có thể gọi từ nhiều QThread.
    """

    def __init__(self, video_path: Path) -> None:
        self._video_path = video_path
        self._thread_lock = threading.RLock()

        self._av_container = None
        self._av_stream = None
        self._frame_iterator = None
        self._last_seek_timestamp_sec: float = -1.0

        self._cv2_capture: cv2.VideoCapture | None = None

        self._open_video()

    def _open_video(self) -> None:
        """Lazy mở PyAV; nếu lỗi thì fallback OpenCV."""
        try:
            import av  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "PyAV chưa cài — PersistentVideoReader dùng OpenCV fallback."
            )
            self._av_container = None
            self._av_stream = None
            return

        try:
            self._av_container = av.open(str(self._video_path))
            video_streams = [
                stream
                for stream in self._av_container.streams
                if stream.type == "video"
            ]
            if not video_streams:
                raise ValueError("Video không chứa luồng hình ảnh nào.")
            self._av_stream = video_streams[0]

            logger.info(
                "PersistentVideoReader đã mở PyAV cho: %s (%dx%d).",
                self._video_path.name,
                self._av_stream.width,
                self._av_stream.height,
            )
        except _get_av_errors() as exc:
            logger.warning(
                "PersistentVideoReader: PyAV không mở được %s — %s. "
                "Sẽ dùng OpenCV fallback.",
                self._video_path.name,
                exc,
            )
            if self._av_container is not None:
                with contextlib.suppress(*_get_av_errors(), AttributeError):
                    self._av_container.close()
            self._av_container = None
            self._av_stream = None

    @property
    def is_open(self) -> bool:
        return (self._av_container is not None and self._av_stream is not None) or self._cv2_capture is not None

    def seek(self, timestamp_sec: float) -> np.ndarray | None:
        """Seek đến timestamp và trả frame RGB gần nhất."""
        with self._thread_lock:
            if self._av_container is None or self._av_stream is None:
                return self._fallback_opencv_seek(timestamp_sec)
            try:
                return self._seek_with_pyav(timestamp_sec)
            except _get_av_errors() as exc:
                logger.debug(
                    "PyAV seek lỗi (%s) — fallback OpenCV.", exc
                )
                return self._fallback_opencv_seek(timestamp_sec)

    def next_frame(self) -> np.ndarray | None:
        """Decode frame tiếp theo mà không seek — tối ưu cho sequential play."""
        with self._thread_lock:
            if self._av_container is None or self._av_stream is None:
                if self._cv2_capture is None:
                    self._cv2_capture = cv2.VideoCapture(str(self._video_path))

                if self._cv2_capture.isOpened():
                    read_success, frame_bgr = self._cv2_capture.read()
                    if read_success and frame_bgr is not None:
                        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                return None

            if self._frame_iterator is None:
                return None

            try:
                decoded_frame = next(self._frame_iterator)
                return decoded_frame.to_ndarray(format="rgb24")
            except StopIteration:
                self._frame_iterator = None
                return None
            except _get_av_errors() as exc:
                logger.debug(
                    "PersistentVideoReader.next_frame lỗi: %s.", exc
                )
                self._frame_iterator = None
                return None

    def _seek_with_pyav(self, timestamp_sec: float) -> np.ndarray | None:
        assert self._av_container is not None
        assert self._av_stream is not None

        time_base = float(self._av_stream.time_base) if self._av_stream.time_base else 1.0/90000.0
        seek_pts = int(timestamp_sec / time_base)

        try:
            self._av_container.seek(seek_pts, backward=True, stream=self._av_stream)
        except _get_av_errors():
            try:
                self._av_container.seek(seek_pts, any_frame=True, backward=True, stream=self._av_stream)
            except _get_av_errors():
                return None

        if hasattr(self._frame_iterator, 'close'):
            self._frame_iterator.close()

        self._frame_iterator = self._av_container.decode(self._av_stream)
        self._last_seek_timestamp_sec = timestamp_sec

        best_matched_frame = None
        minimal_diff = float("inf")
        MAX_SCAN_LIMIT_FRAMES = 60

        for i, frame in enumerate(self._frame_iterator):
            if frame.pts is None:
                continue

            frame_timestamp_sec = float(frame.pts * frame.time_base)
            time_difference = abs(frame_timestamp_sec - timestamp_sec)

            if time_difference < minimal_diff:
                minimal_diff = time_difference
                best_matched_frame = frame

            if frame_timestamp_sec >= timestamp_sec or i >= MAX_SCAN_LIMIT_FRAMES:
                break

        if best_matched_frame is not None and minimal_diff < 2.0:
            try:
                return best_matched_frame.to_ndarray(format="rgb24")
            except ValueError as exc:
                logger.debug("to_ndarray lỗi khi seek: %s.", exc)
        return None

    def _fallback_opencv_seek(self, timestamp_sec: float) -> np.ndarray | None:
        """Fallback OpenCV seek — VFR-safe. Cache VideoCapture để không bị chậm."""
        if self._cv2_capture is None:
            self._cv2_capture = cv2.VideoCapture(str(self._video_path))
            if not self._cv2_capture.isOpened():
                self._cv2_capture = None
                return None
            logger.info("PersistentVideoReader mở OpenCV fallback cho: %s", self._video_path.name)

        try:
            self._cv2_capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
            read_success, frame_bgr = self._cv2_capture.read()
            if read_success and frame_bgr is not None:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except cv2.error as exc:
            logger.debug("OpenCV fallback lỗi giải mã: %s", exc)
        return None

    def close(self) -> None:
        """Đóng container/capture. Idempotent — gọi nhiều lần không lỗi."""
        with self._thread_lock:
            if hasattr(self._frame_iterator, 'close'):
                self._frame_iterator.close()
            self._frame_iterator = None

            if self._av_container is not None:
                with contextlib.suppress(*_get_av_errors(), AttributeError):
                    self._av_container.close()
            self._av_container = None
            self._av_stream = None

            if self._cv2_capture is not None:
                try:
                    self._cv2_capture.release()
                except (cv2.error, RuntimeError) as exc:
                    logger.debug("Lỗi khi release OpenCV capture: %s.", exc)
            self._cv2_capture = None

    def __del__(self) -> None:
        with contextlib.suppress(RuntimeError, OSError, AttributeError):
            self.close()


class SeekWorker(QObject):
    """Seek một frame tại ``timestamp_sec``, phát QImage về UI."""

    frame_ready = Signal(object, int, int)
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        timestamp_sec: float,
        reader: PersistentVideoReader | None = None,
        sequential: bool = False,
    ) -> None:
        super().__init__()
        self._video_path = video_path
        self._timestamp_sec = max(0.0, timestamp_sec)
        self._persistent_reader = reader
        self._is_sequential_play = sequential

    def run(self) -> None:
        frame_rgb = None
        if self._persistent_reader is not None:
            if self._is_sequential_play:
                frame_rgb = self._persistent_reader.next_frame()
            else:
                frame_rgb = self._persistent_reader.seek(self._timestamp_sec)
        else:
            frame_rgb = self._read_frame_without_reader()

        if frame_rgb is None:
            self.failed.emit(
                f"Không thể đọc được khung hình tại mốc {self._timestamp_sec:.2f}s."
            )
            return

        frame_height, frame_width = frame_rgb.shape[:2]
        contiguous_array = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        qt_image = QImage(contiguous_array.data, frame_width, frame_height, frame_width * 3, QImage.Format.Format_RGB888).copy()
        self.frame_ready.emit(qt_image, frame_width, frame_height)

    def _read_frame_without_reader(self) -> np.ndarray | None:
        """Fallback khi không có :class:`PersistentVideoReader`."""
        try:
            import av  # type: ignore[import-not-found]
        except ImportError:
            logger.debug("PyAV không khả dụng — thử OpenCV trực tiếp.")
        else:
            try:
                with av.open(str(self._video_path)) as av_container:
                    video_streams = [
                        s for s in av_container.streams if s.type == "video"
                    ]
                    if not video_streams:
                        return None
                    av_stream = video_streams[0]

                    time_base = float(av_stream.time_base) if av_stream.time_base else 1.0/90000.0
                    seek_pts = int(self._timestamp_sec / time_base)

                    with contextlib.suppress(*_get_av_errors(), AttributeError):
                        av_container.seek(
                            seek_pts, backward=True, stream=av_stream
                        )

                    best_frame = None
                    min_diff = float("inf")
                    MAX_SCAN_LIMIT = 60

                    for i, frame in enumerate(av_container.decode(av_stream)):
                        if frame.pts is None:
                            continue

                        frame_time = float(frame.pts * frame.time_base)
                        diff = abs(frame_time - self._timestamp_sec)

                        if diff < min_diff:
                            min_diff = diff
                            best_frame = frame

                        if frame_time >= self._timestamp_sec or i >= MAX_SCAN_LIMIT:
                            break

                    if best_frame is not None and min_diff < 2.0:
                        return best_frame.to_ndarray(format="rgb24")

            except _get_av_errors() as exc:
                logger.debug("SeekWorker fallback PyAV lỗi: %s.", exc)

        try:
            cv2_capture = cv2.VideoCapture(str(self._video_path))
            cv2_capture.set(cv2.CAP_PROP_POS_MSEC, self._timestamp_sec * 1000.0)
            read_success, bgr_frame = cv2_capture.read()
            cv2_capture.release()
            if read_success and bgr_frame is not None:
                return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        except cv2.error as exc:
            logger.warning("SeekWorker OpenCV lỗi giải mã: %s.", exc)
        return None

__all__ = ["PersistentVideoReader", "SeekWorker"]
