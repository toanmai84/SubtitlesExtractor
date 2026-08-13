"""Giải mã âm thanh bằng PyAV — độc lập Qt, phục vụ trình phát license-clean.

VÌ SAO tồn tại
==============
Hoàn thiện trình phát thay ``libmpv`` (GPL): sau phần hình (v3.23.309) cần phần tiếng.
Module này giải mã audio bằng **PyAV** (libav LGPL, đã có trong bundle) và **tái lấy
mẫu** về đúng định dạng thiết bị phát cần (mặc định 48kHz, stereo, PCM 16-bit) —
đây là định dạng ``QAudioSink`` của PySide6 (LGPL) dùng được trực tiếp.

Nguyên tắc
----------
* **KHÔNG phụ thuộc Qt** → kiểm thử được đầy đủ, không cần thiết bị âm thanh.
* Mốc thời gian luôn từ ``PTS × time_base`` (đồng nhất với phần hình).
* Số mẫu quyết định thời gian phát: ``thời lượng = số_mẫu / tần_số_lấy_mẫu``. Đây là
  cơ sở để dùng âm thanh làm ĐỒNG HỒ CHỦ cho đồng bộ A/V — cách chuẩn của mọi trình
  phát, vì phần cứng âm thanh tiêu thụ mẫu ở tốc độ cố định rất chính xác.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import numpy as np

from subtitles_extractor.domain.exceptions import VideoDecodeError, VideoNotFoundError

logger = logging.getLogger(__name__)

# Định dạng đầu ra mặc định — khớp thứ QAudioSink (PySide6) nhận trực tiếp.
DEFAULT_SAMPLE_RATE: Final[int] = 48_000
DEFAULT_CHANNELS: Final[int] = 2
_OUTPUT_FORMAT: Final[str] = "s16"  # PCM 16-bit interleaved
_BYTES_PER_SAMPLE: Final[int] = 2


def _av_error_types() -> tuple[type[BaseException], ...]:
    """Tuple lớp lỗi PyAV, tương thích nhiều phiên bản (xem ``pyav_frame_decoder``)."""
    import av

    errors: list[type[BaseException]] = [OSError, ValueError, RuntimeError, StopIteration]
    ffmpeg_error = getattr(getattr(av, "error", None), "FFmpegError", None)
    if ffmpeg_error is not None:
        errors.append(ffmpeg_error)
    legacy_error = getattr(av, "AVError", None)
    if legacy_error is not None:
        errors.append(legacy_error)
    return tuple(errors)


class PyAvAudioDecoder:
    """Giải mã + tái lấy mẫu âm thanh, trả PCM 16-bit xen kẽ kênh.

    KHÔNG an toàn đa luồng — mỗi luồng dùng một thực thể riêng (cùng quy ước với
    :class:`PyAvFrameDecoder`).

    Args:
        sample_rate: Tần số lấy mẫu đầu ra mong muốn (Hz).
        channels: Số kênh đầu ra (1 = mono, 2 = stereo).
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
    ) -> None:
        self._sample_rate: int = int(sample_rate)
        self._channels: int = int(channels)
        self._container: Any | None = None
        self._stream: Any | None = None
        self._resampler: Any | None = None
        self._position_sec: float = 0.0
        self._samples_read: int = 0
        self._exhausted: bool = False

    # ── Thuộc tính ───────────────────────────────────────────────────────────
    @property
    def has_audio(self) -> bool:
        """``True`` khi có luồng âm thanh đang mở."""
        return self._stream is not None

    @property
    def sample_rate(self) -> int:
        """Tần số lấy mẫu đầu ra (Hz)."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Số kênh đầu ra."""
        return self._channels

    @property
    def bytes_per_frame(self) -> int:
        """Số byte cho MỘT khung mẫu (một mẫu trên mọi kênh)."""
        return self._channels * _BYTES_PER_SAMPLE

    @property
    def position_sec(self) -> float:
        """Mốc thời gian của gói âm thanh vừa đọc (giây), tính từ ``PTS``."""
        return self._position_sec

    @property
    def samples_read(self) -> int:
        """Tổng số khung mẫu đã đọc kể từ lần seek gần nhất."""
        return self._samples_read

    @property
    def is_exhausted(self) -> bool:
        """``True`` khi đã đọc hết luồng âm thanh."""
        return self._exhausted

    # ── Vòng đời ─────────────────────────────────────────────────────────────
    def open(self, video_path: Path) -> bool:
        """Mở luồng âm thanh của tệp.

        Args:
            video_path: Đường dẫn tệp media.

        Returns:
            ``True`` nếu tệp CÓ luồng âm thanh; ``False`` nếu không có (KHÔNG phải
            lỗi — video câm là hợp lệ, trình phát vẫn chạy phần hình).

        Raises:
            VideoNotFoundError: Khi tệp không tồn tại.
            VideoDecodeError: Khi không mở được tệp.
        """
        if not video_path.is_file():
            raise VideoNotFoundError(f"Không tìm thấy tệp: {video_path}")

        self.release()

        import av

        try:
            container = av.open(str(video_path))
        except _av_error_types() as exc:
            raise VideoDecodeError(f"Không mở được tệp: {exc}") from exc

        if not container.streams.audio:
            container.close()
            logger.info("Tệp %s không có luồng âm thanh.", video_path.name)
            return False

        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"

        from av.audio.resampler import AudioResampler

        layout = "stereo" if self._channels == 2 else "mono"
        self._container = container
        self._stream = stream
        self._resampler = AudioResampler(
            format=_OUTPUT_FORMAT, layout=layout, rate=self._sample_rate
        )
        self._position_sec = 0.0
        self._samples_read = 0
        self._exhausted = False

        logger.info(
            "PyAvAudioDecoder: %s — nguồn %dHz/%d kênh → ra %dHz/%d kênh.",
            video_path.name,
            stream.sample_rate or 0,
            getattr(stream, "channels", 0) or 0,
            self._sample_rate,
            self._channels,
        )
        return True

    def release(self) -> None:
        """Giải phóng tài nguyên. Idempotent."""
        if self._container is not None:
            try:
                self._container.close()
            except Exception as exc:  # noqa: BLE001 — dọn dẹp: log rồi bỏ qua
                logger.debug("Bỏ qua lỗi khi đóng container audio: %s.", exc)
        self._container = None
        self._stream = None
        self._resampler = None
        self._position_sec = 0.0
        self._samples_read = 0
        self._exhausted = False

    # ── Đọc dữ liệu ──────────────────────────────────────────────────────────
    def read_chunk(self) -> bytes | None:
        """Đọc gói âm thanh kế tiếp, đã tái lấy mẫu.

        Returns:
            Chuỗi byte PCM 16-bit xen kẽ kênh; ``None`` khi hết luồng hoặc chưa mở.
        """
        if self._container is None or self._stream is None or self._resampler is None:
            return None
        if self._exhausted:
            return None

        try:
            for frame in self._container.decode(self._stream):
                pts_sec = self._frame_pts_sec(frame)
                for resampled in self._resampler.resample(frame):
                    array = resampled.to_ndarray()
                    if array.size == 0:
                        continue
                    # to_ndarray() cho s16 xen kẽ trả về dạng (1, n*channels).
                    pcm = np.ascontiguousarray(array, dtype=np.int16).tobytes()
                    frame_count = len(pcm) // self.bytes_per_frame
                    self._samples_read += frame_count
                    self._position_sec = pts_sec
                    return pcm
        except _av_error_types() as exc:
            logger.debug("Kết thúc/lỗi giải mã âm thanh: %s.", exc)

        self._exhausted = True
        return None

    def read_duration(self, seconds: float) -> bytes:
        """Đọc gần đúng ``seconds`` giây âm thanh (gộp nhiều gói).

        Args:
            seconds: Thời lượng mong muốn (giây).

        Returns:
            Chuỗi byte PCM; có thể ngắn hơn yêu cầu nếu gặp cuối luồng.
        """
        wanted_bytes = int(seconds * self._sample_rate) * self.bytes_per_frame
        parts: list[bytes] = []
        collected = 0
        while collected < wanted_bytes:
            chunk = self.read_chunk()
            if chunk is None:
                break
            parts.append(chunk)
            collected += len(chunk)
        return b"".join(parts)

    def seek(self, position_sec: float) -> None:
        """Nhảy tới mốc thời gian và xoá bộ đệm tái lấy mẫu.

        Args:
            position_sec: Mốc đích (giây).
        """
        if self._container is None or self._stream is None:
            return

        time_base = self._stream.time_base
        if time_base is not None:
            target_ts = int(max(0.0, position_sec) / float(time_base))
            try:
                self._container.seek(
                    target_ts, stream=self._stream, backward=True, any_frame=False
                )
            except _av_error_types() as exc:
                logger.debug("Seek âm thanh lỗi tại %.3fs: %s.", position_sec, exc)

        # Dựng lại resampler: bộ đệm bên trong còn mẫu CŨ, không xoá sẽ phát nhầm.
        from av.audio.resampler import AudioResampler

        layout = "stereo" if self._channels == 2 else "mono"
        self._resampler = AudioResampler(
            format=_OUTPUT_FORMAT, layout=layout, rate=self._sample_rate
        )
        self._position_sec = max(0.0, position_sec)
        self._samples_read = 0
        self._exhausted = False

    # ── Nội bộ ───────────────────────────────────────────────────────────────
    def _frame_pts_sec(self, frame: Any) -> float:
        """Mốc thời gian của khung âm thanh theo ``PTS × time_base``."""
        pts = getattr(frame, "pts", None)
        time_base = getattr(frame, "time_base", None) or (
            self._stream.time_base if self._stream is not None else None
        )
        if pts is None or time_base is None:
            return self._position_sec
        return float(pts * time_base)


__all__ = ["DEFAULT_CHANNELS", "DEFAULT_SAMPLE_RATE", "PyAvAudioDecoder"]
