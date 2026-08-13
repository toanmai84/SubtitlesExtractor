"""Thiết bị phát âm thanh dùng ``QAudioSink`` (PySide6, LGPL).

Hiện thực :class:`AudioSinkPort` cho trình phát PyAV — thay phần âm thanh của
``libmpv`` (GPL). PySide6 đã có sẵn trong bundle nên KHÔNG thêm phụ thuộc mới.

⚠️ CHƯA KIỂM THỬ TỰ ĐỘNG: lớp này cần thiết bị âm thanh thật nên không chạy được
trong CI. Toàn bộ **logic đồng bộ A/V** đã được kiểm chứng riêng qua
:class:`AudioSinkPort` với thiết bị giả (xem ``test_pyav_audio_sync_v323310.py``);
phần chưa kiểm ở đây chỉ là nối dây với Qt.
"""

from __future__ import annotations

import logging
from typing import Final

from PySide6.QtCore import QIODevice
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

logger = logging.getLogger(__name__)

_BYTES_PER_SAMPLE: Final[int] = 2  # PCM 16-bit

# Đệm ~200ms: đủ chống giật khi hệ thống bận, vẫn đủ nhỏ để seek phản hồi nhanh.
_BUFFER_SECONDS: Final[float] = 0.2


class QtAudioSink:
    """Bọc ``QAudioSink`` theo hợp đồng :class:`AudioSinkPort`.

    Args:
        sample_rate: Tần số lấy mẫu (Hz) — phải khớp đầu ra của bộ giải mã.
        channels: Số kênh (1 hoặc 2).

    Raises:
        RuntimeError: Khi không có thiết bị âm thanh hoặc định dạng không được hỗ trợ.
    """

    def __init__(self, *, sample_rate: int = 48_000, channels: int = 2) -> None:
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._bytes_per_frame = self._channels * _BYTES_PER_SAMPLE

        audio_format = QAudioFormat()
        audio_format.setSampleRate(self._sample_rate)
        audio_format.setChannelCount(self._channels)
        audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        device = QMediaDevices.defaultAudioOutput()
        if device is None or device.isNull():
            raise RuntimeError("Không tìm thấy thiết bị phát âm thanh.")
        if not device.isFormatSupported(audio_format):
            raise RuntimeError(
                f"Thiết bị không hỗ trợ {self._sample_rate}Hz/{self._channels} kênh/Int16."
            )

        self._sink = QAudioSink(device, audio_format)
        self._sink.setBufferSize(
            int(self._sample_rate * _BUFFER_SECONDS) * self._bytes_per_frame
        )
        self._io: QIODevice | None = self._sink.start()
        self._closed = False
        # Mốc trừ hao: QAudioSink đếm micro-giây LUỸ KẾ từ lúc start(), không tự về 0
        # khi reset() -> phải tự trừ đi mốc tại thời điểm reset.
        self._processed_offset_us: int = 0

    # ── Hợp đồng AudioSinkPort ───────────────────────────────────────────────
    @property
    def free_bytes(self) -> int:
        """Số byte thiết bị còn nhận thêm được ngay lúc này."""
        if self._closed or self._io is None:
            return 0
        return max(0, int(self._sink.bytesFree()))

    def write(self, pcm: bytes) -> int:
        """Đẩy PCM xuống thiết bị.

        Args:
            pcm: PCM 16-bit xen kẽ kênh.

        Returns:
            Số byte đã ghi (có thể ít hơn ``len(pcm)``); ``0`` nếu thiết bị đã đóng.
        """
        if self._closed or self._io is None or not pcm:
            return 0
        writable = min(len(pcm), self.free_bytes)
        if writable <= 0:
            return 0
        # Cắt theo bội số khung mẫu — ghi lẻ nửa khung sẽ làm lệch kênh trái/phải.
        writable -= writable % self._bytes_per_frame
        if writable <= 0:
            return 0
        written = self._io.write(pcm[:writable])
        return max(0, int(written))

    def played_seconds(self) -> float:
        """Thời lượng đã phát (giây) kể từ lần :meth:`reset` gần nhất."""
        if self._closed:
            return 0.0
        elapsed_us = int(self._sink.processedUSecs()) - self._processed_offset_us
        return max(0.0, elapsed_us / 1_000_000.0)

    def reset(self) -> None:
        """Xoá bộ đệm và đưa bộ đếm đã-phát về 0 (dùng khi seek)."""
        if self._closed:
            return
        # QAudioSink không cho zero lại processedUSecs -> khởi động lại thiết bị.
        self._sink.stop()
        self._io = self._sink.start()
        self._processed_offset_us = int(self._sink.processedUSecs())

    def set_volume(self, volume: float) -> None:
        """Đặt âm lượng 0.0–1.0."""
        if self._closed:
            return
        self._sink.setVolume(max(0.0, min(1.0, float(volume))))

    def close(self) -> None:
        """Đóng thiết bị. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._sink.stop()
        except Exception as exc:  # noqa: BLE001 — dọn dẹp: log rồi bỏ qua
            logger.debug("Bỏ qua lỗi khi dừng QAudioSink: %s.", exc)
        self._io = None


def try_create_qt_audio_sink(
    *, sample_rate: int = 48_000, channels: int = 2
) -> QtAudioSink | None:
    """Tạo thiết bị âm thanh Qt, trả ``None`` nếu máy không có/không hỗ trợ.

    Video không tiếng vẫn phải phát được — nên thiếu thiết bị âm thanh KHÔNG được
    coi là lỗi nghiêm trọng.

    Returns:
        :class:`QtAudioSink` khi tạo được; ``None`` nếu không.
    """
    try:
        return QtAudioSink(sample_rate=sample_rate, channels=channels)
    except Exception as exc:  # noqa: BLE001 — thiếu tiếng không được chặn phát hình
        logger.warning("Không tạo được thiết bị âm thanh (sẽ phát không tiếng): %s.", exc)
        return None


__all__ = ["QtAudioSink", "try_create_qt_audio_sink"]
