"""Trình phát video dựa trên PyAV — hiện thực :class:`VideoPlayerPort` (license-clean).

VÌ SAO tồn tại
==============
``libmpv-2.dll`` bản Windows dựng sẵn phổ biến là **GPL** (nhúng ffmpeg build kèm
x264/x265 — xem v3.23.308), không dùng được cho phân phối thương mại license-clean.
Adapter này thay thế bằng **PyAV** (libav LGPL, ĐÃ audit, ĐÃ có trong bundle) —
không thêm bất kỳ phụ thuộc mới nào.

Âm thanh & đồng bộ A/V (v3.23.310)
----------------------------------
Khi truyền ``audio_sink``, adapter phát cả tiếng và dùng **âm thanh làm ĐỒNG HỒ CHỦ**
— chuẩn của mọi trình phát, vì phần cứng âm thanh tiêu thụ mẫu ở tốc độ cố định rất
chính xác. Hình được kéo theo cho khớp vị trí âm thanh.

Không truyền ``audio_sink`` (hoặc video câm) → quay về dùng đồng hồ hệ thống, hành vi
y hệt v3.23.309.

**Giới hạn đã biết:** khi ``speed != 1.0``, âm thanh bị TẮT và quay về đồng hồ hệ
thống. Giữ cao độ khi đổi tốc độ cần kéo giãn thời gian (time-stretch) — sẽ làm sau
nếu cần; hiện ưu tiên đúng đắn hơn là đủ tính năng.

* Adapter ``MpvPlayerAdapter`` được GIỮ NGUYÊN song song để so sánh và lùi lại được.

Thiết kế để kiểm thử được
-------------------------
Adapter KHÔNG phụ thuộc Qt và KHÔNG tự tạo timer. Việc phát được điều khiển bằng
:meth:`tick` do tầng trên gọi (Qt dùng ``QTimer``, test dùng đồng hồ giả). Nhờ vậy
toàn bộ logic phát/seek/bước frame kiểm thử được không cần màn hình.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Final

import numpy as np

from subtitles_extractor.infrastructure.video.audio_sink_port import AudioSinkPort
from subtitles_extractor.infrastructure.video.pyav_audio_decoder import PyAvAudioDecoder
from subtitles_extractor.infrastructure.video.pyav_frame_decoder import PyAvFrameDecoder

logger = logging.getLogger(__name__)

# Giới hạn tốc độ phát (khớp hợp đồng VideoPlayerPort.set_speed).
_MIN_SPEED: Final[float] = 0.25
_MAX_SPEED: Final[float] = 4.0

# Khi tụt ít, giải mã tuần tự để bắt kịp (mượt). Giới hạn số frame mỗi nhịp để máy
# yếu không bị dồn việc.
_MAX_CATCHUP_FRAMES: Final[int] = 5

# [v3.23.310] Khi tụt QUÁ ngưỡng này so với đồng hồ chủ, giải mã tuần tự sẽ không bao
# giờ đuổi kịp -> nhảy thẳng (seek) tới đúng vị trí. Tình huống gặp thật: máy khựng,
# cửa sổ bị ẩn, hoặc thiết bị âm thanh nhảy mốc sau khi hệ thống ngủ.
_RESYNC_SEEK_THRESHOLD_SEC: Final[float] = 0.5

FrameCallback = Callable[[np.ndarray], None]
"""Hàm nhận ảnh RGB (H, W, 3) mỗi khi có frame mới để hiển thị."""


class PyAvPlayerAdapter:
    """Trình phát video chỉ-hình dựa trên PyAV, hiện thực :class:`VideoPlayerPort`.

    Tầng trên phải gọi :meth:`tick` định kỳ (vd ``QTimer`` ~10ms) để adapter tự quyết
    khi nào sang frame kế tiếp dựa trên đồng hồ thực và ``PTS`` của frame.

    Args:
        on_frame: Callback nhận ảnh RGB mỗi khi có frame mới (hiển thị lên UI).
        clock: Hàm trả về thời gian đơn điệu (giây). Mặc định
            :func:`time.monotonic`; test truyền đồng hồ giả để kiểm soát thời gian.

    Attributes:
        _decoder: Lõi giải mã PyAV.
        _is_playing: Trạng thái phát.
        _speed: Hệ số tốc độ phát.
        _volume: Âm lượng 0–100 (ghi nhớ, chưa dùng vì chưa có tiếng).
    """

    def __init__(
        self,
        on_frame: FrameCallback | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        audio_sink: AudioSinkPort | None = None,
    ) -> None:
        self._decoder = PyAvFrameDecoder()
        self._audio_decoder = PyAvAudioDecoder()
        self._audio_sink: AudioSinkPort | None = audio_sink
        self._audio_available: bool = False
        self._audio_anchor_media: float = 0.0
        self._pending_pcm: bytes = b""
        self._on_frame: FrameCallback | None = on_frame
        self._clock = clock
        self._is_playing: bool = False
        self._speed: float = 1.0
        self._volume: int = 100
        self._muted: bool = False
        # Mốc neo để tính thời gian phát: (thời điểm đồng hồ, vị trí video tương ứng).
        self._anchor_wall: float = 0.0
        self._anchor_media: float = 0.0

    # ── Hợp đồng VideoPlayerPort: thuộc tính ─────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        """``True`` khi có media đang được nạp."""
        return self._decoder.is_loaded

    @property
    def is_playing(self) -> bool:
        """``True`` khi đang phát (không pause)."""
        return self._is_playing

    @property
    def position_sec(self) -> float:
        """Vị trí phát hiện tại (giây) — lấy từ ``PTS`` của frame đang hiển thị."""
        return self._decoder.position_sec

    @property
    def duration_sec(self) -> float:
        """Tổng thời lượng (giây); 0 nếu chưa nạp."""
        return self._decoder.duration_sec

    # ── Thuộc tính mở rộng (tương đương adapter mpv) ─────────────────────────
    @property
    def video_width(self) -> int:
        """Chiều rộng khung hình (px)."""
        return self._decoder.video_size[0]

    @property
    def video_height(self) -> int:
        """Chiều cao khung hình (px)."""
        return self._decoder.video_size[1]

    @property
    def eof_reached(self) -> bool:
        """``True`` khi đã phát tới cuối video."""
        duration = self._decoder.duration_sec
        if duration <= 0:
            return False
        return self._decoder.position_sec >= duration - self._decoder.frame_duration_sec

    def set_frame_callback(self, on_frame: FrameCallback | None) -> None:
        """Đặt/thay callback nhận frame (cho phép nối UI sau khi khởi tạo)."""
        self._on_frame = on_frame

    # ── Hợp đồng VideoPlayerPort: hành vi ────────────────────────────────────
    def load(self, video_path: Path) -> None:
        """Nạp video — KHÔNG tự phát (đúng hợp đồng port).

        Args:
            video_path: Đường dẫn video.

        Raises:
            VideoNotFoundError: Khi tệp không tồn tại.
            VideoDecodeError: Khi không giải mã được.
        """
        self._is_playing = False
        self._decoder.open(video_path)
        self._open_audio(video_path)
        self._reset_anchor()
        self._emit_current_frame()

    def play(self) -> None:
        """Phát từ vị trí hiện tại."""
        if not self._decoder.is_loaded or self._is_playing:
            return
        self._is_playing = True
        self._resync_audio_to(self._decoder.position_sec)
        self._reset_anchor()

    def pause(self) -> None:
        """Tạm dừng."""
        self._is_playing = False

    def toggle_play_pause(self) -> None:
        """Đảo trạng thái play/pause."""
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def seek(self, position_sec: float) -> None:
        """Nhảy tới mốc thời gian (giây), chính xác tới frame.

        Args:
            position_sec: Mốc đích; tự kẹp vào ``[0, duration]``.
        """
        if not self._decoder.is_loaded:
            return
        image = self._decoder.seek(position_sec)
        self._resync_audio_to(self._decoder.position_sec)
        self._reset_anchor()
        if image is not None:
            self._deliver(image)

    def step_frame(self, forward: bool = True) -> None:
        """Bước đúng 1 frame (tiến hoặc lùi). Tự động tạm dừng.

        Args:
            forward: ``True`` tiến, ``False`` lùi.
        """
        if not self._decoder.is_loaded:
            return
        # Bước frame là thao tác canh chỉnh chính xác -> luôn dừng phát trước.
        self._is_playing = False
        image = (
            self._decoder.next_frame() if forward else self._decoder.previous_frame()
        )
        self._resync_audio_to(self._decoder.position_sec)
        self._reset_anchor()
        if image is not None:
            self._deliver(image)

    def set_volume(self, volume: int) -> None:
        """Đặt âm lượng 0–100.

        Hiện CHƯA phát tiếng nên giá trị chỉ được ghi nhớ (xem docstring module).

        Args:
            volume: Âm lượng 0–100 (tự kẹp).
        """
        self._volume = max(0, min(100, int(volume)))
        if self._audio_sink is not None:
            self._audio_sink.set_volume(0.0 if self._muted else self._volume / 100.0)

    def set_speed(self, speed: float) -> None:
        """Đặt tốc độ phát trong khoảng 0.25–4.0.

        Args:
            speed: Hệ số tốc độ (tự kẹp về khoảng hợp lệ).
        """
        clamped = max(_MIN_SPEED, min(_MAX_SPEED, float(speed)))
        # Neo lại theo vị trí hiện tại để đổi tốc độ không gây nhảy hình.
        self._anchor_media = self._decoder.position_sec
        self._anchor_wall = self._clock()
        self._speed = clamped

    def set_mute(self, mute: bool) -> None:
        """Bật/tắt câm (ghi nhớ; chưa có tiếng nên chưa có tác dụng)."""
        self._muted = bool(mute)
        if self._audio_sink is not None:
            self._audio_sink.set_volume(0.0 if self._muted else self._volume / 100.0)

    def release(self) -> None:
        """Giải phóng tài nguyên. An toàn khi gọi nhiều lần (idempotent)."""
        self._is_playing = False
        self._decoder.release()
        self._audio_decoder.release()
        self._audio_available = False
        self._pending_pcm = b""
        if self._audio_sink is not None:
            self._audio_sink.close()

    # ── Điều khiển phát (tầng trên gọi định kỳ) ──────────────────────────────
    def tick(self) -> None:
        """Cập nhật khung hình (và bơm âm thanh) — gọi định kỳ khi đang phát.

        Chọn ĐỒNG HỒ theo thứ tự ưu tiên:
            1. **Âm thanh** (khi có tiếng và tốc độ = 1.0): vị trí = mốc neo + thời
               lượng thiết bị đã phát. Đây là đồng hồ chuẩn xác nhất vì phần cứng âm
               thanh tiêu thụ mẫu ở tốc độ cố định.
            2. **Đồng hồ hệ thống**: khi câm hoặc đang đổi tốc độ.

        Frame được tiến dựa trên ``PTS`` thật, không giả định khoảng cách đều — nhờ
        vậy đúng cả với video VFR.
        """
        if not self._is_playing or not self._decoder.is_loaded:
            return

        if self._audio_playing:
            self._pump_audio()

        target_media = self._media_target()
        lag = target_media - self._decoder.position_sec

        # Tụt quá xa: giải mã tuần tự không thể đuổi kịp -> nhảy thẳng tới đích.
        if lag > _RESYNC_SEEK_THRESHOLD_SEC:
            if target_media >= self._decoder.duration_sec > 0:
                self._is_playing = False
                logger.debug("PyAvPlayerAdapter: mốc đồng bộ vượt cuối video.")
                return
            logger.debug("Hình tụt %.3fs so với đồng hồ chủ — nhảy thẳng.", lag)
            image = self._decoder.seek(target_media)
            if image is None:
                self._is_playing = False
                return
            self._deliver(image)
            return

        advanced = 0
        while self._decoder.position_sec < target_media:
            if advanced >= _MAX_CATCHUP_FRAMES:
                # Máy yếu tạm thời: dừng bắt kịp ở nhịp này, nhịp sau tiếp tục.
                break
            image = self._decoder.next_frame()
            if image is None:
                # Hết video.
                self._is_playing = False
                logger.debug("PyAvPlayerAdapter: đã tới cuối video.")
                break
            advanced += 1
            self._deliver(image)

    # ── Âm thanh & đồng bộ ───────────────────────────────────────────────────
    @property
    def has_audio(self) -> bool:
        """``True`` khi video có tiếng VÀ đã gắn thiết bị phát."""
        return self._audio_available and self._audio_sink is not None

    @property
    def _audio_playing(self) -> bool:
        """``True`` khi âm thanh đang thực sự được dùng làm đồng hồ chủ.

        Đổi tốc độ sẽ làm sai cao độ nếu phát thẳng, nên khi ``speed != 1.0`` ta tắt
        tiếng và quay về đồng hồ hệ thống (xem giới hạn đã biết ở docstring module).
        """
        return self.has_audio and abs(self._speed - 1.0) < 1e-6

    def _media_target(self) -> float:
        """Mốc thời gian media mà hình PHẢI hiển thị tại thời điểm này."""
        if self._audio_playing and self._audio_sink is not None:
            return self._audio_anchor_media + self._audio_sink.played_seconds()
        elapsed = (self._clock() - self._anchor_wall) * self._speed
        return self._anchor_media + elapsed

    def _open_audio(self, video_path: Path) -> None:
        """Mở luồng âm thanh (nếu có) và chuẩn bị thiết bị phát."""
        self._audio_available = False
        self._pending_pcm = b""
        if self._audio_sink is None:
            return
        try:
            self._audio_available = self._audio_decoder.open(video_path)
        except Exception as exc:  # noqa: BLE001 — thiếu tiếng KHÔNG được chặn phát hình
            logger.warning("Không mở được luồng âm thanh (vẫn phát hình): %s.", exc)
            self._audio_available = False
            return
        if self._audio_available:
            self._audio_sink.reset()
            self._audio_sink.set_volume(
                0.0 if self._muted else self._volume / 100.0
            )
            self._audio_anchor_media = 0.0

    def _resync_audio_to(self, position_sec: float) -> None:
        """Đưa âm thanh về đúng mốc ``position_sec`` (dùng khi seek/step/play).

        Args:
            position_sec: Mốc media đích (giây).
        """
        if not self.has_audio or self._audio_sink is None:
            return
        self._audio_decoder.seek(position_sec)
        self._audio_sink.reset()
        self._pending_pcm = b""
        # Neo lại: từ giờ "đã phát" của thiết bị được tính từ mốc này.
        self._audio_anchor_media = position_sec

    def _pump_audio(self) -> None:
        """Bơm dữ liệu PCM xuống thiết bị cho tới khi thiết bị đầy hoặc hết tiếng."""
        sink = self._audio_sink
        if sink is None:
            return
        while True:
            free = sink.free_bytes
            if free <= 0:
                return
            if not self._pending_pcm:
                chunk = self._audio_decoder.read_chunk()
                if chunk is None:
                    return  # hết tiếng — hình vẫn chạy tiếp bằng đồng hồ hệ thống
                self._pending_pcm = chunk
            written = sink.write(self._pending_pcm)
            if written <= 0:
                return
            self._pending_pcm = self._pending_pcm[written:]

    # ── Nội bộ ───────────────────────────────────────────────────────────────
    def _reset_anchor(self) -> None:
        """Đặt lại mốc neo thời gian về vị trí hiện tại."""
        self._anchor_wall = self._clock()
        self._anchor_media = self._decoder.position_sec

    def _emit_current_frame(self) -> None:
        """Gửi frame hiện tại lên UI (nếu có)."""
        image = self._decoder.current_image()
        if image is not None:
            self._deliver(image)

    def _deliver(self, image: np.ndarray) -> None:
        """Gọi callback hiển thị, nuốt lỗi từ tầng UI để không làm hỏng vòng phát."""
        if self._on_frame is None:
            return
        try:
            self._on_frame(image)
        except Exception as exc:  # noqa: BLE001 — lỗi UI không được phá vòng phát
            logger.warning("Callback hiển thị frame lỗi: %s.", exc)


__all__ = ["FrameCallback", "PyAvPlayerAdapter"]
