"""Widget Qt hiển thị video từ :class:`PyAvPlayerAdapter` (license-clean, không mpv).

VÌ SAO tồn tại
==============
``libmpv-2.dll`` dựng sẵn là GPL (xem v3.23.308) nên không dùng được cho phân phối
thương mại. Widget này hiển thị video bằng **PyAV + Qt** — cả hai đều LGPL và đã có
sẵn trong bundle, không thêm phụ thuộc mới.

TRẠNG THÁI: BẢN ĐÁNH GIÁ
------------------------
Widget này CỐ Ý giữ tối giản để bạn thử chất lượng phát trên video thật TRƯỚC khi
quyết định di trú toàn bộ. Nó **chưa thay thế** ``MpvVideoWidget`` (1093 dòng, có lớp
phủ ROI, OSD…) — widget đó vẫn là mặc định và không bị đụng tới.

Khác biệt so với ``MpvVideoWidget``:
    * CHƯA có OSD nâng cao của mpv (chỉ vẽ ROI cơ bản).
    * Hiển thị bằng ``QPainter`` (CPU) thay vì ``vo=gpu-next`` của mpv.

Cách thử nhanh::

    python -m subtitles_extractor.presentation.widgets.pyav_video_widget video.mp4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from subtitles_extractor.infrastructure.video.pyav_player_adapter import PyAvPlayerAdapter
from subtitles_extractor.presentation.widgets.qt_audio_sink import try_create_qt_audio_sink
from subtitles_extractor.presentation.widgets.roi_geometry import (
    compute_display_geometry,
    video_rect_to_widget,
    widget_rect_to_video,
)

logger = logging.getLogger(__name__)

# Chu kỳ gọi tick (ms). ~4ms cho phép bám sát video tới ~240fps mà vẫn nhẹ CPU.
_TICK_INTERVAL_MS: Final[int] = 4


class PyAvVideoWidget(QWidget):
    """Hiển thị video giải mã bằng PyAV, điều khiển qua :class:`PyAvPlayerAdapter`.

    Signals:
        position_changed: Phát ra mốc thời gian (giây) mỗi khi có frame mới.
        playback_finished: Phát ra khi phát tới cuối video.
    """

    position_changed = Signal(float)
    playback_finished = Signal()
    roi_drawn = Signal(object)
    """Phát ra ``(x, y, rộng, cao)`` theo toạ độ VIDEO khi vẽ xong ROI; ``None`` nếu
    vùng kéo nằm ngoài ảnh."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # [v3.23.310] Gắn thiết bị âm thanh Qt (LGPL). Máy không có thiết bị -> None,
        # trình phát tự chạy không tiếng thay vì hỏng.
        self._audio_sink = try_create_qt_audio_sink()
        self._player = PyAvPlayerAdapter(
            on_frame=self._on_frame, audio_sink=self._audio_sink
        )
        self._image: QImage | None = None
        self._was_playing: bool = False

        # QTimer điều khiển nhịp phát; adapter tự quyết khi nào sang frame kế tiếp
        # dựa trên đồng hồ thực + PTS, nên chu kỳ timer không cần khớp fps video.
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(_TICK_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

        # [v3.23.311] Trạng thái vẽ ROI. Toạ độ ROI luôn lưu theo VIDEO (bất biến khi
        # đổi cỡ cửa sổ); chỉ quy đổi sang toạ độ khung lúc vẽ.
        self._roi_drawing_enabled: bool = False
        self._committed_roi_video: tuple[int, int, int, int] | None = None
        self._secondary_rois_video: list[tuple[int, int, int, int]] = []
        self._drag_origin: QPoint | None = None
        self._drag_current: QPoint | None = None

        self.setAutoFillBackground(True)
        self.setMinimumSize(320, 180)

    # ── API công khai (uỷ quyền cho adapter) ─────────────────────────────────
    @property
    def player(self) -> PyAvPlayerAdapter:
        """Adapter bên dưới — dùng khi cần các thao tác nâng cao."""
        return self._player

    def load(self, video_path: Path) -> None:
        """Nạp video (không tự phát)."""
        self._player.load(video_path)
        self._timer.start()
        self.update()

    def play(self) -> None:
        """Phát từ vị trí hiện tại."""
        self._player.play()
        self._was_playing = True

    def pause(self) -> None:
        """Tạm dừng."""
        self._player.pause()

    def toggle_play_pause(self) -> None:
        """Đảo trạng thái play/pause."""
        self._player.toggle_play_pause()

    def seek(self, position_sec: float) -> None:
        """Nhảy tới mốc thời gian (giây)."""
        self._player.seek(position_sec)

    def step_frame(self, forward: bool = True) -> None:
        """Bước đúng 1 frame (tự động tạm dừng)."""
        self._player.step_frame(forward)

    def set_volume(self, volume: int) -> None:
        """Đặt âm lượng 0–100."""
        self._player.set_volume(volume)

    # ── Vùng ROI ─────────────────────────────────────────────────────────────
    def enable_roi_drawing(self, enabled: bool) -> None:
        """Bật/tắt chế độ kéo chuột để vẽ ROI."""
        self._roi_drawing_enabled = bool(enabled)
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        if not enabled:
            self._drag_origin = None
            self._drag_current = None
        self.update()

    def set_committed_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        """Đặt ROI chính đang áp dụng (toạ độ VIDEO) để hiển thị."""
        self._committed_roi_video = roi
        self.update()

    def set_secondary_rois(self, rois: list[tuple[int, int, int, int]]) -> None:
        """Đặt danh sách ROI phụ (toạ độ VIDEO) để hiển thị."""
        self._secondary_rois_video = list(rois)
        self.update()

    def clear_roi(self) -> None:
        """Xoá mọi ROI đang hiển thị."""
        self._committed_roi_video = None
        self._secondary_rois_video = []
        self._drag_origin = None
        self._drag_current = None
        self.update()

    def _display_geometry(self):
        """Thông số thu phóng/căn giữa hiện tại (theo kích thước video và khung)."""
        return compute_display_geometry(
            video_width=self._player.video_width,
            video_height=self._player.video_height,
            widget_width=self.width(),
            widget_height=self.height(),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        """Bắt đầu kéo vẽ ROI."""
        if self._roi_drawing_enabled and event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.position().toPoint()
            self._drag_current = self._drag_origin
            self.update()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        """Cập nhật khung xem trước khi đang kéo."""
        if self._drag_origin is not None:
            self._drag_current = event.position().toPoint()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override
        """Kết thúc kéo: quy đổi sang toạ độ VIDEO rồi phát tín hiệu."""
        if self._drag_origin is None:
            super().mouseReleaseEvent(event)
            return

        rect = self._drag_rect()
        self._drag_origin = None
        self._drag_current = None

        roi = None
        if rect is not None:
            roi = widget_rect_to_video(
                (rect.x(), rect.y(), rect.width(), rect.height()),
                self._display_geometry(),
            )
        if roi is not None:
            self._committed_roi_video = roi
        self.roi_drawn.emit(roi)
        self.update()

    def _drag_rect(self) -> QRect | None:
        """Hình chữ nhật đang kéo (toạ độ khung), đã chuẩn hoá; ``None`` nếu quá nhỏ."""
        if self._drag_origin is None or self._drag_current is None:
            return None
        rect = QRect(self._drag_origin, self._drag_current).normalized()
        if rect.width() < 2 or rect.height() < 2:
            return None
        return rect

    def set_speed(self, speed: float) -> None:
        """Đặt tốc độ phát 0.25–4.0 (tiếng tự tắt khi khác 1.0)."""
        self._player.set_speed(speed)

    def release(self) -> None:
        """Dừng timer và giải phóng tài nguyên. Idempotent."""
        self._timer.stop()
        self._player.release()
        self._image = None

    # ── Nội bộ ───────────────────────────────────────────────────────────────
    def _on_frame(self, image: np.ndarray) -> None:
        """Nhận ảnh RGB từ adapter, chuyển sang ``QImage`` để vẽ.

        Args:
            image: Mảng ``(H, W, 3)`` dtype ``uint8``, thứ tự RGB.
        """
        height, width, _ = image.shape
        # np.ascontiguousarray + copy(): QImage KHÔNG sở hữu bộ nhớ numpy, nếu không
        # copy thì buffer bị thu hồi -> ảnh rác hoặc crash.
        buffer = np.ascontiguousarray(image)
        qimage = QImage(
            buffer.data, width, height, 3 * width, QImage.Format.Format_RGB888
        ).copy()
        self._image = qimage
        self.update()

    def _on_tick(self) -> None:
        """Nhịp timer: cho adapter tiến frame nếu đang phát."""
        was_playing = self._player.is_playing
        self._player.tick()
        if self._player.is_loaded:
            self.position_changed.emit(self._player.position_sec)
        if was_playing and not self._player.is_playing and self._player.eof_reached:
            self.playback_finished.emit()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 — Qt override
        """Vẽ frame hiện tại (giữ tỉ lệ) rồi vẽ các ROI lên trên."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image is None or self._image.isNull():
            return

        scaled = self._image.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawImage(x, y, scaled)
        self._paint_rois(painter)

    def _paint_rois(self, painter: QPainter) -> None:
        """Vẽ ROI phụ, ROI chính và khung đang kéo.

        Toạ độ ROI lưu theo VIDEO nên phải quy đổi sang khung mỗi lần vẽ — nhờ vậy
        ROI luôn bám đúng vị trí khi người dùng đổi cỡ cửa sổ.
        """
        geometry = self._display_geometry()
        if not geometry.is_valid:
            return

        def to_qrect(video_rect: tuple[int, int, int, int]) -> QRect:
            x, y, width, height = video_rect_to_widget(video_rect, geometry)
            return QRect(x, y, width, height)

        # ROI phụ — nét mảnh, màu nhạt hơn ROI chính.
        painter.setPen(QPen(QColor(255, 200, 0, 180), 1, Qt.PenStyle.DashLine))
        for roi in self._secondary_rois_video:
            painter.drawRect(to_qrect(roi))

        # ROI chính đang áp dụng.
        if self._committed_roi_video is not None:
            painter.setPen(QPen(QColor(0, 200, 255), 2, Qt.PenStyle.SolidLine))
            painter.drawRect(to_qrect(self._committed_roi_video))

        # Khung xem trước khi đang kéo.
        drag_rect = self._drag_rect()
        if drag_rect is not None:
            painter.setPen(QPen(QColor(0, 255, 120), 2, Qt.PenStyle.DashLine))
            painter.drawRect(drag_rect)

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802 — Qt override
        """Giải phóng tài nguyên khi đóng."""
        self.release()
        super().closeEvent(event)


def _run_demo() -> int:
    """Chạy thử nhanh: ``python -m ...pyav_video_widget <video>``."""
    import sys

    from PySide6.QtWidgets import QApplication

    if len(sys.argv) < 2:
        print("Dùng: python -m ...pyav_video_widget <đường-dẫn-video>")
        return 2

    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    widget = PyAvVideoWidget()
    widget.setWindowTitle(
        "Thử trình phát PyAV (LGPL) — Space: play/pause | ←/→: bước frame | R: vẽ ROI"
    )
    widget.resize(960, 540)
    widget.load(Path(sys.argv[1]))
    widget.show()
    widget.play()

    def on_key(event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_Space:
            widget.toggle_play_pause()
        elif event.key() == Qt.Key.Key_Right:
            widget.step_frame(True)
        elif event.key() == Qt.Key.Key_Left:
            widget.step_frame(False)
        elif event.key() == Qt.Key.Key_R:
            widget.enable_roi_drawing(not widget._roi_drawing_enabled)

    widget.roi_drawn.connect(lambda roi: print(f"ROI (toạ độ video): {roi}"))
    widget.keyPressEvent = on_key  # type: ignore[method-assign]
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(_run_demo())


__all__ = ["PyAvVideoWidget"]
