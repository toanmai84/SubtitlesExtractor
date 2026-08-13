"""``VideoCanvas`` — Fallback frame-based video display khi MPV không khả dụng.

TÍCH HỢP PLAYBACK CONTROLLER VÀO LỚP DỰ PHÒNG:
    * Có đầy đủ Signal/Method điều khiển Playback giống như MpvVideoWidget.
    * Sinh sự kiện giả lập Timer để UI vẫn hoạt động mượt mà khi MPV sập.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from PySide6.QtCore import QEvent, QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QEnterEvent,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from subtitles_extractor.domain.value_objects.roi import (
    Roi,
    TextAlignment,
    TextOrientation,
)

_LIVE_COLOR: QColor = QColor(255, 200, 0)
_COMMITTED_COLOR: QColor = QColor(0, 200, 120)
_SECONDARY_COLOR: QColor = QColor(200, 130, 220)
_OCR_COLOR: QColor = QColor(255, 220, 50)


class VideoCanvas(QWidget):
    roi_changed = Signal(object)
    roi_preview = Signal(object)
    video_clicked = Signal(Qt.MouseButton)
    video_double_clicked = Signal(Qt.MouseButton)

    # ── Playback Controller Signals ──
    position_changed = Signal(float)
    duration_changed = Signal(float)
    state_changed = Signal(bool)
    seek_fallback_requested = Signal(float)

    def __init__(self, *, mpv_options: dict[str, Any] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(0, 0, 0))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self._current_pixmap: QPixmap | None = None
        self._video_size: tuple[int, int] | None = None

        self._roi_drawing_enabled: bool = False
        self._dragging: bool = False
        self._start_point: QPoint | None = None
        self._current_rect_widget: QRect | None = None
        self._current_mouse_pos: QPoint | None = None

        self._committed_roi_video: Roi | None = None
        self._secondary_rois_video: list[Roi] = []
        self._ocr_overlay_boxes_video: list[tuple[int, int, int, int]] = []
        self._ocr_overlay_visible: bool = False

        # ── Setup Playback Controller (Dành cho Fallback) ──
        self._virtual_pos = 0.0
        self._duration = 0.0
        self._is_playing = False
        self._auto_loop = True
        self._playback_speed = 1.0  # [#9] hệ số tốc độ phát (fallback)
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(100) # Fallback OpenCV lấy 10fps cho đỡ cháy CPU
        self._playback_timer.timeout.connect(self._on_play_tick)

        # --- CACHE ĐỐI TƯỢNG VẼ ---
        self._live_pen = QPen(_LIVE_COLOR, 2, Qt.PenStyle.DashLine)
        self._live_fill = QColor(_LIVE_COLOR)
        self._live_fill.setAlpha(40)
        self._handle_brush = QColor(255, 200, 0, 200)
        self._committed_handle_brush = QColor(0, 200, 120, 200)
        self._committed_pen = QPen(_COMMITTED_COLOR, 2, Qt.PenStyle.SolidLine)
        self._committed_fill = QColor(_COMMITTED_COLOR)
        self._committed_fill.setAlpha(45)
        self._secondary_pen = QPen(_SECONDARY_COLOR, 2, Qt.PenStyle.DotLine)
        self._secondary_fill = QColor(_SECONDARY_COLOR)
        self._secondary_fill.setAlpha(25)
        self._ocr_pen = QPen(_OCR_COLOR, 1, Qt.PenStyle.SolidLine)
        self._ocr_fill = QColor(_OCR_COLOR)
        self._ocr_fill.setAlpha(35)
        self._crosshair_pen = QPen(QColor(255, 255, 255, 150), 1, Qt.PenStyle.DotLine)
        self._tooltip_font = QFont()
        self._tooltip_font.setBold(True)
        self._tooltip_font.setPointSize(9)
        self._tooltip_bg_color = QColor(0, 0, 0, 190)

    # ── Playback Controller API ──
    @property
    def is_playing(self) -> bool: return self._is_playing
    @property
    def position_sec(self) -> float: return self._virtual_pos
    @property
    def duration_sec(self) -> float: return self._duration

    def set_auto_loop(self, loop: bool) -> None:
        self._auto_loop = loop

    def set_fallback_duration(self, dur: float) -> None:
        self._duration = dur
        self.duration_changed.emit(dur)

    def play(self) -> None:
        self._is_playing = True
        self._playback_timer.start()
        self.state_changed.emit(True)

    def pause(self) -> None:
        self._is_playing = False
        self._playback_timer.stop()
        self.state_changed.emit(False)

    def toggle_play_pause(self) -> None:
        if self._is_playing: self.pause()
        else: self.play()

    def seek(self, sec: float) -> None:
        self._virtual_pos = sec
        self.position_changed.emit(sec)
        self.seek_fallback_requested.emit(sec)

    def _on_play_tick(self) -> None:
        self._virtual_pos += 0.1 * self._playback_speed
        if self._duration > 0 and self._virtual_pos >= self._duration - 0.05:
            if self._auto_loop:
                self._virtual_pos = 0.0
                self.seek(0.0)
            else:
                self.pause()
                self._virtual_pos = self._duration
        self.position_changed.emit(self._virtual_pos)
        self.seek_fallback_requested.emit(self._virtual_pos)

    def send_mpv_command(self, *args: object) -> bool: return False
    def player(self) -> Any | None: return None
    def release_player(self) -> None: self.pause()

    def set_playback_speed(self, speed: float) -> None:
        """[#9] Đặt tốc độ phát. Ở chế độ fallback (không mpv), điều chỉnh bước
        nhảy của bộ đếm thời gian ảo theo hệ số tốc độ."""
        self._playback_speed = max(0.1, min(4.0, float(speed)))

    def load(self, video_path: Path) -> None:
        logger.debug("VideoCanvas.load() no-op (fallback mode).")
        self.position_changed.emit(0.0)
        self.state_changed.emit(False)

    def set_frame(self, image: QImage, video_w: int, video_h: int) -> None:
        if image.isNull(): return
        self._current_pixmap = QPixmap.fromImage(image)
        self._video_size = (video_w, video_h)
        self.update()

    def set_video_size(self, width: int, height: int) -> None:
        self._video_size = (width, height)
        self.update()

    def enable_roi_drawing(self, enabled: bool) -> None:
        self._roi_drawing_enabled = enabled
        self._current_mouse_pos = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.PointingHandCursor)
        self.update()

    def set_committed_roi(self, roi: Roi | None) -> None:
        self._committed_roi_video = roi
        self.update()

    def set_secondary_rois(self, rois: list[Roi]) -> None:
        self._secondary_rois_video = list(rois)
        self.update()

    def set_ocr_overlay(self, boxes: list[tuple[int, int, int, int]], *, visible: bool = True) -> None:
        self._ocr_overlay_boxes_video = list(boxes)
        self._ocr_overlay_visible = visible
        self.update()

    def clear_ocr_overlay(self) -> None:
        self._ocr_overlay_boxes_video = []
        self._ocr_overlay_visible = False
        self.update()

    def clear_roi(self) -> None:
        self._committed_roi_video = None
        self._current_rect_widget = None
        self.update()
        self.roi_changed.emit(None)

    def _clamp_pos(self, pos: QPoint) -> QPoint:
        rect = self._compute_video_display_rect()
        x = max(rect.left(), min(pos.x(), rect.right()))
        y = max(rect.top(), min(pos.y(), rect.bottom()))
        return QPoint(x, y)

    def enterEvent(self, event: QEnterEvent) -> None: super().enterEvent(event)
    def leaveEvent(self, event: QEvent) -> None:
        super().leaveEvent(event)
        self._current_mouse_pos = None
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._roi_drawing_enabled or event.button() != Qt.MouseButton.LeftButton: return
        self._dragging = True
        safe_pos = self._clamp_pos(event.pos())
        self._start_point = safe_pos
        self._current_rect_widget = QRect(self._start_point, safe_pos)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        safe_pos = self._clamp_pos(event.pos())
        self._current_mouse_pos = safe_pos

        if not self._dragging or self._start_point is None:
            if self._roi_drawing_enabled: self.update()
            return

        new_rect = QRect(self._start_point, safe_pos).normalized()
        if self._current_rect_widget != new_rect:
            self._current_rect_widget = new_rect
            self.update()
            preview_roi = self._widget_rect_to_video_roi(new_rect)
            if preview_roi is not None: self.roi_preview.emit(preview_roi)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self._roi_drawing_enabled:
            self.video_clicked.emit(event.button())
            return

        if not self._dragging: return
        self._dragging = False

        if self._current_rect_widget is None: return
        if self._current_rect_widget.width() < 8 or self._current_rect_widget.height() < 8:
            self._current_rect_widget = None
            self.update()
            return

        roi = self._widget_rect_to_video_roi(self._current_rect_widget)
        self._current_rect_widget = None
        if roi is not None:
            self._committed_roi_video = roi
            self.update()
            self.roi_changed.emit(roi)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if not self._roi_drawing_enabled: self.video_double_clicked.emit(event.button())

    def _draw_handles(self, painter: QPainter, rect: QRect, brush_color: QColor) -> None:
        painter.setBrush(brush_color)
        painter.setPen(Qt.PenStyle.NoPen)
        size = 6; half = size // 2
        painter.drawRect(rect.left() - half, rect.top() - half, size, size)
        painter.drawRect(rect.right() - half, rect.top() - half, size, size)
        painter.drawRect(rect.left() - half, rect.bottom() - half, size, size)
        painter.drawRect(rect.right() - half, rect.bottom() - half, size, size)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        video_display_rect: QRect | None = None
        if self._current_pixmap is not None and self._video_size is not None:
            video_display_rect = self._compute_video_display_rect()
            painter.drawPixmap(video_display_rect, self._current_pixmap)
        else:
            painter.fillRect(self.rect(), QColor(0, 0, 0))

        if self._ocr_overlay_visible and self._ocr_overlay_boxes_video and video_display_rect is not None:
            painter.setPen(self._ocr_pen)
            for box in self._ocr_overlay_boxes_video:
                rect = self._video_box_to_widget_rect(box, video_display_rect)
                painter.fillRect(rect, self._ocr_fill)
                painter.drawRect(rect)

        if self._secondary_rois_video and video_display_rect is not None:
            painter.setPen(self._secondary_pen)
            for roi in self._secondary_rois_video:
                rect = self._video_roi_to_widget_rect(roi, video_display_rect)
                painter.fillRect(rect, self._secondary_fill)
                painter.drawRect(rect)

        if self._committed_roi_video is not None and video_display_rect is not None:
            painter.setPen(self._committed_pen)
            rect = self._video_roi_to_widget_rect(self._committed_roi_video, video_display_rect)
            painter.fillRect(rect, self._committed_fill)
            painter.drawRect(rect)
            self._draw_handles(painter, rect, self._committed_handle_brush)

        if self._roi_drawing_enabled and self._current_mouse_pos is not None and video_display_rect is not None:
            painter.setPen(self._crosshair_pen)
            cx, cy = self._current_mouse_pos.x(), self._current_mouse_pos.y()
            gap = 6
            if cx - gap > video_display_rect.left(): painter.drawLine(video_display_rect.left(), cy, cx - gap, cy)
            if cx + gap < video_display_rect.right(): painter.drawLine(cx + gap, cy, video_display_rect.right(), cy)
            if cy - gap > video_display_rect.top(): painter.drawLine(cx, video_display_rect.top(), cx, cy - gap)
            if cy + gap < video_display_rect.bottom(): painter.drawLine(cx, cy + gap, cx, video_display_rect.bottom())

        if self._current_rect_widget is not None:
            painter.setPen(self._live_pen)
            painter.fillRect(self._current_rect_widget, self._live_fill)
            painter.drawRect(self._current_rect_widget)
            self._draw_handles(painter, self._current_rect_widget, self._handle_brush)

            roi_info = self._widget_rect_to_video_roi(self._current_rect_widget)
            if roi_info:
                size_text = f"[{roi_info.x}, {roi_info.y}] {roi_info.width} x {roi_info.height}"
                painter.setFont(self._tooltip_font)
                text_rect = painter.fontMetrics().boundingRect(size_text)
                text_rect.adjust(-8, -4, 8, 4)
                text_rect.moveTopLeft(QPoint(self._current_rect_widget.right() + 10, self._current_rect_widget.bottom() + 10))
                if text_rect.right() > self.width(): text_rect.moveLeft(self._current_rect_widget.left() - text_rect.width() - 10)
                if text_rect.bottom() > self.height(): text_rect.moveTop(self._current_rect_widget.top() - text_rect.height() - 10)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(self._tooltip_bg_color)
                painter.drawRoundedRect(text_rect, 4.0, 4.0)
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, size_text)

    def _compute_video_display_rect(self) -> QRect:
        if self._video_size is None: return QRect(0, 0, 0, 0)
        vw, vh = self._video_size
        ww, wh = self.width(), self.height()
        if vw <= 0 or vh <= 0: return QRect(0, 0, 0, 0)
        scale = min(ww / vw, wh / vh)
        w_disp = int(round(vw * scale)); h_disp = int(round(vh * scale))
        offset_x = int(round((ww - vw * scale) / 2.0)); offset_y = int(round((wh - vh * scale) / 2.0))
        return QRect(offset_x, offset_y, w_disp, h_disp)

    def _video_roi_to_widget_rect(self, roi: Roi, video_display_rect: QRect) -> QRect:
        assert self._video_size is not None
        vw, _ = self._video_size
        scale = video_display_rect.width() / vw if vw > 0 else 1.0
        x1 = int(round(video_display_rect.x() + roi.x * scale))
        y1 = int(round(video_display_rect.y() + roi.y * scale))
        x2 = int(round(video_display_rect.x() + (roi.x + roi.width) * scale))
        y2 = int(round(video_display_rect.y() + (roi.y + roi.height) * scale))
        return QRect(x1, y1, x2 - x1, y2 - y1)

    def _video_box_to_widget_rect(self, box: tuple[int, int, int, int], video_display_rect: QRect) -> QRect:
        assert self._video_size is not None
        vw, _ = self._video_size
        scale = video_display_rect.width() / vw if vw > 0 else 1.0
        x, y, w, h = box
        x1 = int(round(video_display_rect.x() + x * scale))
        y1 = int(round(video_display_rect.y() + y * scale))
        x2 = int(round(video_display_rect.x() + (x + w) * scale))
        y2 = int(round(video_display_rect.y() + (y + h) * scale))
        return QRect(x1, y1, x2 - x1, y2 - y1)

    def _widget_rect_to_video_roi(self, widget_rect: QRect) -> Roi | None:
        if self._video_size is None: return None
        video_display_rect = self._compute_video_display_rect()
        if video_display_rect.width() <= 0 or video_display_rect.height() <= 0: return None
        valid_rect = widget_rect.intersected(video_display_rect)
        if valid_rect.isEmpty(): return None

        vw, vh = self._video_size
        scale = video_display_rect.width() / vw
        left_w = valid_rect.left(); top_w = valid_rect.top()
        right_w = valid_rect.right() + 1; bottom_w = valid_rect.bottom() + 1

        x1 = int(round((left_w - video_display_rect.x()) / scale))
        y1 = int(round((top_w - video_display_rect.y()) / scale))
        x2 = int(round((right_w - video_display_rect.x()) / scale))
        y2 = int(round((bottom_w - video_display_rect.y()) / scale))

        x1 = max(0, min(x1, vw - 1)); y1 = max(0, min(y1, vh - 1))
        x2 = max(x1 + 1, min(x2, vw)); y2 = max(y1 + 1, min(y2, vh))

        try:
            return Roi(x=x1, y=y1, width=x2 - x1, height=y2 - y1, alignment=TextAlignment.CENTER, orientation=TextOrientation.HORIZONTAL)
        except (ValueError, TypeError):
            return None

__all__ = ["VideoCanvas"]
