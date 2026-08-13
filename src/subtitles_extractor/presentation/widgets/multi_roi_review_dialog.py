"""Widget kiểm duyệt Multi-ROI — Human-in-the-Loop.

CẢI TIẾN v2.26 — UI/UX Overhaul:
    1. [TOOLBAR] Thêm toolbar với các tool rõ ràng (Vẽ mới / Xóa / Duplicate /
       Reset zoom).
    2. [SELECTION STATE] Tách rõ "selected" vs "keep/delete". Selected ROI
       highlight đường viền dày 3px màu vàng, có thể edit từ sidebar.
    3. [UNDO/REDO] Ctrl+Z / Ctrl+Y với stack 50 bước. Capture trạng thái sau
       mọi mutate.
    4. [KEYBOARD SHORTCUTS]
       * Delete: xóa hard ROI đang chọn
       * Space / D: toggle keep cluster đang chọn
       * Ctrl+Z / Ctrl+Y: undo / redo
       * Arrow keys: di chuyển 1px (Shift+Arrow = 10px)
       * Ctrl+A: keep tất cả; Ctrl+Shift+A: bỏ keep tất cả
       * Esc: hủy drawing / bỏ chọn
    5. [DRAW SIZE OVERLAY] Hiện size động "245x38" khi đang vẽ ROI mới.
    6. [INFO BAR] Hiển thị: resolution, cursor video coord, số ROI keep/total.
    7. [SIDEBAR EDIT] Click ROI trong sidebar → select. Combo box chỉnh
       alignment, orientation cho ROI được chọn.
    8. [VISUAL] Crosshair khi cursor không trên ROI, edge handles to hơn.

CẢI TIẾN v2.21 (giữ nguyên):
    1. Background pixmap caching: cv2.cvtColor + scale 1 lần duy nhất.
    2. Safe QImage Copy.
    3. Zero-size ROI Prevention.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.domain.value_objects.roi import TextAlignment, TextOrientation
from subtitles_extractor.infrastructure.video.bbox_analyzer import ROICluster

logger = logging.getLogger(__name__)

# ============================================================================
# Color palette (consistent với v2.25, có thêm SELECTED).
# ============================================================================
_COLOR_KEEP = QColor(0, 210, 90, 200)       # xanh lá: cluster keep
_COLOR_DELETE = QColor(220, 50, 50, 180)    # đỏ: cluster sẽ xóa
_COLOR_DRAW = QColor(0, 140, 255, 220)      # xanh dương: rectangle đang vẽ
_COLOR_SELECTED = QColor(255, 215, 0, 230)  # vàng: cluster đang được select
_COLOR_CROSSHAIR = QColor(200, 200, 200, 80)
_COLOR_INFO_BG = QColor(0, 0, 0, 160)

# v2.26: Đặt số bước undo tối đa.
_MAX_UNDO_STACK_SIZE = 50


class _ROICanvas(QLabel):
    """Canvas hiển thị ảnh pha trộn (Composite) và vẽ các khối ROI.

    v2.26 thêm:
        * ``selected_cluster``: con trỏ tới ROI đang được chọn.
        * Signal ``cursor_video_position_changed(int, int)``: phát ra mỗi khi
          chuột di chuyển — để cập nhật info bar.
        * Drawing size overlay: hiện kích thước khi đang vẽ.
        * Crosshair khi cursor không trên ROI.
    """

    roi_changed = Signal()
    cluster_selected = Signal(object)  # ROICluster | None
    cursor_video_position_changed = Signal(int, int)

    def __init__(
        self,
        composite_bgr: np.ndarray,
        clusters: list[ROICluster],
        parent: QWidget | None = None,
        translator: object | None = None,
    ) -> None:
        super().__init__(parent)
        from subtitles_extractor.infrastructure.i18n.null_translator import resolve_translator
        self._translator = resolve_translator(translator)
        self._clusters = clusters
        self._frame_height, self._frame_width = composite_bgr.shape[:2]

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 320)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._is_drawing = False
        self._draw_start_point: QPoint | None = None
        self._draw_current_point: QPoint | None = None

        self._interaction_mode: str | None = None
        self._active_cluster: ROICluster | None = None
        self._interaction_start_pos: QPoint | None = None
        self._cluster_start_rect: tuple[int, int, int, int] | None = None

        self._next_cluster_id = max((c.cluster_id for c in clusters), default=-1) + 1

        # v2.26: selected cluster state.
        self._selected_cluster: ROICluster | None = None

        # v2.26: latest cursor position trong widget coords (cho crosshair).
        self._current_cursor_widget_pos: QPoint | None = None

        # Background pixmap cache.
        rgb_array = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_array.shape
        self._cached_qimage = QImage(
            rgb_array.data, w, h, ch * w, QImage.Format.Format_RGB888
        ).copy()
        self._cached_bg_pixmap: QPixmap | None = None

    # ------------------------------------------------ public selection API

    @property
    def clusters(self) -> list[ROICluster]:
        return self._clusters

    @property
    def selected_cluster(self) -> ROICluster | None:
        return self._selected_cluster

    def select_cluster(self, target_cluster: ROICluster | None) -> None:
        """Chọn 1 cluster (hoặc None để bỏ chọn). Phát signal & redraw."""
        if self._selected_cluster is target_cluster:
            return
        self._selected_cluster = target_cluster
        self.cluster_selected.emit(target_cluster)
        self._update_display()

    def delete_selected_cluster(self) -> bool:
        """Xóa hard cluster đang được chọn (không phải toggle keep).

        Returns:
            True nếu đã xóa, False nếu không có cluster nào được chọn.
        """
        if self._selected_cluster is None:
            return False
        if self._selected_cluster in self._clusters:
            self._clusters.remove(self._selected_cluster)
        self._selected_cluster = None
        self.cluster_selected.emit(None)
        self._update_display()
        self.roi_changed.emit()
        return True

    def duplicate_selected_cluster(self) -> bool:
        """Nhân bản cluster đang chọn, offset 20px xuống-phải.

        Returns:
            True nếu đã duplicate, False nếu không có cluster nào được chọn.
        """
        if self._selected_cluster is None:
            return False
        import copy as _copy_module
        duplicated_cluster = _copy_module.deepcopy(self._selected_cluster)
        duplicated_cluster.cluster_id = self._next_cluster_id
        self._next_cluster_id += 1
        offset_pixels = 20
        duplicated_cluster.coord_x_min = min(
            self._frame_width - 30, duplicated_cluster.coord_x_min + offset_pixels
        )
        duplicated_cluster.coord_y_min = min(
            self._frame_height - 30, duplicated_cluster.coord_y_min + offset_pixels
        )
        duplicated_cluster.coord_x_max = min(
            self._frame_width, duplicated_cluster.coord_x_max + offset_pixels
        )
        duplicated_cluster.coord_y_max = min(
            self._frame_height, duplicated_cluster.coord_y_max + offset_pixels
        )
        self._clusters.append(duplicated_cluster)
        self.select_cluster(duplicated_cluster)
        self.roi_changed.emit()
        return True

    def move_selected_cluster(self, dx_pixels: int, dy_pixels: int) -> bool:
        """Di chuyển cluster đang chọn dx, dy pixel (giới hạn trong frame)."""
        if self._selected_cluster is None:
            return False
        cluster = self._selected_cluster
        cluster_width = cluster.width
        cluster_height = cluster.height
        new_x_min = max(
            0, min(self._frame_width - cluster_width, cluster.coord_x_min + dx_pixels)
        )
        new_y_min = max(
            0, min(self._frame_height - cluster_height, cluster.coord_y_min + dy_pixels)
        )
        cluster.coord_x_min = new_x_min
        cluster.coord_y_min = new_y_min
        cluster.coord_x_max = new_x_min + cluster_width
        cluster.coord_y_max = new_y_min + cluster_height
        self._update_display()
        self.roi_changed.emit()
        return True

    def toggle_keep_selected_cluster(self) -> bool:
        """Toggle keep flag của cluster đang chọn."""
        if self._selected_cluster is None:
            return False
        self._selected_cluster.keep = not self._selected_cluster.keep
        self._update_display()
        self.roi_changed.emit()
        return True

    def keep_all_clusters(self) -> None:
        for cluster in self._clusters:
            cluster.keep = True
        self._update_display()
        self.roi_changed.emit()

    def unkeep_all_clusters(self) -> None:
        for cluster in self._clusters:
            cluster.keep = False
        self._update_display()
        self.roi_changed.emit()

    def replace_clusters(self, new_clusters: list[ROICluster]) -> None:
        """Replace toàn bộ cluster list (dùng cho undo/redo)."""
        self._clusters[:] = new_clusters
        if self._selected_cluster not in self._clusters:
            self._selected_cluster = None
            self.cluster_selected.emit(None)
        self._next_cluster_id = max(
            (c.cluster_id for c in self._clusters), default=-1
        ) + 1
        self._update_display()
        self.roi_changed.emit()

    # ------------------------------------------------ canvas events

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_cached_bg()
        self._update_display()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_cached_bg()
        self._update_display()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._current_cursor_widget_pos = None
        self._update_display()

    # ------------------------------------------------ canvas geometry

    def _get_image_rect(self) -> QRect:
        widget_width, widget_height = self.width(), self.height()
        frame_width, frame_height = self._frame_width, self._frame_height

        scale_factor = (
            min(widget_width / frame_width, widget_height / frame_height)
            if frame_width and frame_height
            else 1.0
        )
        image_width = int(frame_width * scale_factor)
        image_height = int(frame_height * scale_factor)

        offset_x = (widget_width - image_width) // 2
        offset_y = (widget_height - image_height) // 2
        return QRect(offset_x, offset_y, image_width, image_height)

    def _update_cached_bg(self) -> None:
        widget_width, widget_height = self.width(), self.height()
        if widget_width <= 0 or widget_height <= 0:
            return

        self._cached_bg_pixmap = QPixmap(widget_width, widget_height)
        self._cached_bg_pixmap.fill(QColor(20, 20, 20))

        image_rect = self._get_image_rect()
        if image_rect.width() > 0 and image_rect.height() > 0:
            scaled_pixmap = QPixmap.fromImage(self._cached_qimage).scaled(
                image_rect.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter = QPainter(self._cached_bg_pixmap)
            painter.drawPixmap(image_rect, scaled_pixmap)
            painter.end()

    def _update_display(self) -> None:
        if self._cached_bg_pixmap is None:
            return

        canvas_pixmap = self._cached_bg_pixmap.copy()
        painter = QPainter(canvas_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # v2.26: Crosshair khi cursor không trên ROI và không đang vẽ.
        if (
            self._current_cursor_widget_pos is not None
            and not self._is_drawing
            and self._interaction_mode is None
        ):
            self._draw_crosshair_at_cursor(painter)

        for cluster in self._clusters:
            self._draw_roi(painter, cluster)

        if self._is_drawing and self._draw_start_point and self._draw_current_point:
            self._draw_active_rectangle_with_size_overlay(painter)

        painter.end()
        self.setPixmap(canvas_pixmap)

    def _draw_crosshair_at_cursor(self, painter: QPainter) -> None:
        """v2.26: Vẽ crosshair mờ tại vị trí chuột (gợi ý sẽ vẽ ROI mới)."""
        if self._current_cursor_widget_pos is None:
            return
        image_rect = self._get_image_rect()
        if not image_rect.contains(self._current_cursor_widget_pos):
            return
        # Chỉ vẽ crosshair nếu cursor không nằm trên cluster nào.
        hovered_cluster, _ = self._hit_test(self._current_cursor_widget_pos)
        if hovered_cluster is not None:
            return
        painter.setPen(QPen(_COLOR_CROSSHAIR, 1, Qt.PenStyle.DashLine))
        painter.drawLine(
            image_rect.left(),
            self._current_cursor_widget_pos.y(),
            image_rect.right(),
            self._current_cursor_widget_pos.y(),
        )
        painter.drawLine(
            self._current_cursor_widget_pos.x(),
            image_rect.top(),
            self._current_cursor_widget_pos.x(),
            image_rect.bottom(),
        )

    def _draw_active_rectangle_with_size_overlay(self, painter: QPainter) -> None:
        """v2.26: Vẽ rectangle đang được vẽ + overlay kích thước thực (video coord)."""
        if self._draw_start_point is None or self._draw_current_point is None:
            return
        drawing_rect = QRect(
            self._draw_start_point, self._draw_current_point
        ).normalized()
        painter.setPen(QPen(_COLOR_DRAW, 2, Qt.PenStyle.DashLine))
        painter.fillRect(drawing_rect, QColor(0, 140, 255, 30))
        painter.drawRect(drawing_rect)

        # v2.26: Hiện size overlay (kích thước video coord).
        vx_start, vy_start = self._widget_to_video(drawing_rect.topLeft())
        vx_end, vy_end = self._widget_to_video(drawing_rect.bottomRight())
        video_width = abs(vx_end - vx_start)
        video_height = abs(vy_end - vy_start)
        size_text = f"{video_width}×{video_height}"

        size_label_font = QFont()
        size_label_font.setBold(True)
        size_label_font.setPointSize(10)
        painter.setFont(size_label_font)

        text_metrics_rect = painter.fontMetrics().boundingRect(size_text)
        text_metrics_rect.adjust(-6, -2, 6, 2)
        text_metrics_rect.moveTopLeft(
            QPoint(drawing_rect.right() + 6, drawing_rect.bottom() + 6)
        )
        # Adjust nếu tràn ra ngoài widget.
        if text_metrics_rect.right() > self.width():
            text_metrics_rect.moveLeft(drawing_rect.left() - text_metrics_rect.width() - 6)
        if text_metrics_rect.bottom() > self.height():
            text_metrics_rect.moveTop(drawing_rect.top() - text_metrics_rect.height() - 6)

        painter.fillRect(text_metrics_rect, _COLOR_INFO_BG)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(text_metrics_rect, Qt.AlignmentFlag.AlignCenter, size_text)

    # ------------------------------------------------ coord transforms

    def _widget_to_video(self, pos: QPoint) -> tuple[int, int]:
        image_rect = self._get_image_rect()
        if image_rect.width() <= 0 or image_rect.height() <= 0:
            return 0, 0
        video_x = int(
            (pos.x() - image_rect.x()) / image_rect.width() * self._frame_width
        )
        video_y = int(
            (pos.y() - image_rect.y()) / image_rect.height() * self._frame_height
        )
        video_x = max(0, min(video_x, self._frame_width - 1))
        video_y = max(0, min(video_y, self._frame_height - 1))
        return video_x, video_y

    def _video_to_widget_rect(self, cluster: ROICluster) -> QRect:
        image_rect = self._get_image_rect()
        if image_rect.width() <= 0 or image_rect.height() <= 0:
            return QRect()
        scale_x = image_rect.width() / self._frame_width
        scale_y = image_rect.height() / self._frame_height
        return QRect(
            image_rect.x() + int(cluster.coord_x_min * scale_x),
            image_rect.y() + int(cluster.coord_y_min * scale_y),
            int(cluster.width * scale_x),
            int(cluster.height * scale_y),
        )

    # ------------------------------------------------ draw cluster

    def _draw_roi(self, painter: QPainter, cluster: ROICluster) -> None:
        rect = self._video_to_widget_rect(cluster)
        if not rect.isValid() or rect.width() < 2 or rect.height() < 2:
            return

        is_selected_cluster = cluster is self._selected_cluster
        base_color = _COLOR_KEEP if cluster.keep else _COLOR_DELETE

        fill_color = QColor(base_color)
        fill_color.setAlpha(60 if cluster.keep else 90)
        painter.fillRect(rect, fill_color)

        # v2.26: Cluster đang selected → viền vàng dày, ngoài viền màu thường.
        outline_pen_width = 3 if is_selected_cluster else 2
        outline_color = _COLOR_SELECTED if is_selected_cluster else base_color
        outline_color.setAlpha(240)
        painter.setPen(QPen(outline_color, outline_pen_width))
        painter.drawRect(rect)

        # Edge handles (4 corners).
        painter.setBrush(outline_color)
        handle_size = 8 if is_selected_cluster else 6
        for corner in [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight(),
        ]:
            painter.drawRect(
                corner.x() - handle_size // 2,
                corner.y() - handle_size // 2,
                handle_size,
                handle_size,
            )
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Label.
        label_font = QFont()
        label_font.setBold(True)
        label_font.setPointSize(9)
        painter.setFont(label_font)

        direction_text = self._translator.translate("multiroi.horizontal") if cluster.orientation == TextOrientation.HORIZONTAL else self._translator.translate("multiroi.vertical")
        align_map = {
            TextAlignment.CENTER: self._translator.translate("multiroi.align_center"),
            TextAlignment.LEFT: self._translator.translate("multiroi.align_left"),
            TextAlignment.RIGHT: self._translator.translate("multiroi.align_right"),
        }
        align_text = align_map.get(cluster.alignment, "?")
        state_glyph = "✓" if cluster.keep else "✗"
        label_text = f"#{cluster.cluster_id} {state_glyph} {direction_text}-{align_text}"

        painter.setPen(QColor(0, 0, 0, 200))
        painter.drawText(rect.adjusted(6, 5, 0, 0), Qt.AlignmentFlag.AlignTop, label_text)
        painter.setPen(outline_color)
        painter.drawText(rect.adjusted(5, 4, 0, 0), Qt.AlignmentFlag.AlignTop, label_text)

    # ------------------------------------------------ hit testing

    def _hit_test(self, pos: QPoint) -> tuple[ROICluster | None, str | None]:
        hit_results = []
        for cluster in self._clusters:
            rect = self._video_to_widget_rect(cluster)
            margin_px = 8
            out_rect = rect.adjusted(-margin_px, -margin_px, margin_px, margin_px)
            if not out_rect.contains(pos):
                continue

            x, y = pos.x(), pos.y()
            rect_left, rect_right = rect.left(), rect.right()
            rect_top, rect_bottom = rect.top(), rect.bottom()

            handle_mode = "move"
            if abs(x - rect_left) <= margin_px and abs(y - rect_top) <= margin_px:
                handle_mode = "top_left"
            elif abs(x - rect_right) <= margin_px and abs(y - rect_top) <= margin_px:
                handle_mode = "top_right"
            elif abs(x - rect_left) <= margin_px and abs(y - rect_bottom) <= margin_px:
                handle_mode = "bottom_left"
            elif abs(x - rect_right) <= margin_px and abs(y - rect_bottom) <= margin_px:
                handle_mode = "bottom_right"
            elif abs(x - rect_left) <= margin_px:
                handle_mode = "left"
            elif abs(x - rect_right) <= margin_px:
                handle_mode = "right"
            elif abs(y - rect_top) <= margin_px:
                handle_mode = "top"
            elif abs(y - rect_bottom) <= margin_px:
                handle_mode = "bottom"

            hit_results.append((cluster.width * cluster.height, cluster, handle_mode))

        if hit_results:
            hit_results.sort(key=lambda hit_item: hit_item[0])
            return hit_results[0][1], hit_results[0][2]
        return None, None

    def _update_cursor(self, pos: QPoint) -> None:
        if self._is_drawing or self._interaction_mode:
            return

        _, handle = self._hit_test(pos)
        cursor_map = {
            "top_left": Qt.CursorShape.SizeFDiagCursor,
            "bottom_right": Qt.CursorShape.SizeFDiagCursor,
            "top_right": Qt.CursorShape.SizeBDiagCursor,
            "bottom_left": Qt.CursorShape.SizeBDiagCursor,
            "left": Qt.CursorShape.SizeHorCursor,
            "right": Qt.CursorShape.SizeHorCursor,
            "top": Qt.CursorShape.SizeVerCursor,
            "bottom": Qt.CursorShape.SizeVerCursor,
            "move": Qt.CursorShape.OpenHandCursor,
        }
        self.setCursor(cursor_map.get(handle, Qt.CursorShape.CrossCursor))

    # ------------------------------------------------ mouse events

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)

        cluster, handle = self._hit_test(event.pos())
        if cluster and handle:
            self._interaction_mode = handle
            self._active_cluster = cluster
            self._interaction_start_pos = event.pos()
            self._cluster_start_rect = (
                cluster.coord_x_min,
                cluster.coord_y_min,
                cluster.coord_x_max,
                cluster.coord_y_max,
            )
            # v2.26: chọn cluster vừa click.
            self.select_cluster(cluster)
            if handle == "move":
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            # v2.26: bỏ chọn nếu click vào vùng trống, sau đó bắt đầu vẽ.
            self.select_cluster(None)
            self._is_drawing = True
            self._draw_start_point = event.pos()
            self._draw_current_point = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._current_cursor_widget_pos = event.pos()
        # Phát signal cập nhật info bar.
        video_x, video_y = self._widget_to_video(event.pos())
        self.cursor_video_position_changed.emit(video_x, video_y)

        if self._interaction_mode and self._active_cluster and self._cluster_start_rect:
            x_min, y_min, x_max, y_max = self._cluster_start_rect
            mode = self._interaction_mode

            vx_start, vy_start = self._widget_to_video(self._interaction_start_pos)
            vx_current, vy_current = self._widget_to_video(event.pos())
            delta_x = vx_current - vx_start
            delta_y = vy_current - vy_start
            min_size = 10

            if mode == "move":
                cluster_width = x_max - x_min
                cluster_height = y_max - y_min
                delta_x = max(-x_min, min(delta_x, self._frame_width - x_max))
                delta_y = max(-y_min, min(delta_y, self._frame_height - y_max))

                self._active_cluster.coord_x_min = x_min + delta_x
                self._active_cluster.coord_y_min = y_min + delta_y
                self._active_cluster.coord_x_max = x_min + delta_x + cluster_width
                self._active_cluster.coord_y_max = y_min + delta_y + cluster_height
            else:
                if "left" in mode:
                    self._active_cluster.coord_x_min = max(
                        0, min(x_min + delta_x, x_max - min_size)
                    )
                elif "right" in mode:
                    self._active_cluster.coord_x_max = max(
                        x_min + min_size, min(x_max + delta_x, self._frame_width)
                    )

                if "top" in mode:
                    self._active_cluster.coord_y_min = max(
                        0, min(y_min + delta_y, y_max - min_size)
                    )
                elif "bottom" in mode:
                    self._active_cluster.coord_y_max = max(
                        y_min + min_size, min(y_max + delta_y, self._frame_height)
                    )

            self._update_display()

        elif self._is_drawing:
            self._draw_current_point = event.pos()
            self._update_display()
        else:
            self._update_cursor(event.pos())
            self._update_display()  # để vẽ crosshair

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._interaction_mode:
            # v2.26: bỏ "click = toggle keep" — bây giờ chỉ select. Toggle keep
            # qua nút trên toolbar hoặc phím D/Space (rõ ràng & ít sai sót hơn).
            self._interaction_mode = None
            self._active_cluster = None
            self._update_display()
            self.roi_changed.emit()
            self._update_cursor(event.pos())

        elif self._is_drawing:
            self._is_drawing = False
            if self._draw_start_point is None or self._draw_current_point is None:
                return

            drawn_widget_rect = QRect(
                self._draw_start_point, self._draw_current_point
            ).normalized()
            if drawn_widget_rect.width() >= 10 and drawn_widget_rect.height() >= 10:
                video_x_1, video_y_1 = self._widget_to_video(drawn_widget_rect.topLeft())
                video_x_2, video_y_2 = self._widget_to_video(drawn_widget_rect.bottomRight())

                # [LOGIC FIX] Chỉ tạo ROI nếu kích thước thật trên video > 2px.
                if video_x_2 - video_x_1 >= 2 and video_y_2 - video_y_1 >= 2:
                    new_cluster = ROICluster(
                        cluster_id=self._next_cluster_id,
                        orientation=TextOrientation.HORIZONTAL,
                        alignment=TextAlignment.CENTER,
                        coord_x_min=video_x_1,
                        coord_y_min=video_y_1,
                        coord_x_max=video_x_2,
                        coord_y_max=video_y_2,
                        bbox_count=0,
                        frame_count=0,
                        mean_confidence=1.0,
                        keep=True,
                    )
                    self._next_cluster_id += 1
                    self._clusters.append(new_cluster)
                    # v2.26: auto-select cluster vừa vẽ.
                    self.select_cluster(new_cluster)

            self._draw_start_point = None
            self._draw_current_point = None
            self._update_display()
            self.roi_changed.emit()


class MultiROIReviewDialog(QDialog):
    """Dialog kiểm duyệt ROI Human-in-the-Loop.

    v2.26 cải tiến UI/UX toàn diện:
        * Toolbar với các tool rõ ràng.
        * Sidebar có list-widget click chọn + combo box edit alignment/orientation.
        * Info bar: resolution, cursor coord, số ROI keep/total.
        * Undo/redo stack 50 bước.
        * Keyboard shortcuts đầy đủ.
    """

    def __init__(
        self,
        composite_bgr: np.ndarray,
        clusters: list[ROICluster],
        parent: QWidget | None = None,
        translator: object | None = None,
    ) -> None:
        super().__init__(parent)
        from subtitles_extractor.infrastructure.i18n.null_translator import (
            resolve_translator,
        )

        self._translator = resolve_translator(translator)
        self.setWindowTitle(
            self._translator.translate("multiroi.win_title")
        )
        self.setMinimumSize(900, 600)

        screen_geom = QApplication.primaryScreen().availableGeometry()
        max_dialog_height = int(screen_geom.height() * 0.9)
        self.setMaximumHeight(max_dialog_height)
        self.resize(1200, min(800, max_dialog_height))

        self._clusters = clusters

        # v2.26: Undo/Redo stacks.
        self._undo_stack: list[list[ROICluster]] = []
        self._redo_stack: list[list[ROICluster]] = []

        self._build_ui(composite_bgr, clusters)
        self._install_keyboard_shortcuts()

    # ----------------------------------------------------- UI BUILD

    def _build_ui(self, composite_image: np.ndarray, clusters: list[ROICluster]) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # 1. Header.
        header_label = QLabel(
            self._translator.translate("multiroi.header")
            + "<br><small>"
            + self._translator.translate("multiroi.help")
            + "</small>"
        )
        header_label.setWordWrap(True)
        root_layout.addWidget(header_label)

        # 2. Toolbar (v2.26 mới).
        toolbar_widget = self._build_toolbar()
        root_layout.addWidget(toolbar_widget)

        # 3. Splitter: canvas (left) | sidebar (right).
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._canvas = _ROICanvas(composite_image, clusters, self, self._translator)
        self._canvas.roi_changed.connect(self._handle_roi_changed_event)
        self._canvas.cluster_selected.connect(self._handle_cluster_selected_event)
        self._canvas.cursor_video_position_changed.connect(
            self._update_cursor_info_label
        )
        # [BUG FIX v3.6.2] Truyền content_splitter làm parent để tránh
        # Qt tạo native window handle ngay → reparent conflict.
        canvas_scroll_area = QScrollArea(content_splitter)
        canvas_scroll_area.setWidget(self._canvas)
        canvas_scroll_area.setWidgetResizable(True)
        content_splitter.addWidget(canvas_scroll_area)

        sidebar_widget = self._build_sidebar()
        content_splitter.addWidget(sidebar_widget)
        content_splitter.setStretchFactor(0, 4)
        content_splitter.setStretchFactor(1, 1)

        root_layout.addWidget(content_splitter, stretch=1)

        # 4. Info bar.
        self._info_label = QLabel()
        self._info_label.setStyleSheet(
            "QLabel { color: #555; padding: 4px 8px; "
            "background: #f4f4f4; border-radius: 3px; }"
        )
        root_layout.addWidget(self._info_label)

        # 5. Action button row.
        action_button_row = QHBoxLayout()
        keep_all_button = QPushButton(self._translator.translate("multiroi.keep_all"))
        keep_all_button.setShortcut("Ctrl+A")
        keep_all_button.setToolTip(self._translator.translate("multiroi.keep_all_tip"))
        keep_all_button.clicked.connect(self._action_keep_all_rois)

        unkeep_all_button = QPushButton(self._translator.translate("multiroi.unkeep_all"))
        unkeep_all_button.setShortcut("Ctrl+Shift+A")
        unkeep_all_button.setToolTip(self._translator.translate("multiroi.unkeep_all_tip"))
        unkeep_all_button.clicked.connect(self._action_unkeep_all_rois)

        confirm_button = QPushButton(self._translator.translate("multiroi.confirm"))
        confirm_button.setStyleSheet(
            "background-color:#2563eb; color:white; padding:6px 18px; "
            "font-weight:600; border-radius:5px;"
        )
        confirm_button.setDefault(True)
        cancel_button = QPushButton(self._translator.translate("multiroi.cancel"))
        cancel_button.setShortcut("Esc")

        action_button_row.addWidget(keep_all_button)
        action_button_row.addWidget(unkeep_all_button)
        action_button_row.addStretch(1)
        action_button_row.addWidget(cancel_button)
        action_button_row.addWidget(confirm_button)
        root_layout.addLayout(action_button_row)

        confirm_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        # Initial render.
        self._refresh_sidebar_list()
        self._update_info_label()
        # Lưu snapshot ban đầu vào undo stack.
        self._push_undo_snapshot()

    def _build_toolbar(self) -> QWidget:
        """v2.26: Toolbar với các tool quan trọng."""
        toolbar = QToolBar()
        toolbar.setIconSize(toolbar.iconSize())

        delete_action = toolbar.addAction(self._translator.translate("multiroi.act_delete"))
        delete_action.setToolTip(self._translator.translate("multiroi.act_delete_tip"))
        delete_action.triggered.connect(self._action_delete_selected_roi)
        self._toolbar_delete_action = delete_action

        toggle_keep_action = toolbar.addAction(self._translator.translate("multiroi.act_toggle"))
        toggle_keep_action.setToolTip(self._translator.translate("multiroi.act_toggle_tip"))
        toggle_keep_action.triggered.connect(self._action_toggle_keep_selected)
        self._toolbar_toggle_action = toggle_keep_action

        duplicate_action = toolbar.addAction(self._translator.translate("multiroi.act_dup"))
        duplicate_action.setToolTip(self._translator.translate("multiroi.act_dup_tip"))
        duplicate_action.triggered.connect(self._action_duplicate_selected_roi)
        self._toolbar_duplicate_action = duplicate_action

        toolbar.addSeparator()

        undo_action = toolbar.addAction(self._translator.translate("multiroi.act_undo"))
        undo_action.setToolTip(self._translator.translate("multiroi.act_undo_tip"))
        undo_action.triggered.connect(self._action_undo)
        self._toolbar_undo_action = undo_action

        redo_action = toolbar.addAction(self._translator.translate("multiroi.act_redo"))
        redo_action.setToolTip(self._translator.translate("multiroi.act_redo_tip"))
        redo_action.triggered.connect(self._action_redo)
        self._toolbar_redo_action = redo_action

        return toolbar

    def _build_sidebar(self) -> QWidget:
        """v2.26: Sidebar với list ROI + edit panel."""
        sidebar_container = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(8, 0, 0, 0)
        sidebar_layout.setSpacing(6)

        sidebar_layout.addWidget(QLabel(self._translator.translate("multiroi.roi_list")))

        self._roi_list_widget = QListWidget()
        self._roi_list_widget.itemClicked.connect(self._handle_sidebar_item_clicked)
        sidebar_layout.addWidget(self._roi_list_widget, stretch=1)

        # Edit panel — chỉ enable khi có cluster selected.
        edit_panel = QGroupBox(self._translator.translate("multiroi.edit_group"))
        edit_panel.setEnabled(False)
        self._edit_panel_group = edit_panel
        edit_panel_layout = QVBoxLayout(edit_panel)

        # Orientation combo.
        orientation_row = QHBoxLayout()
        orientation_row.addWidget(QLabel(self._translator.translate("multiroi.orientation")))
        self._orientation_combo = QComboBox()
        self._orientation_combo.addItem("Ngang", TextOrientation.HORIZONTAL)
        self._orientation_combo.addItem(self._translator.translate("multiroi.vertical"), TextOrientation.VERTICAL)
        self._orientation_combo.currentIndexChanged.connect(
            self._handle_orientation_combo_changed
        )
        orientation_row.addWidget(self._orientation_combo, stretch=1)
        edit_panel_layout.addLayout(orientation_row)

        # Alignment combo.
        alignment_row = QHBoxLayout()
        alignment_row.addWidget(QLabel(self._translator.translate("multiroi.alignment")))
        self._alignment_combo = QComboBox()
        self._alignment_combo.addItem(self._translator.translate("multiroi.align_center"), TextAlignment.CENTER)
        self._alignment_combo.addItem(self._translator.translate("multiroi.align_left"), TextAlignment.LEFT)
        self._alignment_combo.addItem(self._translator.translate("multiroi.align_right"), TextAlignment.RIGHT)
        self._alignment_combo.currentIndexChanged.connect(
            self._handle_alignment_combo_changed
        )
        alignment_row.addWidget(self._alignment_combo, stretch=1)
        edit_panel_layout.addLayout(alignment_row)

        # Coordinate display (read-only label).
        self._coord_display_label = QLabel(self._translator.translate("multiroi.none_selected"))
        self._coord_display_label.setStyleSheet("font-family: monospace; color: #444;")
        edit_panel_layout.addWidget(self._coord_display_label)

        sidebar_layout.addWidget(edit_panel)

        sidebar_container.setFixedWidth(280)
        return sidebar_container

    # ----------------------------------------------------- shortcuts

    def _install_keyboard_shortcuts(self) -> None:
        """v2.26: Cài đặt phím tắt."""
        # Delete: xóa ROI đang chọn.
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self).activated.connect(
            self._action_delete_selected_roi
        )

        # D hoặc Space: toggle keep.
        QShortcut(QKeySequence(Qt.Key.Key_D), self).activated.connect(
            self._action_toggle_keep_selected
        )
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(
            self._action_toggle_keep_selected
        )

        # Ctrl+Z / Ctrl+Y: undo / redo.
        QShortcut(QKeySequence.StandardKey.Undo, self).activated.connect(
            self._action_undo
        )
        QShortcut(QKeySequence.StandardKey.Redo, self).activated.connect(
            self._action_redo
        )

        # Ctrl+D: duplicate.
        QShortcut(QKeySequence("Ctrl+D"), self).activated.connect(
            self._action_duplicate_selected_roi
        )

        # Arrow keys: di chuyển 1px (Shift+Arrow = 10px).
        for key_code, dx, dy in [
            (Qt.Key.Key_Left, -1, 0),
            (Qt.Key.Key_Right, 1, 0),
            (Qt.Key.Key_Up, 0, -1),
            (Qt.Key.Key_Down, 0, 1),
        ]:
            QShortcut(QKeySequence(key_code), self).activated.connect(
                lambda dx_=dx, dy_=dy: self._action_move_selected_roi(dx_, dy_)
            )
            QShortcut(
                QKeySequence(Qt.KeyboardModifier.ShiftModifier | key_code), self
            ).activated.connect(
                lambda dx_=dx * 10, dy_=dy * 10: self._action_move_selected_roi(dx_, dy_)
            )

    # ----------------------------------------------------- handlers

    def _handle_roi_changed_event(self) -> None:
        """Mỗi khi cluster mutate qua mouse → push snapshot và refresh sidebar."""
        self._push_undo_snapshot()
        self._refresh_sidebar_list()
        self._update_info_label()

    def _handle_cluster_selected_event(self, selected_cluster: object) -> None:
        """Khi canvas đổi selected → sync với sidebar và edit panel."""
        if selected_cluster is None:
            self._edit_panel_group.setEnabled(False)
            self._coord_display_label.setText(self._translator.translate("multiroi.none_selected"))
            self._roi_list_widget.clearSelection()
            self._update_toolbar_button_states()
            return

        cluster_obj: ROICluster = selected_cluster  # type: ignore[assignment]
        self._edit_panel_group.setEnabled(True)

        # Sync combo boxes (block signal để tránh re-trigger).
        self._orientation_combo.blockSignals(True)
        target_orient_idx = (
            0 if cluster_obj.orientation == TextOrientation.HORIZONTAL else 1
        )
        self._orientation_combo.setCurrentIndex(target_orient_idx)
        self._orientation_combo.blockSignals(False)

        self._alignment_combo.blockSignals(True)
        alignment_to_index = {
            TextAlignment.CENTER: 0,
            TextAlignment.LEFT: 1,
            TextAlignment.RIGHT: 2,
        }
        self._alignment_combo.setCurrentIndex(
            alignment_to_index.get(cluster_obj.alignment, 0)
        )
        self._alignment_combo.blockSignals(False)

        # Sync coord display.
        self._coord_display_label.setText(
            f"x: {cluster_obj.coord_x_min} → {cluster_obj.coord_x_max} "
            f"({cluster_obj.width}px)\n"
            f"y: {cluster_obj.coord_y_min} → {cluster_obj.coord_y_max} "
            f"({cluster_obj.height}px)\n"
            f"Conf: {cluster_obj.mean_confidence:.0%}  |  "
            f"{cluster_obj.bbox_count} bbox"
        )

        # Sync sidebar list.
        target_list_index = next(
            (
                idx
                for idx, c in enumerate(self._canvas.clusters)
                if c is cluster_obj
            ),
            -1,
        )
        if target_list_index >= 0:
            self._roi_list_widget.setCurrentRow(target_list_index)

        self._update_toolbar_button_states()

    def _handle_sidebar_item_clicked(self, item: QListWidgetItem) -> None:
        """Khi user click vào item trong sidebar → select tương ứng."""
        clicked_index = self._roi_list_widget.row(item)
        if 0 <= clicked_index < len(self._canvas.clusters):
            self._canvas.select_cluster(self._canvas.clusters[clicked_index])
            self._canvas.setFocus()

    def _handle_orientation_combo_changed(self, index: int) -> None:
        selected_cluster = self._canvas.selected_cluster
        if selected_cluster is None:
            return
        new_orientation = self._orientation_combo.itemData(index)
        if isinstance(new_orientation, TextOrientation):
            selected_cluster.orientation = new_orientation
            self._canvas._update_display()
            self._push_undo_snapshot()
            self._refresh_sidebar_list()

    def _handle_alignment_combo_changed(self, index: int) -> None:
        selected_cluster = self._canvas.selected_cluster
        if selected_cluster is None:
            return
        new_alignment = self._alignment_combo.itemData(index)
        if isinstance(new_alignment, TextAlignment):
            selected_cluster.alignment = new_alignment
            self._canvas._update_display()
            self._push_undo_snapshot()
            self._refresh_sidebar_list()

    def _update_cursor_info_label(self, video_x: int, video_y: int) -> None:
        self._cursor_video_x = video_x
        self._cursor_video_y = video_y
        self._update_info_label()

    def _update_info_label(self) -> None:
        kept_count = sum(1 for c in self._canvas.clusters if c.keep)
        total_count = len(self._canvas.clusters)
        cursor_x = getattr(self, "_cursor_video_x", None)
        cursor_y = getattr(self, "_cursor_video_y", None)
        cursor_text = (
            f"Cursor: ({cursor_x}, {cursor_y})"
            if cursor_x is not None
            else "Cursor: (--, --)"
        )
        self._info_label.setText(
            self._translator.translate("multiroi.info_bar").replace("{w}", str(self._canvas._frame_width)).replace("{h}", str(self._canvas._frame_height)).replace("{kept}", str(kept_count)).replace("{total}", str(total_count)).replace("{cursor}", cursor_text)
        )

    def _update_toolbar_button_states(self) -> None:
        has_selection = self._canvas.selected_cluster is not None
        self._toolbar_delete_action.setEnabled(has_selection)
        self._toolbar_toggle_action.setEnabled(has_selection)
        self._toolbar_duplicate_action.setEnabled(has_selection)
        self._toolbar_undo_action.setEnabled(len(self._undo_stack) > 1)
        self._toolbar_redo_action.setEnabled(len(self._redo_stack) > 0)

    # ----------------------------------------------------- sidebar

    def _refresh_sidebar_list(self) -> None:
        previous_selected_row = self._roi_list_widget.currentRow()
        self._roi_list_widget.clear()
        for cluster in self._canvas.clusters:
            status_icon = "🟢" if cluster.keep else "🔴"
            direction_text = (
                self._translator.translate("multiroi.horizontal") if cluster.orientation == TextOrientation.HORIZONTAL else self._translator.translate("multiroi.vertical")
            )
            align_label = {
                TextAlignment.CENTER: self._translator.translate("multiroi.align_center"),
                TextAlignment.LEFT: self._translator.translate("multiroi.align_left"),
                TextAlignment.RIGHT: self._translator.translate("multiroi.align_right"),
            }.get(cluster.alignment, "?")
            item_text = (
                f"{status_icon} #{cluster.cluster_id}  "
                f"{direction_text}/{align_label}  "
                f"({cluster.width}×{cluster.height}px)"
            )
            list_item = QListWidgetItem(item_text)
            self._roi_list_widget.addItem(list_item)

        if 0 <= previous_selected_row < self._roi_list_widget.count():
            self._roi_list_widget.setCurrentRow(previous_selected_row)

        self._update_info_label()
        self._update_toolbar_button_states()

    # ----------------------------------------------------- actions

    def _action_keep_all_rois(self) -> None:
        self._canvas.keep_all_clusters()
        self._push_undo_snapshot()
        self._refresh_sidebar_list()

    def _action_unkeep_all_rois(self) -> None:
        self._canvas.unkeep_all_clusters()
        self._push_undo_snapshot()
        self._refresh_sidebar_list()

    def _action_delete_selected_roi(self) -> None:
        if self._canvas.delete_selected_cluster():
            self._push_undo_snapshot()
            self._refresh_sidebar_list()

    def _action_toggle_keep_selected(self) -> None:
        if self._canvas.toggle_keep_selected_cluster():
            self._push_undo_snapshot()
            self._refresh_sidebar_list()

    def _action_duplicate_selected_roi(self) -> None:
        if self._canvas.duplicate_selected_cluster():
            self._push_undo_snapshot()
            self._refresh_sidebar_list()

    def _action_move_selected_roi(self, dx: int, dy: int) -> None:
        if self._canvas.move_selected_cluster(dx, dy):
            # Không push snapshot mỗi key press để tránh stack ngập rác —
            # snapshot sẽ được push khi user thực hiện action khác (auto-save).
            self._refresh_sidebar_list()

    # ----------------------------------------------------- undo/redo

    def _push_undo_snapshot(self) -> None:
        """v2.26: Lưu snapshot trạng thái clusters vào undo stack."""
        import copy as _copy_module

        current_state_snapshot = [
            _copy_module.deepcopy(cluster) for cluster in self._canvas.clusters
        ]
        # Tránh push duplicate liên tiếp.
        if self._undo_stack and self._states_equal(
            self._undo_stack[-1], current_state_snapshot
        ):
            return
        self._undo_stack.append(current_state_snapshot)
        if len(self._undo_stack) > _MAX_UNDO_STACK_SIZE:
            self._undo_stack.pop(0)
        # Clear redo stack khi có action mới.
        self._redo_stack.clear()
        self._update_toolbar_button_states()

    @staticmethod
    def _states_equal(
        state_a: list[ROICluster], state_b: list[ROICluster]
    ) -> bool:
        """So sánh 2 snapshot xem có giống nhau không."""
        if len(state_a) != len(state_b):
            return False
        for c_a, c_b in zip(state_a, state_b, strict=True):
            if (
                c_a.cluster_id != c_b.cluster_id
                or c_a.coord_x_min != c_b.coord_x_min
                or c_a.coord_y_min != c_b.coord_y_min
                or c_a.coord_x_max != c_b.coord_x_max
                or c_a.coord_y_max != c_b.coord_y_max
                or c_a.keep != c_b.keep
                or c_a.orientation != c_b.orientation
                or c_a.alignment != c_b.alignment
            ):
                return False
        return True

    def _action_undo(self) -> None:
        """v2.26: Hoàn tác về snapshot trước đó."""
        if len(self._undo_stack) <= 1:
            return
        import copy as _copy_module

        current_state = self._undo_stack.pop()
        self._redo_stack.append(current_state)
        previous_state = self._undo_stack[-1]
        # Replace với bản copy để không chia sẻ tham chiếu với undo stack.
        restored_clusters = [_copy_module.deepcopy(c) for c in previous_state]
        self._canvas.replace_clusters(restored_clusters)
        self._refresh_sidebar_list()
        self._update_toolbar_button_states()

    def _action_redo(self) -> None:
        """v2.26: Làm lại snapshot vừa undo."""
        if not self._redo_stack:
            return
        import copy as _copy_module

        next_state = self._redo_stack.pop()
        self._undo_stack.append(next_state)
        restored_clusters = [_copy_module.deepcopy(c) for c in next_state]
        self._canvas.replace_clusters(restored_clusters)
        self._refresh_sidebar_list()
        self._update_toolbar_button_states()

    # ----------------------------------------------------- public API

    def get_kept_clusters(self) -> list[ROICluster]:
        return [cluster for cluster in self._canvas.clusters if cluster.keep]


__all__ = ["MultiROIReviewDialog"]
