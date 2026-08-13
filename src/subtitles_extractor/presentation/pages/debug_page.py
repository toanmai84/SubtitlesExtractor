"""Trang "Gỡ lỗi OCR" — Hiển thị dữ liệu Raw và Hình ảnh Input/Output.

BẢN CẬP NHẬT ĐỘT PHÁ (V3.30 - The Omni-Tuner):
    * [FEATURE] Bổ sung lựa chọn Mô hình AI (Mobile / Server / V4) vào bảng Live Tuning.
    * [UX] Đồng bộ lựa chọn Mô hình gốc từ Cài đặt hệ thống vào trang Debug.
    * [LỖI 4 JITTER FIX] Bổ sung Nút Bám theo Video (Auto Sync) chống chớp nháy.
    * [LỖI 5 PIXEL BLUR FIX] Chống nhòe hình khi Zoom lớn hơn 200%.
    * [LỖI 6 DATA CORRUPTION FIX] Tự động khóa nút Lưu khi JSON lỗi + Nút Format Code.
    * [LỖI LỆCH BOX KHI UPSCALE] Vẽ Box theo Hệ số Tỷ lệ Thực (Real-Scale Coefficient).
    * [LỖI 9 TYPE MISMATCH FIX] Sử dụng QPointF cho PyQt6 drawText thay vì số Float trần.
"""

from __future__ import annotations

import bisect, re, json, logging
from pathlib import Path
from collections import OrderedDict

from PySide6.QtCore import Qt, QEvent, QTimer, QObject, Signal, QRunnable, QThreadPool, QPointF, QPoint
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeyEvent,
    QImage,
    QPixmap,
    QPainter,
    QWheelEvent,
    QMouseEvent,
    QPaintEvent,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QFormLayout,
    QScrollArea,
)
from subtitles_extractor.presentation.fluent_compat import (
    CaptionLabel,
    FluentIcon,
    InfoBar,
    PrimaryPushButton,
    PushButton,
    Slider,
    TextEdit,
    ToolButton,
    CheckBox,
    DoubleSpinBox,
    SpinBox,
    ComboBox,
)

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.presentation.view_models.debug_page_view_model import (
    DebugPageViewModel,
)
from subtitles_extractor.presentation.widgets import create_video_widget
from subtitles_extractor.presentation.widgets.section_card import SectionCard
from subtitles_extractor.presentation.utils.accessibility import set_accessible_name
from subtitles_extractor.presentation.utils.time_format import seconds_to_display
from subtitles_extractor.presentation.theme import colors as _c
from subtitles_extractor.presentation.theme import metrics as _m
from subtitles_extractor.presentation.theme.styles import mono_label_style


def _mono_style(error: bool = False) -> str:
    """[v3.23.59] Style vùng JSON theo theme (thay màu cứng #1e1e1e/#3f1515)."""
    bg = _c.danger_bg() if error else _c.mono_bg()
    bd = _c.danger() if error else _c.border()
    return (
        f"font-family: Consolas, monospace; font-size: {_m.FONT_SIZE_SMALL}px; "
        f"background-color: {bg}; color: {_c.mono_fg()}; border: 1px solid {bd};"
    )

logger = logging.getLogger(__name__)


# ==============================================================================
# BỘ TÔ MÀU CÚ PHÁP (JSON HIGHLIGHTER)
# ==============================================================================
class JsonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, document: QTextDocument):
        super().__init__(document)
        self.key_format = QTextCharFormat()
        self.key_format.setForeground(QColor("#9CDCFE"))
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("#CE9178"))
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#B5CEA8"))
        self.bracket_format = QTextCharFormat()
        self.bracket_format.setForeground(QColor("#D4D4D4"))

        self.key_regex = re.compile(r'"[^"]*"\s*:')
        self.string_regex = re.compile(r':\s*("[^"]*")')
        self.number_regex = re.compile(r'\b[-+]?[0-9]*\.?[0-9]+\b')
        self.bracket_regex = re.compile(r'[\[\]\{\}]')

    def highlightBlock(self, text: str) -> None:
        if not text: return
        for match in self.bracket_regex.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self.bracket_format)
        for match in self.number_regex.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self.number_format)
        for match in self.string_regex.finditer(text):
            self.setFormat(match.start(1), match.end(1) - match.start(1), self.string_format)
        for match in self.key_regex.finditer(text):
            self.setFormat(match.start(), match.end() - match.start() - 1, self.key_format)


# ==============================================================================
# BẤT ĐỒNG BỘ NẠP ẢNH (ASYNC IMAGE LOADER)
# ==============================================================================
class _ImageLoadSignals(QObject):
    loaded = Signal(str, int, QImage)

class _ImageLoadTask(QRunnable):
    def __init__(self, path: str, task_id: int, signals: _ImageLoadSignals):
        super().__init__()
        self.path = path
        self.task_id = task_id
        self.signals = signals

    def run(self):
        try:
            with open(self.path, 'rb') as f:
                img_data = f.read()
            img = QImage.fromData(img_data)
            if not img.isNull():
                img = img.convertToFormat(QImage.Format.Format_RGB32)
            self.signals.loaded.emit(self.path, self.task_id, img)
        except (OSError, ValueError, RuntimeError) as e:
            logger.debug("Lỗi nạp ảnh QImage: %s", e)
            self.signals.loaded.emit(self.path, self.task_id, QImage())


class InteractiveImagePreviewLabel(QWidget):
    _MAX_CACHE_SIZE = 30

    def __init__(self, translator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._translator = translator
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._current_path: str = ""
        self._fallback_text: str = ""
        self._thread_pool = QThreadPool.globalInstance()
        self._signals = _ImageLoadSignals()
        self._signals.loaded.connect(self._on_image_loaded)
        self._current_task_id: int = 0

        self._zoom: float = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._is_panning = False
        self._last_mouse_pos = QPointF()
        self._cursor_img_pos: QPointF | None = None

        self._direct_qimage: QPixmap | None = None

        self._image_width = 0
        self._image_height = 0

    def set_image_path(self, path: str, fallback_text: str) -> None:
        self._current_path = path
        self._fallback_text = fallback_text
        self._direct_qimage = None
        self._current_task_id += 1

        self.reset_view()

        if not path or not Path(path).exists():
            self.update()
            return

        if path in self._pixmap_cache:
            self._pixmap_cache.move_to_end(path)
            # Update width and height from cache
            cached_pm = self._pixmap_cache[path]
            self._image_width = cached_pm.width()
            self._image_height = cached_pm.height()
            self.update()
        else:
            task = _ImageLoadTask(path, self._current_task_id, self._signals)
            self._thread_pool.start(task)

    def set_direct_qimage(self, img: QImage) -> None:
        self._current_task_id += 1
        if not img.isNull():
            self._direct_qimage = QPixmap.fromImage(img)
            self._image_width = img.width()
            self._image_height = img.height()
        else:
            self._direct_qimage = None
            self._image_width = 0
            self._image_height = 0
        self.reset_view()

    def _on_image_loaded(self, path: str, task_id: int, img: QImage) -> None:
        if task_id != self._current_task_id: return
        if not img.isNull():
            pm = QPixmap.fromImage(img)
            self._pixmap_cache[path] = pm
            self._image_width = img.width()
            self._image_height = img.height()
            if len(self._pixmap_cache) > self._MAX_CACHE_SIZE:
                self._pixmap_cache.popitem(last=False)
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0; self._offset = QPointF(0.0, 0.0); self.update()

    def get_image_size(self) -> tuple[int, int]:
        return self._image_width, self._image_height

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))

        pm = self._direct_qimage
        if pm is None:
            pm = self._pixmap_cache.get(self._current_path)

        if pm is None or pm.isNull() or pm.width() <= 0 or pm.height() <= 0:
            painter.setPen(QColor("#888888")); painter.setFont(QFont("Segoe UI", 11))
            text = self._fallback_text if (self._current_path or self._direct_qimage) else self._translator.translate("debug.dbg_no_frame")
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
            painter.end(); return

        w_factor = self.width() / max(1, pm.width())
        h_factor = self.height() / max(1, pm.height())
        scale_fit = min(w_factor, h_factor)
        actual_zoom = scale_fit * self._zoom

        img_w = pm.width() * actual_zoom
        img_h = pm.height() * actual_zoom
        base_x = (self.width() - img_w) / 2.0
        base_y = (self.height() - img_h) / 2.0

        if actual_zoom > 2.0:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        else:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        painter.translate(base_x + self._offset.x(), base_y + self._offset.y())
        painter.scale(actual_zoom, actual_zoom)
        painter.drawPixmap(0, 0, pm)
        painter.resetTransform()

        if self._cursor_img_pos is not None:
            text = f"X: {int(self._cursor_img_pos.x())} | Y: {int(self._cursor_img_pos.y())}"
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            fm = painter.fontMetrics()
            rect = fm.boundingRect(text).adjusted(-6, -2, 6, 2)
            rect.moveTopLeft(QPoint(10, 10))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 180))
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(QColor("#00FF00"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y() or event.angleDelta().x()
        zoom_in = delta > 0
        zoom_factor = 1.15 if zoom_in else (1.0 / 1.15)

        mouse_pos = event.position()
        old_x = (mouse_pos.x() - self.width() / 2.0 - self._offset.x())
        old_y = (mouse_pos.y() - self.height() / 2.0 - self._offset.y())

        self._zoom = max(0.5, min(30.0, self._zoom * zoom_factor))

        new_x = old_x * zoom_factor; new_y = old_y * zoom_factor
        self._offset.setX(self._offset.x() + (old_x - new_x))
        self._offset.setY(self._offset.y() + (old_y - new_y))

        self._update_radar_pos(mouse_pos)
        self.update()

    def _update_radar_pos(self, mouse_pos: QPointF) -> None:
        pm = self._direct_qimage if self._direct_qimage else self._pixmap_cache.get(self._current_path)
        if pm and not pm.isNull() and pm.width() > 0 and pm.height() > 0:
            scale_fit = min(self.width() / pm.width(), self.height() / pm.height())
            actual_zoom = scale_fit * self._zoom
            base_x = (self.width() - pm.width() * actual_zoom) / 2.0
            base_y = (self.height() - pm.height() * actual_zoom) / 2.0
            img_x = (mouse_pos.x() - base_x - self._offset.x()) / actual_zoom
            img_y = (mouse_pos.y() - base_y - self._offset.y()) / actual_zoom
            if 0 <= img_x <= pm.width() and 0 <= img_y <= pm.height():
                self._cursor_img_pos = QPointF(img_x, img_y)
            else:
                self._cursor_img_pos = None
        else:
            self._cursor_img_pos = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_panning:
            delta = event.position() - self._last_mouse_pos
            self._offset += delta
            self._last_mouse_pos = event.position()
        self._update_radar_pos(event.position())
        self.update()

    def leaveEvent(self, event: QEvent) -> None:
        super().leaveEvent(event)
        self._cursor_img_pos = None
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()

class InteractiveOutputImagePreviewLabel(InteractiveImagePreviewLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame_result: OcrFrameResult | None = None
        self._meta_roi_xywh: list[int] | None = None

    def set_ocr_frame_result(self, frame: OcrFrameResult, meta_roi: list[int] | None) -> None:
        self._frame_result = frame
        self._meta_roi_xywh = meta_roi
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        
        # CHỈ vẽ Box nếu ảnh thật sự đã Load xong
        if self._image_width <= 0 or self._image_height <= 0:
            return

        if self._frame_result is None or not self._frame_result.text_boxes:
            return

        painter = QPainter(self)
        try:
            w_factor = self.width() / max(1, self._image_width)
            h_factor = self.height() / max(1, self._image_height)
            scale_fit = min(w_factor, h_factor)
            actual_zoom = scale_fit * self._zoom

            img_w = self._image_width * actual_zoom
            img_h = self._image_height * actual_zoom
            base_x = (self.width() - img_w) / 2.0
            base_y = (self.height() - img_h) / 2.0

            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            box_pen = QPen(QColor(0, 255, 0), max(1.0, 2.0 * actual_zoom))
            text_fill_pen = QPen(QColor(255, 255, 0), 1)

            font = QFont()
            font.setPointSize(max(10, int(self._image_height * actual_zoom // 20)))
            font.setBold(True)
            painter.setFont(font)

            meta_rx = self._meta_roi_xywh[0] if self._meta_roi_xywh else 0
            meta_ry = self._meta_roi_xywh[1] if self._meta_roi_xywh else 0
            
            # [TÍNH TOÁN LẠI HỆ SỐ TỶ LỆ DỰA TRÊN ẢNH GỐC / ẢNH MỚI]
            # Tính Tỷ lệ Dãn nếu người dùng đã bật chức năng Phóng to (Upscale) ở lần Trích xuất trước
            # Vì tọa độ trong File JSON (self._frame_result) là tọa độ của Video gốc,
            # Ta cần tịnh tiến lại tọa độ này để nó Khớp với Bức ảnh hiện tại (vốn đã bị Scale nếu bật Upscale).
            
            # Mặc định Scale Ratio là 1.0 (Không Upscale).
            scale_ratio_x = 1.0
            scale_ratio_y = 1.0

            # Giả định ROI Crop Width / Height (Từ Meta)
            roi_original_w = self._meta_roi_xywh[2] if self._meta_roi_xywh else 1920
            roi_original_h = self._meta_roi_xywh[3] if self._meta_roi_xywh else 1080

            if roi_original_w > 0 and roi_original_h > 0:
                scale_ratio_x = float(self._image_width) / float(roi_original_w)
                scale_ratio_y = float(self._image_height) / float(roi_original_h)

            for text_box in self._frame_result.text_boxes:
                if not text_box.polygon: continue
                
                drawn_polygon = []
                for pt in text_box.polygon:
                    # 1. Chuyển Tọa độ Tuyệt đối của Video Gốc về Tọa độ Tương đối của ROI Cắt
                    rel_x = pt[0] - meta_rx
                    rel_y = pt[1] - meta_ry
                    
                    # 2. Chuyển Tọa độ Tương đối của ROI Cắt về Tọa độ của Ảnh (Scale nếu có)
                    scaled_x = rel_x * scale_ratio_x
                    scaled_y = rel_y * scale_ratio_y

                    # 3. Phóng to Tọa độ lên Giao diện UI
                    screen_x = base_x + self._offset.x() + scaled_x * actual_zoom
                    screen_y = base_y + self._offset.y() + scaled_y * actual_zoom

                    drawn_polygon.append(QPointF(screen_x, screen_y))

                painter.setPen(box_pen)
                num_pts = len(drawn_polygon)
                for i in range(num_pts):
                    painter.drawLine(drawn_polygon[i], drawn_polygon[(i+1)%num_pts])

                ax = drawn_polygon[0].x()
                ay = drawn_polygon[0].y()
                painter.setPen(text_fill_pen)
                # [FIX]: PyQt6 yêu cầu truyền đối tượng QPointF thay vì số thực đơn lẻ
                painter.drawText(QPointF(ax, max(0.0, ay - 5.0)), text_box.text)

        finally:
            painter.end()

# ==============================================================================
# MAIN DEBUG PAGE
# ==============================================================================
class DebugPage(QWidget):
    def __init__(self, container: ApplicationContainer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("debugPage")
        self._container = container
        self._translator = container.translator
        self._view_model = DebugPageViewModel(container, parent=self)

        self._debounce_validate_timer = QTimer(self)
        self._debounce_validate_timer.setSingleShot(True)
        self._debounce_validate_timer.setInterval(400)
        self._debounce_validate_timer.timeout.connect(self._live_validate_json)

        self._current_raw_json_cache = ""
        self._current_json_path: Path | None = None
        self._custom_draw_roi: Roi | None = None

        self._is_syncing_from_video = False
        self._is_syncing_from_slider = False

        self._build_ui()
        self._connect_signals()
        self.installEventFilter(self)
        self._sync_live_tuning_with_settings()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # 1. Header
        header_lay = QHBoxLayout()
        title = QLabel(self._translator.translate("debug.title"))
        title.setStyleSheet(f"font-size: {_m.FONT_SIZE_HEADING}px; font-weight: 600;")

        self._btn_open = PushButton(self._translator.translate("debug.btn_open_raw"))
        self._lbl_file = CaptionLabel(self._translator.translate("debug.no_file"))

        header_lay.addWidget(title); header_lay.addStretch(1)
        header_lay.addWidget(self._lbl_file); header_lay.addWidget(self._btn_open)

        self._btn_build = PrimaryPushButton(self._translator.translate("debug.btn_build_subs"))
        self._btn_build.setEnabled(False)
        self._btn_build.setStyleSheet(f"background-color: {_c.warning()}; color: {_c.on_accent()};")
        header_lay.addWidget(self._btn_build)

        root.addLayout(header_lay)

        # 2. Main Splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # ====================================================
        # LEFT PANE: DUAL VIEW (Video Context + Crop Images)
        # ====================================================
        left_splitter = QSplitter(Qt.Orientation.Vertical, self)

        self._group_video = SectionCard(self._translator.translate("debug.dbg_sec_video"))
        lay_video = QVBoxLayout()
        lay_video.setContentsMargins(0, 0, 0, 0)

        self._video_widget = create_video_widget(
            mpv_options=self._container.build_mpv_player_kwargs(), parent=self, translator=self._translator
        )
        self._video_widget.enable_roi_drawing(True)
        lay_video.addWidget(self._video_widget)
        self._group_video.add_layout(lay_video)
        left_splitter.addWidget(self._group_video)

        self._group_img = SectionCard(self._translator.translate("debug.dbg_sec_img"))
        lay_img = QVBoxLayout()
        lay_img.setContentsMargins(0, 0, 0, 0)

        self._tabs_img = QTabWidget()
        self._lbl_img_in = InteractiveImagePreviewLabel(self._translator)
        self._lbl_img_out = InteractiveOutputImagePreviewLabel()

        self._tabs_img.addTab(self._lbl_img_in, self._translator.translate("debug.tab_input"))
        self._tabs_img.addTab(self._lbl_img_out, self._translator.translate("debug.tab_output"))
        lay_img.addWidget(self._tabs_img)
        self._group_img.add_layout(lay_img)

        left_splitter.addWidget(self._group_img)
        left_splitter.setStretchFactor(0, 5); left_splitter.setStretchFactor(1, 5)

        main_splitter.addWidget(left_splitter)

        # ====================================================
        # RIGHT PANE: TUNING PANEL + RAW DATA (JSON)
        # ====================================================
        right_panel = QWidget()
        lay_right = QVBoxLayout(right_panel)
        lay_right.setContentsMargins(0, 0, 0, 0); lay_right.setSpacing(6)

        # --- LIVE TUNING PANEL ---
        self._group_tuning = SectionCard(self._translator.translate("debug.dbg_sec_tuning"))

        scroll_tuning = QScrollArea()
        scroll_tuning.setWidgetResizable(True)
        scroll_tuning.setFrameShape(QScrollArea.Shape.NoFrame)
        tuning_content = QWidget()
        form_tuning = QFormLayout(tuning_content)
        form_tuning.setContentsMargins(4, 8, 4, 0)

        # [V3.30] Bổ sung ComboBox lựa chọn Mô hình (AI Model)
        self._cb_model = ComboBox()
        self._cb_model.addItem("PP-OCRv5 (Mobile) - Nhanh", userData="PP-OCRv5_mobile")
        self._cb_model.addItem(self._translator.translate("debug.dbg_m_server"), userData="PP-OCRv5_server")
        self._cb_model.addItem(self._translator.translate("debug.dbg_m_v4"), userData="PP-OCRv4")
        form_tuning.addRow(self._translator.translate("debug.dbg_ai_model"), self._cb_model)

        self._chk_use_custom_roi = CheckBox(self._translator.translate("debug.dbg_chk_roi"))
        self._chk_use_custom_roi.setEnabled(False)
        form_tuning.addRow("", self._chk_use_custom_roi)

        self._spin_limit_side = SpinBox()
        self._spin_limit_side.setRange(0, 4096); self._spin_limit_side.setSingleStep(32)

        self._spin_det_thresh = DoubleSpinBox()
        self._spin_det_thresh.setRange(0.01, 1.0); self._spin_det_thresh.setSingleStep(0.05)

        self._spin_box_thresh = DoubleSpinBox()
        self._spin_box_thresh.setRange(0.01, 1.0); self._spin_box_thresh.setSingleStep(0.05)

        self._spin_unclip_ratio = DoubleSpinBox()
        self._spin_unclip_ratio.setRange(0.5, 3.0); self._spin_unclip_ratio.setSingleStep(0.1)

        form_tuning.addRow(self._translator.translate("debug.dbg_limit_side"), self._spin_limit_side)
        form_tuning.addRow(self._translator.translate("debug.dbg_det_thresh"), self._spin_det_thresh)
        form_tuning.addRow(self._translator.translate("debug.dbg_box_thresh"), self._spin_box_thresh)
        form_tuning.addRow(self._translator.translate("debug.dbg_unclip"), self._spin_unclip_ratio)

        self._chk_upscale = CheckBox("Upscale")
        self._spin_upscale_h = SpinBox(); self._spin_upscale_h.setRange(32, 512); self._spin_upscale_h.setSuffix(" px")
        self._chk_upscale.toggled.connect(self._spin_upscale_h.setEnabled)

        self._chk_contrast = CheckBox("Contrast")
        self._spin_contrast_val = DoubleSpinBox(); self._spin_contrast_val.setRange(0.5, 3.0); self._spin_contrast_val.setSingleStep(0.1)
        self._chk_contrast.toggled.connect(self._spin_contrast_val.setEnabled)

        self._chk_sharpen = CheckBox("Sharpen")

        self._chk_clahe = CheckBox("CLAHE")
        self._spin_clahe_clip = DoubleSpinBox(); self._spin_clahe_clip.setRange(1.0, 10.0); self._spin_clahe_clip.setSingleStep(0.5)
        self._spin_clahe_tile = SpinBox(); self._spin_clahe_tile.setRange(2, 32)
        self._chk_clahe.toggled.connect(self._spin_clahe_clip.setEnabled)
        self._chk_clahe.toggled.connect(self._spin_clahe_tile.setEnabled)

        box_prep_up = QHBoxLayout(); box_prep_up.setContentsMargins(0,0,0,0)
        box_prep_up.addWidget(self._chk_upscale); box_prep_up.addWidget(self._spin_upscale_h)

        box_prep_con = QHBoxLayout(); box_prep_con.setContentsMargins(0,0,0,0)
        box_prep_con.addWidget(self._chk_contrast); box_prep_con.addWidget(self._spin_contrast_val)

        box_prep_clahe = QHBoxLayout(); box_prep_clahe.setContentsMargins(0,0,0,0)
        box_prep_clahe.addWidget(self._chk_clahe); box_prep_clahe.addWidget(self._spin_clahe_clip); box_prep_clahe.addWidget(self._spin_clahe_tile)

        form_tuning.addRow(self._translator.translate("debug.dbg_upscale"), box_prep_up)
        form_tuning.addRow(self._translator.translate("debug.dbg_contrast"), box_prep_con)
        form_tuning.addRow(self._translator.translate("debug.dbg_clahe"), box_prep_clahe)
        form_tuning.addRow(self._translator.translate("debug.dbg_sharpen"), self._chk_sharpen)

        scroll_tuning.setWidget(tuning_content)
        tuning_wrapper = QVBoxLayout()
        tuning_wrapper.setContentsMargins(0, 0, 0, 0)
        tuning_wrapper.addWidget(scroll_tuning)
        self._group_tuning.add_layout(tuning_wrapper)

        lay_right.addWidget(self._group_tuning, stretch=2)

        # --- RAW JSON PANEL ---
        self._group_data = SectionCard(self._translator.translate("debug.group_data"))
        lay_data = QVBoxLayout()
        lay_data.setContentsMargins(0, 0, 0, 0)

        self._txt_json = TextEdit()
        self._txt_json.setReadOnly(False)
        self._txt_json.setStyleSheet(_mono_style())
        self._syntax_highlighter = JsonSyntaxHighlighter(self._txt_json.document())
        lay_data.addWidget(self._txt_json)

        error_lay = QHBoxLayout()
        self._lbl_json_error = QLabel("")
        self._lbl_json_error.setStyleSheet(f"color: {_c.danger()}; font-weight: bold;")
        self._lbl_json_error.hide()

        self._btn_reocr_frame = PushButton(self._translator.translate("debug.dbg_btn_reocr"))
        self._btn_reocr_frame.setStyleSheet(f"background-color: {_c.success()}; color: {_c.on_accent()}; font-weight: bold;")
        self._btn_reocr_frame.setEnabled(False)

        self._btn_format_json = PushButton(self._translator.translate("debug.dbg_btn_format"))
        self._btn_format_json.clicked.connect(self._format_json_clicked)
        
        self._btn_reset_frame = PushButton(self._translator.translate("debug.dbg_btn_reset"))
        self._btn_reset_frame.setEnabled(False)
        self._btn_save_frame = PushButton(self._translator.translate("debug.dbg_btn_save"))
        self._btn_save_frame.setEnabled(False)

        error_lay.addWidget(self._lbl_json_error)
        error_lay.addStretch(1)
        error_lay.addWidget(self._btn_format_json)
        error_lay.addWidget(self._btn_reocr_frame)
        error_lay.addWidget(self._btn_reset_frame)
        error_lay.addWidget(self._btn_save_frame)
        lay_data.addLayout(error_lay)
        self._group_data.add_layout(lay_data)

        lay_right.addWidget(self._group_data, stretch=3)
        main_splitter.addWidget(right_panel)

        main_splitter.setStretchFactor(0, 6)
        main_splitter.setStretchFactor(1, 4)
        root.addWidget(main_splitter, stretch=1)

        # 3. Bottom Controls
        ctrl_lay = QHBoxLayout()
        
        self._btn_auto_sync = ToolButton(FluentIcon.LINK)
        self._btn_auto_sync.setCheckable(True)
        self._btn_auto_sync.setChecked(True)
        self._btn_auto_sync.setToolTip(self._translator.translate("debug.dbg_tt_autosync"))
        
        self._btn_play = ToolButton(FluentIcon.PLAY)
        self._btn_prev = ToolButton(FluentIcon.CARE_LEFT_SOLID)
        self._btn_next = ToolButton(FluentIcon.CARE_RIGHT_SOLID)
        set_accessible_name(self._btn_play, self._translator.translate("debug.dbg_acc_playpause"))
        set_accessible_name(self._btn_prev, self._translator.translate("debug.dbg_acc_prev"))
        set_accessible_name(self._btn_next, self._translator.translate("debug.dbg_acc_next"))

        self._slider = Slider(Qt.Orientation.Horizontal)
        set_accessible_name(self._slider, self._translator.translate("debug.dbg_acc_timeline"), set_tooltip=False)
        self._slider.setRange(0, 0); self._slider.setEnabled(False)

        self._lbl_counter = CaptionLabel("Frame: 0 / 0  |  --:--:--.---")
        self._lbl_counter.setFixedWidth(260)
        self._lbl_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_counter.setStyleSheet(mono_label_style())

        ctrl_lay.addWidget(self._btn_auto_sync)
        ctrl_lay.addWidget(self._btn_play); ctrl_lay.addWidget(self._btn_prev)
        ctrl_lay.addWidget(self._slider, stretch=1); ctrl_lay.addWidget(self._lbl_counter); ctrl_lay.addWidget(self._btn_next)
        root.addLayout(ctrl_lay)

    def _sync_live_tuning_with_settings(self) -> None:
        cfg = self._container.settings_service.current

        idx = self._cb_model.findData(cfg.ocr.version)
        if idx >= 0:
            self._cb_model.setCurrentIndex(idx)

        self._spin_limit_side.setValue(cfg.ocr.limit_side_len)
        self._spin_det_thresh.setValue(cfg.ocr.det_thresh)
        self._spin_box_thresh.setValue(cfg.ocr.det_box_thresh)
        self._spin_unclip_ratio.setValue(cfg.ocr.det_unclip_ratio)

        self._chk_upscale.setChecked(cfg.preprocess.upscale_small_text)
        self._spin_upscale_h.setValue(cfg.preprocess.upscale_target_height_px)

        self._chk_contrast.setChecked(cfg.preprocess.apply_contrast_boost)
        self._spin_contrast_val.setValue(cfg.preprocess.contrast_factor)

        self._chk_sharpen.setChecked(cfg.preprocess.apply_sharpen)

        self._chk_clahe.setChecked(cfg.preprocess.apply_clahe)
        self._spin_clahe_clip.setValue(cfg.preprocess.clahe_clip_limit)
        self._spin_clahe_tile.setValue(cfg.preprocess.clahe_tile_size)

        self._chk_upscale.toggled.emit(self._chk_upscale.isChecked())
        self._chk_contrast.toggled.emit(self._chk_contrast.isChecked())
        self._chk_clahe.toggled.emit(self._chk_clahe.isChecked())

    def _connect_signals(self) -> None:
        self._btn_open.clicked.connect(self._on_open_file)
        self._view_model.file_loaded.connect(self._on_file_loaded)
        self._view_model.frame_changed.connect(self._on_frame_changed)
        self._view_model.action_error.connect(self._on_error)
        self._view_model.action_success.connect(self._on_success)

        self._view_model.live_images_ready.connect(self._on_live_images_ready)
        self._view_model.is_busy.connect(self._on_busy_state_changed)

        self._btn_play.clicked.connect(self._video_widget.toggle_play_pause)
        self._btn_prev.clicked.connect(self._view_model.prev_frame)
        self._btn_next.clicked.connect(self._view_model.next_frame)
        self._slider.valueChanged.connect(self._on_slider_moved_by_user)

        self._txt_json.textChanged.connect(self._debounce_validate_timer.start)
        self._btn_save_frame.clicked.connect(self._on_save_frame_clicked)
        self._btn_reset_frame.clicked.connect(self._on_reset_frame_clicked)
        self._btn_reocr_frame.clicked.connect(self._on_reocr_frame_clicked)
        self._btn_build.clicked.connect(self._view_model.build_subtitles)

        self._video_widget.position_changed.connect(self._on_video_position_changed)
        self._video_widget.state_changed.connect(self._on_video_state_changed)
        self._video_widget.video_clicked.connect(lambda btn: self._video_widget.toggle_play_pause())

        self._video_widget.roi_changed.connect(self._on_custom_roi_drawn)

    def _on_custom_roi_drawn(self, roi: Roi | None) -> None:
        self._custom_draw_roi = roi
        if roi is not None:
            self._chk_use_custom_roi.setEnabled(True)
            self._chk_use_custom_roi.setChecked(True)
        else:
            self._chk_use_custom_roi.setEnabled(False)
            self._chk_use_custom_roi.setChecked(False)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            k = event.key()
            mods = event.modifiers()

            if k == Qt.Key.Key_R and (mods & Qt.KeyboardModifier.ControlModifier) and self._btn_reocr_frame.isEnabled():
                self._btn_reocr_frame.click()
                return True
            if k == Qt.Key.Key_S and (mods & Qt.KeyboardModifier.ControlModifier) and self._btn_save_frame.isEnabled():
                self._btn_save_frame.click()
                return True

            if self._txt_json.hasFocus() or isinstance(QApplication.focusWidget(), (SpinBox, DoubleSpinBox, ComboBox)):
                return super().eventFilter(obj, event)

            step = 10 if (mods & Qt.KeyboardModifier.ShiftModifier) else 1

            if k == Qt.Key.Key_Left:
                self._on_slider_moved_by_user(max(0, self._slider.value() - step))
                return True
            if k == Qt.Key.Key_Right:
                self._on_slider_moved_by_user(min(self._slider.maximum(), self._slider.value() + step))
                return True
            if k == Qt.Key.Key_Space:
                self._video_widget.toggle_play_pause()
                return True

        return super().eventFilter(obj, event)

    def _on_open_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, self._translator.translate("debug.dlg_open_raw"), "", self._translator.translate("debug.dlg_raw_filter"),
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path_str:
            self._current_json_path = Path(path_str)
            self._view_model.load_raw_file(self._current_json_path)

    def _on_file_loaded(self, success: bool, message: str) -> None:
        if success:
            InfoBar.success(self._translator.translate("debug.ib_success"), message, parent=self)
            self._lbl_file.setText(message)
            total = self._view_model.total_frames
            if total > 0:
                self._slider.setEnabled(True); self._slider.setRange(0, total - 1)
                self._btn_build.setEnabled(True); self._btn_save_frame.setEnabled(True)
                self._btn_reset_frame.setEnabled(True); self._btn_reocr_frame.setEnabled(True)

            meta = self._view_model._meta
            if meta and hasattr(meta, 'video_name') and self._current_json_path:
                video_path = self._current_json_path.parent / meta.video_name
                if video_path.exists():
                    self._video_widget.load(video_path)
                    self._video_widget.pause()
                    if meta.roi_xywh:
                        rx, ry, rw, rh = meta.roi_xywh
                        self._video_widget.set_secondary_rois([Roi(x=rx, y=ry, width=rw, height=rh)])
                else:
                    InfoBar.warning(
                        self._translator.translate("debug.ib_no_video_t"),
                        self._translator.translate("debug.ib_no_video_c").replace("{name}", str(meta.video_name)),
                        parent=self, duration=6000
                    )
        else:
            InfoBar.error(self._translator.translate("debug.ib_fail"), message, parent=self)
            self._slider.setEnabled(False); self._btn_build.setEnabled(False)
            self._btn_save_frame.setEnabled(False); self._btn_reset_frame.setEnabled(False)
            self._btn_reocr_frame.setEnabled(False)
            self._video_widget.release_player()

    def _on_slider_moved_by_user(self, value: int) -> None:
        if self._is_syncing_from_video: return
        self._is_syncing_from_slider = True
        self._slider.setValue(value)
        self._view_model.jump_to_frame(value)
        self._is_syncing_from_slider = False

    def _update_mpv_overlay(self, frame: OcrFrameResult) -> None:
        offset_x, offset_y = 0, 0
        if self._view_model._meta and self._view_model._meta.roi_xywh:
            offset_x, offset_y, _, _ = self._view_model._meta.roi_xywh

        boxes_xywh = []
        for box in frame.text_boxes:
            if box.bounding_box:
                x_min, y_min, x_max, y_max = box.bounding_box
                boxes_xywh.append((
                    int(x_min + offset_x),
                    int(y_min + offset_y),
                    int(x_max - x_min),
                    int(y_max - y_min)
                ))
        self._video_widget.set_ocr_overlay(boxes_xywh, visible=True)

    def _on_frame_changed(
        self, current: int, total: int, frame: OcrFrameResult,
        img_in: str, img_out: str, json_txt: str
    ) -> None:
        self._lbl_counter.setText(f"Frame: {current + 1} / {total}  |  {seconds_to_display(frame.timestamp_sec)}")
        self._current_raw_json_cache = json_txt

        self._txt_json.blockSignals(True)
        self._txt_json.setPlainText(json_txt)
        self._lbl_json_error.hide()
        self._txt_json.setStyleSheet(_mono_style())
        self._btn_save_frame.setEnabled(True)
        self._txt_json.blockSignals(False)

        fallback = self._translator.translate("debug.no_image_found")
        self._lbl_img_in.set_image_path(img_in, fallback)
        self._lbl_img_out.set_image_path(img_out, fallback)
        
        self._lbl_img_out.set_ocr_frame_result(frame, self._view_model._meta.roi_xywh if self._view_model._meta else None)

        if not self._is_syncing_from_video:
            self._video_widget.seek(frame.timestamp_sec)

        self._update_mpv_overlay(frame)

    def _on_live_images_ready(self, img_in: QImage, img_out: QImage) -> None:
        self._lbl_img_in.set_direct_qimage(img_in)
        self._lbl_img_out.set_direct_qimage(img_out)

    def _on_video_position_changed(self, pos_sec: float) -> None:
        if not self._view_model._frames or self._is_syncing_from_slider: return
        
        if not getattr(self, '_btn_auto_sync', None) or not self._btn_auto_sync.isChecked():
            return

        timestamps = [f.timestamp_sec for f in self._view_model._frames]
        idx = bisect.bisect_left(timestamps, pos_sec)

        if idx == 0: best_idx = 0
        elif idx == len(timestamps): best_idx = len(timestamps) - 1
        else:
            before, after = timestamps[idx - 1], timestamps[idx]
            best_idx = idx - 1 if (pos_sec - before) < (after - pos_sec) else idx

        if self._slider.value() != best_idx:
            self._is_syncing_from_video = True
            self._slider.setValue(best_idx)
            self._view_model.jump_to_frame(best_idx)
            self._is_syncing_from_video = False

    def _on_video_state_changed(self, is_playing: bool) -> None:
        self._btn_play.setIcon(
                (FluentIcon.PAUSE if is_playing else FluentIcon.PLAY).icon()
            )

    def _format_json_clicked(self) -> None:
        try:
            data = json.loads(self._txt_json.toPlainText())
            formatted = json.dumps(data, ensure_ascii=False, indent=2)
            self._txt_json.setPlainText(formatted)
        except json.JSONDecodeError:
            InfoBar.error(self._translator.translate("debug.ib_error"), self._translator.translate("debug.ib_json_bad"), parent=self)

    def _live_validate_json(self) -> None:
        json_txt = self._txt_json.toPlainText()
        if not json_txt.strip(): return
        try:
            json.loads(json_txt)
            self._lbl_json_error.hide()
            self._txt_json.setStyleSheet(_mono_style())
            self._btn_save_frame.setEnabled(True)
        except json.JSONDecodeError as e:
            self._lbl_json_error.setText(self._translator.translate("debug.json_err_line").replace("{line}", str(e.lineno)).replace("{col}", str(e.colno)))
            self._lbl_json_error.show()
            self._txt_json.setStyleSheet(_mono_style(error=True))
            self._btn_save_frame.setEnabled(False)

    def _on_reocr_frame_clicked(self) -> None:
        tweaks = {
            "model_override": self._cb_model.currentData(),
            "limit_side_len": self._spin_limit_side.value(),
            "det_thresh": self._spin_det_thresh.value(),
            "box_thresh": self._spin_box_thresh.value(),
            "det_unclip_ratio": self._spin_unclip_ratio.value(),
            "upscale": self._chk_upscale.isChecked(),
            "upscale_h": self._spin_upscale_h.value(),
            "sharpen": self._chk_sharpen.isChecked(),
            "contrast": self._chk_contrast.isChecked(),
            "contrast_factor": self._spin_contrast_val.value(),
            "clahe": self._chk_clahe.isChecked(),
            "clahe_clip": self._spin_clahe_clip.value(),
            "clahe_tile": self._spin_clahe_tile.value(),
        }

        if self._chk_use_custom_roi.isChecked() and self._custom_draw_roi is not None:
            tweaks["custom_roi"] = self._custom_draw_roi

        self._view_model.reocr_current_frame(tweaks)

    def _on_busy_state_changed(self, is_busy: bool) -> None:
        self._btn_reocr_frame.setEnabled(not is_busy)
        self._btn_save_frame.setEnabled(not is_busy)
        self._btn_reset_frame.setEnabled(not is_busy)

        if is_busy:
            self._btn_reocr_frame.setText(self._translator.translate("debug.dbg_btn_analyzing"))
        else:
            self._btn_reocr_frame.setText(self._translator.translate("debug.dbg_btn_reocr"))

    def _on_save_frame_clicked(self) -> None:
        json_txt = self._txt_json.toPlainText()
        success = self._view_model.save_current_frame_json(json_txt)
        if success:
            InfoBar.success(self._translator.translate("debug.ib_saved_t"), self._translator.translate("debug.ib_saved_c"), parent=self, duration=1500)
            self._current_raw_json_cache = json_txt
            frame = self._view_model._frames[self._view_model._current_idx]
            
            self._update_mpv_overlay(frame)
            self._lbl_img_out.set_ocr_frame_result(frame, self._view_model._meta.roi_xywh if self._view_model._meta else None)

    def _on_reset_frame_clicked(self) -> None:
        self._txt_json.setPlainText(self._current_raw_json_cache)
        self._live_validate_json()

    def _on_error(self, message: str) -> None:
        InfoBar.error(self._translator.translate("debug.ib_op_error"), message, parent=self, duration=5000)

    def _on_success(self, title: str, detail: str, is_noop: bool = False) -> None:
        # [v3.23.380] Dùng cờ is_noop từ view_model thay vì dò chuỗi tiếng Việt trong
        # detail (trước đây so khớp "Không có sự khác" — vừa là dead code vừa vỡ khi đổi
        # ngôn ngữ). is_noop=True nghĩa là chỉnh tham số không làm đổi kết quả nhận diện.
        if is_noop:
            InfoBar.info(title, detail, parent=self, duration=3000)
        else:
            InfoBar.success(title, detail, parent=self, duration=8000)

    def closeEvent(self, event) -> None:
        if hasattr(self, '_video_widget'):
            self._video_widget.release_player()
        super().closeEvent(event)

__all__ = ["DebugPage"]
