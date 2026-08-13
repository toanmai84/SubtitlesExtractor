"""Các widget nhỏ tái sử dụng cho UI editor.

TỐI ƯU HÓA ĐỈNH CAO GIAO DIỆN VÀ TRẢI NGHIỆM (UI/UX MASTERPIECE - V3.45):
    1. [UX BREAKTHROUGH] Smart Hover Scroll (Chuẩn NLE): Nhận diện tia X vị trí 
       con trỏ chuột trên ô Thời gian. Lăn chuột ở vị trí Giờ/Phút/Giây sẽ thay đổi 
       tương ứng ngay lập tức MÀ KHÔNG CẦN CLICK CHỌN (Focus).
    2. [ROBUSTNESS] Custom Validator: Chống lỗi văng phần mềm khi người dùng lỡ tay
       nhập sai định dạng. Tự fallback về 0 an toàn.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QBrush, QColor, QPainter, QPaintEvent,
    QSyntaxHighlighter, QTextCharFormat, QTextDocument,
    QValidator, QWheelEvent,
)
from PySide6.QtWidgets import QSpinBox, QWidget

from subtitles_extractor.presentation.utils.text_clean import (
    ASS_TAG_REGEX, HTML_TAG_REGEX,
)


class SubtitleTagHighlighter(QSyntaxHighlighter):
    """Tô màu phân loại các tag ASS/HTML trong QPlainTextEdit."""

    def __init__(self, document: QTextDocument | None = None) -> None:
        super().__init__(document)

        self._html_format = QTextCharFormat()
        self._html_format.setForeground(QColor("#4da6ff"))  # Xanh Neon
        self._html_format.setFontItalic(True)

        self._ass_format = QTextCharFormat()
        self._ass_format.setForeground(QColor("#d4b856"))  # Vàng Đất
        self._ass_format.setFontItalic(True)

    def highlightBlock(self, text: str) -> None:
        if not text:
            return

        for match in HTML_TAG_REGEX.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self._html_format)

        for match in ASS_TAG_REGEX.finditer(text):
            self.setFormat(match.start(), match.end() - match.start(), self._ass_format)


class TimeSpinBox(QSpinBox):
    """Spin box format ``HH:MM:SS.mmm`` — value tính bằng mili-giây."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRange(0, 360_000_000)  # 100 giờ
        self.setSingleStep(100)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(110)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def validate(self, input_text: str, pos: int) -> tuple[QValidator.State, str, int]:
        """Tước bỏ quyền chặn ký tự mặc định, tha lỗi cho người dùng nhập thoải mái."""
        return QValidator.State.Acceptable, input_text, pos

    def textFromValue(self, value: int) -> str:
        h, rem = divmod(value, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def valueFromText(self, text: str) -> int:
        """Trình phân tích nội dung siêu mềm dẻo (Robust Parser)."""
        text = text.replace(",", ".").strip()
        if not text: return 0

        parts = text.split(":")
        try:
            if len(parts) == 3: h_str, m_str, s_ms_str = parts[0], parts[1], parts[2]
            elif len(parts) == 2: h_str, m_str, s_ms_str = "0", parts[0], parts[1]
            elif len(parts) == 1: h_str, m_str, s_ms_str = "0", "0", parts[0]
            else: return 0

            sec_parts = str(s_ms_str).split(".")
            s = int(sec_parts[0]) if sec_parts[0] else 0

            ms = 0
            if len(sec_parts) > 1:
                ms_str = sec_parts[1][:3].ljust(3, "0")
                ms = int(ms_str)

            h = int(h_str) if h_str else 0
            m = int(m_str) if m_str else 0

            return h * 3_600_000 + m * 60_000 + s * 1000 + ms
        except (ValueError, TypeError):
            return 0  # Fallback an toàn tuyệt đối

    def stepBy(self, steps: int) -> None:
        """Nhảy thông minh theo vị trí con trỏ chuột."""
        line_edit = self.lineEdit()
        if not line_edit:
            super().stepBy(steps)
            return

        cursor_pos = line_edit.cursorPosition()
        multiplier = 100  # Default 100ms

        # HH:MM:SS.mmm
        if 0 <= cursor_pos <= 2: multiplier = 3_600_000
        elif 3 <= cursor_pos <= 5: multiplier = 60_000
        elif 6 <= cursor_pos <= 8: multiplier = 1_000

        new_value = self.value() + steps * multiplier
        new_value = max(self.minimum(), min(self.maximum(), new_value))

        if new_value != self.value():
            self.setValue(new_value)
            line_edit.setCursorPosition(cursor_pos)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Cho phép lăn chuột để chỉnh thời gian NGAY CẢ KHI CHƯA CLICK VÀO Ô."""
        line_edit = self.lineEdit()
        if not line_edit:
            super().wheelEvent(event)
            return
            
        mouse_x = event.position().x()
        widget_width = max(1, self.width())
        hover_ratio = mouse_x / widget_width
        
        # Ánh xạ tỷ lệ sang vị trí Index của chuỗi (HH:MM:SS.mmm có 12 ký tự)
        if hover_ratio < 0.30: simulated_cursor_pos = 1
        elif hover_ratio < 0.55: simulated_cursor_pos = 4
        elif hover_ratio < 0.80: simulated_cursor_pos = 7
        else: simulated_cursor_pos = 10
        
        original_cursor_pos = line_edit.cursorPosition()
        
        line_edit.setCursorPosition(simulated_cursor_pos)
        steps = 1 if event.angleDelta().y() > 0 else -1
        self.stepBy(steps)
        
        if self.hasFocus():
            line_edit.setCursorPosition(original_cursor_pos)
            
        event.accept()


class CpsGauge(QWidget):
    """Thanh trực quan CPS — bo góc chuẩn Fluent Design."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(8) 
        self.setMinimumWidth(60)
        self._cps: float = 0.0

        self._bg_brush = QBrush(QColor(40, 40, 40, 200))
        self._danger_brush = QBrush(QColor("#FF4D4D"))
        self._warn_brush = QBrush(QColor("#FFB000")) 
        self._safe_brush = QBrush(QColor("#00E676")) 

    def set_cps(self, cps: float) -> None:
        cps = max(0.0, float(cps))
        if abs(self._cps - cps) > 0.05:
            self._cps = cps
            self.setToolTip(f"CPS: {cps:.1f}")
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        radius = h / 2.0

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._bg_brush)
        painter.drawRoundedRect(0, 0, w, h, radius, radius)

        if self._cps > 0:
            ratio = min(1.0, self._cps / 30.0)
            fill_w = max(h, int(ratio * w))

            if self._cps > 20: fill_brush = self._danger_brush
            elif self._cps > 15: fill_brush = self._warn_brush
            else: fill_brush = self._safe_brush

            painter.setBrush(fill_brush)
            painter.drawRoundedRect(0, 0, fill_w, h, radius, radius)
            
        painter.end()

__all__ = ["CpsGauge", "SubtitleTagHighlighter", "TimeSpinBox"]
