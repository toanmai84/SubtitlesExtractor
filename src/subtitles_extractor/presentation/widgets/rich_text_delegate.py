"""Delegate vẽ text phụ đề đã làm sạch tag — giải quyết hiển thị xấu trong table.

TỐI ƯU HÓA ĐỈNH CAO UI/UX (V3.36 - Master Renderer):
    * [PERFORMANCE] Dùng 1 QTextDocument tĩnh duy nhất trong class để render
      toàn bộ View thay vì tạo mới hàng ngàn Object khi cuộn.
"""

from __future__ import annotations

import re
from functools import lru_cache

from PySide6.QtCore import QModelIndex, QRectF, Qt
from PySide6.QtGui import QAbstractTextDocumentLayout, QPainter, QTextDocument, QTextOption
from PySide6.QtWidgets import (
    QApplication, QPlainTextEdit, QStyle, QStyledItemDelegate,
    QStyleOptionViewItem, QWidget,
)

from subtitles_extractor.presentation.widgets.subtitle_table_model import SEARCH_QUERY_ROLE

_ASS_TAG_REGEX = re.compile(r"\{.*?\}")
_HTML_COLOR_REGEX = re.compile(r'color\s*=\s*[\'"]?[^\s>]*[\'"]?', re.IGNORECASE)

@lru_cache(maxsize=2048)
def _compile_html_string(raw_text: str, is_selected: bool, text_color_hex: str, search_query: str) -> str:
    """Memoize xử lý Regex nhanh."""
    html_content = raw_text.replace('\n', '<br>').replace(r'\N', '<br>')
    html_content = _ASS_TAG_REGEX.sub("", html_content)

    if is_selected:
        html_content = _HTML_COLOR_REGEX.sub("", html_content)

    if search_query:
        escaped_query = re.escape(search_query)
        html_content = re.sub(
            f"({escaped_query})",
            r"<span style='background-color: #ffeb3b; color: #000000; font-weight: bold;'>\1</span>",
            html_content,
            flags=re.IGNORECASE
        )

    return f"<div style='color: {text_color_hex}; margin: 0; padding: 0;'>{html_content}</div>"


class RichTextSubtitleDelegate(QStyledItemDelegate):
    def __init__(self, *, text_column: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text_column = text_column
        
        # [MEMORY OPTIMIZE]: Dùng chung 1 Document để render tất cả các Row
        self._shared_document = QTextDocument(self)
        self._shared_document.setDocumentMargin(0)

        text_option = QTextOption()
        text_option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self._shared_document.setDefaultTextOption(text_option)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        if index.column() != self._text_column:
            opt = QStyleOptionViewItem(option)
            opt.state &= ~QStyle.StateFlag.State_HasFocus
            super().paint(painter, opt, index)
            return

        style_option = QStyleOptionViewItem(option)
        self.initStyleOption(style_option, index)
        style_option.state &= ~QStyle.StateFlag.State_HasFocus

        if style_option.font.pointSize() <= 0:
            pixel_size = style_option.font.pixelSize()
            style_option.font.setPointSize(max(1, int(pixel_size * 0.75)) if pixel_size > 0 else 10)

        raw_text_data = str(index.data(Qt.ItemDataRole.UserRole) or style_option.text)
        search_query = str(index.data(SEARCH_QUERY_ROLE) or "")
        is_selected = bool(style_option.state & QStyle.StateFlag.State_Selected)
        text_color_hex = style_option.palette.highlightedText().color().name() if is_selected else style_option.palette.text().color().name()

        styled_html_block = _compile_html_string(raw_text_data, is_selected, text_color_hex, search_query)

        available_width = max(10, style_option.rect.width() - 12)
        
        # Reset Document Status
        self._shared_document.setTextWidth(float(available_width))
        self._shared_document.setDefaultFont(style_option.font)
        self._shared_document.setHtml(styled_html_block)

        style_option.text = ""

        painter.save()
        try:
            active_style = style_option.widget.style() if style_option.widget else QApplication.style()
            active_style.drawControl(
                QStyle.ControlElement.CE_ItemViewItem,
                style_option,
                painter,
                style_option.widget,
            )

            document_height = self._shared_document.documentLayout().documentSize().height()
            vertical_offset = max(0.0, (style_option.rect.height() - document_height) / 2.0)

            painter.translate(
                style_option.rect.left() + 6,
                style_option.rect.top() + vertical_offset,
            )

            paint_context = QAbstractTextDocumentLayout.PaintContext()
            paint_context.palette = style_option.palette

            clip_rect = QRectF(0, 0, available_width, style_option.rect.height())
            paint_context.clip = clip_rect

            self._shared_document.documentLayout().draw(painter, paint_context)
        finally:
            painter.restore()

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex) -> QWidget | None:
        if index.column() != self._text_column: return super().createEditor(parent, option, index)
        # [Newline Stripping Fix] QLineEdit gộp phụ đề 2 dòng thành 1; dùng
        # QPlainTextEdit đa dòng, bỏ viền, bật word-wrap để giữ nguyên xuống dòng.
        editor = QPlainTextEdit(parent)
        editor.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        if index.column() != self._text_column or not isinstance(editor, QPlainTextEdit):
            super().setEditorData(editor, index)
            return
        raw_text = index.data(Qt.ItemDataRole.UserRole) or ""
        editor.setPlainText(raw_text)
        editor.selectAll()

    def setModelData(self, editor: QWidget, model, index: QModelIndex) -> None:
        if index.column() != self._text_column or not isinstance(editor, QPlainTextEdit):
            super().setModelData(editor, model, index)
            return
        new_text = editor.toPlainText()
        model.setData(index, new_text, Qt.ItemDataRole.EditRole)
        model.setData(index, new_text, Qt.ItemDataRole.UserRole)

__all__ = ["RichTextSubtitleDelegate"]
