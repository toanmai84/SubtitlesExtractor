"""Thẻ nội dung (Section Card) tái sử dụng cho giao diện, theo ngôn ngữ Fluent.

[v3.23.64 — Giai đoạn 5 tái thiết UI/UX] Trước đây các trang dùng ``QGroupBox`` thuần Qt
(viền vuông, không nhất quán với Fluent). Widget này bọc ``HeaderCardWidget`` của
qfluentwidgets thành một thẻ có tiêu đề + vùng nội dung XẾP DỌC với khoảng cách chuẩn (lấy
từ :mod:`theme.metrics`), giúp thay thế ``QGroupBox`` đồng nhất trên toàn ứng dụng.

Thiết kế theo SRP: widget chỉ lo phần trình bày (khung thẻ + tiêu đề), nội dung được tiêm
vào qua :meth:`add_widget` / :meth:`add_layout` — không chứa logic nghiệp vụ.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLayout, QToolButton, QVBoxLayout, QWidget
from subtitles_extractor.presentation.fluent_compat import HeaderCardWidget

from subtitles_extractor.presentation.theme import metrics as _metrics

__all__ = ["SectionCard"]


class SectionCard(HeaderCardWidget):
    """Thẻ có tiêu đề với vùng nội dung xếp dọc, khoảng cách theo thang chuẩn.

    Dùng thay cho ``QGroupBox`` để đồng nhất ngôn ngữ thiết kế Fluent. Nội dung được thêm
    qua :meth:`add_widget` hoặc :meth:`add_layout`.

    [v3.23.110] Hỗ trợ THU GỌN (collapsible): với các nhóm tuỳ chọn/nâng cao, đặt
    ``collapsible=True`` (và ``collapsed=True`` để thu gọn sẵn) giúp trang ngắn lại, ít
    cuộn. Một nút mũi tên ở tiêu đề cho phép mở/đóng nội dung.

    Example:
        >>> card = SectionCard("Cấu hình AI")
        >>> card.add_widget(some_form_widget)
        >>> opt = SectionCard("Tuỳ chọn", collapsible=True, collapsed=True)
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        collapsible: bool = False,
        collapsed: bool = False,
        translator: object | None = None,
    ) -> None:
        """Khởi tạo thẻ.

        Args:
            title: Tiêu đề hiển thị trên đầu thẻ.
            parent: Widget cha (tuỳ chọn).
            collapsible: Cho phép thu gọn/mở rộng nội dung.
            collapsed: Thu gọn sẵn khi khởi tạo (chỉ có tác dụng nếu ``collapsible``).
        """
        super().__init__(parent)
        self._translator = translator
        self.setTitle(title)
        # viewLayout của HeaderCardWidget mặc định là QHBoxLayout; ta đặt một QVBoxLayout
        # con để nội dung xếp dọc với khoảng cách chuẩn, đồng nhất giữa các thẻ.
        self._content_layout = QVBoxLayout()
        self._content_layout.setSpacing(_metrics.SPACING_SM)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._collapsible = collapsible
        self._collapsed = False
        self._content_container: QWidget | None = None
        self._toggle_btn: QToolButton | None = None

        if collapsible:
            # Nội dung nằm trong container để ẩn/hiện khi thu gọn.
            self._content_container = QWidget()
            self._content_container.setLayout(self._content_layout)
            self.viewLayout.addWidget(self._content_container)

            self._toggle_btn = QToolButton()
            self._toggle_btn.setAutoRaise(True)
            self._toggle_btn.setToolTip(
                self._translator.translate("common.section_collapse")
                if self._translator
                else "Mở rộng / thu gọn mục này"
            )
            self._toggle_btn.clicked.connect(self.toggle_collapsed)
            # Đặt nút mũi tên ở tiêu đề nếu có (qfluentwidgets thật); môi trường test
            # stub không có headerLayout thật -> bỏ qua an toàn, logic thu gọn vẫn chạy.
            header = getattr(self, "headerLayout", None)
            if header is not None and hasattr(header, "addWidget"):
                header.addWidget(self._toggle_btn)
            self.set_collapsed(collapsed)
        else:
            self.viewLayout.addLayout(self._content_layout)

    def set_collapsed(self, collapsed: bool) -> None:
        """Thu gọn (ẩn) hoặc mở rộng (hiện) vùng nội dung của thẻ."""
        if not self._collapsible or self._content_container is None:
            return
        self._collapsed = collapsed
        self._content_container.setVisible(not collapsed)
        if self._toggle_btn is not None:
            self._toggle_btn.setText("▸" if collapsed else "▾")

    def toggle_collapsed(self) -> None:
        """Đảo trạng thái thu gọn/mở rộng (gọi khi bấm nút mũi tên)."""
        self.set_collapsed(not self._collapsed)

    def add_widget(self, widget: QWidget) -> None:
        """Thêm một widget vào vùng nội dung của thẻ.

        Args:
            widget: Widget cần thêm.
        """
        self._content_layout.addWidget(widget)

    def add_layout(self, layout: QLayout) -> None:
        """Thêm một layout con vào vùng nội dung của thẻ.

        Args:
            layout: Layout cần thêm.
        """
        self._content_layout.addLayout(layout)

    @property
    def content_layout(self) -> QVBoxLayout:
        """Trả về layout nội dung để tuỳ biến nâng cao (vd chèn stretch)."""
        return self._content_layout
