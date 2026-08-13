"""Thanh tiến độ quy trình — hiển thị đang ở khâu nào và việc cần làm tiếp.

VÌ SAO cần
==========
Rà soát v3.23.316 cho thấy ứng dụng KHÔNG hề chỉ dẫn quy trình: người dùng phải tự
biết thứ tự Trích xuất → Biên tập → Dịch → TTS → Xuất bản. Thanh này luôn hiện ở đầu
cửa sổ, cho biết:

* Đang ở khâu nào trong 5 khâu (chấm tròn sáng/mờ).
* Việc nên làm tiếp theo.
* Nút nhảy thẳng tới trang của bước đó.

Widget này chỉ HIỂN THỊ và phát tín hiệu; việc điều hướng do cửa sổ chính quyết định.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QWidget

from subtitles_extractor.domain.entities.project_record import WorkflowStage
from subtitles_extractor.presentation.fluent_compat import (
    BodyLabel,
    CaptionLabel,
    PushButton,
)

logger = logging.getLogger(__name__)

# Các khâu hiển thị trên thanh (bỏ NEW vì đó là "chưa bắt đầu", không phải một bước).
_DISPLAY_STAGES: tuple[tuple[WorkflowStage, str], ...] = (
    (WorkflowStage.EXTRACTED, "wf_step_extract"),
    (WorkflowStage.EDITED, "wf_step_edit"),
    (WorkflowStage.TRANSLATED, "wf_step_translate"),
    (WorkflowStage.TTS_DONE, "wf_step_tts"),
    (WorkflowStage.PUBLISHED, "wf_step_publish"),
)

# Map khâu -> khoá "việc nên làm tiếp" (giữ i18n ở tầng presentation, không đụng domain).
_NEXT_ACTION_KEYS: dict[WorkflowStage, str] = {
    WorkflowStage.NEW: "wf_na_new",
    WorkflowStage.EXTRACTED: "wf_na_extracted",
    WorkflowStage.EDITED: "wf_na_edited",
    WorkflowStage.TRANSLATED: "wf_na_translated",
    WorkflowStage.TTS_DONE: "wf_na_tts",
    WorkflowStage.PUBLISHED: "wf_na_published",
}

_DONE_MARK = "●"
_PENDING_MARK = "○"


class WorkflowBar(QWidget):
    """Thanh ngang hiển thị tiến độ quy trình và nút đi tới bước tiếp theo.

    Signals:
        go_to_page: Phát ra ``objectName`` của trang cần chuyển tới.
    """

    go_to_page = Signal(str)

    def __init__(self, translator, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workflowBar")
        self._translator = translator
        self._stage = WorkflowStage.NEW
        self._build_ui()
        self.set_stage(WorkflowStage.NEW)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(10)

        self._steps_label = BodyLabel("")
        self._steps_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self._steps_label)

        layout.addStretch(1)

        self._next_label = CaptionLabel("")
        layout.addWidget(self._next_label)

        self._go_button = PushButton(self._translator.translate("workflow.go"))
        self._go_button.setToolTip(self._translator.translate("workflow.go_tip"))
        self._go_button.clicked.connect(self._emit_go)
        layout.addWidget(self._go_button)

    def set_stage(self, stage: WorkflowStage) -> None:
        """Cập nhật khâu hiện tại và làm mới hiển thị.

        Args:
            stage: Khâu đã hoàn thành xa nhất của dự án hiện hành.
        """
        self._stage = stage
        self._steps_label.setText(self._render_steps(stage))

        if stage.is_complete:
            self._next_label.setText(self._translator.translate("workflow.completed"))
            self._go_button.setVisible(False)
            return

        self._next_label.setText(
            self._translator.translate("workflow.next_prefix").replace("{action}", self._translator.translate("workflow." + _NEXT_ACTION_KEYS[stage]))
        )
        self._go_button.setVisible(stage.next_page_key is not None)

    def _render_steps(self, stage: WorkflowStage) -> str:
        """Vẽ chuỗi 5 khâu, đánh dấu những khâu đã xong.

        Args:
            stage: Khâu đã hoàn thành xa nhất.

        Returns:
            Chuỗi dạng ``"● Trích xuất → ● Biên tập → ○ Dịch → …"``.
        """
        parts = [
            f"{_DONE_MARK if stage >= step else _PENDING_MARK} {self._translator.translate('workflow.' + label_key)}"
            for step, label_key in _DISPLAY_STAGES
        ]
        return "  →  ".join(parts)

    def _emit_go(self) -> None:
        """Phát tín hiệu chuyển trang cho bước tiếp theo."""
        target = self._stage.next_page_key
        if target:
            self.go_to_page.emit(target)


__all__ = ["WorkflowBar"]
