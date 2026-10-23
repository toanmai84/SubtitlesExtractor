"""[v3.23.625] Hộp thoại "Thêm dòng" cho bảng dòng lồng tiếng: mốc, lời thoại, người nói."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

_MAX_SECONDS = 24 * 3600.0


@dataclass(frozen=True, slots=True)
class NewLineRequest:
    """Dòng người dùng muốn thêm.

    Attributes:
        start_sec: Mốc bắt đầu.
        end_sec: Mốc kết thúc.
        text: Lời thoại.
        speaker: Tên vai; ``""`` = giọng chung; ``None`` = để ứng dụng tự tìm.
    """

    start_sec: float
    end_sec: float
    text: str
    speaker: str | None


class AddLineDialog(QDialog):
    """Hỏi mốc thời gian, lời thoại và người nói của dòng mới."""

    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        labels: dict[str, str],
        start_sec: float,
        end_sec: float,
        speakers: list[tuple[str, str | None]],
        *,
        text: str | None = "",
    ) -> None:
        """Dựng hộp thoại.

        Args:
            parent: Cửa sổ cha.
            title: Tiêu đề.
            labels: Nhãn ``start`` / ``end`` / ``text`` / ``speaker``.
            start_sec: Mốc bắt đầu gợi ý.
            end_sec: Mốc kết thúc gợi ý.
            speakers: ``[(chữ hiện, giá trị)]`` cho ô người nói (giá trị ``None`` = tự động);
                rỗng = không hiện ô người nói.
            text: Lời thoại ban đầu; ``None`` = không hiện ô lời (hộp thoại chỉ đổi mốc).
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.start_box = self._spin(start_sec)
        self.end_box = self._spin(end_sec)
        self._show_text = text is not None
        self.text_edit = QPlainTextEdit(text or "")
        self.text_edit.setMinimumHeight(70)
        self.speaker_box = QComboBox()
        for text, value in speakers:
            self.speaker_box.addItem(text, value)
        form = QFormLayout()
        form.addRow(labels.get("start", "start"), self.start_box)
        form.addRow(labels.get("end", "end"), self.end_box)
        if self._show_text:
            form.addRow(labels.get("text", "text"), self.text_edit)
        if speakers:
            form.addRow(labels.get("speaker", "speaker"), self.speaker_box)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)
        self.text_edit.textChanged.connect(self._validate)
        self.start_box.valueChanged.connect(self._validate)
        self.end_box.valueChanged.connect(self._validate)
        self._validate()

    @staticmethod
    def _spin(value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(3)
        box.setSingleStep(0.1)
        box.setRange(0.0, _MAX_SECONDS)
        box.setSuffix(" s")
        box.setValue(value)
        return box

    def _validate(self) -> None:
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(
            (bool(self.text_edit.toPlainText().strip()) or not self._show_text)
            and self.end_box.value() > self.start_box.value()
        )

    def request(self) -> NewLineRequest:
        """Dòng mới theo giá trị đang nhập."""
        return NewLineRequest(
            start_sec=self.start_box.value(),
            end_sec=self.end_box.value(),
            text=self.text_edit.toPlainText().strip(),
            speaker=self.speaker_box.currentData(),
        )


class SplitLineDialog(QDialog):
    """Hỏi cách tách một dòng thành hai: lời từng nửa và mốc cắt."""

    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        labels: dict[str, str],
        text_a: str,
        text_b: str,
        start_sec: float,
        end_sec: float,
        split_sec: float,
    ) -> None:
        """Dựng hộp thoại (``labels``: ``first`` / ``second`` / ``split``)."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.first_edit = QPlainTextEdit(text_a)
        self.second_edit = QPlainTextEdit(text_b)
        for edit in (self.first_edit, self.second_edit):
            edit.setMinimumHeight(60)
        self.split_box = AddLineDialog._spin(split_sec)
        self.split_box.setRange(start_sec, end_sec)
        self.split_box.setValue(split_sec)
        self._start, self._end = start_sec, end_sec
        form = QFormLayout()
        form.addRow(labels.get("first", "first"), self.first_edit)
        form.addRow(labels.get("second", "second"), self.second_edit)
        form.addRow(labels.get("split", "split"), self.split_box)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.buttons)
        for signal in (
            self.first_edit.textChanged, self.second_edit.textChanged,
            self.split_box.valueChanged,
        ):
            signal.connect(self._validate)
        self._validate()

    def _validate(self) -> None:
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(
            bool(self.first_edit.toPlainText().strip())
            and bool(self.second_edit.toPlainText().strip())
            and self._start < self.split_box.value() < self._end
        )

    def result_values(self) -> tuple[str, str, float]:
        """``(lời nửa đầu, lời nửa sau, mốc cắt)`` theo giá trị đang nhập."""
        return (
            self.first_edit.toPlainText().strip(),
            self.second_edit.toPlainText().strip(),
            self.split_box.value(),
        )


__all__ = ["AddLineDialog", "NewLineRequest", "SplitLineDialog"]
