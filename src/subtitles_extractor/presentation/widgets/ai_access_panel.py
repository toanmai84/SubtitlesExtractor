"""[v3.23.677] Khung "Khoá API và AI" ở trang Cài đặt + dòng tóm tắt khoá cho trang Dịch / Lồng tiếng.

Khung sửa thẳng :class:`AiAccessStore` (lưu ngay, không chờ nút "Lưu" của Cài đặt — key là dữ liệu bí mật, không thuộc
tệp cấu hình ``settings.json``). Trang Dịch và Lồng tiếng chỉ hiện :class:`AiAccessSummary`: một dòng "3 key Gemini"
và nút mở Cài đặt.
"""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import urlparse

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from subtitles_extractor.presentation.ai_access_store import (
    PROVIDER_CUSTOM,
    PROVIDER_GEMINI,
    AiAccessStore,
    shared_ai_access_store,
    split_keys,
)
from subtitles_extractor.presentation.theme.styles import caption_style
from subtitles_extractor.presentation.utils.accessibility import set_accessible_name


def masked_key(key: str) -> str:
    """Che API key, chỉ chừa 4 ký tự cuối để người dùng nhận ra."""
    k = (key or "").strip()
    return f"…{k[-4:]}" if len(k) >= 4 else "…"


def summary_text(store: AiAccessStore, tr: Callable[..., str], *, gemini_only: bool = False) -> tuple[str, bool]:
    """Dòng tóm tắt khoá: ``(chữ, đủ dùng)``. ``gemini_only``: trang chỉ dùng được Gemini (Gemini TTS)."""
    keys = store.gemini_keys()
    if store.is_custom() and not gemini_only:
        if not store.custom_complete():
            return tr("ai.summary_custom_incomplete"), False
        host = urlparse(store.custom_url()).netloc or store.custom_url()
        return tr("ai.summary_custom", host=host), True
    if not keys:
        return tr("ai.summary_none"), False
    return tr("ai.summary_gemini", n=len(keys), last=masked_key(keys[0])), True


class AiAccessSummary(QWidget):
    """Một dòng "Khoá API: 🟢 3 key Gemini (…abcd)  [Sửa trong Cài đặt]" — tự cập nhật theo kho."""

    def __init__(
        self,
        translator: object,
        store: AiAccessStore | None = None,
        *,
        gemini_only: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._tr = translator.translate  # type: ignore[attr-defined]
        self._store = store or shared_ai_access_store()
        self._gemini_only = gemini_only
        self._open_settings: Callable[[], None] | None = None
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.button = QPushButton(self._tr("ai.btn_edit_keys"))
        self.button.clicked.connect(self._on_open)
        row.addWidget(self.label, 1)
        row.addWidget(self.button)
        self._store.changed.connect(self.refresh)
        self.refresh()

    def set_open_settings_callback(self, callback: Callable[[], None] | None) -> None:
        self._open_settings = callback

    def is_ready(self) -> bool:
        return summary_text(self._store, self._tr, gemini_only=self._gemini_only)[1]

    def refresh(self) -> None:
        text, ok = summary_text(self._store, self._tr, gemini_only=self._gemini_only)
        self.label.setText(("🟢 " if ok else "⚠️ ") + text)

    def _on_open(self) -> None:
        if self._open_settings is not None:
            self._open_settings()


class AiAccessPanel(QWidget):
    """Sửa nhà cung cấp AI, key Gemini (nhiều key) và nhà cung cấp tuỳ chỉnh."""

    def __init__(self, translator: object, store: AiAccessStore | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tr = translator.translate  # type: ignore[attr-defined]
        self._store = store or shared_ai_access_store()
        self._quota_provider: Callable[[list[str]], str] | None = None
        self._loading = False
        self._shown_main_key = ""   # [v3.23.700] key mà ô key chính đang hiển thị
        tr = self._tr

        root = QVBoxLayout(self)
        info = QLabel(tr("ai.panel_info"))
        info.setWordWrap(True)
        info.setStyleSheet(caption_style())
        root.addWidget(info)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.provider_combo = QComboBox()
        self.provider_combo.addItem(tr("translate.provider_gemini"), PROVIDER_GEMINI)
        self.provider_combo.addItem(tr("translate.provider_custom"), PROVIDER_CUSTOM)
        self.provider_combo.setToolTip(tr("translate.provider_tip"))
        set_accessible_name(self.provider_combo, tr("translate.provider_label"))
        form.addRow(tr("translate.provider_label"), self.provider_combo)

        # Key Gemini: key chính + nút quản lý nhiều key (dùng cả khi chọn nhà cung cấp tuỳ chỉnh: Gemini TTS).
        self.gemini_key_edit = QLineEdit()
        self.gemini_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key_edit.setPlaceholderText(tr("translate.api_key_ph"))
        set_accessible_name(self.gemini_key_edit, tr("ai.gemini_keys_label"))
        self.gemini_eye = QPushButton("👁")
        self.gemini_eye.setCheckable(True)
        self.gemini_eye.setFixedWidth(40)
        self.gemini_eye.setToolTip(tr("translate.api_key_toggle_tip"))
        set_accessible_name(self.gemini_eye, tr("translate.tr_acc_eye"))
        self.multi_key_button = QPushButton()
        self.multi_key_button.setToolTip(tr("ai.multi_key_tip"))
        set_accessible_name(self.multi_key_button, tr("translate.tr_acc_multikey"))
        key_row = QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        key_row.addWidget(self.gemini_key_edit, 1)
        key_row.addWidget(self.gemini_eye)
        key_row.addWidget(self.multi_key_button)
        holder = QWidget()
        holder.setLayout(key_row)
        form.addRow(tr("ai.gemini_keys_label"), holder)

        self.custom_url_edit = QLineEdit()
        self.custom_url_edit.setToolTip(tr("translate.custom_url_tip"))
        set_accessible_name(self.custom_url_edit, tr("translate.custom_url_label"))
        try:
            from subtitles_extractor.infrastructure.translation.openai_compat_client import EXAMPLE_BASE_URL

            self.custom_url_edit.setPlaceholderText(EXAMPLE_BASE_URL)
        except ImportError:
            pass
        self.custom_key_edit = QLineEdit()
        self.custom_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.custom_key_edit.setPlaceholderText(tr("translate.custom_key_ph"))
        set_accessible_name(self.custom_key_edit, tr("translate.custom_key_label"))
        self.custom_eye = QPushButton("👁")
        self.custom_eye.setCheckable(True)
        self.custom_eye.setFixedWidth(40)
        self.custom_eye.setToolTip(tr("translate.api_key_toggle_tip"))
        set_accessible_name(self.custom_eye, tr("translate.tr_acc_eye"))
        custom_row = QHBoxLayout()
        custom_row.setContentsMargins(0, 0, 0, 0)
        custom_row.addWidget(self.custom_key_edit, 1)
        custom_row.addWidget(self.custom_eye)
        self.custom_key_row = QWidget()
        self.custom_key_row.setLayout(custom_row)
        self.custom_note = QLabel(tr("translate.custom_note"))
        self.custom_note.setWordWrap(True)
        self.custom_note.setStyleSheet(caption_style())
        form.addRow(tr("translate.custom_url_label"), self.custom_url_edit)
        form.addRow(tr("translate.custom_key_label"), self.custom_key_row)
        form.addRow(self.custom_note)
        self.form = form
        root.addLayout(form)

        self.quota_label = QLabel()
        self.quota_label.setWordWrap(True)
        self.quota_label.setStyleSheet(caption_style())
        self.quota_button = QPushButton(tr("ai.btn_quota"))
        self.quota_button.clicked.connect(self.refresh_quota)
        quota_row = QHBoxLayout()
        quota_row.addWidget(self.quota_button)
        quota_row.addStretch(1)
        root.addLayout(quota_row)
        root.addWidget(self.quota_label)
        root.addStretch(1)

        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.gemini_key_edit.editingFinished.connect(self._on_main_key_edited)
        self.gemini_eye.toggled.connect(lambda on: self.gemini_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        self.custom_eye.toggled.connect(lambda on: self.custom_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        self.custom_url_edit.editingFinished.connect(self._on_custom_edited)
        self.custom_key_edit.editingFinished.connect(self._on_custom_edited)
        self.multi_key_button.clicked.connect(self.open_multi_key_dialog)
        self._store.changed.connect(self.load)
        self.load()

    # ── Nạp / ghi ──
    def set_quota_status_provider(self, provider: Callable[[list[str]], str] | None) -> None:
        """Hàm trả mô tả hạn mức từng key (trang Dịch cung cấp — cần bộ đếm quota + model phân tích)."""
        self._quota_provider = provider

    def load(self) -> None:
        """Đổ dữ liệu kho lên các ô (bỏ qua ô đang gõ dở)."""
        self._loading = True
        try:
            index = self.provider_combo.findData(self._store.provider())
            self.provider_combo.setCurrentIndex(max(0, index))
            keys = self._store.gemini_keys()
            if not self.gemini_key_edit.hasFocus():
                self.gemini_key_edit.setText(keys[0] if keys else "")
                self._shown_main_key = keys[0] if keys else ""
            if not self.custom_url_edit.hasFocus():
                self.custom_url_edit.setText(self._store.custom_url())
            if not self.custom_key_edit.hasFocus():
                self.custom_key_edit.setText(self._store.custom_key())
            self.multi_key_button.setText(self._tr("ai.btn_multi_key", n=len(keys)))
            custom = self._store.is_custom()
            self.form.setRowVisible(self.custom_url_edit, custom)
            self.form.setRowVisible(self.custom_key_row, custom)
            self.form.setRowVisible(self.custom_note, custom)
        finally:
            self._loading = False

    def _on_provider_changed(self, _index: int) -> None:
        if not self._loading:
            self._store.set_provider(str(self.provider_combo.currentData() or PROVIDER_GEMINI))

    def _on_main_key_edited(self) -> None:
        if self._loading:
            return
        keys = self._store.gemini_keys()
        main = split_keys(self.gemini_key_edit.text())   # [v3.23.698] dán "key1, key2" vào ô chính → hai key
        # [v3.23.700] Thay đúng key mà ô ĐANG HIỂN THỊ, không phải "key đầu danh sách hiện tại". Ô đang có tiêu điểm thì
        # load() không ghi đè ô: xoá trống ô + Enter (key 2 lên làm key chính) rồi bấm ra ngoài → editingFinished lần hai
        # với ô vẫn trống từng xoá LUÔN key 2.
        shown = self._shown_main_key
        if main == ([shown] if shown else []):
            return
        rest = [k for k in keys if k != shown] if shown else keys
        self._shown_main_key = main[0] if main else ""
        self._store.set_gemini_keys([*main, *rest])

    def _on_custom_edited(self) -> None:
        if not self._loading:
            self._store.set_custom(self.custom_url_edit.text(), self.custom_key_edit.text())

    def refresh_quota(self) -> None:
        keys = self._store.gemini_keys()
        if not keys:
            self.quota_label.setText(self._tr("ai.summary_none"))
            return
        text = ""
        if self._quota_provider is not None:
            try:
                text = self._quota_provider(keys)
            except (AttributeError, KeyError, RuntimeError, ValueError, TypeError):
                text = ""
        self.quota_label.setText(text or self._tr("ai.quota_unknown"))

    def open_multi_key_dialog(self) -> None:
        """Hộp thoại nhập NHIỀU key Gemini (mỗi dòng một key, dòng đầu là key chính)."""
        dlg = QDialog(self)
        dlg.setWindowTitle(self._tr("translate.dlg_apikey_title"))
        dlg.resize(520, 320)
        lay = QVBoxLayout(dlg)
        info = QLabel(self._tr("ai.multi_key_info"))
        info.setWordWrap(True)
        lay.addWidget(info)
        editor = QPlainTextEdit()
        editor.setPlaceholderText("AIza...key1\nAIza...key2\nAIza...key3")
        editor.setPlainText("\n".join(self._store.gemini_keys()))
        lay.addWidget(editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._store.set_gemini_keys(split_keys(editor.toPlainText()))


__all__ = ["AiAccessPanel", "AiAccessSummary", "masked_key", "summary_text"]
