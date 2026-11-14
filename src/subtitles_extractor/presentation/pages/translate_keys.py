"""[v3.23.694] Key API và hạn mức trên trang Dịch: gom key, che key, trạng thái hạn mức, sửa bảng hạn mức.

Tách nguyên văn khỏi ``src/subtitles_extractor/presentation/pages/translate_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TranslatePage``.
"""

from __future__ import annotations

import logging

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from subtitles_extractor.presentation.fluent_compat import PushButton
from subtitles_extractor.presentation.pages.translate_constants import (
    DEFAULT_ANALYSIS_MODEL,
)
from subtitles_extractor.presentation.pages.translate_widgets import parse_quota_rows
from subtitles_extractor.presentation.theme.styles import caption_style

logger = logging.getLogger("subtitles_extractor.presentation.pages.translate_page")



class TranslateKeysMixin:
    """Key API và hạn mức trên trang Dịch: gom key, che key, trạng thái hạn mức, sửa bảng hạn mức."""

    def _all_keys(self) -> list[str]:
        """Key hiệu lực: key chính + key dự phòng, bỏ trùng/rỗng."""
        return self._ai_store.gemini_keys()

    def _combined_keys_text(self) -> str:
        """Chuỗi tất cả key (nối xuống dòng) để truyền xuống adapter (adapter tự tách).

        [v3.23.666] Đang dùng nhà cung cấp tuỳ chỉnh: key của nhà cung cấp (bộ dịch dùng key này).
        """
        if self._custom_mode():
            return self._ai_store.custom_key() if self._custom_provider_ready() else ""
        return "\n".join(self._all_keys())

    @staticmethod
    def _masked_key(key: str) -> str:
        """Che API key, chỉ chừa 4 ký tự cuối để người dùng nhận diện."""
        k = (key or "").strip()
        return f"…{k[-4:]}" if len(k) >= 4 else "…"

    def _keys_quota_status_text(self, keys: list[str] | None = None) -> str:
        """[v3.23.124] Mô tả tình trạng quota/ngày từng key cho model phân tích đang chọn.

        [v3.23.677] Khung "Khoá API và AI" ở Cài đặt gọi hàm này (nút "Xem hạn mức hôm nay").
        """
        keys = self._all_keys() if keys is None else keys
        if not keys:
            return "Chưa có API key nào."
        model = (
            self._analyze_model_combo.currentText().strip() or DEFAULT_ANALYSIS_MODEL
        )
        try:
            stats = self._view_model.quota_status_for_keys(keys, model)
        except (AttributeError, KeyError, RuntimeError):
            return ""
        parts: list[str] = []
        usable_left = 0
        shared_suspect = False
        for key, st in zip(keys, stats):
            masked = self._masked_key(key)
            if st.get("invalid"):
                parts.append(f"⛔ {masked}: Gemini từ chối key này ({st['invalid']})")
            elif st.get("server_blocked") or st["remaining"] <= 0:
                note = ""
                if st.get("server_blocked") and st["used"] < 0.6 * max(1, st["limit"]):
                    shared_suspect = True
                    note = " — server từ chối dù mới đếm ít: nghi dùng chung dự án"
                parts.append(f"🔴 {masked}: {st['used']}/{st['limit']} req, hết hạn mức{note}")
            else:
                usable_left += st["remaining"]
                parts.append(f"🟢 {masked}: {st['used']}/{st['limit']} req")
        # [v3.23.639] Tổng quan: còn bao nhiêu request trên mọi key và khi nào hạn mức hồi.
        from subtitles_extractor.infrastructure.translation.gemini_quota_clock import (
            describe_quota_reset,
        )

        lines = [f"Quota hôm nay ({model}):  " + "   ".join(parts)]
        lines.append(f"Còn tổng cộng ~{usable_left} request trên các key dùng được. {describe_quota_reset()}")
        if shared_suspect:
            lines.append(
                "Lưu ý: hạn mức ngày tính theo DỰ ÁN, không theo key — các key cùng một dự án dùng "
                "chung một hũ quota nên xoay key không thêm được request."
            )
        return "\n".join(lines)

    def _on_edit_quota_clicked(self) -> None:
        """[v3.23.122] Mở hộp thoại xem/sửa/thêm hạn mức quota (RPM·TPM·RPD) của model."""
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
        )
        cfg = self._view_model.get_quota_config()
        custom: dict = cfg.get("custom", {})
        defaults: dict = cfg.get("defaults", {})

        dlg = QDialog(self)
        dlg.setWindowTitle(self._translator.translate("translate.dlg_quota_title"))
        dlg.resize(640, 460)
        lay = QVBoxLayout(dlg)
        info = QLabel(
            "Đặt giới hạn cho TÊN MODEL cụ thể (vd 'gemini-3.6-flash'). Giá trị ở đây sẽ "
            "GHI ĐÈ mặc định.\n"
            "RPM = request/phút · TPM = token/phút · RPD = request/ngày."
        )
        info.setWordWrap(True)
        lay.addWidget(info)
        if defaults:
            ref = ", ".join(
                f"{k}: {v['rpm']}/{v['tpm']}/{v['rpd']}" for k, v in defaults.items()
            )
            ref_lbl = QLabel(self._translator.translate("translate.tr_prefix_default").replace("{ref}", str(ref)))
            ref_lbl.setWordWrap(True)
            ref_lbl.setStyleSheet(caption_style())
            lay.addWidget(ref_lbl)

        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(["Model", "RPM", "TPM", "RPD"])
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        def _add_row(
            name: str = "", rpm: int = 5, tpm: int = 250000, rpd: int = 20
        ) -> None:
            r = table.rowCount()
            table.insertRow(r)
            for col, val in enumerate([name, str(rpm), str(tpm), str(rpd)]):
                table.setItem(r, col, QTableWidgetItem(val))

        for name, lim in custom.items():
            _add_row(name, lim["rpm"], lim["tpm"], lim["rpd"])
        if table.rowCount() == 0:
            _add_row("gemini-3.6-flash")
        lay.addWidget(table, 1)

        btn_row = QHBoxLayout()
        btn_add = PushButton(self._translator.translate("translate.btn_add_model"))
        btn_add.clicked.connect(lambda: _add_row())
        btn_del = PushButton(self._translator.translate("translate.btn_remove_row"))
        btn_del.clicked.connect(
            lambda: (
                table.removeRow(table.currentRow())
                if table.currentRow() >= 0 else None
            )
        )
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_del)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        # [v3.23.609] Nhập sai thì báo rồi MỞ LẠI hộp thoại với đúng các dòng đã nhập.
        # Trước đây hộp thoại đã đóng → mất hết các dòng vừa sửa vì một ô gõ nhầm.
        while True:
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            rows: list[tuple[str, str, str, str]] = []
            for r in range(table.rowCount()):
                cells = [table.item(r, c) for c in range(4)]
                rows.append(tuple((cell.text() if cell else "") for cell in cells))  # type: ignore[arg-type]
            limits, errors = parse_quota_rows(rows)
            if not errors:
                break
            self._show_warning(
                self._translator.translate("translate.warn_invalid_title"),
                "\n".join(errors[:6]),
            )
        self._view_model.save_quota_config(limits)
        self._show_success(
            "Đã lưu hạn mức quota",
            f"Đã áp {len(limits)} model. Áp dụng ngay cho các lần gọi tiếp theo.",
        )


__all__ = ["TranslateKeysMixin"]
