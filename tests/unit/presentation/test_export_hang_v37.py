"""Tests bảo vệ v3.7 fix treo khi Xuất tệp (EXPORT-HANG).

Nguyên nhân gốc: QFileDialog mặc định dùng shell dialog Windows (COM/DWM).
Top-level window chứa widget MPV vo=gpu-next (native child window giữ swapchain
GPU). Sau nhiều lần Re-OCR/seek tạo–hủy GPU context, shell dialog khi khởi tạo
(tạo surface GPU + tương tác DWM) deadlock với context MPV → treo TRƯỚC khi cửa
sổ kịp hiện ("chưa kịp hiện cửa sổ đặt tên").

Fix: ép dùng dialog Qt thuần (DontUseNativeDialog) cho MỌI file dialog ở các
trang host MPV, kèm tạm dừng video trước khi mở (giảm tranh chấp GPU).

Kiểm thử bằng soi mã nguồn (theo mẫu test_dialog_cleanup_v36) để không phải
dựng EditorPage nặng (cần MPV/GPU không có trong môi trường CI).
"""

from pathlib import Path

_PAGES_DIR = Path(__file__).resolve().parents[3] / "src/subtitles_extractor/presentation/pages"
_EDITOR = _PAGES_DIR / "editor_page.py"
_EXTRACT = _PAGES_DIR / "extract_page.py"
_DEBUG = _PAGES_DIR / "debug_page.py"
_TRANSLATE = _PAGES_DIR / "translate_page.py"


def _snip(source: str, method: str) -> str:
    start = source.find(f"def {method}")
    assert start != -1, f"Không tìm thấy method {method}"
    end = source.find("\n    def ", start + 1)
    return source[start:end if end != -1 else len(source)]


# ── Editor: dialog xuất tệp (chỗ treo chính) ─────────────────────────────────

class TestEditorExportDialogNonNative:
    def test_export_uses_non_native_option(self) -> None:
        snippet = _snip(_EDITOR.read_text(encoding="utf-8"), "_on_export_clicked")
        assert "getSaveFileName" in snippet
        assert "options=self._FILE_DIALOG_OPTIONS" in snippet, (
            "EXPORT-HANG: _on_export_clicked PHẢI truyền options=self._FILE_DIALOG_OPTIONS "
            "(DontUseNativeDialog) để tránh deadlock shell dialog với GPU/MPV."
        )

    def test_export_pauses_video_first(self) -> None:
        snippet = _snip(_EDITOR.read_text(encoding="utf-8"), "_on_export_clicked")
        dialog_pos = snippet.find("getSaveFileName")
        prepare_pos = snippet.find("_prepare_for_file_dialog")
        assert prepare_pos != -1, "Phải gọi _prepare_for_file_dialog() trước khi mở dialog."
        assert prepare_pos < dialog_pos, "Phải tạm dừng video TRƯỚC khi mở file dialog."

    def test_file_dialog_options_constant_is_non_native(self) -> None:
        src = _EDITOR.read_text(encoding="utf-8")
        assert "_FILE_DIALOG_OPTIONS = QFileDialog.Option.DontUseNativeDialog" in src, (
            "Phải định nghĩa hằng _FILE_DIALOG_OPTIONS = DontUseNativeDialog."
        )

    def test_prepare_helper_pauses_when_playing(self) -> None:
        snippet = _snip(_EDITOR.read_text(encoding="utf-8"), "_prepare_for_file_dialog")
        assert "is_playing" in snippet and "pause()" in snippet, (
            "_prepare_for_file_dialog phải tạm dừng video khi đang phát."
        )


# ── Editor: dialog mở video / mở phụ đề ──────────────────────────────────────

class TestEditorOpenDialogsNonNative:
    def test_open_video_uses_non_native(self) -> None:
        snippet = _snip(_EDITOR.read_text(encoding="utf-8"), "_on_open_video_clicked")
        assert "getOpenFileName" in snippet
        assert "options=self._FILE_DIALOG_OPTIONS" in snippet

    def test_open_subtitle_uses_non_native(self) -> None:
        snippet = _snip(_EDITOR.read_text(encoding="utf-8"), "_on_open_clicked")
        assert "getOpenFileName" in snippet
        assert "options=self._FILE_DIALOG_OPTIONS" in snippet


# ── Các trang khác host MPV / có dialog ──────────────────────────────────────

class TestOtherPagesNonNative:
    def test_extract_page_all_dialogs_non_native(self) -> None:
        src = _EXTRACT.read_text(encoding="utf-8")
        # Mỗi lời gọi dialog phải kèm DontUseNativeDialog.
        save_count = src.count("getSaveFileName")
        open_count = src.count("getOpenFileName")
        non_native_count = src.count("DontUseNativeDialog")
        assert non_native_count >= save_count + open_count, (
            "Mọi QFileDialog ở extract_page phải dùng DontUseNativeDialog."
        )

    def test_debug_page_open_non_native(self) -> None:
        snippet = _snip(_DEBUG.read_text(encoding="utf-8"), "_on_open_file")
        assert "DontUseNativeDialog" in snippet

    def test_translate_page_dialogs_non_native(self) -> None:
        src = _TRANSLATE.read_text(encoding="utf-8")
        save_count = src.count("getSaveFileName")
        open_count = src.count("getOpenFileName")
        non_native_count = src.count("DontUseNativeDialog")
        assert non_native_count >= save_count + open_count, (
            "Mọi QFileDialog ở translate_page phải dùng DontUseNativeDialog."
        )
