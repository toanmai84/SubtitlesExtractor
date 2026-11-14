"""[v3.23.693] Xuất phụ đề và dựng bản xem trước ASS cho trình phát của trang Chỉnh sửa.

Tách nguyên văn khỏi ``editor_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``EditorPage``.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat
from subtitles_extractor.presentation.fluent_compat import InfoBar, InfoBarPosition
from subtitles_extractor.presentation.theme import colors as _c
from subtitles_extractor.presentation.utils.ass_preview_builder import (
    AssPreviewStyle,
    build_ass_header,
    render_dialogue_line,
)


class EditorExportMixin:
    """Xuất phụ đề và dựng bản xem trước ASS cho trình phát của trang Chỉnh sửa."""

    def _on_export_clicked(self) -> None:
        # [v3.23.374] Log ở đầu để phân biệt "nút bị disable" (không có log) với "handler
        # chạy nhưng dialog không mở" (có log). Bọc try/except vì PySide6 NUỐT exception
        # trong slot → trước đây lỗi ở đây làm nút "không thấy chạy gì" mà không báo.
        logger.debug(
            "Editor: bấm Xuất tệp (events={}, is_busy={}).",
            len(self._table_model.events),
            self._view_model._is_busy,
        )
        try:
            self._do_export_clicked()
        except Exception as exc:  # noqa: BLE001 — chặn Qt nuốt lỗi im lặng; báo cho người dùng
            logger.exception("Lỗi không mong đợi khi mở hộp thoại xuất tệp.")
            self._export_button.setText("💾 " + self._translator.translate("editor.btn_export"))
            self._export_button.setEnabled(bool(self._table_model.events))
            InfoBar.error(
                title=self._translator.translate("editor.error_title"),
                content=(
                    self._translator.translate("editor.export_unexpected_error")
                    + f"\n[{type(exc).__name__}] {exc}"
                ),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=8000,
            )

    def _do_export_clicked(self) -> None:
        default_dir = ""
        if self._current_video_path:
            # Quy ước an toàn: phụ đề gốc từ Biên tập đặt là <tên>.original.srt
            # để không ghi đè bản dịch (.translate.) hay bản TTS (.tts.).
            from subtitles_extractor.domain.value_objects.output_naming import (
                extracted_subtitle_path,
            )

            default_dir = str(extracted_subtitle_path(self._current_video_path))
        self._prepare_for_file_dialog()
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            self._translator.translate("editor.dialog_export"),
            default_dir,
            self._translator.translate("editor.dialog_filter_export"),
            options=self._FILE_DIALOG_OPTIONS,
        )
        if not path_str:
            return

        # [v3.6 bugfix EXPORT-FIX-3]: Kiểm tra _is_busy SAU KHI dialog đóng.
        # QFileDialog.getSaveFileName() là blocking và chạy event loop bên trong.
        # Trong khoảng thời gian dialog mở (~vài giây), các sự kiện khác có thể:
        #   • Re-OCR hoàn tất → set _is_busy=False, sau đó bắt đầu lại → True
        #   • Watchdog hết giờ → force-reset nhưng lại có task khác bắt đầu
        # Nếu _is_busy=True khi export_to_file() được gọi, nó sẽ trả về False
        # NHƯNG nút đã bị disable + text "⏳" → UI kẹt. Guard này ngăn điều đó.
        if self._view_model._is_busy:
            InfoBar.warning(
                title=self._translator.translate("editor.dlg_busy_t"),
                content=self._translator.translate("editor.dlg_busy_b"),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
            )
            return

        output_path = Path(path_str)
        output_format = (
            SubtitleFormat.ASS
            if output_path.suffix.lower() == ".ass"
            else SubtitleFormat.SRT
        )

        # Cập nhật nút trước khi gọi export
        self._export_button.setText(self._translator.translate("editor.ed_btn_saving"))
        self._export_button.setEnabled(False)

        started = self._view_model.export_to_file(output_path, output_format)
        if not started:
            # export_to_file bị từ chối (race condition rất hiếm sau guard ở trên)
            # → phục hồi nút để user thử lại được.
            self._export_button.setText("💾 " + self._translator.translate("editor.btn_export"))
            self._export_button.setEnabled(bool(self._table_model.events))
            InfoBar.warning(
                title=self._translator.translate("editor.dlg_cantexport_t"),
                content=self._translator.translate("editor.dlg_cantexport_b"),
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
            )

    def _on_export_finished(self, output_path: Path) -> None:
        self._status_label.setStyleSheet(f"color: {_c.success()};")
        self._status_label.setText(self._translator.translate("editor.status_exported", path=str(output_path)))
        # [V3.47 PERF FIX] Khôi phục Nút
        self._export_button.setText("💾 " + self._translator.translate("editor.btn_export"))
        self._export_button.setEnabled(True)

    def _schedule_preview(self) -> None: self._preview_timer.start()

    def _generate_ass_preview(self) -> None:
        player = self._video_widget.peek_player()
        if player is None or not player.is_loaded: return
        events = self._view_model.current_events
        if not events: return

        play_res_x, play_res_y = 1920, 1080

        try:
            ass_content = self._build_ass_content(events, play_res_x, play_res_y)

            self._preview_idx ^= 1
            path = self._preview_files[self._preview_idx]

            with open(path, "w", encoding="utf-8") as fp:
                fp.write(ass_content)
                fp.flush()

            resolved_path = str(Path(path).resolve()).replace("\\", "/")

            player.send_command("sub-add", resolved_path, "select")
            QTimer.singleShot(100, lambda: self._cleanup_old_preview_tracks(resolved_path))

        except (OSError, AttributeError) as exc:
            logger.debug("generate_ass_preview thất bại: {}.", exc)

    def _cleanup_old_preview_tracks(self, current_path: str) -> None:
        player = self._video_widget.peek_player()
        if player is None:
            return
        raw_mpv = getattr(player, "_mpv", None)
        if raw_mpv is None:
            return
        try:
            tracks = raw_mpv.track_list or []
        except (AttributeError, RuntimeError):
            return
        for track in tracks:
            if track.get("type") != "sub":
                continue
            ext_path = str(track.get("external-filename", "") or "")
            if (
                "se_preview_" not in ext_path
                or ext_path.replace("\\", "/") == current_path
            ):
                continue
            try:
                player.send_command("sub-remove", str(track["id"]))
            except (RuntimeError, KeyError) as exc:
                logger.debug("Không gỡ được sub track cũ: {}.", exc)

    def _build_ass_content(self, events: list[SubtitleEvent], play_res_x: int = 1920, play_res_y: int = 1080) -> str:
        settings_repo = self._container.settings_service._repo
        style = AssPreviewStyle(
            font_name=str(settings_repo.load("editor_preview_font", "Arial")),
            font_size=int(settings_repo.load("editor_preview_size", 48)),
            color_primary=str(settings_repo.load("editor_preview_color_primary", "&H00FFFFFF")),
            color_outline=str(settings_repo.load("editor_preview_color_outline", "&H00000000")),
            color_background=str(settings_repo.load("editor_preview_color_bg", "&H99000000")),
            margin_vertical=int(settings_repo.load("editor_preview_margin_v", 25)),
            play_res_x=play_res_x,
            play_res_y=play_res_y,
        )
        header = build_ass_header(style)

        active_idx = self._current_row()
        is_editing = self._edit_text.hasFocus()
        active_text = self._edit_text.toPlainText()

        lines = [
            render_dialogue_line(
                active_text if (is_editing and i == active_idx) else ev.text,
                ev.start_sec, ev.end_sec,
            )
            for i, ev in enumerate(events)
        ]
        return header + "\n".join(lines) + "\n"


__all__ = ["EditorExportMixin"]
