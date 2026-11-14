"""[v3.23.693] Re-OCR trên trang Chỉnh sửa: mở hộp thoại nâng cao, Re-OCR nhanh, khoá khi đang trích xuất, tiến độ/huỷ.

Tách nguyên văn khỏi ``editor_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``EditorPage``.
"""

from __future__ import annotations

import contextlib

from PySide6.QtWidgets import QDialog

from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.presentation.fluent_compat import InfoBar
from subtitles_extractor.presentation.pages.editor_dialogs import AdvancedReOcrDialog


class EditorReOcrMixin:
    """Re-OCR trên trang Chỉnh sửa: mở hộp thoại nâng cao, Re-OCR nhanh, khoá khi đang trích xuất, tiến độ/huỷ."""

    def _on_video_roi_changed(self, roi: Roi) -> None: pass

    def _on_reocr_progress(self, current: int, total: int, msg: str) -> None:
        if total > 0:
            self._progress_bar.setRange(0, total); self._progress_bar.setValue(current)
            self._progress_bar.setVisible(True)
        else:
            # [v3.6 bugfix R1]: Chỉ ẩn thanh progress, KHÔNG xóa status_label.
            # Trước đây setVisible(False) + setText("") cùng một lúc → thông báo
            # "Re-OCR hoàn tất trong X.Xs" bị xóa ngay sau khi hiện ra, user
            # không kịp đọc. Giờ giữ lại text nếu msg là chuỗi rỗng.
            self._progress_bar.setVisible(False)
        if msg:  # Chỉ cập nhật status label khi có nội dung thực sự
            self._status_label.setText(msg)

    def set_extraction_running(self, running: bool) -> None:
        """[v3.23.656] Trang Trích xuất đang chạy → khoá Re-OCR (cùng engine OCR; chạy chồng làm hỏng lượt trích xuất)."""
        self._extraction_running = bool(running)
        self._refresh_reocr_buttons()

    def _refresh_reocr_buttons(self) -> None:
        busy = bool(getattr(self, "_reocr_is_busy", False))
        extracting = bool(getattr(self, "_extraction_running", False))
        buttons = (self._btn_reocr, self._btn_fast_reocr)
        if not hasattr(self, "_reocr_tips"):
            self._reocr_tips = {id(button): button.toolTip() for button in buttons}
        locked_tip = self._translator.translate("editor.ed_tt_reocr_locked")
        for button in buttons:
            button.setEnabled(not busy and not extracting)
            button.setToolTip(locked_tip if extracting else self._reocr_tips.get(id(button), ""))

    def _warn_extraction_running(self) -> bool:
        """[v3.23.656] ``True`` (đã báo) nếu đang trích xuất — người gọi dừng lại."""
        if not getattr(self, "_extraction_running", False):
            return False
        InfoBar.warning(
            self._translator.translate("editor.ib_reocr_locked_t"),
            self._translator.translate("editor.ib_reocr_locked_c"), parent=self, duration=4000,
        )
        return True

    def _on_reocr_busy(self, is_busy: bool) -> None:
        self._reocr_is_busy = bool(is_busy)
        self._refresh_reocr_buttons()
        # [v3.23.659] Bật lại "Lưu tệp" khi hết bận (từng bị bỏ ở v3.23.656): xuất lỗi / Re-OCR xong lúc còn bận làm nút
        # kẹt tắt tới khi người dùng sửa một dòng.
        self._export_button.setEnabled(not is_busy and bool(self._table_model.events))
        reocr = bool(is_busy) and bool(getattr(self._view_model, "_busy_is_reocr", False))
        if not is_busy:
            self._view_model._busy_is_reocr = False
        if reocr != getattr(self, "_reocr_running_emitted", False):
            self._reocr_running_emitted = reocr
            self.reocr_running_changed.emit(reocr)

    def _on_reocr_clicked(self) -> None:
        if self._warn_extraction_running():
            return
        if not self._current_video_path:
            InfoBar.warning(self._translator.translate("editor.ib_no_video_t"), self._translator.translate("editor.ib_no_video_open_c"), parent=self, duration=3000)
            return

        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            InfoBar.warning(self._translator.translate("editor.ib_no_sel_t"), self._translator.translate("editor.ib_no_sel_c"), parent=self, duration=3000)
            return

        rows = sorted({self._proxy_model.mapToSource(item).row() for item in indexes})
        events = self._view_model.current_events
        preview_start = min(events[r].start_sec for r in rows)
        preview_end = max(events[r].end_sec for r in rows)

        if self._video_widget.is_playing:
            self._video_widget.toggle_play_pause()

        dlg = AdvancedReOcrDialog(
            video_path=self._current_video_path, start_sec=preview_start, end_sec=preview_end,
            current_roi=getattr(self, "_current_roi", None), container=self._container, parent=self,
            remembered_config=getattr(self, "_reocr_remembered", None),
        )

        # [v3.6 bugfix DLG-1]: PHẢI hủy dialog sau khi dùng. Dialog tạo với
        # parent=self → Qt giữ sống đến khi parent bị hủy. Mỗi lần Re-OCR tạo
        # một AdvancedReOcrDialog mới (kèm MPV player + native window handle).
        # Không hủy → tích lũy 8+ dialog ẩn → khi QFileDialog mở (cùng parent)
        # tương tác với các MPV widget/GPU context tích lũy → TREO ứng dụng.
        # try/finally đảm bảo deleteLater() luôn chạy kể cả khi early-return.
        try:
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            new_start, new_end, tweaks, new_roi, model_override = dlg.get_values()
            # [v3.23.105] Ghi nhớ cấu hình vừa dùng để "Re-OCR Nhanh" lặp lại + Studio mở lại
            # hiển thị đúng lần trước (đáp ứng kỳ vọng "cài đặt đã nhớ trong Re-OCR Studio").
            self._reocr_remembered = {"model": model_override, "tweaks": dict(tweaks)}
        finally:
            dlg.deleteLater()

        overlapping_rows = set(rows)
        for i, ev in enumerate(events):
            if ev.start_sec < new_end and ev.end_sec > new_start:
                overlapping_rows.add(i)

        final_rows = sorted(overlapping_rows)

        # [v3.6 bugfix R2/R4]: Không gọi update_timing() trước Re-OCR nữa.
        # Trước đây expand timing của first/last event để OCR quét đúng vùng:
        #   - Bug R2: khi first_row==last_row, lần 2 update_timing dùng stale
        #     events ref → revert lần 1 → start expansion bị mất.
        #   - Bug R4: nếu Re-OCR không tìm thấy gì, timing đã bị thay đổi vẫn
        #     giữ nguyên trong undo stack → user phải undo thủ công.
        # Thay bằng explicit_time_ranges: truyền vùng quét trực tiếp cho
        # start_reocr(), không cần sửa timing trước.
        from subtitles_extractor.application.dtos.reocr_dto import TimeRange
        explicit_ranges = []
        with contextlib.suppress(ValueError):
            explicit_ranges.append(TimeRange(start_sec=new_start, end_sec=new_end))

        if not explicit_ranges:
            self._show_error(self._translator.translate("editor.err_bad_range"))
            return

        self._waveform_widget.set_reocr_region(new_start, new_end)
        self._view_model.start_reocr(
            video_path=self._current_video_path, rows_to_replace=final_rows, roi=new_roi,
            tweaks=tweaks, model_override=model_override,
            explicit_time_ranges=explicit_ranges,
        )

    def _on_fast_reocr_clicked(self) -> None:
        if self._warn_extraction_running():
            return
        if not self._current_video_path:
            InfoBar.warning(self._translator.translate("editor.ib_no_video_t"), self._translator.translate("editor.ib_no_video_c"), parent=self, duration=3000)
            return

        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            InfoBar.warning(self._translator.translate("editor.ib_no_sel_t"), self._translator.translate("editor.ib_no_sel_c"), parent=self, duration=3000)
            return

        rows_to_process = sorted({self._proxy_model.mapToSource(item).row() for item in indexes})
        current_events_list = self._view_model.current_events

        preview_start_sec = min(current_events_list[row_idx].start_sec for row_idx in rows_to_process)
        preview_end_sec = max(current_events_list[row_idx].end_sec for row_idx in rows_to_process)
        if preview_end_sec - preview_start_sec < 0.5: preview_end_sec = preview_start_sec + 0.5

        if self._video_widget.is_playing:
            self._video_widget.toggle_play_pause()

        self._waveform_widget.set_reocr_region(preview_start_sec, preview_end_sec)

        # [v3.23.105] "Re-OCR Nhanh" = lặp lại cấu hình Re-OCR đã nhớ gần nhất (model +
        # tham số). Nếu chưa từng mở Studio trong phiên này -> dùng cài đặt OCR mặc định của
        # ứng dụng (tweaks rỗng -> _build_reocr_request lấy theo settings; model None -> v6).
        remembered = getattr(self, "_reocr_remembered", None)
        if remembered:
            advanced_default_tweaks = dict(remembered.get("tweaks") or {})
            advanced_model_override = remembered.get("model")
        else:
            advanced_default_tweaks = {}
            advanced_model_override = None

        current_active_roi = None
        saved_video_state = self._container.video_state_repository.get(str(self._current_video_path.resolve()))
        if saved_video_state and saved_video_state.roi:
            current_active_roi = saved_video_state.roi

        overlapping_row_indices = set(rows_to_process)
        for index, event_obj in enumerate(current_events_list):
            if event_obj.start_sec < preview_end_sec and event_obj.end_sec > preview_start_sec:
                overlapping_row_indices.add(index)

        final_rows_to_replace = sorted(overlapping_row_indices)

        # [v3.6 bugfix R2/R4]: Xây explicit_time_ranges từ preview range thay vì
        # gọi update_timing() trước. Loại bỏ hoàn toàn:
        #   - Bug R2: stale events ref khi single-row có cả start lẫn end expand
        #   - Bug R4: orphaned timing entries khi Re-OCR không tìm thấy kết quả
        import contextlib as _ctx

        from subtitles_extractor.application.dtos.reocr_dto import TimeRange
        explicit_ranges = []
        with _ctx.suppress(ValueError):
            explicit_ranges.append(
                TimeRange(start_sec=preview_start_sec, end_sec=preview_end_sec)
            )

        if not explicit_ranges:
            self._waveform_widget.clear_reocr_region()
            InfoBar.warning(self._translator.translate("editor.ib_bad_range_t"), self._translator.translate("editor.ib_range_short_c"), parent=self, duration=3000)
            return

        self._view_model.start_reocr(
            video_path=self._current_video_path, rows_to_replace=final_rows_to_replace,
            roi=current_active_roi, tweaks=advanced_default_tweaks,
            model_override=advanced_model_override,
            explicit_time_ranges=explicit_ranges,
        )

    def cancel_reocr(self) -> None:
        if hasattr(self, "_view_model"): self._view_model.cancel_reocr()


__all__ = ["EditorReOcrMixin"]
