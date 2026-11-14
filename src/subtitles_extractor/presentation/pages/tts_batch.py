"""[v3.23.692] Lồng tiếng cả bộ / dây chuyền phim bộ: hàng đợi các tập, chạy từng tập, xử lý lỗi, báo cáo cuối.

Tách nguyên văn khỏi ``tts_page.py`` (U6). Lớp trộn chỉ chứa phương thức; widget, tín hiệu và trạng thái vẫn thuộc ``TTSPage``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from subtitles_extractor.presentation.utils.error_humanizer import (
    humanize_gemini_error as _humanize_gemini_error,
)
from subtitles_extractor.presentation.utils.task_notifier import notify_task_finished

logger = logging.getLogger("subtitles_extractor.presentation.pages.tts_page")


class TTSBatchMixin:
    """Lồng tiếng cả bộ / dây chuyền phim bộ: hàng đợi các tập, chạy từng tập, xử lý lỗi, báo cáo cuối."""

    def _on_batch_tts_clicked(self) -> None:
        """Quét thư mục phim bộ rồi tổng hợp giọng đọc lần lượt cho từng tập."""
        if self._batch_items or self._view_model.is_busy:
            # [v3.23.658] Nút bật lại một nhịp giữa hai tập: bấm lúc đó từng THAY hàng đợi đang chạy (dây chuyền treo).
            return
        from PySide6.QtWidgets import QFileDialog

        from subtitles_extractor.application.services.batch_tts_plan import (
            build_tts_plan,
            estimate_batch_minutes,
            find_episode_videos,
            summarise_tts_plan,
        )

        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa các tập", "")
        if not folder:
            return

        videos = find_episode_videos(Path(folder))
        if not videos:
            QMessageBox.information(
                self, self._translator.translate("tts.dlg_batch_none_title"),
                self._translator.translate("tts.dlg_batch_none_body"),
            )
            return

        plan = build_tts_plan(videos, skip_existing=True)
        runnable = [item for item in plan if item.will_run]
        summary = summarise_tts_plan(plan)
        if not runnable:
            QMessageBox.information(
                self, self._translator.translate("tts.dlg_batch_nothing_title"),
                self._translator.translate("tts.dlg_batch_nothing_body").replace("{summary}", summary),
            )
            return

        minutes = estimate_batch_minutes(runnable)
        answer = QMessageBox.question(
            self, self._translator.translate("tts.dlg_batch_confirm_title"),
            self._translator.translate("tts.dlg_batch_confirm_body")
            .replace("{n}", str(len(videos))).replace("{summary}", summary)
            .replace("{runnable}", str(len(runnable))).replace("{minutes}", f"{minutes:.0f}"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._batch_items = runnable
        self._batch_index = 0
        self._batch_failures = []
        self._batch_done = 0  # [v3.23.659] đếm tập xong thật (không suy từ chỉ số)
        self._batch_cancelled = False
        self._run_next_batch_tts()

    def start_series_batch(self, videos: list[Path]) -> str:
        """[v3.23.658] Dây chuyền phim bộ: tổng hợp giọng các tập với engine/giọng đang chọn, không hộp thoại.

        Returns:
            ``""`` nếu đã bắt đầu (kết quả về qua ``series_batch_finished``), ngược lại lý do không chạy được.
        """
        from subtitles_extractor.application.services.batch_tts_plan import build_tts_plan
        from subtitles_extractor.presentation.view_models.series_pipeline_controller import (
            StageRunResult,
        )

        if self._view_model.is_busy or self._batch_items:
            return self._translator.translate("series.err_page_busy")
        plan = build_tts_plan([Path(v) for v in videos], skip_existing=True)
        runnable = [item for item in plan if item.will_run]
        invalid = {
            item.video_path.name: item.note or "" for item in plan
            if not item.will_run and item.subtitle_path is None
        }
        if not runnable:
            QTimer.singleShot(0, lambda: self.series_batch_finished.emit(StageRunResult(failed=invalid)))
            return ""
        self._series_batch = True
        self._series_invalid = invalid
        self._batch_stop_message = ""
        self._batch_items = runnable
        self._batch_index = 0
        self._batch_failures = []
        self._batch_done = 0  # [v3.23.659] đếm tập xong thật (không suy từ chỉ số)
        self._batch_cancelled = False
        QTimer.singleShot(0, self._run_next_batch_tts)
        return ""

    def cancel_series_batch(self) -> None:
        """[v3.23.658] Dừng khâu TTS của dây chuyền (như bấm Huỷ)."""
        if getattr(self, "_series_batch", False):
            self._on_cancel_clicked()
            if not self._view_model.is_busy and self._batch_items:
                self._finish_batch_tts()  # đang chuyển tập: không có lượt nào để báo "đã huỷ"

    def _run_next_batch_tts(self) -> None:
        """Nạp phụ đề của tập kế tiếp; việc tổng hợp chạy khi nạp xong."""
        if not self._batch_items:
            return
        if self._batch_cancelled or self._batch_index >= len(self._batch_items):
            self._finish_batch_tts()
            return

        item = self._batch_items[self._batch_index]
        total = len(self._batch_items)
        self._stage_lbl.setText(
            f"Cả bộ {self._batch_index + 1}/{total}: {item.video_path.name}"
        )
        self._output.setText(str(item.output_path))
        # Nạp nguồn là thao tác bất đồng bộ -> đánh dấu để khi `source_changed` bắn ra
        # thì biết đây là tập của hàng đợi và tự bấm "Tổng hợp".
        self._batch_pending_generate = True
        self._clear_results()
        self._view_model.load_source_from_file(item.subtitle_path)

    def _on_source_changed_for_batch(self, count: int) -> None:
        """Nạp phụ đề xong trong lúc chạy hàng loạt -> tổng hợp ngay tập này."""
        if not self._batch_pending_generate:
            return
        self._batch_pending_generate = False
        if count <= 0:
            index = min(self._batch_index, len(self._batch_items) - 1)
            name = self._batch_items[index].video_path.name if self._batch_items else "?"
            logger.warning("Cả bộ: %s không nạp được phụ đề.", name)
            self._advance_batch_tts(failed=name)
            return
        QTimer.singleShot(0, self._generate_for_batch)

    def _generate_for_batch(self) -> None:
        """Bấm "Tổng hợp" cho tập hiện tại của hàng đợi; nếu không khởi động được (engine chưa cài, thiếu khoá, giọng
        mẫu thiếu…) thì dừng HẲN hàng đợi — trước đây ``_on_generate`` thoát sớm và hàng đợi đứng yên vĩnh viễn."""
        index_before = self._batch_index
        self._on_generate()
        # `error_occurred` phát đồng bộ (vd engine không khả dụng) đã tự tính lỗi và sang tập kế (`_batch_index` đổi):
        # khi đó KHÔNG xử lý lần hai (trước đây tập kế bị tính lỗi theo và cả hàng đợi bị huỷ).
        if self._batch_items and self._batch_index == index_before and not self._view_model.is_busy:
            index = min(self._batch_index, len(self._batch_items) - 1)
            name = self._batch_items[index].video_path.name
            logger.warning("Cả bộ: không khởi động được tổng hợp cho %s — dừng hàng đợi.", name)
            self._batch_cancelled = True
            # [v3.23.658] Lý do cho dây chuyền phim bộ (trước đây báo nhầm "Đã dừng theo yêu cầu").
            self._batch_stop_message = self._translator.translate("series.err_tts_start").replace("{name}", name)
            self._advance_batch_tts(failed=name)

    def _on_error_for_batch(self, message: str) -> None:
        """Nhận lỗi TTS: đang chạy hàng loạt thì ghi nhận rồi chạy tiếp."""
        if self._batch_items:
            index = min(self._batch_index, len(self._batch_items) - 1)
            name = self._batch_items[index].video_path.name
            logger.warning("Cả bộ: tập %s lỗi — %s", name, message)
            self._advance_batch_tts(failed=name)
            return
        self._show_warning(self._translator.translate("tts.toast_tts_err_t"), _humanize_gemini_error(message))
        self._stage_lbl.setText(self._translator.translate("tts.status_error"))

    def _advance_batch_tts(self, *, failed: str | None = None) -> bool:
        """Chuyển sang tập kế. Trả ``True`` nếu đang chạy hàng loạt."""
        if not self._batch_items:
            return False
        if failed:
            self._batch_failures.append(failed)
        if self._batch_cancelled:
            self._finish_batch_tts()
            return True
        self._batch_index += 1
        QTimer.singleShot(0, self._run_next_batch_tts)
        return True

    def _finish_batch_tts(self) -> None:
        notify_task_finished(self, self._translator.translate("common.task_done_tts"), "")  # [v3.23.682]
        self._finish_batch_tts_report()

    def _finish_batch_tts_report(self) -> None:
        """Báo tổng kết và dọn trạng thái hàng đợi."""

        total = len(self._batch_items)
        failures = list(self._batch_failures)
        cancelled = self._batch_cancelled
        processed = self._batch_index
        # [v3.23.659] Tập không khởi động được bị tính lỗi mà chỉ số không tăng → "xong" trước đây thiếu một tập.
        done = getattr(self, "_batch_done", max(0, processed - len(failures)))

        self._batch_items = []
        self._batch_index = 0
        self._batch_failures = []
        self._batch_done = 0  # [v3.23.659] đếm tập xong thật (không suy từ chỉ số)
        self._batch_cancelled = False
        self._batch_pending_generate = False

        if getattr(self, "_series_batch", False):
            # [v3.23.658] Dây chuyền phim bộ: báo về bộ điều phối, không hiện hộp thoại tổng kết.
            from subtitles_extractor.presentation.view_models.series_pipeline_controller import (
                StageRunResult,
            )

            self._series_batch = False
            stop_reason = getattr(self, "_batch_stop_message", "")
            self._batch_stop_message = ""
            failed = dict(getattr(self, "_series_invalid", {}) or {})
            failed.update({name: "TTS lỗi — xem Nhật ký" for name in failures})
            self._stage_lbl.setText(f"Phim bộ — TTS xong {done}/{total} tập.")
            self.series_batch_finished.emit(StageRunResult(
                failed=failed, cancelled=cancelled and not stop_reason, stop_reason=stop_reason,
            ))
            return

        if cancelled:
            message = (
                f"Đã huỷ. Tổng hợp xong {done}/{total} tập, "
                f"bỏ {max(0, total - processed)} tập còn lại."
            )
        else:
            message = f"Đã tổng hợp xong {done}/{total} tập."
        if failures:
            message += "\n\nThất bại:\n• " + "\n• ".join(failures[:10])
            if len(failures) > 10:
                message += f"\n… và {len(failures) - 10} tập nữa."

        self._stage_lbl.setText(message.splitlines()[0])
        QMessageBox.information(self, self._translator.translate("tts.dlg_batch_done_title"), message)


__all__ = ["TTSBatchMixin"]
