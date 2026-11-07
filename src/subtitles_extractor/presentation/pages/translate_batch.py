"""[v3.23.689] "Dịch cả bộ" của trang Dịch: hàng đợi các tập, ước lượng hạn mức, tự phân tích từng tập, báo cáo cuối.

Tách nguyên văn khỏi ``translate_page.py`` (U6 trong kế hoạch: trang 4038 dòng trộn bố cục với logic, mỗi lần sửa giao
diện dễ hồi quy). Lớp trộn (mixin) chỉ chứa phương thức; mọi widget, tín hiệu và trạng thái vẫn thuộc ``TranslatePage``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QTimer

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationStageConfig
from subtitles_extractor.presentation.pages.translate_constants import (
    _BATCH_STOP_MARKERS,
    DEFAULT_ANALYSIS_MODEL,
)
from subtitles_extractor.presentation.utils.task_notifier import notify_task_finished

logger = logging.getLogger("subtitles_extractor.presentation.pages.translate_page")


class TranslateBatchMixin:
    """Phương thức "Dịch cả bộ" / dây chuyền phim bộ của :class:`TranslatePage`."""

    def _on_batch_translate_clicked(self) -> None:
        """Quét thư mục phim bộ rồi dịch lần lượt các tập đã trích xuất."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from subtitles_extractor.application.services.batch_translate_plan import (
            build_translate_plan,
            find_episode_videos,
            summarise_translate_plan,
        )

        if self._view_model.is_busy or getattr(self, "_batch_items", None):
            return
        stages = self._collect_enabled_stages()
        if not stages:
            self._show_warning(
                "Chưa bật giai đoạn dịch nào",
                "Hãy bật ít nhất một giai đoạn dịch trước khi dịch cả bộ.",
            )
            return
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục chứa các tập", "")
        if not folder:
            return

        videos = find_episode_videos(Path(folder))
        if not videos:
            QMessageBox.information(
                self, self._translator.translate("translate.no_folder_title"),
                self._translator.translate("translate.batch_no_source_body"),
            )
            return

        # [v3.23.365] BÙ tệp .original.srt từ CSDL cho các tập đã trích nhưng thiếu tệp
        # (vd đã trích ở bản cũ chỉ lưu CSDL) — để KHỎI phải trích xuất lại.
        self._backfill_original_from_db(videos)

        plan = build_translate_plan(videos, skip_existing=True)
        runnable = [item for item in plan if item.will_run]
        summary = summarise_translate_plan(plan)
        if not runnable:
            QMessageBox.information(
                self, self._translator.translate("translate.no_episode_title"),
                self._translator.translate("translate.batch_missing_body").replace("{summary}", summary),
            )
            return

        message = (
            f"Tìm thấy {len(videos)} tập.\n{summary}\n\n"
            + self._batch_request_estimate(plan, stages)
            + f"\n\nDịch {len(runnable)} tập?"
        )

        answer = QMessageBox.question(
            self, self._translator.translate("translate.batch_confirm_title"), message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._batch_items = runnable
        self._batch_index = 0
        self._batch_failures = []
        self._reset_batch_analysis()
        self._batch_done = 0
        self._batch_cancelled = False
        self._batch_quota_stopped = False
        # [v3.23.610] Tổng token của cả hàng đợi (báo ở bảng tổng kết).
        self._batch_token_mark = self._view_model.token_usage_mark()
        self._run_next_batch_translate()

    def start_series_batch(self, videos: list[Path]) -> str:
        """[v3.23.658] Dây chuyền phim bộ: dịch các tập với cấu hình đang đặt (giai đoạn, model, key), không hộp thoại.

        Returns:
            ``""`` nếu đã bắt đầu (kết quả về qua ``series_batch_finished``), ngược lại lý do không chạy được.
        """
        from subtitles_extractor.application.services.batch_translate_plan import (
            build_translate_plan,
        )
        from subtitles_extractor.presentation.view_models.series_pipeline_controller import (
            StageRunResult,
        )

        if self._view_model.is_busy or getattr(self, "_batch_items", None):
            return self._translator.translate("series.err_page_busy")
        if not self._collect_enabled_stages():
            return "Chưa bật giai đoạn dịch nào ở trang Dịch."
        videos = [Path(v) for v in videos]
        self._backfill_original_from_db(videos)
        plan = build_translate_plan(videos, skip_existing=True)
        runnable = [item for item in plan if item.will_run]
        invalid = {
            item.video_path.name: item.note or "" for item in plan
            if not item.will_run and item.source_path is None
        }
        if not runnable:
            QTimer.singleShot(0, lambda: self.series_batch_finished.emit(StageRunResult(failed=invalid)))
            return ""
        self._series_batch = True
        self._series_invalid = invalid
        self._batch_items = runnable
        self._batch_index = 0
        self._batch_failures = []
        self._reset_batch_analysis()
        self._batch_done = 0
        self._batch_cancelled = False
        self._batch_quota_stopped = False
        self._batch_stop_message = ""
        self._batch_token_mark = self._view_model.token_usage_mark()
        QTimer.singleShot(0, self._run_next_batch_translate)
        return ""

    def cancel_series_batch(self) -> None:
        """[v3.23.658] Dừng khâu Dịch của dây chuyền (như bấm Huỷ)."""
        if getattr(self, "_series_batch", False):
            self._on_cancel_clicked()

    def _batch_request_estimate(
        self, plan: list, stages: list[TranslationStageConfig]
    ) -> str:
        """[v3.23.609] Ước lượng request theo từng model + cảnh báo vượt hạn mức còn lại.

        Trước đây đọc ``self.batch_spin``/``self.model_combo`` — trang không có hai ô này
        (cỡ lô và model nằm ở từng giai đoạn) — nên luôn tính MỘT lượt với lô 40 câu và
        không bao giờ cảnh báo hết hạn mức.
        """
        from subtitles_extractor.application.services.batch_translate_plan import (
            estimate_requests_by_model,
            quota_warning,
        )
        from subtitles_extractor.infrastructure.translation import (
            gemini_translation_adapter as adapter_module,
        )

        per_model = estimate_requests_by_model(
            plan,
            [
                (
                    stage.model_name,
                    adapter_module.effective_batch_size(
                        stage.model_name, stage.batch_size
                    ),
                )
                for stage in stages
            ],
        )
        lines = [
            f"Ước tính {sum(per_model.values())} request API cho {len(stages)} giai đoạn:"
        ]
        warnings: list[str] = []
        keys = self._all_keys()
        for model_name, requests in per_model.items():
            lines.append(f"• {model_name}: ~{requests} request")
            if self._custom_mode():
                continue  # [v3.23.666] hạn mức miễn phí Google không áp cho nhà cung cấp tuỳ chỉnh
            remaining = self._remaining_daily_requests(keys, model_name)
            warning = quota_warning(requests, remaining)
            if warning:
                warnings.append(f"{model_name}: {warning}")
        text = "\n".join(lines)
        if warnings:
            text += "\n\n⚠️ " + "\n⚠️ ".join(warnings)
        return text

    def _remaining_daily_requests(self, keys: list[str], model_name: str) -> int | None:
        """Request còn lại hôm nay của ``model_name``, cộng mọi key (None = không rõ).

        Nhiều key cùng một dự án dùng chung hạn mức, nên đây là mức TỐI ĐA có thể.
        """
        if not keys:
            return None
        try:
            stats = self._view_model.quota_status_for_keys(keys, model_name)
        except (AttributeError, KeyError, RuntimeError, ValueError) as exc:
            logger.debug("Không tra được hạn mức ngày của %s: %s", model_name, exc)
            return None
        return sum(max(0, int(stat.get("remaining", 0))) for stat in stats)

    def _run_next_batch_translate(self) -> None:
        """Nạp phụ đề gốc của tập kế tiếp; dịch chạy khi nạp xong."""
        if not getattr(self, "_batch_items", None):
            return
        if self._batch_cancelled or self._batch_index >= len(self._batch_items):
            self._finish_batch_translate()
            return

        item = self._batch_items[self._batch_index]
        total = len(self._batch_items)
        self._stage_label.setText(
            f"Cả bộ {self._batch_index + 1}/{total}: {item.video_path.name}"
        )
        self._prepare_batch_episode(item)
        self._batch_pending_run = True
        self._view_model.load_source_from_file(item.source_path)

    def _prepare_batch_episode(self, item: object) -> None:
        """[v3.23.609] Đặt trang về đúng TẬP sắp dịch trước khi nạp phụ đề của nó.

        Trước đây mọi tập dịch với ngữ cảnh còn trên trang (bảng nhân vật, tóm tắt và
        Visual Cues — gắn theo số dòng — của tập trước hoặc của phim mở trước đó), đính
        cùng MỘT video cho mọi tập, và coi video cũ là video của tập → nạp nhầm phân tích,
        bản dịch của phim cũ. Nay: xoá ngữ cảnh, lấy video của tập làm video phụ đề (đính
        kèm nếu ô đính video đang bật); bước nạp phụ đề tự lấy lại phân tích đã lưu của
        CHÍNH tập đó hoặc ngữ cảnh chung của phim bộ.
        """
        self._flush_context_save()  # [v3.23.652] lưu sửa tay ngữ cảnh của video CŨ trước khi đổi
        video_path = str(getattr(item, "video_path", "") or "")
        self._clear_context_inputs()
        self._source_video_path = video_path
        if self._attach_video_check.isChecked() and video_path:
            self._video_path = video_path
            self._video_path_label.setText(Path(video_path).name)

    def _on_source_changed_for_batch(self, count: int) -> None:
        """Nạp xong trong lúc chạy hàng loạt -> bắt đầu dịch tập này."""
        if not getattr(self, "_batch_pending_run", False):
            return
        self._batch_pending_run = False
        if count <= 0:
            index = min(self._batch_index, len(self._batch_items) - 1)
            name = self._batch_items[index].video_path.name
            self._advance_batch_translate(failed=name)
            return
        QTimer.singleShot(0, self._start_batch_episode)

    def _start_batch_episode(self) -> None:
        """[v3.23.609] Bắt đầu dịch tập vừa nạp; không bắt đầu được thì dừng cả bộ.

        Trước đây gọi thẳng nút Dịch: cảnh báo cấu hình (thiếu key, chưa chọn video…)
        làm hàng đợi đứng im — không tập nào chạy tiếp, không tổng kết, và trạng thái
        "đang dịch cả bộ" còn treo khiến lần dịch tay sau ghi kết quả vào tệp của tập đó.
        """
        if not getattr(self, "_batch_items", None):
            return
        if self._batch_cancelled:
            self._finish_batch_translate()
            return
        if self._view_model.is_busy:
            # Luồng tập trước chưa kịp báo kết thúc — thử lại ngay sau đó.
            QTimer.singleShot(250, self._start_batch_episode)
            return
        index = self._batch_index
        # [v3.23.669] Tập chưa có phân tích: phân tích bằng CHỮ trước (bảng nhân vật, tóm tắt,
        # thuật ngữ, người nói) với ngữ cảnh các tập trước, rồi mới dịch. Trước đây tập đó dịch với
        # bảng nhân vật tích luỹ + tóm tắt của tập KHÁC — đo tập 2-12 phim mẫu: 9/11 tập có nhân vật
        # mới (23 người) mà khâu dịch không biết.
        if self._maybe_analyze_batch_episode(index):
            return
        if (
            not getattr(self, "_batch_items", None) or self._batch_index != index
            or self._batch_cancelled
        ):
            # Bắt đầu phân tích báo lỗi ngay (key sai…) và lỗi đó đã kết thúc / chuyển hàng đợi:
            # không dịch tập này ngoài hàng đợi.
            return
        if self._start_translation_from_ui():
            return
        if not getattr(self, "_batch_items", None) or self._batch_index != index:
            return  # lỗi đã được xử lý qua error_occurred (chuyển tập / dừng)
        name = self._batch_items[min(index, len(self._batch_items) - 1)].video_path.name
        self._batch_cancelled = True
        # [v3.23.658] Không bắt đầu được (thiếu key, cấu hình…) — lý do đã hiện ở cảnh báo của trang Dịch.
        self._batch_stop_message = self._translator.translate("series.err_translate_start").replace("{name}", name)
        self._advance_batch_translate(failed=name)

    def _reset_batch_analysis(self) -> None:
        """[v3.23.669] Trạng thái "tự phân tích tập chưa phân tích" của một lượt dịch cả bộ."""
        self._batch_auto_analyze = True
        self._batch_analyzing_index = -1
        self._batch_analyzed_index = -1
        self._batch_analyzed_count = 0

    def _maybe_analyze_batch_episode(self, index: int) -> bool:
        """[v3.23.669] Bắt đầu phân tích chữ cho tập ``index`` nếu cần; True = đang phân tích (chờ)."""
        if (
            not getattr(self, "_batch_auto_analyze", False)
            or getattr(self, "_batch_analyzed_index", -1) == index
            or getattr(self, "_source_analysis_restored", False)
        ):
            return False
        self._batch_analyzed_index = index      # mỗi tập chỉ thử một lần, lỗi thì dịch luôn
        api_key = self._combined_keys_text()
        if not api_key:
            return False
        model = self._analyze_model_combo.currentText().strip() or DEFAULT_ANALYSIS_MODEL
        self._sync_text_cues()
        started = self._view_model.start_context_analysis(
            api_key=api_key, target_lang=self._lang_combo.currentText() or "Vietnamese",
            model_name=model, video_path=None, enable_visual_cues=False,
            session_path=self._session_video_path(),
        )
        if not started:
            return False
        self._batch_analyzing_index = index
        total = len(self._batch_items)
        name = self._batch_items[min(index, total - 1)].video_path.name
        self._stage_label.setText(
            f"Cả bộ {index + 1}/{total}: phân tích ngữ cảnh {name} (chưa có phân tích)…"
        )
        return True

    def _batch_analysis_done(self, ok: bool, message: str = "", quota: bool = False) -> bool:
        """[v3.23.669] Phân tích của tập đang chạy hàng loạt xong/lỗi → dịch tiếp tập đó.

        Lỗi phân tích không dừng cả bộ: tập vẫn dịch với ngữ cảnh phim bộ như trước. Hết hạn mức ngày
        của model phân tích thì thôi tự phân tích các tập sau (đỡ tốn thời gian thử lại vô ích).

        Returns:
            True nếu đây là phân tích của hàng loạt (đã xử lý).
        """
        if getattr(self, "_batch_analyzing_index", -1) < 0 or not getattr(self, "_batch_items", None):
            return False
        self._batch_analyzing_index = -1
        if quota:
            self._batch_auto_analyze = False   # dừng giữa chừng vì hết hạn mức ngày
        if ok:
            self._batch_analyzed_count = getattr(self, "_batch_analyzed_count", 0) + 1
        else:
            logger.warning("Cả bộ: phân tích tập lỗi — dịch với ngữ cảnh phim bộ: %s", message)
            lowered = (message or "").lower()
            if any(k in lowered for k in ("hạn mức", "quota", "resource_exhausted")):
                self._batch_auto_analyze = False
        if self._batch_cancelled:
            self._finish_batch_translate()
        else:
            QTimer.singleShot(0, self._start_batch_episode)
        return True

    def _on_error_for_batch(self, message: str) -> bool:
        """Xử lý lỗi khi đang chạy hàng loạt.

        ĐIỂM KHÁC BIỆT của khâu Dịch: hết hạn mức NGÀY thì chạy tiếp cũng vô ích —
        phải dừng cả hàng đợi thay vì đốt thêm lỗi cho từng tập còn lại.

        Returns:
            ``True`` nếu đã xử lý (đang chạy hàng loạt).
        """
        if not getattr(self, "_batch_items", None):
            return False
        if getattr(self, "_persisting_batch_translation", False):
            logger.warning("Hàng loạt: lỗi khi ghi bản dịch — %s", message)
            return True  # `_on_translation_finished` tự xử lý (tính tập này là lỗi, sang tập kế đúng một lần)
        index = min(self._batch_index, len(self._batch_items) - 1)
        name = self._batch_items[index].video_path.name

        lowered = message.lower()
        if any(k in lowered for k in ("hạn mức", "quota", "rpd", "resource_exhausted")):
            logger.warning("Cả bộ: hết hạn mức tại tập %s — dừng hàng đợi.", name)
            self._batch_quota_stopped = True
            self._batch_cancelled = True
            self._batch_stop_message = message  # [v3.23.658] lý do dừng cho dây chuyền phim bộ
            self._advance_batch_translate(failed=name)
            return True
        # [v3.23.609] Lỗi cấu hình chung (thiếu thư viện, key sai) áp cho MỌI tập: dừng,
        # thay vì nạp và làm hỏng lần lượt từng tập còn lại.
        if any(k in lowered for k in _BATCH_STOP_MARKERS):
            logger.warning(
                "Cả bộ: lỗi cấu hình tại tập %s — dừng hàng đợi: %s", name, message
            )
            self._batch_cancelled = True
            self._batch_stop_message = message  # [v3.23.658]
            self._advance_batch_translate(failed=name)
            return True

        logger.warning("Cả bộ: tập %s lỗi — %s", name, message)
        self._advance_batch_translate(failed=name)
        return True

    def _advance_batch_translate(self, *, failed: str | None = None) -> bool:
        """Chuyển sang tập kế. Trả ``True`` nếu đang chạy hàng loạt."""
        if not getattr(self, "_batch_items", None):
            return False
        if failed:
            self._batch_failures.append(failed)
        if self._batch_cancelled:
            self._finish_batch_translate()
            return True
        self._batch_index += 1
        QTimer.singleShot(0, self._run_next_batch_translate)
        return True

    def _finish_batch_translate(self) -> None:
        notify_task_finished(self, self._translator.translate("common.task_done_translate"), "")  # [v3.23.682]
        self._finish_batch_translate_report()

    def _finish_batch_translate_report(self) -> None:
        """Báo tổng kết và dọn trạng thái hàng đợi."""
        from PySide6.QtWidgets import QMessageBox

        total = len(self._batch_items)
        failures = list(self._batch_failures)
        cancelled = self._batch_cancelled
        quota_stopped = getattr(self, "_batch_quota_stopped", False)
        # [v3.23.609] Đếm tập xong theo lần dịch xong thật. Trước đây lấy "chỉ số tập
        # hiện tại trừ số tập lỗi", mà khi dừng vì hết hạn mức (hay huỷ) tập lỗi chưa
        # được tính vào chỉ số → báo thiếu một tập đã dịch xong.
        done = self._batch_done

        self._batch_items = []
        self._batch_index = 0
        self._batch_failures = []
        self._batch_done = 0
        self._batch_cancelled = False
        self._batch_quota_stopped = False
        self._batch_pending_run = False
        stop_message = getattr(self, "_batch_stop_message", "")
        self._batch_stop_message = ""

        remaining = max(0, total - done - len(failures))
        if getattr(self, "_series_batch", False):
            # [v3.23.658] Dây chuyền phim bộ: báo về bộ điều phối, không hiện hộp thoại tổng kết.
            from subtitles_extractor.presentation.view_models.series_pipeline_controller import (
                StageRunResult,
            )

            self._series_batch = False
            failed = dict(getattr(self, "_series_invalid", {}) or {})
            failed.update({name: "Dịch lỗi — xem Nhật ký" for name in failures})
            if quota_stopped:
                stop_reason = self._translator.translate("series.stop_quota")
            elif stop_message:
                stop_reason = stop_message
            else:
                stop_reason = ""
            self._stage_label.setText(f"Phim bộ — dịch xong {done}/{total} tập.")
            self._update_action_states()
            self.series_batch_finished.emit(StageRunResult(
                failed=failed, cancelled=cancelled and not stop_reason, stop_reason=stop_reason,
            ))
            return
        if quota_stopped:
            # Tập vướng hạn mức cũng là tập "chưa dịch" sẽ chạy lại vào hôm sau.
            message = (
                f"HẾT HẠN MỨC NGÀY. Đã dịch xong {done}/{total} tập.\n"
                f"Còn {max(0, total - done)} tập chưa dịch — chạy lại vào ngày mai, "
                "hệ thống sẽ "
                "TỰ BỎ QUA các tập đã xong nên không tốn thêm hạn mức."
            )
        elif cancelled:
            message = (
                f"Đã huỷ. Dịch xong {done}/{total} tập, bỏ {remaining} tập còn lại."
            )
        else:
            message = f"Đã dịch xong {done}/{total} tập."
        if failures:
            message += "\n\nThất bại:\n• " + "\n• ".join(failures[:10])
        tokens = self._view_model.describe_token_usage_since(
            getattr(self, "_batch_token_mark", {}) or {}
        )
        if tokens:
            message += f"\n\n{tokens}"

        self._stage_label.setText(message.splitlines()[0])
        self._update_action_states()
        QMessageBox.information(self, self._translator.translate("translate.batch_done_title"), message)


__all__ = ["TranslateBatchMixin"]
