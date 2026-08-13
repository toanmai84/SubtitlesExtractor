"""ViewModel cho trang Dịch phụ đề.

Giữ trạng thái nguồn/đích, điều phối ``TranslateWorker`` và phát signal cho View.
Không chứa widget — chỉ logic trình bày + cầu nối tới use case qua container.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesResponse,
)
from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
)
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat
from subtitles_extractor.presentation.workers.translate_worker import TranslateWorker
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _format_elapsed(seconds: float) -> str:
    """Định dạng thời gian thực thi dạng 'mm:ss' hoặc 'Xs'."""
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}p{s:02d}s"
    return f"{seconds:.1f}s"


class TranslatePageViewModel(QObject):
    """ViewModel điều phối dịch phụ đề.

    Signals:
        source_changed:       ``(int,)``           — số dòng nguồn.
        result_ready:         ``(object, object)`` — (source_events, translated_events).
        progress_changed:     ``(int, str)``       — phần trăm + mô tả giai đoạn.
        busy_changed:         ``(bool,)``          — trạng thái đang bận.
        error_occurred:       ``(str,)``           — thông điệp lỗi dịch.
        analyze_error:        ``(str,)``           — thông điệp lỗi phân tích ngữ cảnh.
        status_message:       ``(str,)``           — trạng thái ngắn.
        translation_cancelled:``()``               — dịch bị huỷ êm.
        resume_detected:      ``(str,)``           — tên giai đoạn resume đầu tiên.
    """

    source_changed = Signal(int)
    result_ready = Signal(object, object)
    progress_changed = Signal(int, str)
    busy_changed = Signal(bool)
    error_occurred = Signal(str)
    analyze_error = Signal(str)         # phân biệt lỗi dịch vs lỗi phân tích
    status_message = Signal(str)
    translation_cancelled = Signal()
    resume_detected = Signal(str)       # phát khi tìm thấy checkpoint
    analysis_restored = Signal(object)  # [v3.23.45] phát khi nạp lại phân tích đã lưu

    def __init__(self, container: ApplicationContainer) -> None:
        super().__init__()
        self._container = container
        self._source_events: list[SubtitleEvent] = []
        self._translated_events: list[SubtitleEvent] = []
        self._stage_outputs: dict = {}  # [v3.23.47] kết quả từng giai đoạn để so sánh
        self._worker: TranslateWorker | None = None
        self._analyze_worker = None
        self._fetch_models_worker = None       # giữ strong ref tránh GC crash
        self._is_busy = False
        self._last_stage_count: int = 0
        self._last_elapsed_sec: float = 0.0
        self._push_to_editor_callback = None

    # ── Truy cập trạng thái ──────────────────────────────────────────────
    @property
    def source_events(self) -> list[SubtitleEvent]:
        return list(self._source_events)

    @property
    def translated_events(self) -> list[SubtitleEvent]:
        return list(self._translated_events)

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    @property
    def has_source(self) -> bool:
        return bool(self._source_events)

    @property
    def has_result(self) -> bool:
        return bool(self._translated_events)

    def export_translation_diagnostics(self, out_path: Path) -> Path:
        """Xuất gói chẩn đoán dịch (JSON) cho video hiện hành.

        Gom phụ đề gốc, kết quả dịch cuối, từng giai đoạn dịch và toàn bộ phân tích
        ngữ cảnh (nhân vật/tóm tắt/glossary/visual cues) cùng ngữ cảnh phim bộ vào MỘT
        file JSON để gửi đi phân tích cải thiện chất lượng dịch.

        Args:
            out_path: Đường dẫn file JSON đích.

        Returns:
            Đường dẫn file đã ghi.

        Raises:
            OSError: Nếu không ghi được file đích.
        """
        import json
        import sqlite3
        from datetime import datetime, timezone

        from subtitles_extractor import __version__
        from subtitles_extractor.application.services.translation_diagnostics import (
            build_diagnostics_bundle,
        )

        video_path = getattr(self, "_last_translate_video_path", "") or ""

        session = None
        if video_path:
            try:
                session = self._container.translation_session_store.get(video_path)
            except (sqlite3.Error, AttributeError, OSError) as exc:
                logger.warning("Không đọc được session dịch khi xuất chẩn đoán: %s", exc)

        series_context = None
        if video_path:
            # restore_series_context tự nuốt lỗi nội bộ và trả None khi thất bại.
            series_context = self.restore_series_context(video_path)

        bundle = build_diagnostics_bundle(
            app_version=__version__,
            exported_at=datetime.now(timezone.utc).isoformat(),
            video_path=str(video_path),
            source_events=self._source_events,
            translated_events=self._translated_events,
            session=session,
            series_context=series_context,
        )

        out_path = Path(out_path)
        out_path.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "Đã xuất gói chẩn đoán dịch (%d nguồn, %d dịch, %d giai đoạn) → %s",
            bundle["counts"]["source_events"],
            bundle["counts"]["translated_events"],
            bundle["counts"]["stages"],
            out_path,
        )
        return out_path

    def check_glossary_consistency(self, glossary_text: str) -> list:
        """[v3.23.54] Đối chiếu bảng thuật ngữ với bản dịch, trả danh sách vi phạm.

        Mỗi vi phạm là một :class:`GlossaryViolation` chỉ ra dòng có thuật ngữ gốc nhưng
        thiếu bản dịch chuẩn. Rỗng nếu nhất quán hoặc chưa có bản dịch.
        """
        if not self._translated_events:
            return []
        from subtitles_extractor.application.services.glossary_consistency import (
            check_glossary_consistency,
        )
        source_by_index = {ev.index: ev.text for ev in self._source_events}
        source_texts: list[str] = []
        translated_texts: list[str] = []
        for ev in self._translated_events:
            source_texts.append(source_by_index.get(ev.index, ""))
            translated_texts.append(ev.text)
        return check_glossary_consistency(glossary_text, source_texts, translated_texts)

    def list_memory_series(self) -> list[tuple[str, int]]:
        """[v3.23.57] Liệt kê các phim bộ có bộ nhớ dịch (tên, số cặp câu)."""
        try:
            return self._container.translation_memory_store.list_series()
        except (OSError, ValueError) as exc:
            logger.warning("Không liệt kê được bộ nhớ phim bộ: %s", exc)
            return []

    def clear_memory_series(self, series_key: str) -> bool:
        """[v3.23.57] Xoá bộ nhớ dịch + ngữ cảnh của một phim bộ. Trả True nếu thành công."""
        if not series_key:
            return False
        try:
            self._container.translation_memory_store.clear_series(series_key)
            logger.info("Đã xoá bộ nhớ dịch của phim bộ '%s'.", series_key)
            return True
        except (OSError, ValueError) as exc:
            logger.warning("Không xoá được bộ nhớ phim bộ: %s", exc)
            return False

    def _accumulate_translation_memory(self) -> None:
        """[v3.23.55] Lưu cặp (câu nguồn → câu dịch) vào Bộ nhớ dịch theo phim bộ."""
        video_path = getattr(self, "_last_translate_video_path", "") or ""
        if not video_path or not self._translated_events:
            return
        try:
            from subtitles_extractor.application.services.translation_memory import (
                TranslationMemoryEntry,
                derive_series_key,
            )
            series_key = derive_series_key(video_path)
            if not series_key:
                return
            source_by_index = {ev.index: ev.text for ev in self._source_events}
            entries = [
                TranslationMemoryEntry(
                    source_text=source_by_index.get(ev.index, ""), target_text=ev.text
                )
                for ev in self._translated_events
                if source_by_index.get(ev.index, "").strip() and ev.text.strip()
            ]
            if not entries:
                return
            store = self._container.translation_memory_store
            saved = store.add_entries(series_key, entries)
            logger.info(
                "Đã lưu %d cặp câu vào Bộ nhớ dịch cho phim bộ '%s'.", saved, series_key
            )
        except (OSError, ValueError, ImportError) as exc:
            logger.warning("Không lưu được Bộ nhớ dịch: %s", exc)

    def accumulate_series_context(
        self, video_path: str, glossary: str, characters: str, overview: str
    ) -> None:
        """[v3.23.56] Gộp & lưu ngữ cảnh (glossary/roster/tóm tắt) chung cho phim bộ.

        Glossary được GỘP (giữ cách dịch cũ, thêm mục mới); roster/tóm tắt lấy bản mới nhất
        nếu không rỗng. Gọi sau khi phân tích/dịch một tập để các tập sau dùng lại.
        """
        if not video_path:
            return
        try:
            from subtitles_extractor.application.services.translation_memory import (
                derive_series_key,
                merge_characters,
                merge_glossary,
            )
            series_key = derive_series_key(video_path)
            if not series_key:
                return
            store = self._container.translation_memory_store
            prior = store.get_series_context(series_key)
            if prior is not None:
                glossary = merge_glossary(prior.glossary, glossary)
                # [v3.23.91] GỘP roster (không ghi đè) -> danh sách tên chuẩn tích luỹ
                # xuyên tập, giúp tên riêng nhất quán hơn giữa các tập.
                characters = merge_characters(prior.characters, characters)
                overview = overview.strip() or prior.overview
            store.save_series_context(series_key, glossary, characters, overview)
            logger.info("Đã cập nhật ngữ cảnh chung cho phim bộ '%s'.", series_key)
        except (OSError, ValueError, ImportError) as exc:
            logger.warning("Không lưu được ngữ cảnh phim bộ: %s", exc)

    def _build_prior_series_context(self, video_path: str) -> str:
        """[v3.23.92] Dựng chuỗi ngữ cảnh phim bộ ĐÃ TÍCH LUỸ (roster + glossary) để mồi
        vào prompt phân tích tập hiện tại, phục vụ phân tích tuần tự tích luỹ.

        Trả chuỗi rỗng nếu chưa có ngữ cảnh tích luỹ / không xác định được phim bộ.
        """
        prior = self.restore_series_context(video_path)
        if prior is None:
            return ""
        parts: list[str] = []
        chars = (getattr(prior, "characters", "") or "").strip()
        gloss = (getattr(prior, "glossary", "") or "").strip()
        if chars:
            parts.append("# Nhân vật đã thiết lập:\n" + chars)
        if gloss:
            parts.append("# Thuật ngữ đã thiết lập:\n" + gloss)
        return "\n\n".join(parts)

    def restore_series_context(self, video_path: str):
        """[v3.23.56] Lấy ngữ cảnh chung của phim bộ (để điền vào UI khi chọn tập mới).

        Trả ``SeriesContext`` hoặc None nếu chưa có/không xác định được phim bộ.
        """
        if not video_path:
            return None
        try:
            from subtitles_extractor.application.services.translation_memory import (
                derive_series_key,
            )
            series_key = derive_series_key(video_path)
            if not series_key:
                return None
            return self._container.translation_memory_store.get_series_context(series_key)
        except (OSError, ValueError, ImportError) as exc:
            logger.warning("Không nạp được ngữ cảnh phim bộ: %s", exc)
            return None

    def retrieve_memory_for_lines(
        self, video_path: str, source_lines: list[str], *, per_line_k: int = 2,
        max_query_lines: int = 400, max_references: int = 40,
    ) -> str:
        """[v3.23.55] Truy hồi tham chiếu TM cho tập câu nguồn, trả khối chèn vào prompt.

        Gom các câu đã dịch liên quan nhất từ các tập trước của cùng phim bộ. Trả chuỗi
        rỗng nếu không có TM hoặc không xác định được phim bộ.

        [v3.23.58] Giới hạn hiệu năng: với phụ đề lớn (hàng nghìn dòng) + TM lớn, truy hồi
        cho MỌI dòng sẽ rất chậm (treo UI). Vì vậy chỉ lấy mẫu tối đa ``max_query_lines``
        câu dài nhất (câu dài thường chứa tên/thuật ngữ cần nhất quán) và dừng khi đã đủ
        ``max_references`` tham chiếu.
        """
        if not video_path or not source_lines:
            return ""
        try:
            from subtitles_extractor.application.services.translation_memory import (
                derive_series_key,
                format_reference_block,
                retrieve_relevant,
            )
            series_key = derive_series_key(video_path)
            if not series_key:
                return ""
            store = self._container.translation_memory_store
            entries = store.get_entries(series_key)
            if not entries:
                return ""
            # Lấy mẫu các câu DÀI nhất (giàu danh từ riêng/thuật ngữ) để giảm số truy vấn.
            query_lines = sorted(
                (ln for ln in source_lines if ln.strip()), key=len, reverse=True
            )[:max_query_lines]
            collected: list = []
            seen: set[str] = set()
            for line in query_lines:
                for entry in retrieve_relevant(line, entries, top_k=per_line_k):
                    if entry.source_text not in seen:
                        seen.add(entry.source_text)
                        collected.append(entry)
                if len(collected) >= max_references:
                    break
            return format_reference_block(collected, max_entries=max_references)
        except (OSError, ValueError, ImportError) as exc:
            logger.warning("Không truy hồi được Bộ nhớ dịch: %s", exc)
            return ""

    def _persist_current_result(self) -> None:
        """[v3.23.49] Lưu BẢN DỊCH HIỆN HÀNH (sau khi sửa tay / dịch lại dòng) vào session
        dưới stage_id 'final', để không mất khi đóng app và được ưu tiên khi nạp lại."""
        video_path = getattr(self, "_last_translate_video_path", "") or ""
        if not video_path or not self._translated_events:
            return
        try:
            import json

            from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
                StageResult,
            )
            store = self._container.translation_session_store
            lines_json = json.dumps(
                [{"index": ev.index, "text": ev.text} for ev in self._translated_events],
                ensure_ascii=False,
            )
            store.save_stage(
                video_path,
                StageResult(stage_id="final", input_hash="", lines_json=lines_json),
            )
        except (OSError, ValueError, ImportError) as exc:
            logger.warning("Không lưu được bản dịch hiện hành vào session: %s", exc)

    def _persist_stage_outputs(self) -> None:
        """[v3.23.48] Lưu kết quả từng giai đoạn vào session store (theo video_path)."""
        video_path = getattr(self, "_last_translate_video_path", "") or ""
        if not video_path or not self._stage_outputs:
            return
        try:
            import json

            from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
                StageResult,
            )
            store = self._container.translation_session_store
            for kind, events in self._stage_outputs.items():
                stage_id = kind.value if hasattr(kind, "value") else str(kind)
                lines_json = json.dumps(
                    [{"index": ev.index, "text": ev.text} for ev in events],
                    ensure_ascii=False,
                )
                store.save_stage(
                    video_path,
                    StageResult(stage_id=stage_id, input_hash="", lines_json=lines_json),
                )
            logger.info(
                "Đã lưu %d giai đoạn dịch vào session cho %s.",
                len(self._stage_outputs), video_path,
            )
        except (OSError, ValueError, ImportError) as exc:
            logger.warning("Không lưu được kết quả giai đoạn vào session: %s", exc)

    def restore_stage_outputs_for_video(self, video_path: str) -> int:
        """[v3.23.48] Nạp lại kết quả từng giai đoạn đã lưu cho video. Trả số giai đoạn."""
        if not video_path:
            return 0
        try:
            import json

            from subtitles_extractor.domain.ports.subtitle_translator_port import (
                TranslationStageKind,
            )
            store = self._container.translation_session_store
            session = store.get(video_path)
            if session is None or not session.stages:
                return 0
            # [v3.23.50] An toàn: nếu phụ đề nguồn hiện tại có số dòng KHÁC NHIỀU so với
            # dữ liệu đã lưu, việc gán theo index sẽ lệch → bỏ qua để không làm hỏng bản
            # dịch. Cho phép sai lệch nhỏ (vài dòng) vì index dựa trên giá trị gốc.
            max_saved_index = 0
            for stage in session.stages:
                try:
                    lines_preview = json.loads(stage.lines_json) if stage.lines_json else []
                    for ln in lines_preview:
                        max_saved_index = max(max_saved_index, int(ln.get("index", 0)))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
            if max_saved_index and self._source_events:
                source_max = max(ev.index for ev in self._source_events)
                # Lệch >20% số dòng → coi như nguồn đã đổi, không khôi phục (tránh sai).
                if abs(source_max - max_saved_index) > max(5, source_max * 0.2):
                    logger.info(
                        "Bỏ qua nạp lại giai đoạn: nguồn đổi (hiện %d dòng vs đã lưu %d).",
                        source_max, max_saved_index,
                    )
                    return 0
            restored: dict = {}
            final_events: list[SubtitleEvent] = []
            for stage in session.stages:
                lines = json.loads(stage.lines_json) if stage.lines_json else []
                events = []
                for ln in lines:
                    src_idx = ln.get("index", 0) - 1
                    if 0 <= src_idx < len(self._source_events):
                        base = self._source_events[src_idx]
                        events.append(
                            SubtitleEvent(
                                index=base.index, text=ln.get("text", ""),
                                interval=base.interval,
                            )
                        )
                # [v3.23.49] stage_id 'final' = bản dịch hiện hành (đã sửa/dịch lại) →
                # khôi phục vào _translated_events; các stage còn lại dùng để so sánh.
                if stage.stage_id == "final":
                    final_events = events
                    continue
                try:
                    kind = TranslationStageKind(stage.stage_id)
                except ValueError:
                    continue
                if events:
                    restored[kind] = events
            self._stage_outputs = restored
            self._last_translate_video_path = video_path  # [v3.23.49] cho persist sau sửa
            if final_events:
                self._translated_events = final_events
            elif restored:
                # Không có 'final' → dùng giai đoạn cuối theo thứ tự pipeline làm kết quả.
                order = [
                    TranslationStageKind.LOCALIZE, TranslationStageKind.STYLE,
                    TranslationStageKind.LITERAL, TranslationStageKind.PREPROCESS,
                ]
                for k in order:
                    if k in restored:
                        self._translated_events = list(restored[k])
                        break
            return len(restored)
        except (OSError, ValueError, ImportError) as exc:
            logger.warning("Không nạp được kết quả giai đoạn từ session: %s", exc)
            return 0

    def stage_comparison(self) -> list[tuple[str, list[str]]]:
        """[v3.23.47] Trả dữ liệu so sánh các giai đoạn dịch.

        Mỗi phần tử là (tên_giai_đoạn, danh_sách_văn_bản_theo_dòng). Chỉ trả khi có ÍT
        NHẤT 2 giai đoạn được lưu (để so sánh có nghĩa); ngược lại trả danh sách rỗng.
        """
        if len(self._stage_outputs) < 2:
            return []
        labels = {
            "preprocess": "Tiền xử lý",
            "literal": "Dịch thô",
            "style": "Tinh chỉnh",
            "localize": "Bản địa hoá",
        }
        order = ["preprocess", "literal", "style", "localize"]
        result: list[tuple[str, list[str]]] = []
        for kind in self._stage_outputs:
            key = kind.value if hasattr(kind, "value") else str(kind)
            events = self._stage_outputs[kind]
            result.append((labels.get(key, key), [ev.text for ev in events]))
        # Sắp theo thứ tự pipeline.
        result.sort(key=lambda x: next(
            (i for i, k in enumerate(order) if labels.get(k) == x[0]), 99
        ))
        return result

    # ── Chỉnh sửa / dịch lại dòng (trang Dịch chuyên biệt) ───────────────
    def update_translation_text(self, line_index: int, new_text: str) -> bool:
        """[v3.23.46] Sửa bản dịch của MỘT dòng tại chỗ (theo index 0-based trong list).

        Trả True nếu cập nhật thành công. Không gọi AI — chỉ sửa dữ liệu hiện có.
        """
        if not (0 <= line_index < len(self._translated_events)):
            return False
        self._translated_events[line_index].text = new_text
        self._persist_current_result()  # [v3.23.49] lưu để không mất khi đóng app
        return True

    def retranslate_lines(
        self, api_key: str, retry_count: int, line_indices: list[int],
        stages: list[TranslationStageConfig], context: TranslationContext,
    ) -> bool:
        """[v3.23.46] Dịch lại CHỈ các dòng được chọn (theo index 0-based), giữ nguyên
        các dòng khác. Dùng cùng pipeline dịch nhưng trên tập con sự kiện nguồn.

        Trả True nếu đã khởi động; False nếu đầu vào không hợp lệ (đang bận/thiếu key…).
        """
        if self._is_busy:
            self.status_message.emit("Đang xử lý, vui lòng chờ tiến trình hiện tại.")
            return False
        if not self._source_events or not self._translated_events:
            self.error_occurred.emit("Chưa có bản dịch để dịch lại.")
            return False
        valid = sorted({i for i in line_indices if 0 <= i < len(self._source_events)})
        if not valid:
            self.error_occurred.emit("Không có dòng hợp lệ để dịch lại.")
            return False
        if not stages:
            self.error_occurred.emit("Hãy bật ít nhất một giai đoạn dịch.")
            return False
        probe = self._container.make_subtitle_translator(api_key, retry_count)
        if not probe.is_available():
            self.error_occurred.emit(
                "Bộ dịch chưa sẵn sàng. Kiểm tra đã cài 'google-genai' và nhập đúng API Key."
            )
            return False

        subset = [self._source_events[i] for i in valid]
        use_case = self._container.make_translate_subtitles_use_case(api_key, retry_count)
        request = TranslateSubtitlesRequest(
            events=list(subset), stages=stages, context=context,
        )
        self._retranslate_indices = valid
        self._set_busy(True)
        self._worker = TranslateWorker(use_case=use_case, request=request)
        self._worker.progress_changed.connect(
            lambda pct, msg: self.progress_changed.emit(pct, f"Dịch lại: {msg}")
        )
        self._worker.finished_ok.connect(self._on_retranslate_done)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._on_thread_finished)
        self._worker.start()
        self.status_message.emit(f"Đang dịch lại {len(valid)} dòng…")
        return True

    def _on_retranslate_done(self, response: object) -> None:
        """Ghép kết quả dịch lại vào đúng vị trí các dòng đã chọn (không đụng dòng khác)."""
        if self.sender() is not self._worker:
            return
        new_events = list(getattr(response, "events", []))
        indices = getattr(self, "_retranslate_indices", [])
        applied = 0
        for offset, line_idx in enumerate(indices):
            if offset < len(new_events) and 0 <= line_idx < len(self._translated_events):
                self._translated_events[line_idx].text = new_events[offset].text
                applied += 1
        self.result_ready.emit(self.source_events, self.translated_events)
        self._persist_current_result()  # [v3.23.49] lưu bản dịch sau khi dịch lại
        self.status_message.emit(f"Đã dịch lại {applied} dòng.")

    # ── Nạp nguồn ────────────────────────────────────────────────────────
    def set_source_events(self, events: list[SubtitleEvent]) -> None:
        """Nạp danh sách phụ đề nguồn (vd lấy từ trang Biên tập)."""
        self._source_events = list(events)
        self._translated_events = []
        self.source_changed.emit(len(self._source_events))

    def load_source_from_file(self, source_path: Path) -> None:
        """Nạp phụ đề nguồn từ tệp SRT/ASS qua use case import."""
        try:
            use_case = self._container.make_import_subtitles_use_case()
            events = use_case.execute(source_path)
        except FileNotFoundError:
            self.error_occurred.emit(f"Không tìm thấy tệp: {source_path}")
            return
        except (ValueError, KeyError, OSError) as exc:
            self.error_occurred.emit(f"Không thể đọc tệp phụ đề: {exc}")
            return

        if not events:
            self.error_occurred.emit("Tệp phụ đề rỗng hoặc không có dòng hợp lệ.")
            return

        self.set_source_events(events)
        self.status_message.emit(f"Đã nạp {len(events)} dòng từ {source_path.name}.")

    # ── Dịch ─────────────────────────────────────────────────────────────
    def start_translation(
        self,
        api_key: str,
        retry_count: int,
        stages: list[TranslationStageConfig],
        context: TranslationContext,
        video_path: str | None = None,
        attach_video_stages: frozenset | None = None,
        enable_visual_cues: bool = False,
    ) -> bool:
        if self._is_busy:
            self.status_message.emit("Đang dịch, vui lòng chờ tiến trình hiện tại hoàn tất.")
            return False
        if not self._source_events:
            self.error_occurred.emit("Chưa có phụ đề nguồn để dịch.")
            return False
        if not stages:
            self.error_occurred.emit("Hãy bật ít nhất một giai đoạn dịch.")
            return False

        use_case = self._container.make_translate_subtitles_use_case(api_key, retry_count)
        # [fix B4] Dùng make_subtitle_translator riêng để kiểm tra availability
        # thay vì truy cập thuộc tính private _translator của use case.
        probe = self._container.make_subtitle_translator(api_key, retry_count)
        if not probe.is_available():
            self.error_occurred.emit(
                "Bộ dịch chưa sẵn sàng. Kiểm tra đã cài 'google-genai' và nhập đúng API Key."
            )
            return False

        # [v3.23.34] Visual Cues dùng CÙNG model với các giai đoạn dịch (model người
        # dùng đang dùng, vd flash-lite-latest) thay vì gemini-2.0-flash cứng — tránh
        # 429 khi tài khoản không có quota cho 2.0-flash.
        # [v3.23.58] Lưu context GỐC (trước khi chèn TM) để tích luỹ ngữ cảnh phim bộ —
        # tránh khối TM bị ghi vào overview rồi phình to lũy tiến qua mỗi tập.
        original_context = context

        # [v3.23.55] Truy hồi Bộ nhớ dịch (TM) của phim bộ và chèn vào ngữ cảnh để giữ
        # nhất quán tên riêng/thuật ngữ/lối xưng hô giữa các tập. Chỉ khi có video_path.
        if video_path:
            tm_block = self.retrieve_memory_for_lines(
                video_path, [ev.text for ev in self._source_events]
            )
            if tm_block:
                from dataclasses import replace as _dc_replace

                try:
                    context = _dc_replace(
                        context, overview=(context.overview or "") + "\n" + tm_block
                    )
                    logger.info("Đã chèn Bộ nhớ dịch phim bộ vào ngữ cảnh dịch.")
                except (TypeError, ValueError):
                    pass  # context không hỗ trợ replace → bỏ qua, không chặn dịch

        visual_cues_model = stages[0].model_name if stages else "gemini-flash-lite-latest"
        request = TranslateSubtitlesRequest(
            events=list(self._source_events), stages=stages, context=context,
            enable_visual_cues=bool(enable_visual_cues and video_path),
            visual_cues_model=visual_cues_model,
        )

        # Chuẩn bị provider video nếu người dùng bật đính video HOẶC bật visual cues
        # (visual cues cũng cần video_refs đã tải lên).
        video_provider = None
        if video_path and (attach_video_stages or enable_visual_cues):
            video_provider = self._container.make_video_context_provider(api_key)

        self._set_busy(True)
        # [v3.23.48] Ghi nhớ video_path để lưu kết quả từng giai đoạn vào session sau khi
        # dịch xong (cho phép nạp lại khi mở lại video lần sau).
        self._last_translate_video_path = video_path or ""
        self._last_translate_context = original_context  # [v3.23.58] context GỐC (không TM)
        self._worker = TranslateWorker(
            use_case=use_case, request=request,
            video_provider=video_provider, video_path=video_path,
            attach_video_stages=attach_video_stages or frozenset(),
        )
        self._worker.progress_changed.connect(self.progress_changed.emit)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._on_thread_finished)
        self._worker.start()
        self.status_message.emit("Bắt đầu dịch...")
        return True

    def clear_translation_checkpoint(self, api_key: str, retry_count: int,
                                     stages: list[TranslationStageConfig],
                                     context: TranslationContext) -> None:
        """Xóa checkpoint dịch tương ứng với request hiện tại."""
        if not self._source_events:
            return
        try:
            from subtitles_extractor.application.use_cases.translate_subtitles import (
                _compute_checkpoint_key, _StageCheckpoint, TranslateSubtitlesRequest,
            )
            request = TranslateSubtitlesRequest(
                events=list(self._source_events), stages=stages, context=context
            )
            checkpoint_dir = self._container.user_data_dir / "translation_checkpoints"
            key = _compute_checkpoint_key(request)
            cp = _StageCheckpoint(checkpoint_dir, key)
            cp.delete()
            self.status_message.emit("Đã xoá bộ nhớ đệm dịch. Lần dịch tới sẽ bắt đầu lại từ đầu.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không xoá được checkpoint: %s", exc)

    # ── Phân tích ngữ cảnh ───────────────────────────────────────────────────
    analyze_context_ready = Signal(object)   # SubtitleContextAnalysis — kết quả phân tích

    def start_context_analysis(
        self, api_key: str, target_lang: str, model_name: str = "gemini-3.1-flash-lite",
        video_path: str | None = None, enable_visual_cues: bool = False,
    ) -> bool:
        """Phân tích TOÀN BỘ phụ đề nguồn bằng AI trên thread nền.

        Args:
            api_key:     Khoá API Gemini.
            target_lang: Ngôn ngữ viết phần tóm tắt.
            model_name:  Model AI dùng cho phân tích.
        """
        if self._is_busy:
            self.status_message.emit("Đang bận, vui lòng chờ.")
            return False
        if not self._source_events:
            self.error_occurred.emit("Chưa có phụ đề nguồn để phân tích.")
            return False

        use_case = self._container.make_analyze_context_use_case(api_key)
        probe = self._container.make_subtitle_translator(api_key, retry_count=3)
        if not probe.is_available():
            self.error_occurred.emit("API Key Gemini chưa hợp lệ hoặc thiếu thư viện google-genai.")
            return False

        from subtitles_extractor.application.use_cases.translate_subtitles import _events_to_lines_helper
        source_lines = _events_to_lines_helper(self._source_events)

        from subtitles_extractor.presentation.workers.analyze_context_worker import AnalyzeContextWorker
        video_provider = (
            self._container.make_video_context_provider(api_key) if video_path else None
        )
        # [v3.23.92] PHÂN TÍCH TUẦN TỰ TÍCH LUỸ: nạp ngữ cảnh phim bộ đã tích luỹ từ các
        # tập TRƯỚC (roster + glossary) và mồi vào prompt phân tích, để model TÁI SỬ DỤNG
        # đúng tên/thuật ngữ đã thiết lập -> nhất quán xuyên tập.
        prior_context = self._build_prior_series_context(video_path)
        self._analyze_worker = AnalyzeContextWorker(
            use_case, source_lines, target_lang, model_name=model_name,
            video_provider=video_provider, video_path=video_path,
            enable_visual_cues=bool(enable_visual_cues and video_path),
            prior_context=prior_context,
        )
        self._analyze_worker.finished_ok.connect(self._on_analyze_finished)
        self._analyze_worker.video_refs_ready.connect(self._on_video_refs_ready)
        self._analyze_worker.cancelled.connect(lambda: self.status_message.emit("Đã huỷ phân tích."))
        self._analyze_worker.failed.connect(self.analyze_error.emit)
        self._analyze_worker.progress_message.connect(self.status_message.emit)
        self._analyze_worker.finished.connect(self._on_analyze_thread_finished)
        # [v3.23.15] Ghi nhớ ngữ cảnh phiên để lưu kết quả phân tích sau khi xong.
        self._pending_analysis_video_path = video_path
        self._pending_analysis_target_lang = target_lang
        self._pending_analysis_source_lines = [ln.text for ln in source_lines]
        self._set_busy(True)
        self._analyze_worker.start()
        n = len(source_lines)
        self.status_message.emit(f"Đang phân tích toàn bộ {n} dòng bằng {model_name}…")
        return True

    def _on_analyze_finished(self, result) -> None:
        self.analyze_context_ready.emit(result)
        self.status_message.emit("Phân tích ngữ cảnh hoàn tất.")
        # [v3.23.15] Lưu kết quả phân tích vào phiên dịch (theo video) để mở lại video
        # là khôi phục được, không phân tích lại. Bọc an toàn — lỗi lưu không phá UI.
        self._persist_analysis_result(result)

    def _persist_analysis_result(self, result) -> None:
        video_path = getattr(self, "_pending_analysis_video_path", None)
        if not video_path:
            return
        try:
            from subtitles_extractor.application.services.translation_session_hashing import (
                hash_analysis_input,
            )

            target_lang = getattr(self, "_pending_analysis_target_lang", "") or ""
            source_lines = getattr(self, "_pending_analysis_source_lines", []) or []
            input_hash = hash_analysis_input(source_lines, target_lang)
            store = self._container.translation_session_store
            store.save_analysis(
                video_path,
                characters=getattr(result, "characters", "") or "",
                overview=getattr(result, "overview", "") or "",
                source_lang=getattr(result, "source_lang", "") or "",
                target_lang=target_lang,
                input_hash=input_hash,
                glossary=getattr(result, "glossary", "") or "",
                visual_cues=getattr(result, "visual_cues", "") or "",
            )
            logger.info("Đã lưu phân tích ngữ cảnh vào phiên dịch cho %s.", video_path)
        except Exception as exc:  # noqa: BLE001 — lưu phiên không được phá luồng UI
            logger.warning("Không lưu được phân tích vào phiên dịch: %s", exc)

    def try_restore_saved_analysis(self, video_path: str, target_lang: str) -> bool:
        """[v3.23.45] Nạp lại phân tích ngữ cảnh đã lưu cho video (nếu còn hợp lệ).

        Phát :attr:`analysis_restored` kèm kết quả để UI tự điền các ô ngữ cảnh, glossary
        và visual cues. Trả True nếu khôi phục được, False nếu không có/không hợp lệ.
        """
        result = self.restore_analysis_for_video(video_path, target_lang)
        # [v3.23.48] Nạp lại luôn kết quả từng giai đoạn đã lưu (để so sánh được ngay).
        self.restore_stage_outputs_for_video(video_path)
        # [v3.23.49] Nếu có bản dịch đã lưu → hiển thị lại trên bảng kết quả.
        if self._translated_events:
            self.result_ready.emit(self.source_events, self.translated_events)
        if result is None:
            return False
        self.analysis_restored.emit(result)
        return True

    def restore_analysis_for_video(
        self, video_path: str, target_lang: str
    ) -> object | None:
        """[v3.23.15] Khôi phục phân tích ngữ cảnh đã lưu cho video (nếu còn hợp lệ).

        Trả về :class:`SubtitleContextAnalysis` nếu phiên đã có phân tích VÀ đầu vào
        (phụ đề nguồn + ngôn ngữ đích) khớp hash đã lưu; ngược lại trả None để báo
        cần phân tích lại.
        """
        if not video_path or not self._source_events:
            return None
        try:
            from subtitles_extractor.application.services.translation_session_hashing import (
                hash_analysis_input,
            )
            from subtitles_extractor.domain.ports.subtitle_translator_port import (
                SubtitleContextAnalysis,
            )

            store = self._container.translation_session_store
            session = store.get(video_path)
            if session is None:
                return None
            source_lines = [ev.text for ev in self._source_events]
            current_hash = hash_analysis_input(source_lines, target_lang or "")
            if not session.has_valid_analysis(current_hash):
                return None
            logger.info("Khôi phục phân tích ngữ cảnh đã lưu cho %s.", video_path)
            return SubtitleContextAnalysis(
                source_lang=session.analysis_source_lang,
                characters=session.analysis_characters,
                overview=session.analysis_overview,
                glossary=session.analysis_glossary,
                visual_cues=getattr(session, "analysis_visual_cues", "") or "",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không khôi phục được phân tích đã lưu: %s", exc)
            return None

    def _on_video_refs_ready(self, video_refs: object) -> None:
        """[v3.23.18] Lưu danh sách file cloud đã upload vào phiên dịch (theo video)."""
        video_path = getattr(self, "_pending_analysis_video_path", None)
        if not video_path or not video_refs:
            return
        try:
            from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
                CloudVideoFile,
            )

            cloud_files = [
                CloudVideoFile(
                    remote_name=getattr(r, "remote_name", ""),
                    start_sec=float(getattr(r, "start_sec", 0.0)),
                    end_sec=float(getattr(r, "end_sec", 0.0)),
                )
                for r in video_refs
                if getattr(r, "remote_name", "")
            ]
            if cloud_files:
                self._container.translation_session_store.save_cloud_files(
                    video_path, cloud_files
                )
                logger.info("Đã lưu %d file cloud vào phiên dịch cho %s.",
                            len(cloud_files), video_path)
        except Exception as exc:  # noqa: BLE001 — lưu phiên không được phá UI
            logger.warning("Không lưu được danh sách file cloud: %s", exc)

    def delete_cloud_files_for_video(self, video_path: str, api_key: str) -> int:
        """[v3.23.18] Xoá các file cloud đã upload của một video (giải phóng dung lượng).

        Returns:
            Số file đã xoá thành công.
        """
        if not video_path:
            return 0
        try:
            store = self._container.translation_session_store
            session = store.get(video_path)
            if session is None or not session.cloud_files:
                self.status_message.emit("Không có file cloud nào để xoá cho video này.")
                return 0
            remote_names = [cf.remote_name for cf in session.cloud_files if cf.remote_name]
            provider = self._container.make_video_context_provider(api_key)
            results = provider.delete_remote_files(remote_names)
            deleted = sum(1 for ok in results.values() if ok)
            store.clear_cloud_files(video_path)
            self.status_message.emit(f"Đã xoá {deleted}/{len(remote_names)} file cloud.")
            return deleted
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lỗi khi xoá file cloud: %s", exc)
            self.error_occurred.emit(f"Không xoá được file cloud: {exc}")
            return 0

    def _on_analyze_thread_finished(self) -> None:
        self._set_busy(False)
        if self._analyze_worker is not None:    # hasattr không cần, đã init trong __init__
            self._analyze_worker.deleteLater()
        self._analyze_worker = None

    def cancel_translation(self) -> None:
        """Huỷ tiến trình đang chạy (dịch hoặc phân tích ngữ cảnh)."""
        if self._worker is not None and self._is_busy:
            self._worker.request_cancel()
            self.status_message.emit("Đã yêu cầu huỷ, đang dừng ở ranh giới lô kế tiếp...")
        # [fix B2] Cũng huỷ analyze worker nếu đang chạy.
        if self._analyze_worker is not None and self._analyze_worker.isRunning():
            self._analyze_worker.request_cancel()
            self.status_message.emit("Đã yêu cầu huỷ phân tích ngữ cảnh...")

    def _on_worker_finished(self, response: TranslateSubtitlesResponse) -> None:
        # [Cross-Thread Contamination] Bỏ qua tín hiệu đến muộn từ worker CŨ đã bị
        # huỷ — nó không được phép ghi đè kết quả/trạng thái của worker hiện tại.
        if self.sender() is not self._worker:
            return
        self._translated_events = list(response.events)
        self._last_stage_count = len(response.completed_stages)
        self._last_elapsed_sec = response.elapsed_seconds
        # [v3.23.47] Giữ kết quả từng giai đoạn để so sánh (nếu có >1 giai đoạn).
        self._stage_outputs = dict(getattr(response, "stage_outputs", {}) or {})
        # [v3.23.48] Lưu BỀN VỮNG kết quả từng giai đoạn vào session để nạp lại lần sau.
        self._persist_stage_outputs()
        # [v3.23.55] Tích luỹ cặp câu đã dịch vào Bộ nhớ dịch (Translation Memory) theo
        # phim bộ — để các tập sau truy hồi làm tham chiếu, giữ nhất quán.
        self._accumulate_translation_memory()
        # [v3.23.56] Gộp glossary + roster + tóm tắt vào ngữ cảnh chung của phim bộ.
        ctx = getattr(self, "_last_translate_context", None)
        if ctx is not None:
            self.accumulate_series_context(
                getattr(self, "_last_translate_video_path", "") or "",
                getattr(ctx, "glossary", ""), getattr(ctx, "characters", ""),
                getattr(ctx, "overview", ""),
            )
        # [Q3] Thông báo resume nếu checkpoint được dùng
        if response.resumed_from is not None:
            self.resume_detected.emit(response.resumed_from.value)
        self.result_ready.emit(self.source_events, self.translated_events)
        elapsed_str = _format_elapsed(response.elapsed_seconds)
        resumed = f" (resume từ {response.resumed_from.value})" if response.resumed_from else ""
        self.status_message.emit(
            f"Dịch hoàn tất qua {self._last_stage_count} giai đoạn{resumed} — {elapsed_str}."
        )

    def _on_worker_failed(self, message: str) -> None:
        self.error_occurred.emit(message)

    def _on_worker_cancelled(self) -> None:
        # [v3.7 fix] Huỷ là kết thúc ÊM — KHÔNG phát error_occurred (tránh hiện
        # cảnh báo lỗi đỏ). Chỉ báo trạng thái trung tính.
        self.status_message.emit("Đã huỷ dịch. Giữ nguyên kết quả trước đó (nếu có).")
        self.translation_cancelled.emit()

    def _on_thread_finished(self) -> None:
        # [Cross-Thread Contamination] Chỉ worker HIỆN TẠI mới được phép tắt trạng
        # thái bận. Tín hiệu finished bay đến muộn từ worker cũ bị chặn hoàn toàn.
        if self.sender() is not self._worker:
            return
        self._set_busy(False)
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None

    def cleanup(self) -> None:
        """Huỷ + chờ tất cả worker an toàn khi đóng app (tránh crash QThread).

        [2.2] Ngắt mọi tín hiệu TRƯỚC khi dừng để slot không chạy sau khi VM/page đã
        bị huỷ (tránh segfault), rồi yêu cầu huỷ và chờ; quá hạn thì terminate.
        """
        for worker in (self._worker, self._analyze_worker, self._fetch_models_worker):
            if worker is None:
                continue
            # Ngắt toàn bộ tín hiệu của worker (an toàn nếu chưa kết nối).
            for sig_name in ("progress_changed", "finished_ok", "failed", "cancelled",
                             "progress_message", "finished"):
                sig = getattr(worker, sig_name, None)
                if sig is not None:
                    try:
                        sig.disconnect()
                    except (TypeError, RuntimeError):
                        pass
        for worker in (self._worker, self._analyze_worker):
            if worker is not None and worker.isRunning():
                if hasattr(worker, "request_cancel"):
                    worker.request_cancel()
                if not worker.wait(8000):
                    logger.warning("Worker %s không dừng kịp trong 8s — buộc dừng.",
                                   type(worker).__name__)
                    worker.terminate()
                    worker.wait(2000)
        if self._fetch_models_worker is not None and self._fetch_models_worker.isRunning():
            if not self._fetch_models_worker.wait(5000):
                logger.warning("FetchModelsWorker không dừng kịp trong 5s.")

    # ── Xuất tệp ────────────────────────────────────────────────────────────
    def export_translation(self, output_path: Path, output_format: SubtitleFormat) -> bool:
        """Xuất bản dịch ra tệp SRT / ASS.

        Dùng :class:`ExportSubtitlesUseCase` qua container — nhất quán với
        cách trang Biên tập xuất tệp.

        Returns:
            True nếu xuất thành công, False nếu thất bại (đã emit error_occurred).
        """
        if not self._translated_events:
            self.error_occurred.emit("Chưa có bản dịch để xuất.")
            return False
        try:
            export_uc = self._container.make_export_subtitles_use_case()
            export_uc.execute(
                events=self._translated_events,
                output_path=output_path,
                output_format=output_format,
            )
            logger.info(
                "Đã xuất %d dòng dịch → %s", len(self._translated_events), output_path
            )
            return True
        except (KeyError, OSError) as exc:
            self.error_occurred.emit(f"Không xuất được tệp: {exc}")
            return False
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"Lỗi khi xuất tệp: {exc}")
            return False

    def get_analysis_media_resolution(self) -> str:
        """[v3.23.129] Mức độ phân giải video khi phân tích ngữ cảnh/cues (low/medium/high)."""
        try:
            return (
                self._container.settings_service.current.translation
                .analysis_media_resolution
            )
        except AttributeError:
            return "medium"

    def set_analysis_media_resolution(self, level: str) -> None:
        """[v3.23.129] Lưu mức phân giải video phân tích (áp cho lần phân tích kế tiếp)."""
        if level not in ("low", "medium", "high"):
            level = "medium"
        self._container.settings_service.update(
            translation={"analysis_media_resolution": level}
        )

    def get_analysis_thinking_level(self) -> str:
        """[v3.23.140] Mức Thinking khi phân tích ngữ cảnh toàn cục (low/medium/high)."""
        try:
            return (
                self._container.settings_service.current.translation
                .analysis_thinking_level
            )
        except AttributeError:
            return "medium"

    def set_analysis_thinking_level(self, level: str) -> None:
        """[v3.23.140] Lưu mức Thinking phân tích (áp cho lần phân tích kế tiếp)."""
        if level not in ("low", "medium", "high"):
            level = "medium"
        self._container.settings_service.update(
            translation={"analysis_thinking_level": level}
        )

    def get_translation_parallel_batches(self) -> int:
        """[v3.23.149] Số batch dịch chạy song song mỗi giai đoạn (1 = tuần tự)."""
        try:
            return int(
                self._container.settings_service.current.translation
                .translation_parallel_batches
            )
        except (AttributeError, TypeError, ValueError):
            return 1

    def set_translation_parallel_batches(self, count: int) -> None:
        """[v3.23.149] Lưu mức song song dịch (áp dụng khi tạo translator kế tiếp).

        Raises:
            ValueError: không xảy ra — giá trị ngoài [1, 4] được kẹp về biên.
        """
        clamped = max(1, min(int(count or 1), 4))
        self._container.settings_service.update(
            translation={"translation_parallel_batches": clamped}
        )

    def get_quota_config(self) -> dict:
        """[v3.23.122] Lấy giới hạn quota: 'custom' (người dùng đặt) + 'defaults' (mặc định)."""
        mgr = self._container.gemini_quota_manager
        return {
            "custom": mgr.export_limits_dict(),
            "defaults": mgr.default_tier_limits(),
        }

    def save_quota_config(self, limits: dict) -> None:
        """[v3.23.122] Lưu giới hạn quota tuỳ chỉnh (áp ngay + ghi JSON để dùng lần sau)."""
        self._container.save_quota_limits(limits)

    def quota_status_for_keys(self, keys: list[str], model: str) -> list[dict]:
        """[v3.23.124] Tình trạng quota request/NGÀY của từng API key cho ``model``.

        Trả về danh sách cùng thứ tự ``keys``: ``{"used","limit","remaining"}``.
        """
        from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
            GeminiQuotaManager,
        )
        mgr = self._container.gemini_quota_manager
        out: list[dict] = []
        for key in keys:
            fp = GeminiQuotaManager.key_fingerprint(key)
            rem = mgr.get_remaining(model, key_id=fp)
            out.append({
                "used": rem["rpd_used"],
                "limit": rem["rpd_limit"],
                "remaining": rem["rpd_remaining"],
            })
        return out

    def export_bilingual(
        self,
        output_path: Path,
        output_format: SubtitleFormat,
        *,
        translation_on_top: bool = False,
    ) -> bool:
        """[v3.23.116] Xuất phụ đề SONG NGỮ (gốc + dịch) ra SRT / ASS.

        Ghép nguyên văn và bản dịch (căn theo index) thành mỗi câu hai dòng, rồi xuất
        qua cùng use-case như :meth:`export_translation`.

        Returns:
            True nếu xuất thành công, False nếu thất bại (đã emit error_occurred).
        """
        if not self._translated_events:
            self.error_occurred.emit("Chưa có bản dịch để xuất song ngữ.")
            return False
        if not self._source_events:
            self.error_occurred.emit("Chưa có phụ đề gốc để ghép song ngữ.")
            return False
        from subtitles_extractor.application.services.bilingual_builder import (
            build_bilingual_events,
        )
        try:
            bilingual = build_bilingual_events(
                self._source_events, self._translated_events,
                translation_on_top=translation_on_top,
            )
            export_uc = self._container.make_export_subtitles_use_case()
            export_uc.execute(
                events=bilingual, output_path=output_path, output_format=output_format,
            )
            logger.info("Đã xuất %d dòng song ngữ → %s", len(bilingual), output_path)
            return True
        except (KeyError, OSError) as exc:
            self.error_occurred.emit(f"Không xuất được tệp song ngữ: {exc}")
            return False
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"Lỗi khi xuất tệp song ngữ: {exc}")
            return False

    # ── Lấy danh sách model ──────────────────────────────────────────────────
    models_ready = Signal(list)   # list[str]

    def fetch_available_models(self, api_key: str) -> None:
        """Lấy danh sách model Gemini trên thread nền.

        **Lỗi đã sửa**: worker trước đây là biến cục bộ → Python GC huỷ ngay
        khi method return trong khi thread vẫn chạy → crash
        ``QThread: Destroyed while thread is still running``.
        Nay lưu vào ``self._fetch_models_worker`` để giữ strong reference.
        """
        if not api_key.strip():
            self.error_occurred.emit("Hãy nhập API Key trước khi tải danh sách model.")
            return
        # Guard: không start thêm nếu đang có worker chạy
        if self._fetch_models_worker is not None and self._fetch_models_worker.isRunning():
            self.status_message.emit("Đang tải danh sách model, vui lòng chờ…")
            return

        from subtitles_extractor.presentation.workers.fetch_models_worker import FetchModelsWorker
        translator = self._container.make_subtitle_translator(api_key, retry_count=1)
        self._fetch_models_worker = FetchModelsWorker(translator)   # ← strong ref
        self._fetch_models_worker.models_ready.connect(self.models_ready.emit)
        self._fetch_models_worker.failed.connect(self.error_occurred.emit)
        self._fetch_models_worker.finished.connect(self._on_fetch_models_finished)
        self.status_message.emit("Đang tải danh sách model từ Gemini API…")
        self._fetch_models_worker.start()

    def _on_fetch_models_finished(self) -> None:
        """Dọn dẹp _fetch_models_worker sau khi thread kết thúc."""
        if self._fetch_models_worker is not None:
            self._fetch_models_worker.deleteLater()
        self._fetch_models_worker = None

    # ── Gửi kết quả sang trang Biên tập ─────────────────────────────────────
    def push_to_editor(self) -> bool:
        """Gửi bản dịch sang trang Biên tập qua callback đã được tiêm."""
        if not self._translated_events:
            self.error_occurred.emit("Chưa có bản dịch để gửi sang trang Biên tập.")
            return False
        if self._push_to_editor_callback is None:
            self.error_occurred.emit("Chưa kết nối được với trang Biên tập.")
            return False
        try:
            self._push_to_editor_callback(self._translated_events)
            self.status_message.emit(
                f"Đã gửi {len(self._translated_events)} dòng dịch sang trang Biên tập."
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"Lỗi gửi sang Biên tập: {exc}")
            return False

    def set_push_to_editor_callback(
        self, callback: "Callable[[list[SubtitleEvent]], None] | None"
    ) -> None:
        """Tiêm callback gửi bản dịch sang trang Biên tập."""
        self._push_to_editor_callback = callback

    # ── Properties cho Page tham chiếu không dùng private ───────────────────
    @property
    def last_stage_count(self) -> int:
        return self._last_stage_count

    @property
    def last_elapsed_sec(self) -> float:
        return self._last_elapsed_sec

    # ── Nội bộ ───────────────────────────────────────────────────────────────
    def _set_busy(self, busy: bool) -> None:
        if busy != self._is_busy:
            self._is_busy = busy
            self.busy_changed.emit(busy)


__all__ = ["TranslatePageViewModel"]
