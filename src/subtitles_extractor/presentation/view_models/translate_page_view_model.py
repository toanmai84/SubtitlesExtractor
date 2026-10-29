"""ViewModel cho trang Dịch phụ đề.

Giữ trạng thái nguồn/đích, điều phối ``TranslateWorker`` và phát signal cho View.
Không chứa widget — chỉ logic trình bày + cầu nối tới use case qua container.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import replace as _copy_event
from pathlib import Path

from PySide6.QtCore import QTimer, QObject, Signal

from subtitles_extractor.application.services.analysis_completeness import (
    describe_incomplete_analysis,
)
from subtitles_extractor.application.services.gemini_pricing import (
    PricingTable,
    describe_token_usage,
    estimate_cost,
)
from subtitles_extractor.application.services.token_usage import (
    TokenUsage,
    TokenUsageLedger,
)
from subtitles_extractor.application.services.final_line_edits import (
    FINAL_EDITS_STAGE_ID,
    FinalLineEdit,
    apply_final_edits,
    discard_final_edits,
    final_edits_from_json,
    final_edits_to_json,
    record_final_edit,
)
from subtitles_extractor.application.services.cue_completion import (
    SpeakerCoverage,
    speaker_coverage,
)
from subtitles_extractor.application.services.name_reading_check import (
    NameReadings,
    apply_name_corrections,
    correct_context_names,
    describe_fix,
    name_corrections,
)
from subtitles_extractor.application.services.native_address import (
    describe_changes as describe_native_changes,
)
from subtitles_extractor.application.services.native_address import (
    nativize_context,
    nativize_lines,
)
from subtitles_extractor.application.services.title_reconciliation import (
    describe_title_fix,
    reconcile_context_titles,
)
from subtitles_extractor.application.services.stage_cache import (
    STAGE_LABELS,
    StageCacheScope,
)
from subtitles_extractor.application.services.stage_checkpoint import (
    line_input_digest,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesResponse,
    _lines_to_events_helper,
    build_stage_cache_scope,
)
from subtitles_extractor.application.use_cases.complete_speakers import (
    SpeakerCompletionResult,
)
from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat
from subtitles_extractor.presentation.view_models.stage_cache_session import (
    StageCacheSession,
)
from subtitles_extractor.presentation.view_models.stage_cache_texts import (
    describe_cache_reuse,
)
from subtitles_extractor.presentation.workers.translate_worker import TranslateWorker
from collections.abc import Callable

logger = logging.getLogger(__name__)

#: [v3.23.621] Model phân tích dự phòng khi model đã chọn hết hạn mức ngày (RPD 500/key).
ANALYSIS_FALLBACK_MODEL = "gemini-3.1-flash-lite"


def _format_elapsed(seconds: float) -> str:
    """Định dạng thời gian thực thi dạng 'mm:ss' hoặc 'Xs'."""
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}p{s:02d}s"
    return f"{seconds:.1f}s"


def _capitalize_retranslated(
    current_texts: list[str], new_texts: dict[int, str], target_lang: str
) -> dict[int, str]:
    """[v3.23.616] Viết hoa chữ đầu các dòng VỪA DỊCH LẠI khi chúng mở câu mới.

    Lượt dịch lại chỉ thấy tập dòng chọn (dòng kề trong tập không phải dòng kề trong
    phim) nên không tự sửa được; ở đây dòng trước là dòng thật của bản dịch. Dòng không
    được chọn giữ nguyên.

    Args:
        current_texts: Bản dịch hiện tại của cả phim, theo vị trí.
        new_texts: ``vị trí → câu dịch lại``.
        target_lang: Ngôn ngữ đích.

    Returns:
        ``new_texts`` với chữ đầu đã viết hoa ở các dòng cần sửa.
    """
    from subtitles_extractor.application.services.sentence_case_guard import (
        capitalize_sentence_starts,
        supports_sentence_case,
    )

    if not new_texts or not supports_sentence_case(target_lang):
        return new_texts
    merged = list(current_texts)
    for position, text in new_texts.items():
        merged[position] = text
    fixed = capitalize_sentence_starts(merged)
    return {
        position: fixed.get(position, text) for position, text in new_texts.items()
    }


#: Model quét Visual Cues khi chưa bật giai đoạn nào (không xảy ra khi dịch thật).
DEFAULT_VISUAL_CUES_MODEL = "gemini-flash-lite-latest"

BUSY_WATCHDOG_INTERVAL_MS = 5000
"""Chu kỳ kiểm trạng thái bận (ms) — đủ thưa để không tốn gì, đủ dày để người dùng không chờ lâu."""

BUSY_WATCHDOG_CONFIRM_TICKS = 2
"""[v3.23.550] Số nhịp LIÊN TIẾP phải thấy "bận mà không luồng nào sống" mới tự mở khoá.

Một nhịp đơn lẻ không đủ tin: tín hiệu kết thúc hợp lệ có thể vẫn đang xếp hàng. Hai nhịp = mở
khoá sau 5-10 giây trong ca kẹt thật — vẫn hơn hẳn phải End Task."""



def _tm_video_key(video_path: str) -> str:
    """[v3.23.654] Khoá video cho Bộ nhớ dịch (đường dẫn chuẩn hoá, không phân biệt hoa thường trên Windows)."""
    import os

    try:
        return os.path.normcase(os.path.abspath(str(video_path or "")))
    except (TypeError, ValueError):
        return str(video_path or "")


def _drop_own_legacy_entries(entries: list, source_lines: list[str], min_overlap: float = 0.3) -> list:
    """[v3.23.654] Mục bản cũ (không ghi video) nhưng phần lớn trùng đúng câu của phụ đề đang dịch → bỏ các câu trùng.

    Bộ nhớ lưu trước 3.23.654 không biết câu thuộc video nào; dịch lại cùng một tập thì ≥ 30% câu của tập nằm nguyên
    văn trong bộ nhớ — khi đó các câu trùng nguyên văn là bản dịch cũ của chính tập này.
    """
    current = {line.strip() for line in source_lines if line and line.strip()}
    if not current or not entries:
        return entries
    overlap = sum(1 for entry in entries if entry.source_text.strip() in current)
    if overlap < min_overlap * len(current):
        return entries
    return [entry for entry in entries if entry.source_text.strip() not in current]

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
        resume_detected:      ``(str,)``           — [v3.23.608] tóm tắt phần lấy từ bộ
                                                   nhớ đệm giai đoạn của lượt dịch.
        translation_finished: ``()``               — [v3.23.609] một lượt dịch vừa xong.
    """

    source_changed = Signal(int)
    result_ready = Signal(object, object)
    progress_changed = Signal(int, str)
    busy_changed = Signal(bool)
    error_occurred = Signal(str)
    analyze_error = Signal(str)         # phân biệt lỗi dịch vs lỗi phân tích
    status_message = Signal(str)
    translation_cancelled = Signal()
    resume_detected = Signal(str)       # phần dùng lại từ bộ nhớ đệm giai đoạn
    # [v3.23.609] Một lượt DỊCH (không phải nạp lại phiên / dịch lại vài dòng / dùng một
    # giai đoạn) vừa xong. ``result_ready`` chỉ để vẽ bảng: nó còn phát khi mở lại phiên đã
    # lưu, nên dịch cả bộ bám vào nó đã ghi nhầm bản dịch cũ thành bản dịch của tập mới.
    translation_finished = Signal()
    analysis_restored = Signal(object)  # [v3.23.45] phát khi nạp lại phân tích đã lưu
    visual_cues_topped_up = Signal(str)  # [v3.23.559] cues đã được quét bổ sung khi dịch
    speaker_completion_done = Signal(str)  # [v3.23.601] kết quả nút "Bổ sung người nói"
    context_names_corrected = Signal(object)  # [v3.23.567] ngữ cảnh đã sửa phiên âm tên

    def __init__(self, container: ApplicationContainer) -> None:
        super().__init__()
        self._container = container
        self._source_events: list[SubtitleEvent] = []
        self._translated_events: list[SubtitleEvent] = []
        self._stage_outputs: dict = {}  # [v3.23.47] kết quả từng giai đoạn để so sánh
        self._worker: TranslateWorker | None = None
        self._analyze_worker = None
        self._fetch_models_worker = None       # giữ strong ref tránh GC crash
        # [v3.23.623] Gán người nói bằng chữ (+ giọng nói nếu có video cạnh phụ đề).
        self._text_cues = True
        self._voice_source_path = ""
        self._is_busy = False
        # [v3.23.535] Đồng hồ canh chống kẹt trạng thái bận (xem ``_check_busy_watchdog``).
        self._busy_watchdog = QTimer(self)
        self._busy_watchdog.setInterval(BUSY_WATCHDOG_INTERVAL_MS)
        self._busy_watchdog.timeout.connect(self._check_busy_watchdog)
        self._busy_idle_ticks = 0
        self._last_stage_count: int = 0
        self._last_elapsed_sec: float = 0.0
        self._push_to_editor_callback = None
        # [v3.23.567] Cụm tên sai → đúng tìm thấy khi nạp lại phân tích, để sửa bản dịch.
        self._restored_name_corrections: dict[str, str] = {}
        # [v3.23.609] Đếm số lần thay phụ đề nguồn + kết quả độ phủ người nói gần nhất —
        # trang hỏi số dòng thiếu người nói mỗi lần chọn dòng / gõ phím, mà mỗi lần tính
        # là giải mã cả JSON Visual Cues (hàng nghìn mục).
        self._source_version = 0
        # [v3.23.609] Dòng sửa tay trên bảng kết quả (theo chỉ số câu gốc) + ngôn ngữ đích
        # của bản dịch đang hiện.
        self._final_edits: dict[int, FinalLineEdit] = {}
        self._result_target_lang = ""
        # [v3.23.610] Mốc sổ token lúc bắt đầu việc đang chạy + báo cáo của việc vừa xong.
        self._token_mark: dict[str, TokenUsage] = {}
        self._last_token_usage: dict[str, TokenUsage] = {}
        self._last_token_report = ""
        self._coverage_cache: tuple[tuple[int, str], SpeakerCoverage | None] | None = None

    # ── Truy cập trạng thái ──────────────────────────────────────────────
    @property
    def source_events(self) -> list[SubtitleEvent]:
        """Bản SAO phụ đề nguồn.

        [v3.23.609] Bản sao từng dòng, không chỉ bản sao danh sách: trang Biên tập/TTS
        nhận các dòng này và đánh số lại / sửa tại chỗ — trước đây là sửa luôn dữ liệu
        của trang Dịch (và ngược lại).
        """
        return [_copy_event(event) for event in self._source_events]

    @property
    def translated_events(self) -> list[SubtitleEvent]:
        """Bản SAO bản dịch hiện hành (xem :attr:`source_events`)."""
        return [_copy_event(event) for event in self._translated_events]

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    @property
    def has_source(self) -> bool:
        return bool(self._source_events)

    @property
    def has_result(self) -> bool:
        return bool(self._translated_events)

    @property
    def stage_output_count(self) -> int:
        """[v3.23.609] Số giai đoạn có kết quả để so sánh (không dựng lại danh sách chữ —
        trang gọi mỗi lần chọn dòng / gõ phím vào bảng thuật ngữ)."""
        return len(self._stage_outputs)

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
        # [v3.23.610] Token của việc chạy gần nhất (để đối chiếu token suy nghĩ, cache…).
        last_usage = getattr(self, "_last_token_usage", {}) or {}
        if last_usage:
            bundle["token_usage_last_run"] = {
                model: {
                    "requests": usage.requests, "prompt": usage.prompt,
                    "cached": usage.cached, "thoughts": usage.thoughts,
                    "output": usage.output,
                }
                for model, usage in last_usage.items()
            }

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
        except (OSError, ValueError, sqlite3.Error) as exc:
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
        except (OSError, ValueError, sqlite3.Error) as exc:
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
            saved = store.add_entries(series_key, entries, video_key=_tm_video_key(video_path))
            logger.info(
                "Đã lưu %d cặp câu vào Bộ nhớ dịch cho phim bộ '%s'.", saved, series_key
            )
        except (OSError, ValueError, ImportError, sqlite3.Error) as exc:
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
        except (OSError, ValueError, ImportError, sqlite3.Error) as exc:
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
        except (OSError, ValueError, ImportError, sqlite3.Error) as exc:
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
                retrieve_for_lines,
            )
            series_key = derive_series_key(video_path)
            if not series_key:
                return ""
            store = self._container.translation_memory_store
            # [v3.23.654] Không lấy câu của CHÍNH video đang dịch (bộ nhớ tự nuôi chính nó, kể cả lỗi cũ).
            entries = store.get_entries(series_key, exclude_video=_tm_video_key(video_path))
            entries = _drop_own_legacy_entries(entries, source_lines)
            if not entries:
                return ""
            # Lấy mẫu các câu DÀI nhất (giàu danh từ riêng/thuật ngữ) để giảm số truy vấn.
            query_lines = sorted(
                (ln for ln in source_lines if ln.strip()), key=len, reverse=True
            )[:max_query_lines]
            collected = retrieve_for_lines(
                query_lines, entries, per_line_k=per_line_k, max_references=max_references,
            )
            return format_reference_block(collected, max_entries=max_references)
        except (OSError, ValueError, ImportError, sqlite3.Error) as exc:
            logger.warning("Không truy hồi được Bộ nhớ dịch: %s", exc)
            return ""

    def _persist_current_result(self) -> None:
        """[v3.23.49] Lưu BẢN DỊCH HIỆN HÀNH (sau khi sửa tay / dịch lại dòng) vào session
        dưới stage_id 'final', để không mất khi đóng app và được ưu tiên khi nạp lại."""
        video_path = getattr(self, "_last_translate_video_path", "") or ""
        if not video_path or not self._translated_events:
            return
        try:
            from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
                StageResult,
            )
            store = self._container.translation_session_store
            lines_json = self._session_lines_json(self._translated_events)
            store.save_stage(
                video_path,
                StageResult(stage_id="final", input_hash="", lines_json=lines_json),
            )
        except (OSError, ValueError, ImportError, sqlite3.Error) as exc:
            logger.warning("Không lưu được bản dịch hiện hành vào session: %s", exc)

    @property
    def name_readings(self) -> NameReadings | None:
        """[v3.23.567] Bảng âm Hán-Việt (cho file chuẩn đoán); None nếu không có."""
        readings = getattr(self._container, "name_readings", None)
        return readings if isinstance(readings, NameReadings) else None

    def _correct_context_names(
        self, context: object, video_path: str = ""
    ) -> tuple[object, dict[str, str]]:
        """[v3.23.567] Sửa phiên âm tên sai âm Hán-Việt trong ngữ cảnh, lưu lại vào phiên.

        Phiên đã phân tích bằng bản cũ (vd bảng nhân vật ghi "Liễu Mộc Vân" cho 柳慕云)
        được sửa ngay khi nạp lại/khi dịch, không phải tốn quota phân tích lại cả phim.

        Args:
            context: ``SubtitleContextAnalysis`` hoặc ``TranslationContext``.
            video_path: Video của phiên để lưu bản đã sửa (rỗng = không lưu).

        Returns:
            (ngữ cảnh đã sửa, ánh xạ cụm tên sai → đúng để sửa tiếp văn bản khác như bản
            dịch đã có hay bộ nhớ dịch). Không có gì để sửa thì trả ĐÚNG đối tượng ngữ
            cảnh cũ và ``{}``. [v3.23.589] Ánh xạ có thể rỗng mà ngữ cảnh vẫn đổi (chỉ
            đổi cách gọi chức danh) — nơi gọi so đối tượng trả về để biết có cần báo
            giao diện.
        """
        readings = self.name_readings
        corrected = context
        fixes: list = []
        title_fixes: list = []
        if readings is not None:
            corrected, fixes = correct_context_names(context, readings)  # type: ignore[type-var]
            # [v3.23.589] Cùng chỗ, cùng lý do: phiên đã lưu có bảng nhân vật gọi chức
            # danh khác bảng thuật ngữ ("gọi 'Sở trưởng'" với "所長 = Trưởng đồn công an")
            # được đổi theo bảng thuật ngữ mà không phải phân tích lại.
            corrected, title_fixes = reconcile_context_titles(corrected, readings)
        # [v3.23.612] Tuỳ chọn xưng hô thuần Việt: phân tích đã lưu (kiểu "Kiếm ca") được
        # đổi ngay, không phải phân tích lại.
        native_changes: list = []
        if self.get_native_address():
            corrected, native_changes = nativize_context(corrected)
        if not fixes and not title_fixes and not native_changes:
            return context, {}
        corrections = name_corrections(fixes, context.characters)  # type: ignore[attr-defined]
        for fix in fixes:
            logger.warning("Sửa phiên âm tên sai âm Hán-Việt — %s.", describe_fix(fix))
        for title_fix in title_fixes:
            logger.warning(
                "Bảng nhân vật gọi chức danh khác bảng thuật ngữ — đổi theo bảng thuật "
                "ngữ: %s.",
                describe_title_fix(title_fix),
            )
        messages: list[str] = []
        if fixes:
            messages.append(
                "Đã sửa phiên âm tên theo âm Hán-Việt: "
                + "; ".join(describe_fix(fix) for fix in fixes)
            )
        if title_fixes:
            messages.append(
                "Đã đổi cách gọi chức danh trong bảng nhân vật theo bảng thuật ngữ: "
                + "; ".join(describe_title_fix(fix) for fix in title_fixes)
                + " (muốn giữ cách gọi cũ thì sửa bảng thuật ngữ)"
            )
        if native_changes:
            logger.info(
                "Xưng hô thuần Việt trong ngữ cảnh: %s.",
                describe_native_changes(native_changes),
            )
            messages.append(
                "Đã đổi tiếng gọi sang thuần Việt trong bảng nhân vật/tóm tắt/thuật ngữ: "
                + describe_native_changes(native_changes)
            )
        self.status_message.emit(". ".join(messages))
        if video_path:
            try:
                self._container.translation_session_store.save_analysis_texts(
                    video_path,
                    characters=corrected.characters,
                    overview=corrected.overview,
                    glossary=corrected.glossary,
                    visual_cues=corrected.visual_cues,
                )
            except (OSError, ValueError, sqlite3.Error) as exc:
                logger.warning("Không lưu được ngữ cảnh đã sửa tên vào phiên: %s", exc)
        return corrected, corrections

    def save_context_edits(
        self, video_path: str, *, characters: str, overview: str, glossary: str, visual_cues: str,
    ) -> None:
        """[v3.23.652] Lưu sửa tay các ô ngữ cảnh vào phiên dịch (giữ hash phân tích như cũ)."""
        if not video_path:  # [v3.23.652] xoá trắng các ô cũng là một sửa tay cần lưu
            return
        try:
            self._container.translation_session_store.save_analysis_texts(
                video_path, characters=characters, overview=overview, glossary=glossary, visual_cues=visual_cues,
            )
        except (OSError, ValueError, sqlite3.Error) as exc:
            logger.warning("Không lưu được ngữ cảnh sửa tay vào phiên: %s", exc)

    def get_native_address(self) -> bool:
        """[v3.23.612] Đang bật xưng hô thuần Việt ("anh Kiếm" thay "Kiếm ca")?"""
        try:
            return bool(
                self._container.settings_service.current.translation.native_address
            )
        except AttributeError:
            return False

    def get_clip_video_to_batch(self) -> bool:
        """[v3.23.648] Dịch đính video chỉ gửi khoảng video của lô (mặc định bật)?"""
        try:
            return bool(self._container.settings_service.current.translation.clip_video_to_batch)
        except AttributeError:
            return True

    def set_clip_video_to_batch(self, enabled: bool) -> None:
        """[v3.23.648] Lưu tuỳ chọn cắt video theo lô (áp dụng từ lượt dịch kế tiếp)."""
        self._container.settings_service.update(translation={"clip_video_to_batch": bool(enabled)})

    def set_native_address(self, enabled: bool) -> None:
        """[v3.23.612] Lưu tuỳ chọn xưng hô thuần Việt (cho lượt phân tích/dịch sau)."""
        self._container.settings_service.update(
            translation={"native_address": bool(enabled)}
        )

    def apply_native_address(self, context: object, video_path: str = "") -> int:
        """[v3.23.612] Đổi NGAY ngữ cảnh và bản dịch đang có sang xưng hô thuần Việt.

        Gọi khi người dùng vừa bật tuỳ chọn: không phải phân tích hay dịch lại (không tốn
        hạn mức). Ngữ cảnh đã đổi được phát qua :attr:`context_names_corrected` để giao
        diện điền lại các ô, và được lưu vào phiên như khi sửa phiên âm tên.

        Args:
            context: Ngữ cảnh đang hiển thị (``TranslationContext``).
            video_path: Video của phiên để lưu (rỗng = không lưu ngữ cảnh).

        Returns:
            Số dòng bản dịch đã đổi (0 khi đang dịch/phân tích: kết quả sắp về sẽ ghi
            đè, lượt kế tiếp mới áp tuỳ chọn).
        """
        if self._is_busy:
            self.status_message.emit(
                "Đã lưu cách gọi tên — áp dụng từ lượt phân tích/dịch kế tiếp."
            )
            return 0
        corrected, _ = self._correct_context_names(context, video_path)
        if corrected is not context:
            self.context_names_corrected.emit(corrected)
        changed = self._nativize_translation(corrected)
        if changed:
            self.result_ready.emit(self.source_events, self.translated_events)
        return changed

    def _nativize_translation(self, context: object) -> int:
        """[v3.23.612] Đổi tiếng gọi Hán-Việt trong BẢN DỊCH đã có (không tốn hạn mức).

        Từ vai vế theo người nói/người nghe của Visual Cues và bảng nhân vật — như bước
        cuối của lượt dịch (xem ``TranslateSubtitlesUseCase._apply_native_address``).

        Args:
            context: Ngữ cảnh (bảng nhân vật, Visual Cues).

        Returns:
            Số dòng đã đổi (đã lưu lại vào phiên nếu > 0).
        """
        if not self._translated_events or not self.get_native_address():
            return 0
        from subtitles_extractor.application.services.visual_cue_serializer import (
            deserialize_visual_cues,
        )

        cues = deserialize_visual_cues(str(getattr(context, "visual_cues", "") or ""))
        sources = {
            position: event.text
            for position, event in enumerate(self._source_events, start=1)
        }
        changed, changes = nativize_lines(
            {
                position: event.text
                for position, event in enumerate(self._translated_events, start=1)
            },
            sources,
            str(getattr(context, "characters", "") or ""),
            speakers={cue.line_no: cue.speaker for cue in cues if cue.speaker},
            listeners={cue.line_no: cue.addressee for cue in cues if cue.addressee},
        )
        if not changed:
            return 0
        self._translated_events = [
            _copy_event(event, text=changed[position]) if position in changed else event
            for position, event in enumerate(self._translated_events, start=1)
        ]
        logger.info(
            "Xưng hô thuần Việt: đổi %d dòng bản dịch đã có — %s.",
            len(changed), describe_native_changes(changes),
        )
        self.status_message.emit(
            f"Đã đổi tiếng gọi sang thuần Việt ở {len(changed)} dòng bản dịch: "
            + describe_native_changes(changes)
        )
        self._persist_current_result()
        return len(changed)

    def _correct_names_in_translation(self, corrections: dict[str, str]) -> int:
        """[v3.23.567] Sửa tên sai trong BẢN DỊCH đã có (không tốn quota dịch lại).

        Bản dịch v3.23.566 chép đúng bảng nhân vật sai: 44 dòng "Liễu Mộc Vân"/"Mộc Vân".
        Chỉ thay các cụm ≥ 2 âm tiết của tên sai nên không đụng từ thường.

        Args:
            corrections: Ánh xạ cụm sai → đúng từ :meth:`_correct_context_names`.

        Returns:
            Số dòng đã sửa (đã lưu lại vào phiên nếu > 0).
        """
        if not corrections:
            return 0
        changed = 0
        corrected: list[SubtitleEvent] = []
        for event in self._translated_events:
            fixed = apply_name_corrections(event.text, corrections)
            if fixed != event.text:
                event = _copy_event(event, text=fixed)
                changed += 1
            corrected.append(event)
        self._translated_events = corrected
        if changed:
            logger.info("Đã sửa tên sai âm Hán-Việt ở %d dòng bản dịch đã lưu.", changed)
            self._persist_current_result()
        return changed

    def _persist_topped_up_cues(self, visual_cues: str, video_path: str = "") -> None:
        """[v3.23.559] Lưu Visual Cues đã quét bổ sung vào phiên, báo trang cập nhật.

        Không lưu thì lần dịch sau lại quét bổ sung từ đầu (tốn quota), còn file chuẩn
        đoán ngữ cảnh vẫn báo độ phủ cũ — người dùng không biết cues đã được lấp.

        Args:
            visual_cues: JSON Visual Cues đã gộp; rỗng nghĩa là không có gì bổ sung.
            video_path: [v3.23.601] Video của phiên; rỗng = video của lượt dịch gần nhất.
        """
        if not visual_cues:
            return
        video_path = video_path or getattr(self, "_last_translate_video_path", "") or ""
        # [v3.23.655] LƯU TRƯỚC rồi mới báo trang: trang gộp 3 chiều với sửa tay trong lúc chạy rồi lưu bản đã gộp —
        # lưu sau khi báo thì bản chưa gộp đè lên bản đã gộp (mất sửa tay).
        if video_path:
            try:
                self._container.translation_session_store.save_visual_cues(
                    video_path, visual_cues
                )
            except (OSError, ValueError, sqlite3.Error) as exc:
                logger.warning("Không lưu được Visual Cues bổ sung vào session: %s", exc)
        self.visual_cues_topped_up.emit(visual_cues)

    def _persist_stage_outputs(self) -> None:
        """[v3.23.48] Lưu kết quả từng giai đoạn vào session store (theo video_path)."""
        video_path = getattr(self, "_last_translate_video_path", "") or ""
        if not video_path or not self._stage_outputs:
            return
        try:
            from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
                StageResult,
            )
            store = self._container.translation_session_store
            for kind, events in self._stage_outputs.items():
                stage_id = kind.value if hasattr(kind, "value") else str(kind)
                lines_json = self._session_lines_json(events)
                store.save_stage(
                    video_path,
                    StageResult(stage_id=stage_id, input_hash="", lines_json=lines_json),
                )
            logger.info(
                "Đã lưu %d giai đoạn dịch vào session cho %s.",
                len(self._stage_outputs), video_path,
            )
        except (OSError, ValueError, ImportError, sqlite3.Error) as exc:
            logger.warning("Không lưu được kết quả giai đoạn vào session: %s", exc)

    def _session_lines_json(self, events: list[SubtitleEvent]) -> str:
        """[v3.23.609] Dòng lưu vào phiên: chỉ số, câu dịch, vân tay câu GỐC của dòng.

        Vân tay câu gốc (``h``) cho phép lúc mở lại kiểm từng dòng còn khớp phụ đề nguồn
        không — trước đây chỉ so số dòng (lệch ≤ 20%), nên bản dịch của phim KHÁC cùng
        cỡ hay của phụ đề đã thêm/bớt dòng ở giữa vẫn được gán theo số thứ tự.
        """
        import json

        from subtitles_extractor.application.services.stage_checkpoint import (
            line_input_digest,
        )

        source_by_index = {ev.index: ev.text for ev in self._source_events}
        return json.dumps(
            [
                {
                    "index": ev.index,
                    "text": ev.text,
                    "h": line_input_digest(source_by_index.get(ev.index, "")),
                }
                for ev in events
            ],
            ensure_ascii=False,
        )

    def _saved_lines_match_source(self, rows: list[dict]) -> bool:
        """[v3.23.609] Các dòng đã lưu có thuộc phụ đề nguồn đang nạp không.

        Dòng có vân tay câu gốc: ≥ 90% dòng phải khớp câu gốc cùng chỉ số (cho phép sửa
        lỗi chính tả vài dòng sau khi dịch). Dòng của bản cũ (không có vân tay): giữ cách
        cũ — số dòng lệch không quá 20% (tối thiểu 5).
        """
        from subtitles_extractor.application.services.stage_checkpoint import (
            line_input_digest,
        )

        if not rows or not self._source_events:
            return False
        with_digest = [row for row in rows if isinstance(row, dict) and row.get("h")]
        if with_digest:
            source_by_index = {ev.index: ev.text for ev in self._source_events}
            matched = sum(
                1 for row in with_digest
                if row.get("index") in source_by_index
                and line_input_digest(source_by_index[row.get("index")]) == row.get("h")
            )
            return matched >= 0.9 * len(with_digest)
        saved_max = max(int(row.get("index", 0) or 0) for row in rows)
        source_max = max(ev.index for ev in self._source_events)
        return abs(source_max - saved_max) <= max(5, source_max * 0.2)

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
            restored: dict = {}
            final_events: list[SubtitleEvent] = []
            self._final_edits = {}
            for stage in session.stages:
                if stage.stage_id == FINAL_EDITS_STAGE_ID:
                    self._final_edits = final_edits_from_json(stage.lines_json)
                    continue
                try:
                    lines = json.loads(stage.lines_json) if stage.lines_json else []
                except json.JSONDecodeError:
                    continue
                # [v3.23.50]/[v3.23.609] Không gán bản dịch của phụ đề KHÁC theo số thứ tự.
                if not isinstance(lines, list) or not self._saved_lines_match_source(lines):
                    logger.info(
                        "Bỏ qua nạp lại '%s' của %s: không khớp phụ đề nguồn đang nạp.",
                        stage.stage_id, video_path,
                    )
                    continue
                events = self._events_aligned_to_source(lines)
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
            # Dòng sửa của câu gốc đã đổi thì không còn hiệu lực.
            source_by_index = {ev.index: ev for ev in self._source_events}
            self._final_edits = {
                index: edit for index, edit in self._final_edits.items()
                if index in source_by_index
                and line_input_digest(source_by_index[index].text) == edit.source_digest
            }
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
            # [v3.23.580] Phiên chỉ lưu CHỮ, còn thời gian lấy lại từ phụ đề nguồn — nên
            # bản dịch mở lại mất phần giãn giờ đọc đã làm lúc dịch xong. Ca thật: cùng
            # một bản dịch 3902 dòng, mở lại thì 758 dòng đọc không kịp (vừa dịch: 426).
            if final_events or restored:
                self._apply_reading_time_extension()
            return len(restored)
        except (OSError, ValueError, ImportError, sqlite3.Error) as exc:
            logger.warning("Không nạp được kết quả giai đoạn từ session: %s", exc)
            return 0

    def _events_aligned_to_source(self, rows: list[dict]) -> list[SubtitleEvent]:
        """[v3.23.609] Dòng đã lưu → bản dịch SONG SONG với phụ đề nguồn.

        Mọi thao tác theo vị trí (bảng ghép câu gốc với bản dịch, "Dịch lại các dòng đã
        chọn", sửa một dòng) giả định dòng thứ i của bản dịch là dòng thứ i của nguồn.
        Trước đây dòng nguồn không có bản lưu (phụ đề được thêm dòng sau khi dịch) bị bỏ
        khỏi bản dịch → từ đó trở đi lệch hết. Nay dòng thiếu để trống, đúng vị trí.

        Args:
            rows: Dòng đã lưu (``index`` = chỉ số câu gốc, ``text``).

        Returns:
            Một dòng cho mỗi câu gốc, cùng thứ tự.
        """
        text_by_index: dict[int, str] = {}
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("index"), int):
                text_by_index[row["index"]] = str(row.get("text", "") or "")
        return [
            SubtitleEvent(
                index=source.index,
                text=text_by_index.get(source.index, ""),
                interval=source.interval,
            )
            for source in self._source_events
        ]

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
    def update_translation_text(
        self, line_index: int, new_text: str, *, target_lang: str = ""
    ) -> bool:
        """[v3.23.46] Sửa bản dịch của MỘT dòng tại chỗ (theo index 0-based trong list).

        [v3.23.609] Dòng sửa được ghi nhớ và áp lại sau mỗi lần bấm Dịch lại (xem
        :mod:`final_line_edits`).

        Args:
            line_index: Vị trí dòng (từ 0).
            new_text: Câu mới.
            target_lang: Ngôn ngữ đích đang chọn (để không áp câu sửa vào bản dịch sang
                ngôn ngữ khác); rỗng = ngôn ngữ của lượt dịch gần nhất.

        Returns:
            True nếu cập nhật thành công. Không gọi AI — chỉ sửa dữ liệu hiện có.
        """
        if not (0 <= line_index < len(self._translated_events)):
            return False
        self._set_final_texts({line_index: new_text}, target_lang)
        self._persist_current_result()  # [v3.23.49] lưu để không mất khi đóng app
        return True

    def _set_final_texts(self, texts: dict[int, str], target_lang: str) -> None:
        """Đặt câu cho các dòng (theo vị trí) và ghi chúng là dòng sửa tay."""
        language = target_lang or self._result_target_lang
        for position, text in texts.items():
            current = self._translated_events[position]
            source = (
                self._source_events[position]
                if position < len(self._source_events)
                else current
            )
            self._final_edits = record_final_edit(
                self._final_edits,
                index=current.index,
                new_text=text,
                current_text=current.text,
                source_text=source.text,
                target_lang=language,
            )
            self._translated_events[position] = _copy_event(current, text=text)
        self._persist_final_edits()

    def final_edit_positions(self) -> set[int]:
        """[v3.23.609] Vị trí (từ 0) các dòng đã sửa tay trên bảng kết quả."""
        return {
            position for position, event in enumerate(self._translated_events)
            if event.index in self._final_edits
        }

    def discard_final_edits(self, positions: Iterable[int]) -> int:
        """[v3.23.609] Trả các dòng sửa tay (theo vị trí) về câu máy dịch gần nhất.

        Returns:
            Số dòng đã trả lại.
        """
        indices = [
            self._translated_events[position].index
            for position in positions
            if 0 <= position < len(self._translated_events)
        ]
        events, edits, reverted = discard_final_edits(
            self._translated_events, self._final_edits, indices
        )
        if not reverted:
            return 0
        self._translated_events = events
        self._final_edits = edits
        self._persist_final_edits()
        self._persist_current_result()
        self.result_ready.emit(self.source_events, self.translated_events)
        self.status_message.emit(f"Đã bỏ sửa tay {len(reverted)} dòng.")
        return len(reverted)

    def _persist_final_edits(self) -> None:
        """Lưu các dòng sửa tay vào phiên (cạnh bản dịch 'final')."""
        video_path = getattr(self, "_last_translate_video_path", "") or ""
        if not video_path:
            return
        try:
            from subtitles_extractor.infrastructure.database import (
                sqlite_translation_session_store as session_store,
            )

            self._container.translation_session_store.save_stage(
                video_path,
                session_store.StageResult(
                    stage_id=FINAL_EDITS_STAGE_ID,
                    input_hash="",
                    lines_json=final_edits_to_json(self._final_edits),
                ),
            )
        except (OSError, ValueError, sqlite3.Error) as exc:
            logger.warning("Không lưu được các dòng sửa tay vào phiên: %s", exc)

    def _apply_final_edits_to_result(self, target_lang: str) -> int:
        """Áp các dòng sửa tay lên bản dịch vừa dựng lại. Trả số dòng đã áp."""
        if not self._final_edits:
            return 0
        outcome = apply_final_edits(
            self._translated_events, self._source_events, self._final_edits,
            target_lang,
        )
        self._translated_events = outcome.events
        self._final_edits = outcome.edits
        if outcome.dropped:
            logger.info(
                "Bỏ %d dòng sửa tay vì câu gốc đã đổi: %s.",
                len(outcome.dropped), list(outcome.dropped)[:10],
            )
        self._persist_final_edits()
        return len(outcome.applied)

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
        corrected, name_fixes = self._correct_context_names(
            context, getattr(self, "_last_translate_video_path", "") or ""
        )
        if corrected is not context:
            self.context_names_corrected.emit(corrected)
        context = corrected
        # [v3.23.609] Cues gắn theo số thứ tự dòng của CẢ phim, còn tập con được đánh số
        # lại 1..k → chỉ giữ cues của các dòng chọn, đánh số theo tập con.
        if (context.visual_cues or "").strip():
            from dataclasses import replace as _dc_replace

            from subtitles_extractor.application.services.visual_cue_serializer import (
                remap_visual_cues_to_subset,
            )

            context = _dc_replace(
                context,
                visual_cues=remap_visual_cues_to_subset(context.visual_cues, valid),
            )
        self._correct_names_in_translation(name_fixes)
        # [v3.23.608] Không dùng bộ nhớ đệm giai đoạn cho tập dòng con (xem container).
        use_case = self._container.make_translate_subtitles_use_case(
            api_key, retry_count, use_stage_cache=False
        )
        request = TranslateSubtitlesRequest(
            events=list(subset), stages=stages, context=context, contiguous=False,
        )
        self._retranslate_indices = valid
        self._retranslate_target_lang = context.target_lang
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
        # [v3.23.609] Dòng dịch lại theo yêu cầu được giữ như dòng sửa tay: bấm Dịch lại
        # cả bản (dựng lại từ bộ nhớ đệm giai đoạn) không trả chúng về bản dịch cũ.
        texts = {
            line_idx: new_events[offset].text
            for offset, line_idx in enumerate(indices)
            if offset < len(new_events) and 0 <= line_idx < len(self._translated_events)
        }
        target_lang = getattr(self, "_retranslate_target_lang", "")
        texts = _capitalize_retranslated(
            [event.text for event in self._translated_events], texts, target_lang
        )
        self._set_final_texts(texts, target_lang)
        applied = len(texts)
        # [v3.23.580] Dòng dịch lại có độ dài mới → giãn giờ đọc lại (dòng khác giữ
        # nguyên vì đã giãn rồi, phép giãn không làm gì thêm).
        self._apply_reading_time_extension()
        self.result_ready.emit(self.source_events, self.translated_events)
        self._persist_current_result()  # [v3.23.49] lưu bản dịch sau khi dịch lại
        tokens = self._close_token_report()
        self.status_message.emit(
            f"Đã dịch lại {applied} dòng." + (f" {tokens}" if tokens else "")
        )

    # ── Nạp nguồn ────────────────────────────────────────────────────────
    def set_source_events(self, events: list[SubtitleEvent]) -> None:
        """Nạp danh sách phụ đề nguồn (vd lấy từ trang Biên tập).

        [v3.23.609] Bỏ luôn kết quả từng giai đoạn và phiên đã gắn của phụ đề CŨ. Trước
        đây chỉ bỏ bản dịch: nút "So sánh giai đoạn" vẫn hiện các giai đoạn của phim cũ
        cạnh câu gốc của phim mới, và sửa/dịch lại dòng của phim mới ghi vào phiên của
        video cũ. Trang nạp lại phiên đúng của phụ đề mới ngay sau tín hiệu này.
        """
        self._source_events = [_copy_event(event) for event in events]
        self._source_version += 1
        self._translated_events = []
        self._final_edits = {}
        self._stage_outputs = {}
        self._last_translate_video_path = ""
        self._last_stage_count = 0
        self._last_elapsed_sec = 0.0
        self.source_changed.emit(len(self._source_events))

    def read_source_file(self, source_path: Path) -> list[SubtitleEvent] | None:
        """[v3.23.609] Đọc tệp SRT/ASS mà CHƯA thay phụ đề nguồn (báo lỗi nếu hỏng).

        Trang cần biết tệp đọc được trước khi xoá ngữ cảnh và đổi video của phụ đề cũ —
        đọc hỏng thì phụ đề cũ và ngữ cảnh của nó phải còn nguyên.

        Returns:
            Danh sách phụ đề, hoặc ``None`` khi không đọc được / tệp rỗng.
        """
        try:
            use_case = self._container.make_import_subtitles_use_case()
            events = use_case.execute(source_path)
        except FileNotFoundError:
            self.error_occurred.emit(f"Không tìm thấy tệp: {source_path}")
            return None
        except (ValueError, KeyError, OSError) as exc:
            self.error_occurred.emit(f"Không thể đọc tệp phụ đề: {exc}")
            return None
        if not events:
            self.error_occurred.emit("Tệp phụ đề rỗng hoặc không có dòng hợp lệ.")
            return None
        return list(events)

    def load_source_from_file(self, source_path: Path) -> bool:
        """Nạp phụ đề nguồn từ tệp SRT/ASS qua use case import.

        Returns:
            True khi đã nạp (False: đã báo lỗi qua ``error_occurred``).
        """
        events = self.read_source_file(source_path)
        if events is None:
            return False
        self.set_source_events(events)
        self.status_message.emit(f"Đã nạp {len(events)} dòng từ {source_path.name}.")
        return True

    # ── Dịch ─────────────────────────────────────────────────────────────
    def set_text_cues(self, enabled: bool, voice_source_path: str = "") -> None:
        """[v3.23.623] Bật/tắt gán ai nói / nói với ai từ phụ đề; video cạnh phụ đề (nếu có
        tiếng) được dùng để tách giọng — KHÔNG tải video lên Gemini."""
        self._text_cues = bool(enabled)
        self._voice_source_path = str(voice_source_path or "") if enabled else ""

    def start_translation(
        self,
        api_key: str,
        retry_count: int,
        stages: list[TranslationStageConfig],
        context: TranslationContext,
        video_path: str | None = None,
        attach_video_stages: frozenset | None = None,
        enable_visual_cues: bool = False,
        speaker_model: str = "",
        session_path: str = "",
    ) -> bool:
        """Bắt đầu dịch trên thread nền.

        Args:
            api_key: Khoá API Gemini.
            retry_count: Số lần thử lại khi lỗi.
            stages: Các giai đoạn dịch đã bật.
            context: Ngữ cảnh dịch.
            video_path: Video ĐÍNH KÈM gửi Gemini (tuỳ chọn).
            attach_video_stages: Giai đoạn được đính video.
            enable_visual_cues: Bật Visual Cues (cần video).
            speaker_model: [v3.23.602] Model hỏi lại người nói các dòng còn thiếu (model
                phân tích ngữ cảnh); rỗng = model của giai đoạn đầu.
            session_path: [v3.23.609] Video mà phụ đề thuộc về — khoá lưu phiên (từng giai
                đoạn, bản dịch cuối), Bộ nhớ dịch và ngữ cảnh phim bộ. Rỗng = dùng
                ``video_path``. Trước đây chỉ có ``video_path``, nên dịch KHÔNG đính video
                thì không lưu gì: mở lại mất bản dịch, Bộ nhớ dịch phim bộ không tích luỹ.

        Returns:
            True khi đã bắt đầu.
        """
        if self._is_busy:
            self.status_message.emit("Đang dịch, vui lòng chờ tiến trình hiện tại hoàn tất.")
            return False
        if self.custom_provider_active() and (attach_video_stages or enable_visual_cues):
            # [v3.23.666] Nhà cung cấp tuỳ chỉnh không xem được video: dịch bằng văn bản + cues có sẵn.
            attach_video_stages = frozenset()
            enable_visual_cues = False
            self.status_message.emit(self._CUSTOM_NO_VIDEO_NOTE)
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
        # [v3.23.567] Sửa phiên âm tên sai TRƯỚC khi chèn TM/tạo request: mọi giai đoạn,
        # lưới xưng hô và ngữ cảnh phim bộ tích luỹ đều dùng tên đúng.
        session_key = session_path or video_path or ""
        corrected, name_fixes = self._correct_context_names(context, session_key)
        if corrected is not context:
            self.context_names_corrected.emit(corrected)
        context = corrected
        original_context = context

        # [v3.23.55] Truy hồi Bộ nhớ dịch (TM) của phim bộ và chèn vào ngữ cảnh để giữ
        # nhất quán tên riêng/thuật ngữ/lối xưng hô giữa các tập. Chỉ khi biết video.
        if session_key:
            tm_block = self.retrieve_memory_for_lines(
                session_key, [ev.text for ev in self._source_events]
            )
            if tm_block and name_fixes:
                # Bộ nhớ dịch của các lần dịch trước còn tên sai — làm ví dụ cho model thì
                # model sẽ chép lại đúng cái tên vừa sửa.
                tm_block = apply_name_corrections(tm_block, name_fixes)
            if tm_block:
                from dataclasses import replace as _dc_replace

                try:
                    context = _dc_replace(
                        context, overview=(context.overview or "") + "\n" + tm_block
                    )
                    logger.info("Đã chèn Bộ nhớ dịch phim bộ vào ngữ cảnh dịch.")
                except (TypeError, ValueError):
                    pass  # context không hỗ trợ replace → bỏ qua, không chặn dịch

        request = self._translation_request(
            stages, context,
            video_path=video_path,
            enable_visual_cues=enable_visual_cues,
            speaker_model=speaker_model,
        )

        # Chuẩn bị provider video nếu người dùng bật đính video HOẶC bật visual cues
        # (visual cues cũng cần video_refs đã tải lên).
        video_provider = None
        if video_path and (attach_video_stages or enable_visual_cues):
            video_provider = self._container.make_video_context_provider(api_key)

        self._set_busy(True)
        # [v3.23.48] Ghi nhớ video_path để lưu kết quả từng giai đoạn vào session sau khi
        # dịch xong (cho phép nạp lại khi mở lại video lần sau).
        self._last_translate_video_path = session_key
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
        self._worker.video_refs_ready.connect(self._on_translate_video_refs_ready)
        self._worker.finished.connect(self._on_thread_finished)
        self._worker.start()
        self.status_message.emit("Bắt đầu dịch...")
        return True

    def _translation_request(
        self,
        stages: list[TranslationStageConfig],
        context: TranslationContext,
        *,
        video_path: str | None,
        enable_visual_cues: bool,
        speaker_model: str = "",
        attach_video_stages: frozenset[TranslationStageKind] | None = None,
    ) -> TranslateSubtitlesRequest:
        """[v3.23.608] Yêu cầu dịch từ cấu hình trang — DÙNG CHUNG cho nút Dịch, nút xoá
        bộ nhớ đệm và màn Quản lý giai đoạn, để cả ba tính cùng một khoá bộ nhớ đệm.

        Trước bản này nút "Xoá bộ nhớ đệm dịch" tự dựng yêu cầu thiếu ngôn ngữ gốc và
        Visual Cues → tính ra khoá KHÁC khoá lúc dịch → bấm xoá mà không xoá được gì.

        Args:
            stages: Các giai đoạn đang bật.
            context: Ngữ cảnh dịch (đã sửa phiên âm tên).
            video_path: Video ngữ cảnh (rỗng = không có).
            enable_visual_cues: Ô Visual Cues đang bật (chỉ có hiệu lực khi có video).
            speaker_model: [v3.23.602] Model hỏi lại người nói.
            attach_video_stages: Giai đoạn đính video (chỉ có hiệu lực khi có video).

        Returns:
            Yêu cầu dịch.
        """
        # [v3.23.34] Visual Cues dùng CÙNG model với giai đoạn đầu (model người dùng đang
        # dùng) thay vì model cứng — tránh 429 khi tài khoản không có quota cho model đó.
        return TranslateSubtitlesRequest(
            events=list(self._source_events), stages=list(stages), context=context,
            enable_visual_cues=bool(enable_visual_cues and video_path),
            visual_cues_model=(
                stages[0].model_name if stages else DEFAULT_VISUAL_CUES_MODEL
            ),
            speaker_model=speaker_model.strip(),
            attach_video_stages=(
                frozenset(attach_video_stages or ()) if video_path else frozenset()
            ),
            # getattr: vài test dựng ViewModel bằng __new__ (không qua __init__).
            text_cues=bool(getattr(self, "_text_cues", False)),
            voice_source_path=getattr(self, "_voice_source_path", "") or "",
        )

    def _effective_context(self, context: TranslationContext) -> TranslationContext:
        """[v3.23.608] Ngữ cảnh sau khi sửa phiên âm tên — đúng như lúc bấm Dịch, nhưng
        không lưu/không báo gì (chỉ để tính vân tay bảng nhân vật cho bộ nhớ đệm)."""
        readings = self.name_readings
        corrected = context
        if readings is not None:
            corrected, _ = correct_context_names(context, readings)
            corrected, _ = reconcile_context_titles(corrected, readings)
        if self.get_native_address():
            corrected, _ = nativize_context(corrected)
        return corrected

    def _stage_cache_scope(
        self,
        stages: list[TranslationStageConfig],
        context: TranslationContext,
        *,
        video_path: str | None,
        enable_visual_cues: bool,
        attach_video_stages: frozenset[TranslationStageKind] | None,
    ) -> StageCacheScope:
        request = self._translation_request(
            stages, self._effective_context(context),
            video_path=video_path,
            enable_visual_cues=enable_visual_cues,
            attach_video_stages=attach_video_stages,
        )
        return build_stage_cache_scope(request, self._container.stage_checkpoint_dir)

    def open_stage_cache(
        self,
        stages: list[TranslationStageConfig],
        context: TranslationContext,
        *,
        video_path: str | None = None,
        enable_visual_cues: bool = False,
        attach_video_stages: frozenset[TranslationStageKind] | None = None,
        session_path: str = "",
    ) -> StageCacheSession | None:
        """[v3.23.608] Mở bộ nhớ đệm từng giai đoạn của phụ đề hiện tại (màn Quản lý).

        Args:
            stages: Các giai đoạn đang bật (để dự đoán lần dịch tới).
            context: Ngữ cảnh đang nhập trên trang.
            video_path: Video ĐÍNH KÈM (quyết định khoá bộ nhớ đệm như lúc dịch).
            enable_visual_cues: Ô Visual Cues.
            attach_video_stages: Giai đoạn đính video.
            session_path: [v3.23.609] Video của phụ đề — khoá phiên để lưu nội dung sửa /
                bản dịch hiện hành; rỗng = ``video_path``. Trước đây chỉ dùng video đính
                kèm, nên không đính video thì sửa giai đoạn / dùng làm bản dịch hiện hành
                không được lưu vào phiên.

        Returns:
            Phiên làm việc cho hộp thoại; ``None`` khi chưa có phụ đề nguồn.
        """
        if not self._source_events:
            return None
        scope = self._stage_cache_scope(
            stages, context,
            video_path=video_path,
            enable_visual_cues=enable_visual_cues,
            attach_video_stages=attach_video_stages,
        )
        display_context = self._effective_context(context)
        session_video = session_path or video_path or ""

        def _edited(kind: TranslationStageKind, lines: list[TranslationLine]) -> None:
            self._on_cached_stage_edited(kind, lines, display_context, session_video)

        def _apply(kind: TranslationStageKind, lines: list[TranslationLine]) -> None:
            self._apply_cached_stage(kind, lines, display_context, session_video)

        return StageCacheSession(
            self._container.make_stage_cache_service(),
            scope,
            on_stage_edited=_edited,
            on_apply_stage=_apply,
        )

    def _on_cached_stage_edited(
        self,
        kind: TranslationStageKind,
        lines: list[TranslationLine],
        context: TranslationContext,
        video_path: str,
    ) -> None:
        """[v3.23.608] Giai đoạn vừa sửa tay → bảng so sánh, phiên đã lưu theo kịp."""
        self._stage_outputs[kind] = _lines_to_events_helper(
            self._source_events, lines, context
        )
        if video_path:
            self._last_translate_video_path = video_path
        self._persist_stage_outputs()

    def _apply_cached_stage(
        self,
        kind: TranslationStageKind,
        lines: list[TranslationLine],
        context: TranslationContext,
        video_path: str,
    ) -> None:
        """[v3.23.608] Dùng nội dung một giai đoạn đã lưu làm bản dịch hiện hành.

        [v3.23.609] Người dùng đã xác nhận thay cả các dòng sửa trên bảng kết quả → bỏ
        luôn danh sách dòng sửa tay (kẻo lần dịch sau áp lại chúng).
        """
        self._translated_events = _lines_to_events_helper(
            self._source_events, lines, context
        )
        self._apply_reading_time_extension()
        if video_path:
            self._last_translate_video_path = video_path
        self._final_edits = {}
        self._persist_final_edits()
        self._persist_current_result()
        self.result_ready.emit(self.source_events, self.translated_events)
        self.status_message.emit(
            f"Đã dùng bản '{STAGE_LABELS[kind]}' ({len(lines)} dòng) làm bản dịch hiện "
            "hành."
        )

    def clear_translation_checkpoint(
        self,
        stages: list[TranslationStageConfig],
        context: TranslationContext,
        *,
        video_path: str | None = None,
        enable_visual_cues: bool = False,
        attach_video_stages: frozenset[TranslationStageKind] | None = None,
        session_path: str = "",
    ) -> int:
        """Xoá TOÀN BỘ bộ nhớ đệm giai đoạn của phụ đề hiện tại (kể cả file khoá cũ).

        Args:
            stages: Các giai đoạn đang bật.
            context: Ngữ cảnh đang nhập trên trang.
            video_path: Video ngữ cảnh.
            enable_visual_cues: Ô Visual Cues.
            attach_video_stages: Giai đoạn đính video.
            session_path: Video của phụ đề (không dùng để tính khoá).

        Returns:
            Số file bộ nhớ đệm đã xoá.
        """
        session = self.open_stage_cache(
            stages, context,
            video_path=video_path,
            enable_visual_cues=enable_visual_cues,
            attach_video_stages=attach_video_stages,
            session_path=session_path,
        )
        if session is None:
            return 0
        try:
            removed = session.delete_all()
        except OSError as exc:
            logger.warning("Không xoá được bộ nhớ đệm dịch: %s", exc)
            self.error_occurred.emit(f"Không xoá được bộ nhớ đệm dịch: {exc}")
            return 0
        self.status_message.emit(
            "Đã xoá bộ nhớ đệm dịch. Lần dịch tới sẽ dịch lại mọi giai đoạn."
            if removed
            else "Phụ đề này chưa có bộ nhớ đệm dịch nào để xoá."
        )
        return removed

    # ── Phân tích ngữ cảnh ───────────────────────────────────────────────────
    analyze_context_ready = Signal(object)   # SubtitleContextAnalysis — kết quả phân tích

    def start_context_analysis(
        self, api_key: str, target_lang: str, model_name: str = "gemini-3.5-flash-lite",
        video_path: str | None = None, enable_visual_cues: bool = False,
        session_path: str = "",
    ) -> bool:
        """Phân tích TOÀN BỘ phụ đề nguồn bằng AI trên thread nền.

        Args:
            api_key:     Khoá API Gemini.
            target_lang: Ngôn ngữ viết phần tóm tắt.
            model_name:  Model AI dùng cho phân tích.
            video_path:  Video ĐÍNH KÈM gửi Gemini (tuỳ chọn).
            enable_visual_cues: Quét người nói/người nghe (cần video).
            session_path: [v3.23.609] Video mà phụ đề thuộc về — khoá lưu phân tích và ngữ
                cảnh phim bộ; rỗng = ``video_path``. Trước đây phân tích không đính video
                thì không được lưu, mở lại phải phân tích lại.
        """
        if self._is_busy:
            self.status_message.emit("Đang bận, vui lòng chờ.")
            return False
        if not self._source_events:
            self.error_occurred.emit("Chưa có phụ đề nguồn để phân tích.")
            return False
        if self.custom_provider_active() and video_path:
            # [v3.23.666] Phân tích bằng phụ đề; vẫn lưu theo video của phim.
            session_path = session_path or video_path
            video_path = None
            enable_visual_cues = False
            self.status_message.emit(self._CUSTOM_NO_VIDEO_NOTE)

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
        session_key = session_path or video_path or ""
        prior_context = self._build_prior_series_context(session_key)
        self._analyze_worker = AnalyzeContextWorker(
            use_case, source_lines, target_lang, model_name=model_name,
            video_provider=video_provider, video_path=video_path,
            enable_visual_cues=bool(enable_visual_cues and video_path),
            prior_context=prior_context,
            fallback_model=self.analysis_fallback_model(),
            audit_model=self.audit_chain(model_name)[0],
            audit_fallbacks=self.audit_chain(model_name)[1],
            # Có video đính kèm + Visual Cues thì nhãn đến từ video; còn lại gán từ phụ đề.
            text_cues=bool(
                getattr(self, "_text_cues", False) and not (enable_visual_cues and video_path)
            ),
            voice_source_path=getattr(self, "_voice_source_path", "") or "",
        )
        self._analyze_worker.finished_ok.connect(self._on_analyze_finished)
        self._analyze_worker.video_refs_ready.connect(self._on_video_refs_ready)
        self._analyze_worker.cancelled.connect(lambda: self.status_message.emit("Đã huỷ phân tích."))
        self._analyze_worker.failed.connect(self.analyze_error.emit)
        self._analyze_worker.progress_message.connect(self.status_message.emit)
        self._analyze_worker.finished.connect(self._on_analyze_thread_finished)
        # [v3.23.15] Ghi nhớ ngữ cảnh phiên để lưu kết quả phân tích sau khi xong.
        self._pending_analysis_video_path = session_key
        self._pending_analysis_target_lang = target_lang
        self._pending_analysis_source_lines = [ln.text for ln in source_lines]
        self._set_busy(True)
        self._analyze_worker.start()
        n = len(source_lines)
        self.status_message.emit(f"Đang phân tích toàn bộ {n} dòng bằng {model_name}…")
        return True

    def set_fallback_model(self, value: str) -> None:
        """[v3.23.638] Lưu lựa chọn model dự phòng của người dùng (``""`` tự động, ``"-"`` tắt).

        Áp cho MỌI lời gọi Gemini của ứng dụng (bộ chọn dự phòng khi model quá tải dùng chung) và
        cho bước phân tích khi model phân tích hết hạn mức ngày.
        """
        self._fallback_choice = (value or "").strip()
        try:
            self._container.gemini_model_health.set_preferred_fallback(self._fallback_choice)
            # [v3.23.666] Cùng lựa chọn cho nhà cung cấp tuỳ chỉnh (khi đang dùng).
            if self.custom_provider_active():
                self._container.custom_model_health.set_preferred_fallback(self._fallback_choice)
        except Exception as exc:  # noqa: BLE001 — cấu hình phụ: lỗi không được làm hỏng giao diện
            logger.warning("Không áp được model dự phòng '%s': %s", value, exc)

    #: [v3.23.666] Thông báo khi nhà cung cấp tuỳ chỉnh bỏ qua video.
    _CUSTOM_NO_VIDEO_NOTE = (
        "Nhà cung cấp tuỳ chỉnh không xem được video — chạy bằng phụ đề (và Visual Cues đã có)."
    )
    _CUSTOM_VIDEO_ONLY_NOTE = (
        "Tính năng này cần Gemini xem video — nhà cung cấp tuỳ chỉnh không hỗ trợ. Chuyển "
        "'Nhà cung cấp AI' về Google Gemini để dùng."
    )

    def set_custom_provider(self, enabled: bool, base_url: str, api_key: str) -> bool:
        """[v3.23.666] Bật/tắt nhà cung cấp AI tuỳ chỉnh (API kiểu OpenAI, vd OxionAI).

        Args:
            enabled: Người dùng chọn "Tuỳ chỉnh".
            base_url: Địa chỉ API gốc (vd ``https://oxionai.com/v1``).
            api_key: Khoá API của nhà cung cấp.

        Returns:
            True nếu nhà cung cấp tuỳ chỉnh đang có hiệu lực (bật + đủ địa chỉ và key).
        """
        from subtitles_extractor.infrastructure.translation.openai_compat_client import (
            CustomProviderConfig,
        )

        config = CustomProviderConfig(base_url=base_url or "", api_key=api_key or "")
        self._container.set_custom_text_provider(config if enabled else None)
        active = self.custom_provider_active()
        if active:
            self._container.custom_model_health.set_preferred_fallback(
                getattr(self, "_fallback_choice", "") or ""
            )
        return active

    def custom_provider_active(self) -> bool:
        """[v3.23.666] True khi các lời gọi AI văn bản đang đi tới nhà cung cấp tuỳ chỉnh."""
        try:
            return self._container.custom_text_provider is not None
        except AttributeError:  # container giả trong kiểm thử cũ
            return False

    def set_audit_model(self, value: str) -> None:
        """[v3.23.640] Model tự kiểm tra phân tích do người dùng chọn (``""`` = mặc định rẻ)."""
        self._audit_choice = (value or "").strip()

    def audit_chain(self, analysis_model: str = "") -> tuple[str, tuple[str, ...]]:
        """[v3.23.640] ``(model tự kiểm tra, các model dự phòng của nó)`` theo lựa chọn hiện tại.

        Model chính = lựa chọn của người dùng (mặc định ``DEFAULT_AUDIT_MODEL``: rẻ, 500 request/ngày).
        Dự phòng: model dự phòng người dùng chọn (nếu là một model cụ thể), rồi model mặc định.

        [v3.23.666] Nhà cung cấp tuỳ chỉnh: mặc định là chính ``analysis_model`` (tên Gemini cố
        định có thể không được nhà cung cấp bán → 404), không thêm model mặc định của Gemini.
        """
        from subtitles_extractor.application.services.analysis_audit import DEFAULT_AUDIT_MODEL

        custom = self.custom_provider_active()
        default = (analysis_model or DEFAULT_AUDIT_MODEL) if custom else DEFAULT_AUDIT_MODEL
        primary = getattr(self, "_audit_choice", "") or default
        backup = getattr(self, "_fallback_choice", "")
        candidates = [backup] if backup and backup != "-" else []
        if not custom:
            candidates.append(DEFAULT_AUDIT_MODEL)
        return primary, tuple(dict.fromkeys(m for m in candidates if m and m != primary))

    def analysis_fallback_model(self) -> str:
        """Model dự phòng cho bước phân tích: lựa chọn của người dùng, mặc định gemini-3.1-flash-lite;
        ``""`` (tắt) khi người dùng chọn không chuyển model."""
        choice = getattr(self, "_fallback_choice", "")
        if choice == "-":
            return ""
        if self.custom_provider_active():
            return choice  # [v3.23.666] không tự đoán tên model Gemini cho nhà cung cấp tuỳ chỉnh
        return choice or ANALYSIS_FALLBACK_MODEL

    def start_analysis_refinement(
        self, api_key: str, target_lang: str, model_name: str, source_lang: str,
        characters: str, overview: str, glossary: str, visual_cues: str,
        session_path: str = "",
    ) -> bool:
        """[v3.23.624] Tự kiểm tra rồi vá phân tích HIỆN CÓ (không phân tích lại từ đầu).

        Lấy nguyên các ô người dùng đang thấy (kể cả chỗ đã sửa tay) làm gốc, đối chiếu với
        phụ đề nguồn bằng kiểm tra tất định, rồi hỏi model CHỈ phần cần sửa.

        Returns:
            True khi đã bắt đầu chạy nền.
        """
        if self._is_busy:
            self.status_message.emit("Đang bận, vui lòng chờ.")
            return False
        if not self._source_events:
            self.error_occurred.emit("Chưa có phụ đề nguồn để đối chiếu.")
            return False
        probe = self._container.make_subtitle_translator(api_key, retry_count=3)
        if not probe.is_available():
            self.error_occurred.emit("API Key Gemini chưa hợp lệ hoặc thiếu thư viện google-genai.")
            return False
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            _events_to_lines_helper,
        )
        from subtitles_extractor.presentation.workers.refine_analysis_worker import (
            RefineAnalysisWorker,
        )

        source_lines = _events_to_lines_helper(self._source_events)
        current = SubtitleContextAnalysis(
            source_lang=source_lang, characters=characters, overview=overview,
            glossary=glossary, visual_cues=visual_cues,
        )
        audit_primary, audit_backups = self.audit_chain(model_name)
        self._analyze_worker = RefineAnalysisWorker(
            self._container.make_analyze_context_use_case(api_key),
            source_lines, target_lang, model_name, current,
            audit_model=audit_primary, audit_fallbacks=audit_backups,
        )
        self._analyze_worker.finished_ok.connect(self._on_analyze_finished)
        self._analyze_worker.cancelled.connect(
            lambda: self.status_message.emit("Đã huỷ tự kiểm tra phân tích.")
        )
        self._analyze_worker.failed.connect(self.analyze_error.emit)
        self._analyze_worker.progress_message.connect(self.status_message.emit)
        self._analyze_worker.finished.connect(self._on_analyze_thread_finished)
        self._pending_analysis_video_path = session_path
        self._pending_analysis_target_lang = target_lang
        self._pending_analysis_source_lines = [ln.text for ln in source_lines]
        self._set_busy(True)
        self._analyze_worker.start()
        # [v3.23.640] Nói đúng model THẬT sự dùng (trước đây hiện model phân tích đang chọn, trong khi
        # bước kiểm tra chạy bằng model kiểm định cố định → người dùng tưởng nó dùng model đã chọn).
        chain = " → ".join((audit_primary, *audit_backups))
        self.status_message.emit(f"Đang tự kiểm tra & tối ưu phân tích bằng {chain}…")
        return True

    @staticmethod
    def incomplete_analysis_message(result: object) -> str:
        """[v3.23.585] Thông báo khi kết quả phân tích còn THIẾU đoạn video.

        Args:
            result: Kết quả phân tích (``SubtitleContextAnalysis``).

        Returns:
            Câu thông báo cho người dùng; rỗng khi phân tích đã đủ.
        """
        if not isinstance(result, SubtitleContextAnalysis):
            return ""
        return describe_incomplete_analysis(result)

    def _speaker_coverage(self, visual_cues: str) -> SpeakerCoverage | None:
        """Độ phủ người nói của cues trên phụ đề nguồn đang nạp (None khi thiếu dữ liệu)."""
        if not (visual_cues or "").strip() or not self._source_events:
            return None
        # getattr: vài test dựng ViewModel bằng __new__ (không qua __init__).
        cache_key = (getattr(self, "_source_version", 0), visual_cues)
        cached = getattr(self, "_coverage_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        coverage = self._compute_speaker_coverage(visual_cues)
        self._coverage_cache = (cache_key, coverage)
        return coverage

    def _compute_speaker_coverage(self, visual_cues: str) -> SpeakerCoverage | None:
        from subtitles_extractor.application.services.visual_cue_serializer import (
            deserialize_visual_cues,
        )

        lines = [
            TranslationLine(
                index=int(event.index), start_ms=0, end_ms=0, text=str(event.text or "")
            )
            for event in self._source_events
        ]
        return speaker_coverage(deserialize_visual_cues(visual_cues), lines)

    def speaker_coverage_message(self, visual_cues: str) -> str:
        """[v3.23.600] "Người nói: x/y dòng có lời" cho phụ đề nguồn đang nạp.

        Lồng tiếng cần người nói cho MỌI dòng có lời — báo ngay sau khi phân tích để người
        dùng biết còn bao nhiêu dòng (khâu dịch có kèm video sẽ hỏi lại các dòng đó).

        Args:
            visual_cues: JSON Visual Cues.

        Returns:
            Câu ngắn; rỗng khi không có cues hoặc chưa có phụ đề nguồn.
        """
        coverage = self._speaker_coverage(visual_cues)
        if coverage is None:
            return ""
        message = (
            f"Người nói: {coverage.with_speaker}/{coverage.spoken_lines} dòng có lời "
            f"({coverage.ratio:.1%})"
        )
        if coverage.voice_only:
            message += f", {coverage.voice_only} dòng chỉ biết giọng"
        if coverage.missing:
            message += f" — còn {len(coverage.missing)} dòng chưa có người nói"
        return message + "."

    def missing_speaker_count(self, visual_cues: str) -> int:
        """[v3.23.601] Số dòng có lời chưa có người nói (0 khi chưa có cues/phụ đề)."""
        coverage = self._speaker_coverage(visual_cues)
        return len(coverage.missing) if coverage is not None else 0

    def start_speaker_completion(
        self,
        api_key: str,
        model_name: str,
        video_path: str | None,
        visual_cues: str,
        characters: str,
        target_lang: str,
        session_path: str = "",
    ) -> bool:
        """[v3.23.601] Hỏi lại người nói cho các dòng còn thiếu (nút "Bổ sung người nói").

        Phân tích khôi phục từ phiên cũ không qua bước hỏi lại của khâu phân tích/dịch
        (chuẩn đoán thật: 39/1246 dòng lồng tiếng bằng giọng chung). Chỉ hỏi các dòng
        thiếu, tối đa hai lượt; kết quả lưu vào phiên như cues quét bổ sung khi dịch.

        Args:
            api_key: Khoá API Gemini (có thể nhiều khoá).
            model_name: Model Gemini có thị giác.
            video_path: Video của phim (bắt buộc).
            visual_cues: JSON Visual Cues hiện có.
            characters: Bảng nhân vật.
            target_lang: Ngôn ngữ đích.
            session_path: [v3.23.609] Khoá phiên để lưu cues; rỗng = ``video_path``.

        Returns:
            True khi đã bắt đầu chạy nền.
        """
        if self._is_busy:
            self.status_message.emit("Đang bận, vui lòng chờ.")
            return False
        if not self._source_events:
            self.error_occurred.emit("Chưa có phụ đề nguồn.")
            return False
        if self.custom_provider_active():
            self.analyze_error.emit(self._CUSTOM_VIDEO_ONLY_NOTE)
            return False
        if not video_path:
            self.analyze_error.emit(
                "Cần video của phim để biết ai đang nói — hãy chọn video ngữ cảnh."
            )
            return False
        if not self.missing_speaker_count(visual_cues):
            self.status_message.emit("Mọi dòng có lời đã có người nói.")
            return False
        probe = self._container.make_subtitle_translator(api_key, retry_count=3)
        if not probe.is_available():
            self.error_occurred.emit(
                "API Key Gemini chưa hợp lệ hoặc thiếu thư viện google-genai."
            )
            return False
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            _events_to_lines_helper,
        )
        from subtitles_extractor.presentation.workers.complete_speakers_worker import (
            CompleteSpeakersWorker,
        )

        self._analyze_worker = CompleteSpeakersWorker(
            self._container.make_complete_speakers_use_case(api_key),
            _events_to_lines_helper(self._source_events),
            target_lang,
            model_name,
            video_provider=self._container.make_video_context_provider(api_key),
            video_path=video_path,
            visual_cues=visual_cues,
            characters=characters,
        )
        self._speaker_completion_video_path = session_path or video_path
        self._analyze_worker.finished_ok.connect(self._on_speaker_completion_finished)
        self._analyze_worker.video_refs_ready.connect(self._on_video_refs_ready)
        self._analyze_worker.cancelled.connect(
            lambda: self.status_message.emit("Đã huỷ bổ sung người nói.")
        )
        self._analyze_worker.failed.connect(self.analyze_error.emit)
        self._analyze_worker.progress_message.connect(self.status_message.emit)
        self._analyze_worker.finished.connect(self._on_analyze_thread_finished)
        self._pending_analysis_video_path = session_path or video_path
        self._set_busy(True)
        self._analyze_worker.start()
        self.status_message.emit(
            f"Đang hỏi lại người nói {self.missing_speaker_count(visual_cues)} dòng "
            f"bằng {model_name}…"
        )
        return True

    def start_cue_recheck(
        self, *, api_key: str, model_name: str, video_path: str, visual_cues: str,
        characters: str, target_lang: str, session_path: str = "",
    ) -> bool:
        """[v3.23.655] Hỏi lại AI CHỈ các dòng Visual Cues bị chấm nghi (rẻ hơn phân tích lại cả phim)."""
        if self._is_busy:
            self.status_message.emit("Đang bận, vui lòng chờ.")
            return False
        if not self._source_events:
            self.error_occurred.emit("Chưa có phụ đề nguồn.")
            return False
        if self.custom_provider_active():
            self.analyze_error.emit(self._CUSTOM_VIDEO_ONLY_NOTE)
            return False
        if not video_path:
            self.analyze_error.emit("Cần video của phim để kiểm lại ai đang nói — hãy chọn video ngữ cảnh.")
            return False
        probe = self._container.make_subtitle_translator(api_key, retry_count=3)
        if not probe.is_available():
            self.error_occurred.emit("API Key Gemini chưa hợp lệ hoặc thiếu thư viện google-genai.")
            return False
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            _events_to_lines_helper,
        )
        from subtitles_extractor.presentation.workers.complete_speakers_worker import RecheckCuesWorker

        self._analyze_worker = RecheckCuesWorker(
            self._container.make_recheck_cues_use_case(api_key),
            _events_to_lines_helper(self._source_events),
            target_lang,
            model_name,
            video_provider=self._container.make_video_context_provider(api_key),
            video_path=video_path,
            visual_cues=visual_cues,
            characters=characters,
        )
        self._speaker_completion_video_path = session_path or video_path
        self._analyze_worker.finished_ok.connect(self._on_cue_recheck_finished)
        self._analyze_worker.video_refs_ready.connect(self._on_video_refs_ready)
        self._analyze_worker.cancelled.connect(lambda: self.status_message.emit("Đã huỷ kiểm lại dòng nghi."))
        self._analyze_worker.failed.connect(self.analyze_error.emit)
        self._analyze_worker.progress_message.connect(self.status_message.emit)
        self._analyze_worker.finished.connect(self._on_analyze_thread_finished)
        self._pending_analysis_video_path = session_path or video_path
        self._set_busy(True)
        self._analyze_worker.start()
        self.status_message.emit(f"Đang kiểm lại các dòng nghi bằng {model_name}…")
        return True

    def _on_cue_recheck_finished(self, result: object) -> None:
        from subtitles_extractor.application.use_cases.complete_speakers import CueRecheckResult

        if not isinstance(result, CueRecheckResult):
            return
        if result.changed:
            self._persist_topped_up_cues(
                result.visual_cues, getattr(self, "_speaker_completion_video_path", "") or "",
            )
        message = (
            f"Kiểm lại {result.suspicious} dòng nghi: sửa {len(result.fixed)}, giữ nguyên {len(result.unchanged)}"
            f" — còn {result.still_suspicious} dòng nghi"
            + (" (mở 📋 Bảng người nói để xem/sửa tay)" if result.still_suspicious else "")
        )
        tokens = self._close_token_report()
        self.speaker_completion_done.emit(message + "." + (f" {tokens}" if tokens else ""))

    def _on_speaker_completion_finished(self, result: object) -> None:
        """[v3.23.601] Lưu cues đã bổ sung vào phiên rồi báo kết quả cho trang."""
        if not isinstance(result, SpeakerCompletionResult):
            return
        if result.changed:
            self._persist_topped_up_cues(
                result.visual_cues,
                getattr(self, "_speaker_completion_video_path", "") or "",
            )
        message = (
            f"Người nói: {result.before.with_speaker} → {result.after.with_speaker}/"
            f"{result.after.spoken_lines} dòng có lời"
        )
        if result.dropped:
            message += f", bỏ {len(result.dropped)} nhãn lạc cảnh"
        if result.demoted:
            message += (
                f", {len(result.demoted)} dòng model gán tên người không có mặt trong "
                "cảnh → chỉ giữ giọng nam/nữ"
            )
        if result.after.missing:
            message += (
                f" — còn {len(result.after.missing)} dòng model không xác định được "
                "(soát ở trang TTS: ⚠ Soát người nói)"
            )
        tokens = self._close_token_report()
        self.speaker_completion_done.emit(
            message + "." + (f" {tokens}" if tokens else "")
        )

    def _on_analyze_finished(self, result) -> None:
        tokens = self._close_token_report()
        self.analyze_context_ready.emit(result)
        missing = len(getattr(result, "missing_segments", ()) or ())
        audit_note = getattr(getattr(self, "_analyze_worker", None), "audit_note", "")
        self.status_message.emit(
            (
                f"Phân tích ngữ cảnh CHƯA đủ — còn thiếu {missing} đoạn video."
                if missing
                else "Phân tích ngữ cảnh hoàn tất."
            )
            + (f" {audit_note}" if audit_note else "")
            + (f" {tokens}" if tokens else "")
        )
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
        self._restored_name_corrections = {}
        result = self.restore_analysis_for_video(video_path, target_lang)
        # [v3.23.48] Nạp lại luôn kết quả từng giai đoạn đã lưu (để so sánh được ngay).
        self.restore_stage_outputs_for_video(video_path)
        # [v3.23.567] Bảng nhân vật đã lưu vừa được sửa tên → sửa luôn bản dịch đã lưu.
        self._correct_names_in_translation(self._restored_name_corrections)
        # [v3.23.612] Bản dịch đã lưu (kiểu "Kiếm ca") đổi theo tuỳ chọn xưng hô.
        if result is not None:
            self._nativize_translation(result)
        # [v3.23.49] Nếu có bản dịch đã lưu → hiển thị lại trên bảng kết quả.
        if self._translated_events:
            self.result_ready.emit(self.source_events, self.translated_events)
        if result is None:
            return False
        self.analysis_restored.emit(result)
        return True

    @staticmethod
    def roster_without_address_rules(characters: str) -> int:
        """[v3.23.578] Số nhân vật nếu bảng KHÔNG có quy tắc xưng hô nào, ngược lại 0.

        Phân tích đã lưu được khôi phục theo hash phụ đề — bản tạo trước khi khâu phân
        tích chốt xưng hô (v3.23.524) được dùng lại mãi mà mọi lưới xưng hô đều im lặng.
        Giao diện dùng số này để khuyên người dùng Phân tích lại.

        Args:
            characters: Bảng nhân vật đã khôi phục.

        Returns:
            Số nhân vật trong bảng, hoặc 0 nếu bảng có quy tắc (hoặc quá ít nhân vật).
        """
        from subtitles_extractor.application.services.roster_format import (
            lacks_address_rules,
            parse_roster,
        )

        if not lacks_address_rules(characters):
            return 0
        return len(parse_roster(characters))

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

            store = self._container.translation_session_store
            session = store.get(video_path)
            if session is None:
                return None
            source_lines = [ev.text for ev in self._source_events]
            current_hash = hash_analysis_input(source_lines, target_lang or "")
            if not session.has_valid_analysis(current_hash):
                return None
            logger.info("Khôi phục phân tích ngữ cảnh đã lưu cho %s.", video_path)
            restored = SubtitleContextAnalysis(
                source_lang=session.analysis_source_lang,
                characters=session.analysis_characters,
                overview=session.analysis_overview,
                glossary=session.analysis_glossary,
                visual_cues=getattr(session, "analysis_visual_cues", "") or "",
            )
            corrected, corrections = self._correct_context_names(restored, video_path)
            self._restored_name_corrections = corrections
            return corrected
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không khôi phục được phân tích đã lưu: %s", exc)
            return None

    def _on_video_refs_ready(self, video_refs: object) -> None:
        """[v3.23.18] Lưu danh sách file cloud đã upload vào phiên dịch (theo video)."""
        self._record_cloud_files(
            getattr(self, "_pending_analysis_video_path", None), video_refs
        )

    def _on_translate_video_refs_ready(self, video_refs: object) -> None:
        """[v3.23.609] Ghi các đoạn video tải lên khi DỊCH (kể cả tải lại khi xoay key).

        Trước đây chỉ khâu phân tích ghi lại, nên đoạn tải khi dịch không bao giờ được nút
        "Xoá file cloud" xoá.
        """
        self._record_cloud_files(
            getattr(self, "_last_translate_video_path", None), video_refs
        )

    def _record_cloud_files(self, session_key: str | None, video_refs: object) -> None:
        """Thêm các đoạn vừa tải vào danh sách file cloud của phiên (không ghi đè)."""
        if not session_key or not video_refs:
            return
        from datetime import UTC, datetime

        from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
            CloudVideoFile,
            merge_cloud_files,
        )

        now = datetime.now(UTC)
        uploaded = [
            CloudVideoFile(
                remote_name=getattr(ref, "remote_name", ""),
                start_sec=float(getattr(ref, "start_sec", 0.0)),
                end_sec=float(getattr(ref, "end_sec", 0.0)),
                uploaded_at=now.isoformat(),
            )
            for ref in video_refs  # type: ignore[union-attr]
            if getattr(ref, "remote_name", "")
        ]
        if not uploaded:
            return
        try:
            store = self._container.translation_session_store
            session = store.get(session_key)
            existing = list(session.cloud_files) if session is not None else []
            merged = merge_cloud_files(existing, uploaded, now)
            store.save_cloud_files(session_key, merged)
            logger.info(
                "Đã ghi %d file cloud vào phiên dịch cho %s (tổng %d).",
                len(uploaded), session_key, len(merged),
            )
        except Exception as exc:  # noqa: BLE001 — lưu phiên không được phá UI
            logger.warning("Không lưu được danh sách file cloud: %s", exc)

    def delete_cloud_files_for_video(self, video_path: str, api_key: str) -> int:
        """[v3.23.18] Xoá các file cloud đã upload của một video (giải phóng dung lượng).

        [v3.23.609] Dùng MỌI key trong ô API key (file thuộc key đã tải nó), và giữ lại
        trong phiên các file xoá lỗi để bấm lại lần sau.

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
            remote_names = [
                cf.remote_name for cf in session.cloud_files if cf.remote_name
            ]
            provider = self._container.make_video_context_provider(api_key)
            results = provider.delete_remote_files(remote_names)
            deleted = sum(1 for ok in results.values() if ok)
            remaining = [
                cf for cf in session.cloud_files
                if cf.remote_name and not results.get(cf.remote_name, False)
            ]
            if remaining:
                store.save_cloud_files(video_path, remaining)
                self.status_message.emit(
                    f"Đã xoá {deleted}/{len(remote_names)} file cloud. {len(remaining)} "
                    "file chưa xoá được (lỗi mạng hoặc máy chủ) — bấm Xoá file cloud để "
                    "thử lại."
                )
            else:
                store.clear_cloud_files(video_path)
                self.status_message.emit(
                    f"Đã xoá {deleted}/{len(remote_names)} file cloud."
                )
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

    def _apply_reading_time_extension(self) -> None:
        """[v3.23.440] Giãn thời gian hiển thị dòng đọc-không-kịp vào khoảng lặng phía sau.

        Bổ trợ dịch CPS-aware: đòn bẩy KHÔNG đụng chữ (0 rủi ro nghĩa). Bật/tắt qua Cài đặt
        ``extend_display_for_reading``; mục tiêu CPS lấy theo ``reading_cps_budget`` (mặc định
        21) hoặc 20 nếu CPS-aware đang tắt. Lỗi thì bỏ qua an toàn, giữ nguyên timing gốc.
        """
        try:
            tr = self._container.settings_service.current.translation
        except AttributeError:
            return
        if not getattr(tr, "extend_display_for_reading", False):
            return
        if not self._translated_events:
            return
        target_cps = float(getattr(tr, "reading_cps_budget", 0.0)) or 20.0
        from subtitles_extractor.application.services.reading_time_extender import (
            GLITCH_WINDOW_SEC,
            extend_events_for_reading,
        )

        def _glitches(events: list[SubtitleEvent]) -> int:
            return sum(
                1 for ev in events if ev.end_sec - ev.start_sec < GLITCH_WINDOW_SEC
            )

        try:
            extended_events, extended_count = extend_events_for_reading(
                self._translated_events, target_cps
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Giãn giờ đọc thất bại (%s) — giữ nguyên timing.", exc)
            return
        if extended_count:
            glitches_before = _glitches(self._translated_events)
            self._translated_events = extended_events
            logger.info(
                "Giãn giờ đọc: %d dòng đọc-không-kịp được kéo dài vào khoảng lặng "
                "(mục tiêu %.0f CPS); khung OCR chớp nhoáng < %.1f s: %d → %d dòng.",
                extended_count, target_cps, GLITCH_WINDOW_SEC,
                glitches_before, _glitches(extended_events),
            )

    def _on_worker_finished(self, response: TranslateSubtitlesResponse) -> None:
        # [Cross-Thread Contamination] Bỏ qua tín hiệu đến muộn từ worker CŨ đã bị
        # huỷ — nó không được phép ghi đè kết quả/trạng thái của worker hiện tại.
        if self._from_stale_worker():
            return
        self._translated_events = [_copy_event(event) for event in response.events]
        # [v3.23.609] Dòng sửa tay / dịch lại trên bảng kết quả thắng bản dựng lại.
        context = getattr(self, "_last_translate_context", None)
        self._result_target_lang = str(getattr(context, "target_lang", "") or "")
        kept_edits = self._apply_final_edits_to_result(self._result_target_lang)
        self._apply_reading_time_extension()
        self._last_stage_count = len(response.completed_stages)
        self._last_elapsed_sec = response.elapsed_seconds
        # [v3.23.47] Giữ kết quả từng giai đoạn để so sánh (nếu có >1 giai đoạn).
        self._stage_outputs = dict(getattr(response, "stage_outputs", {}) or {})
        # [v3.23.48] Lưu BỀN VỮNG kết quả từng giai đoạn vào session để nạp lại lần sau.
        self._persist_stage_outputs()
        # [v3.23.551] GHI ĐÈ 'final' bằng kết quả vừa dịch. Trước đây 'final' chỉ được ghi khi
        # sửa tay/dịch lại dòng, còn lúc nạp lại thì 'final' được ƯU TIÊN — nên mở lại dự án sẽ
        # hiện bản dịch CŨ của lần sửa tay trước. Chuẩn đoán thật: 'final' hoàn tất từ 3 ngày
        # trước, chỉ khớp 29% bản dịch mới. Kết quả ở đây còn gồm các bước hậu kỳ (dọn chữ Hán
        # sót, dịch lại dòng trùng, nới thời gian đọc) mà các giai đoạn đã lưu KHÔNG có.
        self._persist_current_result()
        self._persist_topped_up_cues(getattr(response, "visual_cues", "") or "")
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
        # [v3.23.608] Báo phần lấy từ bộ nhớ đệm giai đoạn (dùng lại nguyên / chỉ dịch lại
        # phần đổi / áp lại dòng sửa tay) thay cho "resume từ <giai đoạn>".
        reuse = describe_cache_reuse(
            getattr(response, "reused_stages", ()),
            getattr(response, "carried_lines", {}) or {},
            getattr(response, "reapplied_edits", {}) or {},
        )
        if kept_edits:
            reuse = (
                f"{reuse} " if reuse else ""
            ) + f"Giữ {kept_edits} dòng đã sửa trên bảng kết quả."
        if reuse:
            self.resume_detected.emit(reuse)
        self.result_ready.emit(self.source_events, self.translated_events)
        elapsed_str = _format_elapsed(response.elapsed_seconds)
        tokens = self._close_token_report()
        self.status_message.emit(
            f"Dịch hoàn tất qua {self._last_stage_count} giai đoạn — {elapsed_str}."
            + (f" {reuse}" if reuse else "")
            + (f" {tokens}" if tokens else "")
        )
        self.translation_finished.emit()

    def _from_stale_worker(self) -> bool:
        """Tín hiệu đến từ một worker CŨ (không phải worker hiện tại).

        Gọi trực tiếp (không qua tín hiệu, vd trong test) thì ``sender()`` là None —
        coi như của worker hiện tại.
        """
        sender = self.sender()
        return sender is not None and sender is not self._worker

    def _on_worker_failed(self, message: str) -> None:
        # [v3.23.609] Cùng lý do với ``_on_worker_finished``: lỗi đến muộn của worker cũ
        # không được hiện thành lỗi của lượt hiện tại (dịch cả bộ còn chuyển tập theo nó).
        if self._from_stale_worker():
            return
        self.error_occurred.emit(message)
        tokens = self._close_token_report()
        if tokens:
            self.status_message.emit(f"Lượt dịch dừng vì lỗi. {tokens}")

    def _on_worker_cancelled(self) -> None:
        if self._from_stale_worker():
            return
        # [v3.7 fix] Huỷ là kết thúc ÊM — KHÔNG phát error_occurred (tránh hiện
        # cảnh báo lỗi đỏ). Chỉ báo trạng thái trung tính.
        tokens = self._close_token_report()
        self.status_message.emit(
            "Đã huỷ dịch. Giữ nguyên kết quả trước đó (nếu có)."
            + (f" {tokens}" if tokens else "")
        )
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
            # [v3.23.609] Thêm video_refs_ready / models_ready: còn nối thì luồng phân tích
            # vẫn ghi phiên sau khi kho dữ liệu đã đóng lúc thoát ứng dụng.
            for sig_name in ("progress_changed", "finished_ok", "failed", "cancelled",
                             "progress_message", "video_refs_ready", "models_ready",
                             "finished"):
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
                # [v3.23.639] Key bị server từ chối vĩnh viễn / báo hết hạn mức ngày hôm nay.
                "invalid": mgr.invalid_key_reason(fp),
                "server_blocked": mgr.is_server_daily_blocked(model, key_id=fp),
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
        self.status_message.emit(
            "Đang tải danh sách model từ nhà cung cấp tuỳ chỉnh…" if self.custom_provider_active()
            else "Đang tải danh sách model từ Gemini API…"
        )
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
            # [v3.23.609] Gửi BẢN SAO: trang Biên tập đánh số lại / giữ trong ngăn hoàn
            # tác các dòng này; sửa tiếp ở trang Dịch không được đổi ngầm chúng.
            self._push_to_editor_callback(self.translated_events)
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

    @property
    def last_token_report(self) -> str:
        """[v3.23.610] Token và chi phí ước tính của việc vừa xong (rỗng: không gọi)."""
        return self._last_token_report

    def token_usage_mark(self) -> dict[str, TokenUsage]:
        """[v3.23.610] Mốc sổ token hiện tại (vd lúc bắt đầu "Dịch cả bộ")."""
        ledger = self._token_ledger()
        return ledger.snapshot() if ledger is not None else {}

    def describe_token_usage_since(self, mark: dict[str, TokenUsage]) -> str:
        """[v3.23.610] Câu tóm tắt token đã dùng từ mốc ``mark``.

        Args:
            mark: Kết quả :meth:`token_usage_mark` lúc bắt đầu.

        Returns:
            Câu tiếng Việt; rỗng nếu không có lời gọi nào.
        """
        return self._describe_usage(self._usage_since(mark))

    def _token_ledger(self) -> TokenUsageLedger | None:
        ledger = getattr(self._container, "token_usage_ledger", None)
        return ledger if isinstance(ledger, TokenUsageLedger) else None

    def _usage_since(self, mark: dict[str, TokenUsage]) -> dict[str, TokenUsage]:
        ledger = self._token_ledger()
        return ledger.since(mark) if ledger is not None else {}

    def _describe_usage(self, usage: dict[str, TokenUsage]) -> str:
        if not usage:
            return ""
        table = getattr(self._container, "gemini_pricing_table", None)
        estimate = (
            estimate_cost(usage, table) if isinstance(table, PricingTable) else None
        )
        return describe_token_usage(usage, estimate)

    def _close_token_report(self) -> str:
        """Báo cáo token của việc vừa xong (tính từ mốc lúc bắt đầu)."""
        self._last_token_usage = self._usage_since(self._token_mark)
        self._last_token_report = self._describe_usage(self._last_token_usage)
        self._token_mark = self.token_usage_mark()
        return self._last_token_report

    # ── Nội bộ ───────────────────────────────────────────────────────────────
    def _set_busy(self, busy: bool) -> None:
        if busy and not self._is_busy:
            # [v3.23.610] Bắt đầu một việc: đánh dấu sổ token để báo phần việc này dùng.
            self._token_mark = self.token_usage_mark()
        if busy != self._is_busy:
            self._is_busy = busy
            self.busy_changed.emit(busy)
        # [v3.23.535] LỚP BẢO VỆ THỨ HAI cho lỗi "không bấm được gì, phải End Task": giao diện chỉ
        # được mở khoá trong handler ``finished`` của luồng, nên nếu ``run()`` treo (lời gọi mạng
        # không thời hạn, tiến trình con kẹt…) thì giao diện khoá VĨNH VIỄN. Đồng hồ canh này định
        # kỳ kiểm luồng còn sống thật không; luồng đã chết mà cờ bận còn bật thì tự mở khoá.
        self._busy_idle_ticks = 0
        if busy:
            self._busy_watchdog.start(BUSY_WATCHDOG_INTERVAL_MS)
        else:
            self._busy_watchdog.stop()

    def _check_busy_watchdog(self) -> None:
        """Mở khoá giao diện nếu đang bận mà KHÔNG còn luồng nào chạy.

        Không đoán "treo" theo thời gian — chỉ xử lý trường hợp chắc chắn sai: cờ bận còn bật
        trong khi mọi luồng đã kết thúc (tín hiệu ``finished`` bị mất, worker bị thu gom sớm…).
        """
        if not self._is_busy:
            self._busy_watchdog.stop()
            return
        workers = (
            getattr(self, "_worker", None),
            getattr(self, "_analyze_worker", None),
        )
        alive = any(
            worker is not None and getattr(worker, "isRunning", lambda: False)()
            for worker in workers
        )
        if alive:
            self._busy_idle_ticks = 0
            return
        # [v3.23.550] Đòi HAI nhịp liên tiếp mới kết luận. Luồng thoát ``run()`` thì ``isRunning``
        # về False ngay, nhưng tín hiệu kết thúc của nó có thể vẫn đang XẾP HÀNG trong vòng lặp sự
        # kiện. App_log thật: watchdog bắn 158ms sau khi phân tích xong — không có tín hiệu nào bị
        # mất cả, chỉ là đồng hồ canh "nhanh tay" hơn handler hợp lệ. Nhịp thứ hai (sau
        # ``BUSY_WATCHDOG_INTERVAL_MS``) thừa thời gian cho handler chạy xong.
        self._busy_idle_ticks = getattr(self, "_busy_idle_ticks", 0) + 1
        if self._busy_idle_ticks < BUSY_WATCHDOG_CONFIRM_TICKS:
            return
        logger.warning(
            "Giao diện vẫn ở trạng thái BẬN nhưng không còn luồng nào chạy — tự mở khoá "
            "(tín hiệu kết thúc có thể đã bị mất)."
        )
        self._set_busy(False)


__all__ = ["TranslatePageViewModel"]
