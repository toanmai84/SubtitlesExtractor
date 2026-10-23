"""[v3.23.624] Worker QThread cho nút "Tự kiểm tra & tối ưu phân tích".

Lấy phân tích HIỆN CÓ trên giao diện (kể cả phần người dùng đã sửa) làm gốc, đối chiếu với
phụ đề rồi vá khiếm khuyết — không tải video, không phân tích lại từ đầu.
"""

from __future__ import annotations

from PySide6.QtCore import QObject

from subtitles_extractor.application.services.analysis_audit import (
    DEFAULT_AUDIT_MODEL,
    describe_audit_report,
)
from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    TranslationLine,
)
from subtitles_extractor.presentation.workers.analyze_context_worker import (
    AnalyzeContextWorker,
)
from subtitles_extractor.presentation.workers.video_reupload import VideoReuploadCallback


class RefineAnalysisWorker(AnalyzeContextWorker):
    """Vá phân tích cũ trên thread nền. ``finished_ok`` phát :class:`SubtitleContextAnalysis`."""

    def __init__(
        self,
        use_case: AnalyzeSubtitleContextUseCase,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str,
        current: SubtitleContextAnalysis,
        parent: QObject | None = None,
        audit_model: str = "",
        audit_fallbacks: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            use_case, source_lines, target_lang, model_name=model_name, parent=parent,
            audit_model=audit_model, audit_fallbacks=audit_fallbacks,
        )
        self._current = current
        #: [v3.23.631] Câu tóm tắt lần kiểm tra vừa xong (cho thanh trạng thái).
        self.audit_note = ""

    def _needed_requests(self) -> int:
        """Số lời gọi tối đa (theo ``AUDIT_MAX_CALLS``)."""
        return 6

    def _execute(
        self, video_refs: list | None, video_reupload_cb: VideoReuploadCallback | None
    ) -> object:
        """Tự kiểm tra rồi vá phân tích hiện có.

        [v3.23.631] Mặc định dùng model kiểm định rẻ (``DEFAULT_AUDIT_MODEL``, 500 request/ngày) thay
        vì model phân tích đang chọn (3.8-flash chỉ 20 request/ngày/key, mà 24 đoạn video đã cần):
        bản vá là việc cơ học. [v3.23.640] Người dùng chọn được model kiểm định riêng + dự phòng.
        """
        result = self._use_case.refine(
            self._current, self._source_lines, self._target_lang,
            self._audit_model or DEFAULT_AUDIT_MODEL,
            cancel_cb=self._is_cancelled, fallback_models=self._audit_fallbacks,
        )
        self.audit_note = describe_audit_report(
            getattr(self._use_case, "last_audit_report", None)
        )
        return result


__all__ = ["RefineAnalysisWorker"]
