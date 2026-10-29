"""[v3.23.601] Worker QThread cho nút "Bổ sung người nói" (hỏi lại dòng thiếu người nói).

Cùng khâu chuẩn bị video với phân tích ngữ cảnh (chọn key còn hạn mức, dùng lại đoạn video
đã tải, tải lại khi adapter xoay key) — chỉ khác việc chạy sau đó, và video là BẮT BUỘC:
người nói đọc từ hình, không đoán từ chữ.
"""

from __future__ import annotations

from PySide6.QtCore import QObject

from subtitles_extractor.application.services.cue_completion import (
    SPEAKER_COMPLETION_PASSES,
)
from subtitles_extractor.application.use_cases.complete_speakers import (
    CompleteSpeakersUseCase,
    SpeakerCompletionResult,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.presentation.workers.analyze_context_worker import (
    AnalyzeContextWorker,
)
from subtitles_extractor.presentation.workers.video_reupload import VideoReuploadCallback


class CompleteSpeakersWorker(AnalyzeContextWorker):
    """Hỏi lại người nói trên thread nền.

    ``finished_ok`` phát :class:`SpeakerCompletionResult`.
    """

    _VIDEO_REQUIRED = True

    def __init__(
        self,
        use_case: CompleteSpeakersUseCase,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str,
        *,
        video_provider: object,
        video_path: str,
        visual_cues: str,
        characters: str,
        parent: QObject | None = None,
    ) -> None:
        """Khởi tạo worker.

        Args:
            use_case: Use case bổ sung người nói.
            source_lines: Toàn bộ dòng phụ đề gốc.
            target_lang: Ngôn ngữ đích.
            model_name: Model Gemini có thị giác.
            video_provider: Bộ chuẩn bị/tải video ngữ cảnh.
            video_path: Đường dẫn video của phim.
            visual_cues: JSON Visual Cues hiện có.
            characters: Bảng nhân vật.
            parent: QObject cha.
        """
        super().__init__(
            use_case,  # type: ignore[arg-type]  # cùng hai câu hỏi hạn mức
            source_lines,
            target_lang,
            model_name=model_name,
            parent=parent,
            video_provider=video_provider,
            video_path=video_path,
            enable_visual_cues=True,
        )
        self._speaker_use_case = use_case
        self._visual_cues = visual_cues
        self._characters = characters

    def _needed_requests(self) -> int:
        """Chỉ các dòng thiếu, tối đa hai lượt hỏi."""
        return SPEAKER_COMPLETION_PASSES

    def _execute(
        self, video_refs: list | None, video_reupload_cb: VideoReuploadCallback | None
    ) -> SpeakerCompletionResult:
        """Hỏi lại người nói với video đã chuẩn bị."""
        return self._speaker_use_case.execute(
            self._source_lines,
            self._visual_cues,
            characters=self._characters,
            target_lang=self._target_lang,
            model_name=self._model_name,
            video_refs=list(video_refs or []),
            cancel_cb=self._is_cancelled,
            progress_cb=lambda _fraction, message: self.progress_message.emit(message),
            video_reupload_cb=video_reupload_cb,
        )


class RecheckCuesWorker(CompleteSpeakersWorker):
    """[v3.23.655] Hỏi lại các dòng Visual Cues đáng nghi trên thread nền (``finished_ok`` phát ``CueRecheckResult``)."""

    def _needed_requests(self) -> int:
        return 2

    def _execute(self, video_refs, video_reupload_cb):
        return self._speaker_use_case.execute(
            self._source_lines,
            self._visual_cues,
            characters=self._characters,
            target_lang=self._target_lang,
            model_name=self._model_name,
            video_refs=list(video_refs or []),
            cancel_cb=self._is_cancelled,
            progress_cb=lambda _fraction, message: self.progress_message.emit(message),
            video_reupload_cb=video_reupload_cb,
        )


__all__ = ["CompleteSpeakersWorker", "RecheckCuesWorker"]
