"""[v3.23.601] Hỏi lại NGƯỜI NÓI cho các dòng có lời còn thiếu, dựa trên video đã tải lên.

Khâu phân tích (và khâu dịch có kèm video) đã tự hỏi lại các dòng bỏ trống từ v3.23.600.
Nhưng phân tích KHÔI PHỤC từ phiên cũ thì không qua hai khâu đó: chuẩn đoán thật
灭门惨案II (v3.23.600, 18:05) khôi phục phân tích cũ rồi lồng tiếng ngay — 39/1246 dòng
không có người nói, đọc bằng giọng chung (dòng 714 "Đừng!" của Tần Khả Tú đọc giọng
nam). Use case này là đường riêng, rẻ (chỉ các dòng thiếu, tối đa hai lượt) cho trường hợp
đó, và là phần lõi dùng chung với khâu phân tích.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from subtitles_extractor.application.services.cue_completion import (
    SpeakerCompletionOutcome,
    SpeakerCoverage,
    complete_and_check_speakers,
)
from subtitles_extractor.application.services.cue_label_canonicalizer import (
    canonicalize_cue_labels,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    serialize_visual_cues,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    SubtitleTranslatorPort,
    TranslationCancelledError,
    TranslationLine,
    VisualCue,
)

logger = logging.getLogger(__name__)

CancellationCallback = Callable[[], bool]
ProgressCallback = Callable[[float, str], None]

#: Lỗi một lượt hỏi không được làm đổ cả khâu (mạng, hạn mức, model trả rác).
_SCAN_ERRORS = (RuntimeError, ValueError, OSError, SubtitleTranslationError)


class SpeakerCompletionUnsupportedError(SubtitleTranslationError):
    """Bộ dịch hiện tại không có khả năng hỏi người nói từ video."""


@dataclass(frozen=True, slots=True)
class SpeakerCompletionResult:
    """Kết quả bổ sung người nói.

    Attributes:
        visual_cues: JSON Visual Cues sau khi điền (nguyên bản khi không điền thêm dòng
            nào).
        before: Độ phủ người nói trước khi hỏi.
        after: Độ phủ sau khi hỏi.
        dropped: Các dòng có nhãn vừa hỏi được nhưng bị bỏ vì lạc cảnh.
        demoted: [v3.23.602] Các dòng có nhãn tên vừa hỏi được nhưng người đó không có
            mặt trong cảnh → đã hạ xuống nhãn giọng.
    """

    visual_cues: str
    before: SpeakerCoverage
    after: SpeakerCoverage
    dropped: tuple[int, ...] = ()
    demoted: tuple[int, ...] = ()

    @property
    def changed(self) -> bool:
        """Có thêm dòng nào được điền người nói không."""
        return self.after.with_speaker != self.before.with_speaker


class GeminiQuotaAwareUseCase:
    """Hai câu hỏi hạn mức mà worker hỏi TRƯỚC khi nén/tải video (uỷ cho adapter)."""

    _translator: SubtitleTranslatorPort

    def ensure_viable_key(self, model_name: str, needed_requests: int = 1) -> str | None:
        """[v3.23.145] Chọn API key còn quota ngày TRƯỚC khi upload video (uỷ adapter).

        [v3.23.157] ``needed_requests``: số request dự trù cho trọn phiên (vd số đoạn
        phân tích) để adapter chọn key ĐỦ quota ngay từ đầu.

        Args:
            model_name: Model sẽ dùng.
            needed_requests: Số request dự trù.

        Returns:
            API key đã chọn, hoặc None nếu adapter không hỗ trợ/không chọn được.
        """
        fn = getattr(self._translator, "ensure_viable_key", None)
        if fn is None:
            return None
        try:
            return fn(model_name, needed_requests=needed_requests)
        except TypeError:
            try:
                return fn(model_name)  # adapter cũ chưa có tham số
            except (AttributeError, TypeError, ValueError):
                return None
        except (AttributeError, ValueError):
            return None

    def has_any_daily_quota(self, model_name: str) -> bool:
        """[v3.23.146] True nếu còn ÍT NHẤT một key có quota ngày (để fail nhanh nếu cạn).

        Args:
            model_name: Model sẽ dùng.

        Returns:
            True khi còn quota hoặc adapter không trả lời được câu hỏi này.
        """
        fn = getattr(self._translator, "has_any_daily_quota", None)
        if fn is None:
            return True
        try:
            return bool(fn(model_name))
        except (AttributeError, TypeError, ValueError):
            return True


class _ModelCueScanner:
    """Hàm ``scan`` cho :func:`complete_and_check_speakers`: hỏi model các dòng thiếu.

    Lỗi của một lượt được ghi lại (không làm đổ khâu chính); người dùng THẬT SỰ huỷ thì
    ném tiếp.
    """

    def __init__(
        self,
        analyze: Callable[..., list[VisualCue]],
        *,
        target_lang: str,
        model_name: str,
        video_refs: list[Any],
        batch_size: int,
        characters: str,
        context_lines: list[TranslationLine],
        cancel_cb: CancellationCallback | None,
        progress_cb: ProgressCallback | None,
        scanned_sink: list[VisualCue] | None = None,
    ) -> None:
        self._analyze = analyze
        self._scanned_sink = scanned_sink
        self._target_lang = target_lang
        self._model_name = model_name
        self._video_refs = video_refs
        self._batch_size = batch_size
        self._characters = characters
        self._context_lines = context_lines
        self._cancel_cb = cancel_cb
        self._progress_cb = progress_cb
        self.calls = 0
        self.errors: list[Exception] = []

    def _report(self, fraction: float) -> None:
        if self._progress_cb is not None:
            self._progress_cb(
                fraction, f"Hỏi lại người nói các dòng còn thiếu: {fraction:.0%}"
            )

    def __call__(
        self, lines: list[TranslationLine], known: list[VisualCue]
    ) -> list[VisualCue]:
        self.calls += 1
        try:
            found = self._analyze(
                lines, self._target_lang, model_name=self._model_name,
                video_refs=list(self._video_refs), batch_size=self._batch_size,
                cancel_cb=self._cancel_cb, progress_cb=self._report,
                characters=self._characters, known_cues=known,
                context_lines=self._context_lines,
            )
        except TranslationCancelledError as exc:
            if self._cancel_cb is not None and self._cancel_cb():
                raise
            logger.warning("Hỏi lại người nói báo 'đã huỷ' — bỏ qua lượt này: %s", exc)
            self.errors.append(exc)
            return []
        except _SCAN_ERRORS as exc:
            logger.warning("Hỏi lại người nói thất bại — giữ nhãn sẵn có: %s", exc)
            self.errors.append(exc)
            return []
        canonical = canonicalize_cue_labels(found, self._characters)
        if self._scanned_sink is not None:
            self._scanned_sink.extend(canonical)
        return canonical


class CompleteSpeakersUseCase(GeminiQuotaAwareUseCase):
    """Điền người nói cho mọi dòng có lời còn thiếu, giữ nguyên nhãn đã có."""

    def __init__(self, translator: SubtitleTranslatorPort) -> None:
        """Khởi tạo use case.

        Args:
            translator: Adapter dịch/phân tích (có ``analyze_visual_cues``).
        """
        self._translator = translator

    def complete(
        self,
        cues: list[VisualCue],
        source_lines: list[TranslationLine],
        *,
        characters: str,
        target_lang: str,
        model_name: str,
        video_refs: list[Any],
        batch_size: int = 150,
        cancel_cb: CancellationCallback | None = None,
        progress_cb: ProgressCallback | None = None,
        raise_when_all_failed: bool = False,
        scanned_sink: list[VisualCue] | None = None,
    ) -> SpeakerCompletionOutcome:
        """Lõi dùng chung: hỏi lại các dòng thiếu người nói (cues đã quy nhãn).

        Args:
            cues: Visual Cues hiện có (đã quy về bảng nhân vật).
            source_lines: Toàn bộ dòng phụ đề gốc.
            characters: Bảng nhân vật.
            target_lang: Ngôn ngữ đích.
            model_name: Model Gemini có thị giác.
            video_refs: Các đoạn video đã tải lên.
            batch_size: Cỡ lô hỏi.
            cancel_cb: Callback huỷ.
            progress_cb: Callback tiến độ ``(tỉ lệ, thông điệp)``.
            raise_when_all_failed: Mọi lượt hỏi đều lỗi thì ném lỗi cuối (nút riêng cần
                báo cho người dùng; khâu phân tích thì giữ kết quả, không ném).
            scanned_sink: Nơi nhận MỌI nhãn model trả về (khâu dịch cộng vào cache
                checkpoint để dịch lại không phải hỏi lại).

        Returns:
            :class:`SpeakerCompletionOutcome`.

        Raises:
            SpeakerCompletionUnsupportedError: Adapter không có ``analyze_visual_cues``.
            TranslationCancelledError: Người dùng huỷ.
            RuntimeError | ValueError | OSError | SubtitleTranslationError: Khi
                ``raise_when_all_failed`` và không lượt hỏi nào thành công.
        """
        analyze = getattr(self._translator, "analyze_visual_cues", None)
        if analyze is None:
            raise SpeakerCompletionUnsupportedError(
                "Bộ dịch hiện tại không hỏi được người nói từ video."
            )
        scanner = _ModelCueScanner(
            analyze, target_lang=target_lang, model_name=model_name,
            video_refs=video_refs, batch_size=batch_size, characters=characters,
            context_lines=source_lines, cancel_cb=cancel_cb, progress_cb=progress_cb,
            scanned_sink=scanned_sink,
        )
        outcome = complete_and_check_speakers(cues, source_lines, characters, scanner)
        all_failed = bool(scanner.calls) and len(scanner.errors) == scanner.calls
        if raise_when_all_failed and all_failed:
            raise scanner.errors[-1]
        after = outcome.after
        logger.info(
            "Người nói sau khi hỏi lại: %d → %d/%d dòng có lời (%.1f%%)%s%s%s%s.",
            outcome.before.with_speaker, after.with_speaker, after.spoken_lines,
            100 * after.ratio,
            f", {after.voice_only} dòng chỉ biết giọng" if after.voice_only else "",
            f", bỏ {len(outcome.dropped)} nhãn lạc cảnh" if outcome.dropped else "",
            f", hạ {len(outcome.demoted)} nhãn tên không có mặt trong cảnh về nhãn giọng "
            f"(dòng {', '.join(map(str, outcome.demoted[:12]))})"
            if outcome.demoted else "",
            f"; CÒN THIẾU {len(after.missing)} dòng: "
            + ", ".join(str(line) for line in after.missing[:15])
            if after.missing else "",
        )
        return outcome

    def execute(
        self,
        source_lines: list[TranslationLine],
        visual_cues: str,
        *,
        characters: str,
        target_lang: str,
        model_name: str,
        video_refs: list[Any],
        batch_size: int = 150,
        cancel_cb: CancellationCallback | None = None,
        progress_cb: ProgressCallback | None = None,
        video_reupload_cb: Callable[..., list] | None = None,
    ) -> SpeakerCompletionResult:
        """Bổ sung người nói cho Visual Cues đã có (nút "Bổ sung người nói").

        Args:
            source_lines: Toàn bộ dòng phụ đề gốc.
            visual_cues: JSON Visual Cues hiện có.
            characters: Bảng nhân vật.
            target_lang: Ngôn ngữ đích.
            model_name: Model Gemini có thị giác.
            video_refs: Các đoạn video đã tải lên (bắt buộc — người nói đọc từ hình).
            batch_size: Cỡ lô hỏi.
            cancel_cb: Callback huỷ.
            progress_cb: Callback tiến độ.
            video_reupload_cb: Tải lại video bằng key hiện tại khi adapter xoay key.

        Returns:
            :class:`SpeakerCompletionResult`.

        Raises:
            SubtitleTranslationError: Không có cues/video, adapter không hỗ trợ, hoặc mọi
                lượt hỏi đều lỗi (kể cả hết hạn mức).
            TranslationCancelledError: Người dùng huỷ.
        """
        if not (visual_cues or "").strip():
            raise SubtitleTranslationError(
                "Chưa có Visual Cues — hãy Phân tích ngữ cảnh (bật phân tích hình ảnh) "
                "trước."
            )
        if not video_refs:
            raise SubtitleTranslationError(
                "Cần video ngữ cảnh để biết ai đang nói — hãy chọn video của phim."
            )
        if video_reupload_cb is not None and hasattr(
            self._translator, "set_video_reupload_callback"
        ):
            self._translator.set_video_reupload_callback(video_reupload_cb)
        cues = canonicalize_cue_labels(deserialize_visual_cues(visual_cues), characters)
        outcome = self.complete(
            cues, source_lines, characters=characters, target_lang=target_lang,
            model_name=model_name, video_refs=video_refs, batch_size=batch_size,
            cancel_cb=cancel_cb, progress_cb=progress_cb, raise_when_all_failed=True,
        )
        return SpeakerCompletionResult(
            visual_cues=(
                serialize_visual_cues(outcome.cues) if outcome.changed else visual_cues
            ),
            before=outcome.before,
            after=outcome.after,
            dropped=outcome.dropped,
            demoted=outcome.demoted,
        )


__all__ = [
    "CompleteSpeakersUseCase",
    "GeminiQuotaAwareUseCase",
    "SpeakerCompletionResult",
    "SpeakerCompletionUnsupportedError",
]
