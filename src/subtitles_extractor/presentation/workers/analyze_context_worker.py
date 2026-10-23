"""Worker QThread để chạy phân tích ngữ cảnh toàn cục trên thread nền."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Event

from PySide6.QtCore import QThread, Signal

from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_clock import (
    describe_quota_reset,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    QuotaExhaustedError,
)
from subtitles_extractor.presentation.workers.cancellation_outcome import (
    CancellationOutcome,
    classify_cancellation_outcome,
)
from subtitles_extractor.presentation.workers.video_reupload import (
    VideoReuploadCallback,
    make_video_reupload_callback,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    SubtitleTranslationError,
    TranslationCancelledError,
    TranslationLine,
)

logger = logging.getLogger(__name__)


class _WorkerStopped(Exception):
    """[v3.23.601] Worker đã phát tín hiệu huỷ/lỗi trong lúc chuẩn bị video — dừng êm."""


class AnalyzeContextWorker(QThread):
    """Chạy phân tích ngữ cảnh trên thread nền, phát kết quả qua signal.

    Signals:
        finished_ok: Phát ``SubtitleContextAnalysis`` khi hoàn tất.
        cancelled:   Phát khi người dùng huỷ.
        failed:      Phát thông điệp lỗi khi thất bại.
    """

    finished_ok = Signal(object)   # SubtitleContextAnalysis
    cancelled = Signal()
    failed = Signal(str)
    progress_message = Signal(str)   # báo tiến trình chuẩn bị video
    video_refs_ready = Signal(object)  # [v3.23.18] list[RemoteVideoRef] đã upload

    #: [v3.23.601] Phân tích vẫn chạy được từ văn bản khi nén video hỏng; lớp con cần hình
    #: (hỏi người nói) đặt True để báo lỗi thay vì chạy mù.
    _VIDEO_REQUIRED = False

    def __init__(
        self,
        use_case: AnalyzeSubtitleContextUseCase,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str = "gemini-3.5-flash-lite",
        parent=None,
        video_provider=None,
        video_path=None,
        enable_visual_cues: bool = False,
        prior_context: str = "",
        fallback_model: str = "",
        text_cues: bool = False,
        voice_source_path: str = "",
        audit_model: str = "",
        audit_fallbacks: tuple[str, ...] = (),
    ) -> None:
        super().__init__(parent)
        #: [v3.23.640] Model tự kiểm tra (rỗng = mặc định rẻ) và các model dự phòng của nó.
        self._audit_model = audit_model
        self._audit_fallbacks = tuple(audit_fallbacks)
        self._use_case = use_case
        self._source_lines = source_lines
        self._target_lang = target_lang
        self._model_name = model_name
        self._cancel_event = Event()
        # Ngữ cảnh video (tuỳ chọn): phân tích GỘP tất cả đoạn.
        self._video_provider = video_provider
        self._video_path = video_path
        self._enable_visual_cues = enable_visual_cues
        self._prior_context = prior_context
        self._text_cues = bool(text_cues)
        self._voice_source_path = voice_source_path
        # [v3.23.621] Model dự phòng khi model đã chọn hết hạn mức NGÀY (vd 3.5-flash chỉ
        # 20 request/ngày/key) — phân tích tiếp bằng model hạn mức cao thay vì dừng.
        self._fallback_model = (
            fallback_model if fallback_model and fallback_model != model_name else ""
        )

    def _switch_to_fallback(self, reason: str) -> bool:
        """Chuyển sang model dự phòng (một lần). Trả ``False`` nếu không còn dự phòng."""
        if not self._fallback_model:
            return False
        logger.warning(
            "Phân tích: model '%s' %s → dùng model dự phòng '%s'.",
            self._model_name, reason, self._fallback_model,
        )
        self.progress_message.emit(
            f"Model '{self._model_name}' {reason} → phân tích bằng '{self._fallback_model}'."
        )
        self._model_name, self._fallback_model = self._fallback_model, ""
        return True

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def _is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def run(self) -> None:
        try:
            video_refs = self._prepare_video_refs()
            # [v3.23.132] Callback tải lại video bằng key HIỆN TẠI của adapter (chống 403
            # khi xoay key: file của key cũ không truy cập được bằng key mới).
            video_reupload_cb = None
            if self._video_provider is not None and self._video_path:
                video_reupload_cb = make_video_reupload_callback(
                    self._video_provider, self._video_path, self._is_cancelled,
                    on_uploaded=self.video_refs_ready.emit,
                )
            try:
                result = self._execute(video_refs, video_reupload_cb)
            except QuotaExhaustedError as exc:
                logger.warning("Hết hạn mức ngày khi phân tích ngữ cảnh: %s", exc)
                if not self._switch_to_fallback("đã hết hạn mức ngày"):
                    raise
                result = self._execute(video_refs, video_reupload_cb)
            self.finished_ok.emit(result)
        except _WorkerStopped:
            return
        except TranslationCancelledError as exc:
            # [v3.23.609] Như khâu dịch (v3.23.175): chỉ là "người dùng huỷ" khi cờ huỷ
            # THẬT SỰ bật. Gián đoạn do timeout/lỗi mạng kéo dài cũng ném lỗi này — trước
            # đây chỉ hiện "Đã huỷ phân tích." ở dòng trạng thái, lỗi thật bị giấu.
            if (
                classify_cancellation_outcome(self._is_cancelled())
                is CancellationOutcome.USER_CANCELLED
            ):
                logger.info("Người dùng đã huỷ phân tích ngữ cảnh.")
                self.cancelled.emit()
            else:
                logger.warning(
                    "Phân tích dừng do gián đoạn KHÔNG phải người dùng huỷ: %s", exc
                )
                self.failed.emit(
                    "Phân tích dừng do gián đoạn kết nối tới Gemini (timeout hoặc lỗi "
                    "mạng), không phải do bạn huỷ. Hãy thử lại — các đoạn video đã tải "
                    "lên được dùng lại nên sẽ nhanh hơn."
                )
        except QuotaExhaustedError as exc:
            # [v3.23.527] Hết hạn mức NGÀY là tình huống BÌNH THƯỜNG của free-tier, không phải
            # lỗi lập trình. Trước đây rơi vào nhánh "Lỗi không lường trước" nên giao diện báo
            # như crash, người dùng không biết phải làm gì. Nay báo rõ và gợi ý cách xử lý.
            logger.warning("Hết hạn mức ngày khi phân tích ngữ cảnh: %s", exc)
            self.failed.emit(
                f"{exc}\n\n"
                "Gợi ý: hạn mức miễn phí theo NGÀY của Gemini rất nhỏ (ví dụ 20 request/ngày "
                "cho các model Flash mới), trong khi phân tích một phim cần vài chục request. "
                "Có thể: (1) chọn model có hạn mức ngày cao hơn như bản 'flash-lite'; "
                "(2) thêm API key thuộc DỰ ÁN KHÁC — nhiều key cùng một dự án vẫn dùng chung "
                "hạn mức; (3) chờ tới mốc hạn mức hồi nêu ở trên; "
                "(4) bật thanh toán cho dự án để nâng hạn mức."
            )
        except SubtitleTranslationError as exc:
            logger.warning("Phân tích ngữ cảnh thất bại: %s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi không lường trước khi phân tích ngữ cảnh.")
            self.failed.emit(f"Lỗi không xác định: {exc}")

    def _needed_requests(self) -> int:
        """[v3.23.157] Số request dự trù cho trọn phiên (~1 request mỗi đoạn video)."""
        # [v3.23.645] Bước bổ sung người nói chạy NGAY SAU các đoạn phân tích (trên cùng model,
        # cùng key) nên cũng tính vào dự trù — trước đây chọn key chỉ đủ cho các đoạn, key vừa
        # đủ cạn quota giữa lượt bổ sung và phải xoay key giữa chừng.
        from subtitles_extractor.application.services.cue_completion import (
            SPEAKER_COMPLETION_PASSES,
        )

        extra = SPEAKER_COMPLETION_PASSES if self._enable_visual_cues else 0
        estimator = getattr(self._video_provider, "estimate_chunk_count", None)
        if estimator is None:
            return 1 + extra
        try:
            return int(estimator(Path(self._video_path))) + extra
        except (OSError, ValueError, TypeError):
            return 1 + extra

    def _prepare_video_refs(self) -> list | None:
        """Chọn key còn hạn mức rồi nén/tải video ngữ cảnh (hoặc dùng lại bản đã tải).

        Returns:
            Tham chiếu các đoạn video; None khi không có video (hoặc nén hỏng mà video
            không bắt buộc — phân tích từ văn bản phụ đề).

        Raises:
            _WorkerStopped: Đã phát ``cancelled``/``failed`` — dừng worker.
        """
        if self._video_provider is None or not self._video_path:
            return None
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            VideoContextError,
        )

        # [v3.23.145] CHỌN KEY CÒN QUOTA TRƯỚC KHI UPLOAD: nếu key hiện tại đã hết
        # quota ngày, xoay sang key còn hạn NGAY và set CÙNG key đó cho video
        # provider -> video được upload/tra-cache đúng dưới key sẽ dùng, tránh 403 (file
        # thuộc key cũ) + tải lại (nén lại) nhiều lần.
        # [v3.23.146] DỰ ĐOÁN: nếu MỌI key đã cạn quota ngày, fail NGAY thay vì phí
        # ~2 phút nén+upload rồi mới 429. [v3.23.621] Còn model dự phòng thì đổi sang nó
        # TRƯỚC khi chọn key, để video được tải dưới key còn hạn mức cho model sẽ dùng.
        if not self._use_case.has_any_daily_quota(self._model_name) and not (
            self._switch_to_fallback("đã hết hạn mức ngày ở mọi key")
            and self._use_case.has_any_daily_quota(self._model_name)
        ):
            self.failed.emit(
                "Tất cả API key đã hết quota ngày cho model này. "
                f"{describe_quota_reset()} Hãy thử lại lúc đó hoặc thêm API key mới."
            )
            raise _WorkerStopped
        viable_key = self._use_case.ensure_viable_key(
            self._model_name, needed_requests=self._needed_requests()
        )
        if viable_key:
            self._video_provider.set_active_key(viable_key)
        self.progress_message.emit("Chuẩn bị ngữ cảnh video cho phân tích…")
        try:
            video_refs = self._video_provider.prepare_and_upload(
                Path(self._video_path),
                progress_cb=self.progress_message.emit,
                cancel_cb=self._is_cancelled,
            )
        except VideoContextError as exc:
            if self._is_cancelled():
                self.cancelled.emit()
                raise _WorkerStopped from exc
            if self._VIDEO_REQUIRED:
                self.failed.emit(f"Không chuẩn bị được video ngữ cảnh: {exc}")
                raise _WorkerStopped from exc
            # [v3.23.359] SUY BIẾN MỀM: nén/upload video ngữ cảnh hỏng (vd ffmpeg TREO cả
            # GPU lẫn CPU trên một số MKV/driver) KHÔNG được làm hỏng CẢ phân tích. Bỏ
            # video, phân tích TỪ VĂN BẢN phụ đề — vẫn ra source_lang, bảng nhân vật, tóm
            # tắt, thuật ngữ (chỉ thiếu gợi ý hình ảnh từ video).
            logger.warning(
                "Chuẩn bị ngữ cảnh video thất bại (%s) → chuyển sang phân tích TỪ "
                "VĂN BẢN phụ đề (không kèm video).", exc,
            )
            self.progress_message.emit(
                "Không nén được video ngữ cảnh → phân tích từ phụ đề (bỏ video)."
            )
            return None
        if video_refs:
            # [v3.23.18] Phát refs để luồng chính lưu vào phiên dịch (cloud_files).
            self.video_refs_ready.emit(list(video_refs))
        return video_refs

    def _execute(
        self, video_refs: list | None, video_reupload_cb: VideoReuploadCallback | None
    ) -> object:
        """Chạy use case với video đã chuẩn bị (lớp con thay bằng việc riêng)."""
        result: SubtitleContextAnalysis = self._use_case.execute(
            source_lines=self._source_lines,
            target_lang=self._target_lang,
            model_name=self._model_name,
            cancel_cb=self._is_cancelled,
            video_refs=video_refs,
            enable_visual_cues=self._enable_visual_cues,
            progress_cb=lambda _p, msg: self.progress_message.emit(msg),
            prior_context=self._prior_context,
            video_reupload_cb=video_reupload_cb,
            text_cues=self._text_cues,
            voice_source_path=self._voice_source_path,
            audit_model=self._audit_model,
            audit_fallbacks=self._audit_fallbacks,
        )
        return result


__all__ = ["AnalyzeContextWorker"]
