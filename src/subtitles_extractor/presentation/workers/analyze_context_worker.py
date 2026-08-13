"""Worker QThread để chạy phân tích ngữ cảnh toàn cục trên thread nền."""

from __future__ import annotations

import logging
from threading import Event

from PySide6.QtCore import QThread, Signal

from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    SubtitleTranslationError,
    TranslationCancelledError,
    TranslationLine,
)

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        use_case: AnalyzeSubtitleContextUseCase,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str = "gemini-3.1-flash-lite",
        parent=None,
        video_provider=None,
        video_path=None,
        enable_visual_cues: bool = False,
        prior_context: str = "",
    ) -> None:
        super().__init__(parent)
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

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def _is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def run(self) -> None:
        try:
            video_refs = None
            if self._video_provider is not None and self._video_path:
                from pathlib import Path

                from subtitles_extractor.infrastructure.translation.gemini_video_context import (
                    VideoContextError,
                )
                # [v3.23.145] CHỌN KEY CÒN QUOTA TRƯỚC KHI UPLOAD: nếu key hiện tại đã hết
                # quota ngày, xoay sang key còn hạn NGAY và set CÙNG key đó cho video
                # provider -> video được upload/tra-cache đúng dưới key sẽ dùng phân tích,
                # tránh 403 (file thuộc key cũ) + tải lại (nén lại) nhiều lần. Không còn dò mù.
                # [v3.23.157] DỰ TRÙ số request = số đoạn video (phân tích tuần tự tốn
                # ~1 request/đoạn) -> chọn key ĐỦ quota đi trọn phiên ngay từ đầu.
                needed_requests = 1
                estimator = getattr(self._video_provider, "estimate_chunk_count", None)
                if estimator is not None:
                    try:
                        needed_requests = int(estimator(Path(self._video_path)))
                    except (OSError, ValueError, TypeError):
                        needed_requests = 1
                viable_key = self._use_case.ensure_viable_key(
                    self._model_name, needed_requests=needed_requests
                )
                if viable_key:
                    self._video_provider.set_active_key(viable_key)
                # [v3.23.146] DỰ ĐOÁN: nếu MỌI key đã cạn quota ngày, fail NGAY thay vì phí
                # ~2 phút nén+upload rồi mới 429. Quota reset lúc nửa đêm giờ Thái Bình Dương.
                if not self._use_case.has_any_daily_quota(self._model_name):
                    self.failed.emit(
                        "Tất cả API key đã hết quota ngày cho model này. Hãy thử lại sau "
                        "(quota reset lúc nửa đêm giờ Thái Bình Dương) hoặc thêm API key mới."
                    )
                    return
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
                        return
                    # [v3.23.359] SUY BIẾN MỀM: nén/upload video ngữ cảnh hỏng (vd ffmpeg
                    # TREO cả GPU lẫn CPU trên một số MKV/driver) KHÔNG được làm hỏng CẢ
                    # phân tích. Bỏ video, phân tích TỪ VĂN BẢN phụ đề — vẫn ra source_lang,
                    # bảng nhân vật, tóm tắt, thuật ngữ (chỉ thiếu gợi ý hình ảnh từ video).
                    logger.warning(
                        "Chuẩn bị ngữ cảnh video thất bại (%s) → chuyển sang phân tích TỪ "
                        "VĂN BẢN phụ đề (không kèm video).", exc,
                    )
                    self.progress_message.emit(
                        "Không nén được video ngữ cảnh → phân tích từ phụ đề (bỏ video)."
                    )
                    video_refs = None
                if video_refs:
                    # [v3.23.18] Phát refs để luồng chính lưu vào phiên dịch (cloud_files).
                    self.video_refs_ready.emit(list(video_refs))

            # [v3.23.132] Callback tải lại video bằng key HIỆN TẠI của adapter (chống 403
            # khi xoay key: file của key cũ không truy cập được bằng key mới).
            video_reupload_cb = None
            if self._video_provider is not None and self._video_path:
                from pathlib import Path as _Path

                def video_reupload_cb(api_key: str) -> list:
                    self._video_provider.set_active_key(api_key)
                    return list(
                        self._video_provider.prepare_and_upload(
                            _Path(self._video_path),
                            cancel_cb=self._is_cancelled,
                        )
                    )

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
            )
            self.finished_ok.emit(result)
        except TranslationCancelledError:
            logger.info("Người dùng đã huỷ phân tích ngữ cảnh.")
            self.cancelled.emit()
        except SubtitleTranslationError as exc:
            logger.warning("Phân tích ngữ cảnh thất bại: %s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lỗi không lường trước khi phân tích ngữ cảnh.")
            self.failed.emit(f"Lỗi không xác định: {exc}")


__all__ = ["AnalyzeContextWorker"]
