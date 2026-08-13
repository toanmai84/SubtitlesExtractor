"""Worker chạy :class:`TranslateSubtitlesUseCase` trên QThread.

Tách dịch thuật (gọi mạng, có thể kéo dài nhiều phút) khỏi UI thread để giao
diện không bị treo. Worker phát signal tiến độ, hoàn tất hoặc lỗi; hỗ trợ huỷ
hợp tác (cooperative cancel) qua cờ ``_cancel_requested``.

Mẫu sử dụng (theo quy ước dự án — QObject + QThread.moveToThread không bắt buộc,
ở đây dùng QThread.run() trực tiếp cho gọn vì worker là tác vụ một lần):
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QThread, Signal

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesResponse,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationCancelledError,
)
from subtitles_extractor.presentation.workers.cancellation_outcome import (
    CancellationOutcome,
    classify_cancellation_outcome,
)

logger = logging.getLogger(__name__)


class TranslateWorker(QThread):
    """Chạy pipeline dịch trên thread riêng, phát kết quả qua signal.

    Signals:
        progress_changed: ``(int, str)`` — phần trăm ``0..100`` và mô tả giai đoạn.
        finished_ok:      ``(object,)`` — phát ``TranslateSubtitlesResponse`` khi xong.
        cancelled:        ``()`` — phát khi người dùng huỷ (kết thúc êm, KHÔNG lỗi).
        failed:           ``(str,)`` — thông điệp lỗi tiếng Việt khi thất bại.
    """

    progress_changed = Signal(int, str)
    finished_ok = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        use_case: TranslateSubtitlesUseCase,
        request: TranslateSubtitlesRequest,
        video_provider: Any = None,
        video_path: Any = None,
        attach_video_stages: frozenset | None = None,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._request = request
        self._cancel_requested = False
        # Ngữ cảnh video (tuỳ chọn): nếu có, chuẩn bị + tải lên trước khi dịch.
        self._video_provider = video_provider
        self._video_path = video_path
        self._attach_video_stages = attach_video_stages or frozenset()

    def request_cancel(self) -> None:
        """Yêu cầu huỷ hợp tác. Worker sẽ dừng ở ranh giới lô/giai đoạn kế tiếp."""
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    def _emit_progress(self, fraction: float, stage_label: str) -> None:
        percent = max(0, min(100, int(round(fraction * 100))))
        self.progress_changed.emit(percent, stage_label)

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            request = self._request
            # Chuẩn bị ngữ cảnh video (cắt + tải lên) nếu được yêu cầu.
            if self._video_provider is not None and self._video_path:
                import dataclasses
                from pathlib import Path

                from subtitles_extractor.infrastructure.translation.gemini_video_context import (
                    VideoContextError,
                )
                self._emit_progress(0.0, "Chuẩn bị ngữ cảnh video…")
                # [v3.23.145] Chọn key còn quota TRƯỚC upload (đồng bộ video provider) để
                # video gắn đúng key sẽ dùng dịch -> tránh 403 + tải lại. Lấy model của
                # giai đoạn CÓ đính video (nếu có) làm mốc xét quota.
                video_stage_model = next(
                    (
                        s.model_name
                        for s in request.stages
                        if s.kind in self._attach_video_stages
                    ),
                    request.stages[0].model_name if request.stages else "",
                )
                if video_stage_model:
                    # [v3.23.294] Dự trù số request cho trọn phiên dịch = số giai đoạn
                    # (mỗi giai đoạn ~1 request/đoạn video; video ngắn thường 1 đoạn).
                    # avoid_reupload=True: video sẽ tái dùng bản đã upload ở phase phân
                    # tích → adapter KHÔNG xoay key cơ hội (tránh cắt+nén+upload lại).
                    session_needed_requests = len(request.stages) if request.stages else 1
                    viable_key = self._use_case.ensure_viable_key(
                        video_stage_model,
                        needed_requests=session_needed_requests,
                        avoid_reupload=True,
                    )
                    if viable_key:
                        self._video_provider.set_active_key(viable_key)
                    # [v3.23.146] DỰ ĐOÁN: mọi key cạn quota ngày -> fail ngay, khỏi phí
                    # nén+upload rồi mới 429.
                    if not self._use_case.has_any_daily_quota(video_stage_model):
                        self.failed.emit(
                            "Tất cả API key đã hết quota ngày cho model dịch có đính video. "
                            "Hãy thử lại sau (reset lúc nửa đêm giờ Thái Bình Dương), thêm API "
                            "key, hoặc tắt đính video ở giai đoạn dịch."
                        )
                        return
                try:
                    refs = self._video_provider.prepare_and_upload(
                        Path(self._video_path),
                        progress_cb=lambda msg: self.progress_changed.emit(0, msg),
                        cancel_cb=self._is_cancelled,
                    )
                except VideoContextError as exc:
                    logger.exception("Tác vụ nền translate_worker thất bại.")
                    if self._is_cancelled():
                        self.cancelled.emit()
                        return
                    self.failed.emit(f"Lỗi chuẩn bị ngữ cảnh video: {exc}")
                    return
                request = dataclasses.replace(
                    request,
                    video_refs=tuple(refs),
                    attach_video_stages=self._attach_video_stages,
                )
            response: TranslateSubtitlesResponse = self._use_case.execute(
                request=request,
                progress_cb=self._emit_progress,
                cancel_cb=self._is_cancelled,
            )
            self.finished_ok.emit(response)
        except TranslationCancelledError as exc:
            # [v3.23.175] CHỈ coi là "người dùng huỷ" khi cờ huỷ THẬT SỰ được bật.
            # Trước đây mọi TranslationCancelledError đều báo huỷ, kể cả khi nó phát
            # sinh gián tiếp từ timeout/lỗi mạng kéo dài (phân tích ngữ cảnh 8 đoạn có
            # thể mất 30+ phút kèm 503/504) -> người dùng thấy "đã huỷ" dù không hề bấm
            # huỷ, và lỗi thật bị che giấu. Phân loại bằng hàm thuần để báo đúng.
            if (
                classify_cancellation_outcome(self._is_cancelled())
                is CancellationOutcome.USER_CANCELLED
            ):
                logger.info("Người dùng đã huỷ tiến trình dịch.")
                self.cancelled.emit()
            else:
                logger.warning(
                    "Tiến trình dừng do gián đoạn KHÔNG phải người dùng huỷ "
                    "(thường là timeout/lỗi mạng khi chờ Gemini): %s", exc,
                )
                self.failed.emit(
                    "Tiến trình dừng do gián đoạn kết nối tới Gemini (timeout hoặc "
                    "lỗi mạng), không phải do bạn huỷ. Hãy thử lại — các đoạn video đã "
                    "nén sẽ được tái dùng từ cache nên sẽ nhanh hơn."
                )
        except SubtitleTranslationError as exc:
            logger.warning("Dịch phụ đề thất bại: %s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - chống treo UI nếu lỗi bất ngờ
            logger.exception("Lỗi không lường trước khi dịch phụ đề.")
            self.failed.emit(f"Lỗi không xác định khi dịch: {exc}")


__all__ = ["TranslateWorker"]
