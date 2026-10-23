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
from subtitles_extractor.infrastructure.translation.gemini_quota_clock import (
    describe_quota_reset,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    QuotaExhaustedError,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationCancelledError,
    TranslationStageKind,
)
from subtitles_extractor.presentation.workers.video_reupload import (
    make_video_reupload_callback,
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
        video_refs_ready: ``(object,)`` — [v3.23.609] ``list[RemoteVideoRef]`` vừa tải
            lên (lần đầu hoặc tải lại khi xoay key), để ghi vào phiên dịch.
    """

    progress_changed = Signal(int, str)
    finished_ok = Signal(object)
    cancelled = Signal()
    failed = Signal(str)
    video_refs_ready = Signal(object)

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

    def _stages_to_translate(
        self, request: TranslateSubtitlesRequest
    ) -> list[TranslationStageKind] | None:
        """[v3.23.609] Giai đoạn phải gọi model; ``None`` khi không kiểm được bộ nhớ đệm
        (khi đó coi như mọi giai đoạn đều phải dịch — vẫn chuẩn bị video như trước)."""
        try:
            return list(self._use_case.stages_needing_model(request))
        except (OSError, ValueError) as exc:
            logger.warning(
                "Không kiểm được bộ nhớ đệm giai đoạn (%s) — chuẩn bị video như thường.",
                exc,
            )
            return None

    def _quota_model(
        self,
        request: TranslateSubtitlesRequest,
        needed_kinds: set[TranslationStageKind],
    ) -> str:
        """Model dùng để xét hạn mức trước khi tải video: giai đoạn đính video SẼ dịch,
        không có thì giai đoạn đầu sẽ dịch, không có nữa thì giai đoạn đầu."""
        will_run = [stage for stage in request.stages if stage.kind in needed_kinds]
        attached = [s for s in will_run if s.kind in self._attach_video_stages]
        for candidates in (attached, will_run, list(request.stages)):
            if candidates:
                return candidates[0].model_name
        return ""

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            request = self._request
            # [v3.23.609] Mọi giai đoạn lấy nguyên từ bộ nhớ đệm → không cần video: bỏ
            # bước nén + tải (vài phút) và bước chặn khi hết hạn mức. Vẫn ghi giai đoạn
            # nào đính video, vì đó là một phần điều kiện dùng lại kết quả đã lưu.
            needed_kinds = {stage.kind for stage in request.stages}
            if self._video_provider is not None and self._video_path:
                import dataclasses

                with_attach = dataclasses.replace(
                    request, attach_video_stages=self._attach_video_stages
                )
                needed = self._stages_to_translate(with_attach)
                if needed is not None:
                    needed_kinds = set(needed)
                if needed == []:
                    logger.info(
                        "Mọi giai đoạn lấy nguyên từ bộ nhớ đệm — bỏ qua chuẩn bị video."
                    )
                    request = with_attach
                    self._video_provider = None
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
                # [v3.23.609] Chỉ xét các giai đoạn SẼ gọi model: giai đoạn lấy từ bộ nhớ
                # đệm không gọi gì — trước đây model của giai đoạn đính video đầu tiên
                # (vd dịch thô đã xong, model đó hết hạn mức) chặn cả lượt dịch.
                video_stage_model = self._quota_model(request, needed_kinds)
                if video_stage_model:
                    # [v3.23.294] Dự trù số request cho trọn phiên dịch = số giai đoạn
                    # (mỗi giai đoạn ~1 request/đoạn video; video ngắn thường 1 đoạn).
                    # avoid_reupload=True: video sẽ tái dùng bản đã upload ở phase phân
                    # tích → adapter KHÔNG xoay key cơ hội (tránh cắt+nén+upload lại).
                    session_needed_requests = max(
                        1,
                        sum(1 for stage in request.stages if stage.kind in needed_kinds),
                    )
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
                            f"{describe_quota_reset()} Hãy thử lại lúc đó, thêm API "
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
                    # [v3.23.615] Lỗi đã phân loại (mạng, hết hạn, ffmpeg) — thông điệp
                    # đủ rõ, không cần cả traceback trong nhật ký.
                    logger.warning("Chuẩn bị video ngữ cảnh cho khâu dịch lỗi: %s", exc)
                    if self._is_cancelled():
                        self.cancelled.emit()
                        return
                    self.failed.emit(f"Lỗi chuẩn bị ngữ cảnh video: {exc}")
                    return
                if refs:
                    self.video_refs_ready.emit(list(refs))
                request = dataclasses.replace(
                    request,
                    video_refs=tuple(refs),
                    attach_video_stages=self._attach_video_stages,
                )
                # [v3.23.482] Đăng ký callback tải-lại-video cho adapter DỊCH: khi 403 do xoay key
                # (file thuộc key cũ), adapter tự tải lại đoạn bằng key hiện tại thay vì để cả tập
                # dịch thất bại (như khâu phân tích đã làm). Sửa app_log "Dịch phụ đề thất bại: 403".
                _translate_video_reupload_cb = make_video_reupload_callback(
                    self._video_provider, self._video_path, self._is_cancelled,
                    on_uploaded=self.video_refs_ready.emit,
                )
                self._use_case.set_video_reupload_callback(_translate_video_reupload_cb)
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
        except QuotaExhaustedError as exc:
            # [v3.23.527] Hết hạn mức NGÀY là tình huống bình thường của free-tier, không phải
            # lỗi lập trình — báo rõ kèm gợi ý thay vì hiện như crash.
            logger.warning("Hết hạn mức ngày khi dịch phụ đề: %s", exc)
            self.failed.emit(
                f"{exc}\n\n"
                "Gợi ý: chọn model có hạn mức ngày cao hơn (bản 'flash-lite'), thêm API key "
                "thuộc DỰ ÁN KHÁC (nhiều key cùng dự án dùng chung hạn mức), chờ tới "
                "mốc hạn mức hồi nêu ở trên rồi dịch tiếp bằng model mạnh, hoặc bật "
                "thanh toán cho dự án.\n"
                # [v3.23.606] Đổi xuống model yếu hơn không còn làm mất phần đã dịch.
                "Phần đã dịch xong được giữ: đổi sang model yếu hơn thì các giai đoạn/lô "
                "đã xong bằng model mạnh hơn được dùng lại, chỉ dịch phần còn thiếu."
            )
        except Exception as exc:  # noqa: BLE001 - chống treo UI nếu lỗi bất ngờ
            logger.exception("Lỗi không lường trước khi dịch phụ đề.")
            self.failed.emit(f"Lỗi không xác định khi dịch: {exc}")


__all__ = ["TranslateWorker"]
