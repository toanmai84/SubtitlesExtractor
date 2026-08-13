"""Use case: phân tích ngữ cảnh toàn cục của phụ đề bằng AI.

Gửi TOÀN BỘ phụ đề nguồn (sau khử trùng lặp liên tiếp) đến Gemini để tự động
điền vào 3 trường ngữ cảnh: ngôn ngữ gốc, danh sách nhân vật đầy đủ, và tóm tắt
bối cảnh/cốt truyện đầy đủ. Người dùng chọn model AI dùng cho phân tích.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    SubtitleTranslatorPort,
    TranslationLine,
)

logger = logging.getLogger(__name__)

CancellationCallback = Callable[[], bool]


class AnalyzeSubtitleContextUseCase:
    """Phân tích ngữ cảnh toàn cục từ toàn bộ phụ đề bằng AI."""

    def __init__(self, translator: SubtitleTranslatorPort) -> None:
        self._translator = translator

    def ensure_viable_key(self, model_name: str, needed_requests: int = 1) -> str | None:
        """[v3.23.145] Chọn API key còn quota ngày TRƯỚC khi upload video (uỷ adapter).

        [v3.23.157] ``needed_requests``: số request dự trù cho trọn phiên (vd số đoạn
        phân tích) để adapter chọn key ĐỦ quota ngay từ đầu.
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
        """[v3.23.146] True nếu còn ÍT NHẤT một key có quota ngày (để fail nhanh nếu cạn)."""
        fn = getattr(self._translator, "has_any_daily_quota", None)
        if fn is None:
            return True
        try:
            return bool(fn(model_name))
        except (AttributeError, TypeError, ValueError):
            return True

    def execute(
        self,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str = "gemini-3.1-flash-lite",
        cancel_cb: CancellationCallback | None = None,
        video_refs: list | None = None,
        enable_visual_cues: bool = False,
        visual_cues_batch_size: int = 150,
        progress_cb: Callable[[float, str], None] | None = None,
        prior_context: str = "",
        video_reupload_cb: Callable[[str], list] | None = None,
    ) -> SubtitleContextAnalysis:
        """Gửi toàn bộ phụ đề đến AI, nhận ngôn ngữ gốc + nhân vật + tóm tắt.

        Args:
            source_lines: Toàn bộ dòng phụ đề nguồn.
            target_lang:  Ngôn ngữ viết tóm tắt (vd ``"Vietnamese"``).
            model_name:   Model AI thực hiện phân tích (DÙNG CHUNG cho cả Visual Cues).
            cancel_cb:    Callback kiểm tra huỷ.
            video_refs:   (Tuỳ chọn) các đoạn video đã tải lên để đính làm ngữ cảnh —
                          gửi GỘP tất cả đoạn cho phân tích toàn cục.
            enable_visual_cues: Nếu True VÀ có video_refs, phân tích thêm "ai nói/nói
                          với ai" NGAY trong bước này, dùng CÙNG model_name.
            visual_cues_batch_size: Cỡ lô cho Visual Cues.
            progress_cb:  Callback tiến độ (cho bước Visual Cues).

        Returns:
            :class:`SubtitleContextAnalysis` với source_lang, characters, overview,
            glossary và (tuỳ chọn) visual_cues.
        """
        if not source_lines:
            logger.warning("Không có dòng phụ đề nào để phân tích ngữ cảnh.")
            return SubtitleContextAnalysis()

        # [v3.23.132] Đăng ký callback tải lại video bằng key hiện tại (chống 403 khi
        # adapter xoay sang API key khác — file của key cũ không truy cập được).
        if video_reupload_cb is not None and hasattr(
            self._translator, "set_video_reupload_callback"
        ):
            self._translator.set_video_reupload_callback(video_reupload_cb)

        logger.info(
            "Phân tích ngữ cảnh: %d dòng nguồn → model='%s'%s%s.",
            len(source_lines), model_name,
            f", kèm {len(video_refs)} đoạn video" if video_refs else "",
            " + Visual Cues (gộp chung)" if (enable_visual_cues and video_refs) else "",
        )
        # [v3.23.37] Khi bật Visual Cues VÀ có video: lấy LUÔN cues trong cùng request
        # phân tích (đã gửi video) → tiết kiệm quota, KHÔNG gọi analyze_visual_cues riêng.
        want_cues_inline = bool(enable_visual_cues and video_refs)
        result = self._translator.analyze_global_context(
            source_lines=source_lines,
            target_lang=target_lang,
            model_name=model_name,
            cancel_cb=cancel_cb,
            video_refs=video_refs,
            with_visual_cues=want_cues_inline,
            prior_context=prior_context,
        )
        char_preview = result.characters[:80] + "…" if len(result.characters) > 80 else result.characters
        logger.info(
            "Phân tích hoàn tất: lang=%r, nhân vật=%r, tóm tắt=%d ký tự%s.",
            result.source_lang, char_preview, len(result.overview),
            f", {len(result.visual_cues)} ký tự cues" if result.visual_cues else "",
        )
        return result


__all__ = ["AnalyzeSubtitleContextUseCase"]
