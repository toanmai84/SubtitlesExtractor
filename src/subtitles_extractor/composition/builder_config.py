"""[v3.23.720] Một nguồn duy nhất cho ánh xạ cài đặt ứng dụng → :class:`SubtitleBuilderConfig`.

Trước bản này có 4 bản chép tay của cùng ánh xạ (container, trang Trích xuất, trang Chỉnh sửa / ReOCR, trang Debug), còn
các công cụ (phát lại, thử sức, hiệu chỉnh) dùng mặc định của DTO — lệch mặc định của ứng dụng ở 5 trường
(``merge_gap_sec`` 0,6 / 0,2; ``use_viterbi`` tắt / bật; ``min_text_chars`` 2 / 0; ``temporal_padding_sec`` 0,05 / 0;
``sample_step_sec``). Mọi số đo phát lại seraw không có ``meta.run`` vì thế đo một cấu hình ứng dụng không bao giờ chạy.
"""

from __future__ import annotations

from typing import Any

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig


def builder_config_from_settings(snapshot: Any, *, min_confidence: float | None = None) -> SubtitleBuilderConfig:
    """Cấu hình dựng phụ đề từ ảnh chụp cài đặt (``ApplicationSettings``).

    Args:
        snapshot: Ảnh chụp cài đặt hiện tại.
        min_confidence: Ghi đè ngưỡng tin cậy OCR (ReOCR cho phép chỉnh riêng); ``None`` = theo cài đặt.
    """
    post = snapshot.post_process
    threshold = snapshot.threshold
    return SubtitleBuilderConfig(
        similarity_threshold=min(post.similarity_threshold, threshold.text_similarity),
        min_duration_sec=post.min_duration_sec,
        max_duration_sec=post.max_duration_sec,
        min_confidence=threshold.ocr_min_confidence if min_confidence is None else min_confidence,
        merge_gap_sec=post.merge_gap_sec,
        merge_progressive=bool(getattr(post, "merge_progressive", False)),
        use_viterbi=post.use_viterbi,
        viterbi_open_penalty=post.viterbi_open_penalty,
        min_text_chars=threshold.drop_short_text_chars,
        line_similarity_threshold=threshold.line_similarity,
        sample_step_sec=snapshot.frame.sample_step_sec,
        temporal_padding_sec=post.temporal_padding_sec,
        y_clustering_tolerance_ratio=post.y_clustering_tolerance_ratio,
        y_clustering_tolerance_min_px=post.y_clustering_tolerance_min_px,
        alignment_center_tolerance_ratio=post.alignment_center_tolerance_ratio,
        alignment_margin_tolerance_ratio=post.alignment_margin_tolerance_ratio,
        alignment_tolerance_min_px=post.alignment_tolerance_min_px,
    )


def app_default_builder_config(sample_step_sec: float | None = None) -> SubtitleBuilderConfig:
    """Cấu hình dựng theo MẶC ĐỊNH của ứng dụng (cài đặt chưa chỉnh) — dùng cho công cụ khi seraw không ghi
    ``meta.run.builder``. ``sample_step_sec``: bước lấy mẫu của lượt OCR (mặc định ứng dụng là 0,12 s)."""
    from dataclasses import replace

    from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings

    config = builder_config_from_settings(ApplicationSettings())
    return replace(config, sample_step_sec=sample_step_sec) if sample_step_sec else config


__all__ = ["app_default_builder_config", "builder_config_from_settings"]
