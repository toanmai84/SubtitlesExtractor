"""Tương thích ngược: trỏ PaddleOCR về model đã prefetch (uỷ quyền ``model_store``).

[v3.23.301] Logic đã chuyển sang :mod:`infrastructure.model_store` — nơi quản lý TẤT CẢ
kho model tập trung (``models/paddle``, ``models/huggingface``) song song với ``vendor/``
cho binary native. Module này giữ lại tên hàm cũ để không phá code/test hiện có.

Điểm mới so với bản v3.23.293: kho model hoạt động **cả khi chạy nguồn** (``models/paddle``
ở gốc dự án), không chỉ khi đóng gói — nên dev cũng OCR offline được sau khi prefetch.
"""

from __future__ import annotations

from pathlib import Path

from subtitles_extractor.infrastructure.model_store import configure_paddle_model_store


def configure_bundled_paddle_models() -> Path | None:
    """Trỏ ``PADDLE_PDX_CACHE_HOME`` về ``models/paddle`` nếu đã có model (idempotent).

    Returns:
        Đường dẫn gốc cache đã áp dụng, hoặc ``None`` nếu không áp dụng (chưa prefetch,
        hoặc biến môi trường đã được đặt sẵn).
    """
    return configure_paddle_model_store()


__all__ = ["configure_bundled_paddle_models"]
