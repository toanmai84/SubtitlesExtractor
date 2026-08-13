"""Mô tả bộ dữ liệu hiệu chuẩn: một cặp (OCR thô, ground-truth)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CalibrationDataset:
    """Một cặp dữ liệu để hiệu chuẩn.

    Attributes:
        label: Nhãn dễ đọc (vd ``"test4"``).
        seraw_path: File OCR thô ``*_seraw.json``.
        srt_path: File phụ đề chuẩn ``*.srt`` đã ghép cặp.
        sample_step_sec: Bước lấy mẫu OCR (lấy từ meta seraw).
        time_window: Cửa sổ thời gian ``(start, end)`` để giới hạn đánh giá nhằm
            tăng tốc vòng dò thô; ``None`` = dùng toàn bộ.
    """

    label: str
    seraw_path: Path
    srt_path: Path
    sample_step_sec: float = 0.04
    time_window: tuple[float, float] | None = None
