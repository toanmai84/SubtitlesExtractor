"""[v3.23.737] Cổng xuất GÓI KIỂM ĐỊNH CHẤT LƯỢNG đi kèm dữ liệu OCR thô (``.seraw.json.gz``).

Mục tiêu (chủ dự án, 17/11/2026): chạy ứng dụng trên máy có GPU là ra đủ dữ liệu để đối chiếu phụ đề với hardsub mà
không phải dò mắt từng dòng. Gói gồm:

* ``phim.qa.json.gz`` — mỗi câu đã dựng được PaddleOCR-VL (mô hình khác hẳn PP-OCR) đọc lại trên ẢNH GỐC lấy từ video
  vùng ROI: "người đọc thứ hai" độc lập để tìm câu đọc sai chữ;
* ``phim.audio.opus`` — tiếng của phim (mono 16 kHz, ≈ 10 MB/giờ): nguồn thứ ba không phụ thuộc OCR — tìm câu sót
  (có lời mà không có phụ đề), câu thừa, và phân xử chữ giống hình khác âm (昊 hào / 吴 wú).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig


def qa_paths_for(raw_ocr_path: Path) -> tuple[Path, Path]:
    """``(gói đọc lại, tệp tiếng)`` đi kèm tệp OCR thô.

    ``phim.seraw.json.gz`` → ``phim.qa.json.gz`` + ``phim.audio.opus``.
    """
    name = raw_ocr_path.name
    for suffix in (".seraw.json.gz", ".seraw.json", ".json.gz", ".json", ".gz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return raw_ocr_path.with_name(f"{name}.qa.json.gz"), raw_ocr_path.with_name(f"{name}.audio.opus")


@runtime_checkable
class QualityAuditPort(Protocol):
    """Đọc lại MỌI câu đã dựng bằng mô hình thứ hai trên video vùng ROI, ghi kết quả ra tệp."""

    def audit(
        self, events: Sequence[SubtitleEvent], clip_path: Path, sidecar_path: Path, output_path: Path, *,
        config: OcrEngineConfig, progress: Callable[[float], bool] | None = None,
        frames: Sequence[OcrFrameResult] | None = None,
    ) -> Path | None:
        """Ghi ``output_path``; ``progress(0..1)`` trả ``False`` để huỷ. ``None`` nếu không làm được.

        ``frames``: [v3.23.744] kết quả OCR thô của lượt (hộp chữ theo toạ độ ROI) — để cắt sát hộp phụ đề trước khi
        đọc lại (bỏ chữ trong cảnh khỏi ảnh). ``None`` → đọc cả khung video ROI.
        """
        ...


class GlmAuditorFactory(Protocol):
    """[v3.23.745] Mở bộ kiểm định đọc bằng GLM-OCR cho một lượt: cài nếu thiếu (``progress(thông điệp, 0..1)`` →
    ``False`` để huỷ), bật bộ đọc, đóng khi ra khỏi khối ``with``. Lỗi → ném ngoại lệ (lượt lùi về PaddleOCR-VL)."""

    def __call__(
        self, *, use_gpu: bool, progress: Callable[[str, float], bool], cancelled: Callable[[], bool],
    ) -> AbstractContextManager[QualityAuditPort]: ...


class LineReader(Protocol):
    """[v3.23.758] Bộ đọc DÒNG chữ (ảnh BGR cắt sát một hộp chữ → chữ), cùng giao diện bộ đọc PaddleOCR-VL."""

    @property
    def model_name(self) -> str: ...

    def read(self, crops_bgr: Sequence[Any], device: str = "gpu") -> list[str]: ...


class GlmLineReaderFactory(Protocol):
    """[v3.23.758] Mở bộ đọc dòng GLM-OCR cho bước SOÁT CÂU (cài nếu thiếu, bật, đóng khi ra khỏi ``with``)."""

    def __call__(
        self, *, use_gpu: bool, progress: Callable[[str, float], bool], cancelled: Callable[[], bool],
    ) -> AbstractContextManager[LineReader]: ...


@runtime_checkable
class AudioExporterPort(Protocol):
    """Tách tiếng của phim thành tệp nhỏ (mono 16 kHz) để đối chiếu lời thoại."""

    def export(self, video_path: Path, output_path: Path) -> Path | None:
        """Ghi tệp tiếng; ``None`` nếu phim không có tiếng / không tách được."""
        ...


__all__ = [
    "AudioExporterPort", "GlmAuditorFactory", "GlmLineReaderFactory", "LineReader", "QualityAuditPort", "qa_paths_for",
]
