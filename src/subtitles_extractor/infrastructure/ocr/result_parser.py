"""Parse output thô của PaddleOCR sang :class:`OcrFrameResult` (v3.3).

Robustness:
    * Hỗ trợ MỌI format mà PaddleOCR v2.6 → v3.x đã xuất:
      - Dict có key ``"res"`` bao bọc (PaddleX Pipeline mới).
      - Dict phẳng có key ``rec_texts``/``rec_scores``/``rec_polys``.
      - Object có thuộc tính ``rec_texts`` (kiểu ``OCRResult`` mới).
      - List of tuples format cũ ``[(polygon, (text, score)), ...]``.
    * Validate strict: 3 mảng phải cùng độ dài.
    * Polygon được normalize về ``list[tuple[int, int]]`` immutable-friendly.

Changelog v3.3:
    * [STYLE] Fix toàn bộ ``= [`` → ``= []`` (Black-compliant).
    * [QUALITY] Tách hàm ``_extract_from_dict`` ra (SRP).
    * [QUALITY] Type alias ``RawPolygon`` cho rõ ý.
"""

from __future__ import annotations

import logging
from typing import Any

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.exceptions import OcrInferenceError
from subtitles_extractor.domain.value_objects.confidence import Confidence

logger = logging.getLogger(__name__)

# Polygon thô từ PaddleOCR: ``list[list[int]]`` hoặc ``numpy.ndarray``.
RawPolygon = Any

# ── Public API ────────────────────────────────────────────────────────────


def parse_paddle_result(
    raw: Any,
    frame_index: int,
    timestamp_sec: float,
) -> OcrFrameResult:
    """Parse 1 phần tử raw output của PaddleOCR thành :class:`OcrFrameResult`.

    Args:
        raw: 1 phần tử trong output ``model.predict(...)``. Có thể là:
            ``dict``, object có ``.res``, object có ``.rec_texts``, hoặc
            list-of-tuples (format cũ).
        frame_index: Thứ tự khung hình.
        timestamp_sec: Mốc thời gian.

    Returns:
        :class:`OcrFrameResult` rỗng nếu ``raw`` là ``None``, hoặc đầy đủ
        text boxes nếu parse thành công.

    Raises:
        OcrInferenceError: Khi 3 mảng ``texts``/``scores``/``polygons``
            không cùng độ dài, hoặc khi không nhận diện được format.
    """
    if raw is None:
        return OcrFrameResult(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            text_boxes= [],
        )

    texts, scores, polygons = _extract_fields(raw)

    if not (len(texts) == len(scores) == len(polygons)):
        raise OcrInferenceError(
            f"Độ dài không khớp giữa texts ({len(texts)}), "
            f"scores ({len(scores)}) và polygons ({len(polygons)}) "
            f"trên frame #{frame_index}."
        )

    text_boxes: list[OcrTextBox] = []
    for text, score, polygon in zip(texts, scores, polygons, strict=True):
        normalized_score = max(0.0, min(1.0, float(score)))
        text_boxes.append(
            OcrTextBox(
                text=str(text).strip(),
                confidence=Confidence(normalized_score),
                polygon=_normalize_polygon(polygon),
            )
        )

    return OcrFrameResult(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        text_boxes=text_boxes,
    )


# ── Private — Field extraction ────────────────────────────────────────────


def _extract_fields(
    raw: Any,
) -> tuple[list[str], list[float], list[RawPolygon]]:
    """Trả về ``(texts, scores, polygons)`` từ mọi dạng output của PaddleOCR.

    Args:
        raw: Output thô.

    Returns:
        Tuple 3 list cùng độ dài (về mặt ngữ nghĩa — caller phải validate).

    Raises:
        OcrInferenceError: Khi không nhận diện được format.
    """
    if isinstance(raw, dict):
        return _extract_from_dict(raw)

    # Object PaddleX với attribute .res là dict.
    res_attr = getattr(raw, "res", None)
    if isinstance(res_attr, dict):
        return _extract_from_dict(res_attr, allow_unwrap=False)

    # Object có attribute .rec_texts trực tiếp.
    if hasattr(raw, "rec_texts"):
        return _extract_from_attrs(raw)

    # Fallback cho format cũ: list of (polygon, (text, score)).
    if isinstance(raw, list):
        return _extract_from_legacy_list(raw)

    raise OcrInferenceError(
        f"Không nhận diện được format kết quả PaddleOCR: "
        f"type={type(raw).__name__}."
    )


def _extract_from_dict(
    raw_dict: dict[str, Any],
    *,
    allow_unwrap: bool = True,
) -> tuple[list[str], list[float], list[RawPolygon]]:
    """Trích xuất từ dict — tự unwrap key ``"res"`` nếu có.

    Args:
        raw_dict: Dict raw từ PaddleOCR.
        allow_unwrap: Nếu ``True`` và ``raw_dict["res"]`` là dict, dùng
            nội bộ đó. Tránh recursion vô tận khi caller đã unwrap.
    """
    target: dict[str, Any] = raw_dict
    if allow_unwrap:
        inner = raw_dict.get("res")
        if isinstance(inner, dict):
            target = inner

    return (
        list(target.get("rec_texts", []) or []),
        list(target.get("rec_scores", []) or []),
        list(
            target.get("rec_polys", target.get("dt_polys", [])) or []
        ),
    )


def _extract_from_attrs(
    raw_obj: Any,
) -> tuple[list[str], list[float], list[RawPolygon]]:
    """Trích xuất từ object có ``rec_texts``/``rec_scores``/``rec_polys``."""
    return (
        list(getattr(raw_obj, "rec_texts", []) or []),
        list(getattr(raw_obj, "rec_scores", []) or []),
        list(
            getattr(raw_obj, "rec_polys", getattr(raw_obj, "dt_polys", []))
            or []
        ),
    )


def _extract_from_legacy_list(
    raw_list: list[Any],
) -> tuple[list[str], list[float], list[RawPolygon]]:
    """Parse format cũ ``[(polygon, (text, score)), ...]`` (PaddleOCR ≤ 2.5)."""
    texts: list[str] = []
    scores: list[float] = []
    polygons: list[RawPolygon] = []

    for entry in raw_list:
        try:
            polygon, (text, score) = entry
        except (TypeError, ValueError):
            continue
        polygons.append(polygon)
        texts.append(text)
        scores.append(score)

    return texts, scores, polygons


# ── Private — Polygon normalization ───────────────────────────────────────


def _normalize_polygon(raw_polygon: RawPolygon) -> list[tuple[int, int]]:
    """Chuyển polygon thô (list-of-list / numpy / list-of-tuple) về
    ``list[tuple[int, int]]``.

    Args:
        raw_polygon: Dữ liệu polygon thô, có thể là ``None``.

    Returns:
        List các điểm ``(x, y)`` int. Trả về list rỗng nếu ``None`` hoặc
        không parse được điểm nào.
    """
    if raw_polygon is None:
        return []

    points: list[tuple[int, int]] = []
    for point in raw_polygon:
        try:
            x, y = point[0], point[1]
            points.append((int(round(float(x))), int(round(float(y)))))
        except (TypeError, IndexError, ValueError):
            continue
    return points


__all__ = ["parse_paddle_result"]
