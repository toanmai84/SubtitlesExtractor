"""Serializer cho dữ liệu OCR thô (:class:`OcrFrameResult`).

Mục đích:
    Lưu toàn bộ đầu ra OCR (trước khi SubtitleBuilder xử lý) ra file JSON.
    Cho phép test/tune SubtitleBuilder với các tham số khác nhau mà không
    phải chạy lại OCR (vốn mất hàng phút).

Format file (.seraw.json):
    ``version``: phiên bản schema, tăng khi thay đổi không tương thích.
    ``meta``:    thông tin video + cấu hình OCR đã dùng.
    ``frames``:  danh sách OcrFrameResult đã serialize.

Tính năng:
    * Compact JSON — không có indent thừa (file 1-5 MB thay vì 10+ MB).
    * Round-trip lossless: deserialize ra đúng object domain ban đầu.
    * Backward compat: loader bỏ qua field không nhận ra (unknown_fields).
    * Compression optional: nén gzip nếu output_path kết thúc bằng ``.gz``.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
    Polygon,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence

# ── Schema version ──────────────────────────────────────────────────────
# Tăng khi thay đổi format không tương thích ngược (breaking change).
SCHEMA_VERSION: str = "1.0"

# Dung lượng tối đa hợp lý cho file raw OCR (100 MB). Vượt quá → cảnh báo.
_MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RawOcrMeta:
    """Metadata đi kèm file dữ liệu thô.

    Attributes:
        video_name:        Tên file video (không có đường dẫn đầy đủ).
        video_duration_sec: Thời lượng video (giây).
        frame_count:       Tổng số frame đã OCR.
        sample_step_sec:   Bước lấy mẫu (giây).
        detection_model:   Tên model detection đã dùng.
        recognition_model: Tên model recognition đã dùng.
        score_threshold:   Ngưỡng confidence của OCR adapter.
        saved_at:          Thời điểm lưu file (ISO 8601).
        roi_xywh:          ROI dạng ``[x,y,w,h]`` hoặc ``null`` nếu full frame.
        app_version:       Phiên bản ứng dụng (để debug tương thích).
        preprocess:        [NEW] Cấu hình tiền xử lý ảnh lúc OCR.
        advanced_ocr:      [NEW] Các thông số OCR nâng cao (Det Thresh, Unclip...).
    """

    video_name: str
    video_duration_sec: float
    frame_count: int
    sample_step_sec: float
    detection_model: str
    recognition_model: str
    score_threshold: float
    saved_at: str
    roi_xywh: list[int] | None = None
    app_version: str = "3.23.399"
    preprocess: dict[str, Any] | None = None
    advanced_ocr: dict[str, Any] | None = None


def save_raw_ocr(
    frames: list[OcrFrameResult],
    output_path: Path,
    meta: RawOcrMeta,
) -> None:
    """Lưu danh sách OcrFrameResult ra file JSON.

    Args:
        frames:      Danh sách kết quả OCR (đầu vào của SubtitleBuilder).
        output_path: Đường dẫn file đầu ra. Nếu kết thúc bằng ``.gz``
                     → nén gzip tự động.
        meta:        Metadata video + OCR config.

    Raises:
        OSError:     Khi không ghi được file (full disk, permission, …).
        ValueError:  Khi ``frames`` rỗng.
    """
    if not frames:
        raise ValueError("frames rỗng — không có gì để lưu.")

    payload: dict[str, Any] = {
        "version": SCHEMA_VERSION,
        "meta": asdict(meta),
        "frames": [_serialize_frame(f) for f in frames],
    }

    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw_bytes = serialized.encode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".gz":
        with gzip.open(output_path, "wb", compresslevel=6) as fp:
            fp.write(raw_bytes)
    else:
        output_path.write_bytes(raw_bytes)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    if output_path.stat().st_size > _MAX_FILE_SIZE_BYTES:
        logger.warning(
            "File dữ liệu thô lớn hơn bình thường: {:.1f} MB ({}).",
            file_size_mb, output_path.name,
        )
    logger.info(
        "Đã lưu {} frame OCR vào {} ({:.2f} MB).",
        len(frames), output_path.name, file_size_mb,
    )


def load_raw_ocr(input_path: Path) -> tuple[list[OcrFrameResult], RawOcrMeta]:
    """Nạp OcrFrameResult từ file đã lưu bởi :func:`save_raw_ocr`.

    Args:
        input_path: Đường dẫn file ``.seraw.json`` hoặc ``.seraw.json.gz``.

    Returns:
        Tuple ``(frames, meta)`` đã deserialize.

    Raises:
        FileNotFoundError: Khi file không tồn tại.
        ValueError:        Khi schema version không tương thích hoặc JSON sai.
        OSError:           Khi không đọc được file.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    if input_path.suffix.lower() == ".gz":
        with gzip.open(input_path, "rb") as fp:
            raw_bytes = fp.read()
    else:
        raw_bytes = input_path.read_bytes()

    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"File JSON không hợp lệ: {exc}") from exc

    version = payload.get("version", "")
    if not _is_compatible_version(version):
        raise ValueError(
            f"Schema version không tương thích: file={version!r}, "
            f"app={SCHEMA_VERSION!r}. Hãy xuất lại dữ liệu thô."
        )

    meta_dict = payload.get("meta", {})
    meta = _deserialize_meta(meta_dict)

    frames_raw = payload.get("frames", [])
    frames = [_deserialize_frame(f) for f in frames_raw]

    logger.info(
        "Nạp {} frame OCR từ {} (video: {}).",
        len(frames), input_path.name, meta.video_name,
    )
    return frames, meta


# ── Serialization helpers ────────────────────────────────────────────────


def _serialize_frame(frame: OcrFrameResult) -> dict[str, Any]:
    """Serialize 1 OcrFrameResult thành dict JSON-safe."""
    return {
        "fi": frame.frame_index,
        "ts": round(frame.timestamp_sec, 6),
        "boxes": [_serialize_box(b) for b in frame.text_boxes],
    }


def _serialize_box(box: OcrTextBox) -> dict[str, Any]:
    """Serialize 1 OcrTextBox — compact key names để giảm kích thước file."""
    result: dict[str, Any] = {
        "t": box.text,
        "c": round(float(box.confidence), 6),
    }
    if box.polygon:
        result["p"] = box.polygon  # list[list[int]] tự nhiên JSON-safe
    return result


def _deserialize_frame(data: dict[str, Any]) -> OcrFrameResult:
    """Deserialize dict → OcrFrameResult. Bỏ qua key không nhận ra."""
    frame_index = int(data.get("fi", 0))
    timestamp_sec = float(data.get("ts", 0.0))
    boxes_raw = data.get("boxes", [])
    text_boxes = [_deserialize_box(b) for b in boxes_raw]
    return OcrFrameResult(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        text_boxes=text_boxes,
    )


def _deserialize_box(data: dict[str, Any]) -> OcrTextBox:
    """Deserialize dict → OcrTextBox."""
    text = str(data.get("t", ""))
    confidence = Confidence(float(data.get("c", 0.0)))
    polygon_raw = data.get("p", [])
    polygon: Polygon = [
        (int(pt[0]), int(pt[1])) for pt in polygon_raw if len(pt) >= 2
    ]
    return OcrTextBox(text=text, confidence=confidence, polygon=polygon)


def _deserialize_meta(data: dict[str, Any]) -> RawOcrMeta:
    """Deserialize meta dict → RawOcrMeta. Tolerant với field thiếu."""
    return RawOcrMeta(
        video_name=str(data.get("video_name", "unknown")),
        video_duration_sec=float(data.get("video_duration_sec", 0.0)),
        frame_count=int(data.get("frame_count", 0)),
        sample_step_sec=float(data.get("sample_step_sec", 0.05)),
        detection_model=str(data.get("detection_model", "")),
        recognition_model=str(data.get("recognition_model", "")),
        score_threshold=float(data.get("score_threshold", 0.45)),
        saved_at=str(data.get("saved_at", "")),
        roi_xywh=data.get("roi_xywh"),
        app_version=str(data.get("app_version", "")),
        preprocess=data.get("preprocess"),
        advanced_ocr=data.get("advanced_ocr"),
    )


def _is_compatible_version(version: str) -> bool:
    """Kiểm tra version tương thích — chỉ major version phải khớp."""
    if not version:
        return False
    try:
        file_major = int(version.split(".")[0])
        app_major = int(SCHEMA_VERSION.split(".")[0])
        return file_major == app_major
    except (ValueError, IndexError):
        return False


__all__ = [
    "SCHEMA_VERSION",
    "RawOcrMeta",
    "load_raw_ocr",
    "save_raw_ocr",
]
