"""Serializer cho dữ liệu phiên âm THÔ của WhisperX (.sestt.json).

Mục đích: lưu toàn bộ segment + words + timestamp từ WhisperX ra file JSON, để:
  * Hiệu chuẩn/tối ưu thuật toán tự tách câu OFFLINE mà không phải chạy lại STT
    (vốn mất nhiều phút + cần GPU).
  * Người dùng gửi file này cho dev làm dữ liệu thật.

Format (.sestt.json):
    ``version``:  phiên bản schema.
    ``meta``:     media path, ngôn ngữ nhận diện, model, thời điểm xuất.
    ``segments``: danh sách segment (start/end/text/speaker/words).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

SCHEMA_VERSION: str = "1.0"


@dataclass(frozen=True, slots=True)
class RawSttMeta:
    """Thông tin kèm theo dữ liệu phiên âm thô."""

    media_path: str
    detected_language: str
    model_size: str
    app_version: str
    exported_at: str
    segment_count: int
    word_count: int


def save_raw_stt(
    output_path: Path,
    segments: list[dict],
    media_path: str,
    detected_language: str,
    model_size: str,
    app_version: str,
) -> Path:
    """Lưu dữ liệu phiên âm thô ra file ``.sestt.json``.

    Args:
        output_path:       Đường dẫn file đích.
        segments:          Danh sách segment đã chuẩn hoá (start/end/text/words).
        media_path:        Đường dẫn media nguồn.
        detected_language: Ngôn ngữ WhisperX nhận diện.
        model_size:        Model đã dùng.
        app_version:       Phiên bản app.

    Returns:
        Đường dẫn file đã ghi.
    """
    word_count = sum(len(s.get("words", [])) for s in segments)
    meta = RawSttMeta(
        media_path=media_path,
        detected_language=detected_language,
        model_size=model_size,
        app_version=app_version,
        exported_at=datetime.now(timezone.utc).isoformat(),
        segment_count=len(segments),
        word_count=word_count,
    )
    payload = {
        "version": SCHEMA_VERSION,
        "meta": asdict(meta),
        "segments": segments,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    logger.info(
        "Đã xuất dữ liệu STT thô: {} ({} segment, {} từ).",
        output_path.name, len(segments), word_count,
    )
    return output_path


def load_raw_stt(input_path: Path) -> tuple[RawSttMeta, list[dict]]:
    """Đọc file ``.sestt.json`` → (meta, segments). Phục vụ hiệu chuẩn offline."""
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    meta_dict = payload.get("meta", {})
    # Bỏ field lạ để tương thích ngược.
    known = RawSttMeta.__dataclass_fields__.keys()
    meta = RawSttMeta(**{k: meta_dict.get(k, "") if k not in
                         ("segment_count", "word_count") else meta_dict.get(k, 0)
                         for k in known})
    return meta, payload.get("segments", [])


__all__ = ["SCHEMA_VERSION", "RawSttMeta", "save_raw_stt", "load_raw_stt"]
