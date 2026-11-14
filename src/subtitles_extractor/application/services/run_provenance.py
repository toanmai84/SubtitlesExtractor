"""[v3.23.714] Hồ sơ lượt trích xuất — ghi kèm tệp OCR thô để phân tích / gỡ lỗi tái lập được đúng lượt chạy.

Vì sao cần: lượt chạy thật của chủ dự án (bản 3.23.711 và 3.23.712, phim nối 86 tập) lệch mốc 4 câu so với lượt
dựng lại từ CHÍNH tệp OCR thô của nó. Mất nhiều giờ mới tìm ra nguyên nhân là cài đặt hậu xử lý trên máy chủ dự án
khác mặc định (khoảng gộp ≈ 0,1 s thay vì 0,2 s; dung sai căn giữa nhỏ hơn) — tệp chỉ ghi model, vùng chữ, tiền xử
lý và 4 ngưỡng dò chữ, KHÔNG ghi cấu hình dựng phụ đề, cấu hình lấy mẫu / lọc khung, hướng và căn lề của vùng chữ,
cũng không ghi kết quả đã dựng. Từ bản này tệp mang đủ hai phần:

* ``run``: mọi thứ quyết định kết quả (ứng dụng, thư viện, phần cứng, video, vùng chữ, lấy mẫu, OCR, dựng phụ đề);
* ``result``: các câu đã dựng (mốc, chữ, cờ nghi ngờ) và nhật ký vòng soát — đối chiếu được với tệp phụ đề người
  dùng gửi kèm mà không phải đoán.

Hàm ở đây thuần dữ liệu (không đụng giao diện / mô hình) và không bao giờ ném lỗi ra ngoài: thiếu một mục thì bỏ mục đó.
"""

from __future__ import annotations

import dataclasses
import enum
import platform
import sys
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

#: Thư viện ảnh hưởng tới kết quả OCR / giải mã — ghi phiên bản đang cài (không có thì bỏ qua).
_TRACKED_PACKAGES: tuple[str, ...] = (
    "paddleocr", "paddlepaddle-gpu", "paddlepaddle", "paddlex", "numpy", "opencv-python", "opencv-python-headless",
    "av", "PyNvVideoCodec", "cupy-cuda12x", "rapidfuzz", "PySide6",
)
#: Khoá trong ``extra_kwargs`` / cấu hình có thể mang bí mật — không bao giờ ghi giá trị.
_SECRET_MARKERS: tuple[str, ...] = (
    "api_key", "apikey", "secret", "password", "authorization", "access_token", "auth_token", "credential",
)

PROVENANCE_SCHEMA: int = 1


def _text_attr(obj: Any, name: str) -> str | None:
    """Thuộc tính chuỗi (chỉ nhận ``str`` — đối tượng lạ / giả lập không được làm hỏng tệp JSON)."""
    value = getattr(obj, name, None) if obj is not None else None
    return value if isinstance(value, str) else None


def _sampler_name(frame_sampler: Any) -> str | None:
    """[v3.23.723] Bộ giải mã THẬT SỰ chạy (bộ bọc NVDEC ghi rõ nếu đã lùi PyAV giữa chừng)."""
    if frame_sampler is None:
        return None
    return _text_attr(frame_sampler, "effective_sampler") or type(frame_sampler).__name__


def to_jsonable(value: Any, _depth: int = 0) -> Any:
    """Đổi dataclass / enum / đường dẫn / tuple… về kiểu JSON ghi được; khoá trông như bí mật bị che."""
    if _depth > 8:
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, PurePath):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name, None), _depth + 1)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): ("***" if any(marker in str(key).lower() for marker in _SECRET_MARKERS)
                       else to_jsonable(item, _depth + 1))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item, _depth + 1) for item in value]
    return str(value)


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]


def _package_versions() -> dict[str, str]:
    from importlib import metadata

    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:  # noqa: BLE001 — gói không cài / metadata hỏng: bỏ qua, không được làm hỏng lượt chạy
            continue
    return versions


def _video_info(video_path: Path, metadata: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"name": Path(video_path).name}
    for key in ("width", "height", "fps", "total_frames", "duration_sec", "codec"):
        value = getattr(metadata, key, None)
        if value is not None:
            info[key] = to_jsonable(value)
    try:
        stat = Path(video_path).stat()
        info["size_bytes"] = int(stat.st_size)
        info["modified_at"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    except OSError:
        pass
    return info


def _roi_info(roi: Any) -> dict[str, Any] | None:
    if roi is None:
        return None
    return {
        "x": int(roi.x), "y": int(roi.y), "width": int(roi.width), "height": int(roi.height),
        "alignment": to_jsonable(getattr(roi, "alignment", None)),
        "orientation": to_jsonable(getattr(roi, "orientation", None)),
    }


def frame_statistics(frames: Iterable[Any]) -> dict[str, int]:
    """Đếm khung theo loại — biết ngay bộ lọc khung đã bỏ qua bao nhiêu khung, bao nhiêu khung thật sự có chữ."""
    total = with_text = continued = 0
    for frame in frames:
        total += 1
        if getattr(frame, "text_boxes", None):
            with_text += 1
        if getattr(frame, "continues_previous", False):
            continued += 1
    return {"total": total, "with_text": with_text, "empty": total - with_text, "same_pixels_as_previous": continued}


def collect_run_info(
    *, run_id: str, app_version: str, request: Any, metadata: Any, builder_config: Any,
    frame_sampler: Any = None, ocr_engine: Any = None, started_at: datetime | None = None,
    ocr_elapsed_sec: float | None = None, frames: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Gom mọi thông tin quyết định kết quả của một lượt trích xuất thành ``dict`` JSON ghi được."""
    info: dict[str, Any] = {"schema": PROVENANCE_SCHEMA, "run_id": run_id, "app_version": app_version}
    try:
        info["started_at"] = (started_at or datetime.now(tz=UTC)).isoformat()
        if ocr_elapsed_sec is not None:
            info["ocr_elapsed_sec"] = round(float(ocr_elapsed_sec), 3)
        info["environment"] = {
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "frozen": bool(getattr(sys, "frozen", False)),
            "packages": _package_versions(),
            # [v3.23.723] Bộ giải mã THẬT SỰ chạy (bộ bọc NVDEC ghi rõ nếu đã lùi PyAV và lý do).
            "frame_sampler": _sampler_name(frame_sampler),
            "frame_sampler_fallback_reason": _text_attr(frame_sampler, "last_fallback_reason"),
            "ocr_engine": type(ocr_engine).__name__ if ocr_engine is not None else None,
        }
        info["video"] = _video_info(request.video_path, metadata)
        info["roi"] = _roi_info(getattr(request, "roi", None))
        info["sampling"] = to_jsonable(getattr(request, "sampling", None))
        info["ocr"] = to_jsonable(getattr(request, "ocr", None))
        info["builder"] = to_jsonable(builder_config)
        info["options"] = {
            "verify_suspect_lines": bool(getattr(request, "verify_suspect_lines", False)),
            "auto_tune_batch": bool(getattr(request, "auto_tune_batch", False)),
            "use_frame_cache": bool(getattr(request, "use_frame_cache", False)),
            "skip_export": bool(getattr(request, "skip_export", False)),
            "output_format": to_jsonable(getattr(request, "output_format", None)),
        }
        if frames is not None:
            info["frames"] = frame_statistics(frames)
    except Exception as exc:  # noqa: BLE001 — hồ sơ là phần phụ: thiếu mục còn hơn hỏng lượt trích xuất
        info["collect_error"] = f"{type(exc).__name__}: {exc}"
    return info


def summarize_result(
    events: Iterable[Any], verify_log: Iterable[Mapping[str, Any]] | None = None, *, build_elapsed_sec: float | None = None,
) -> dict[str, Any]:
    """Các câu đã dựng (sau vòng soát) ở dạng gọn + nhật ký vòng soát."""
    lines: list[dict[str, Any]] = []
    for event in events:
        try:
            line: dict[str, Any] = {
                "s": round(float(event.start_sec), 3), "e": round(float(event.end_sec), 3), "t": str(event.text or ""),
            }
            notes = tuple(getattr(event, "ocr_notes", ()) or ())
            if notes:
                line["n"] = list(notes)
            confidence = getattr(event, "confidence", None)
            if confidence is not None:
                line["c"] = round(float(confidence), 4)
            frame_count = int(getattr(event, "frame_count", 0) or 0)
            if frame_count:
                line["f"] = frame_count
            lines.append(line)
        except Exception:  # noqa: BLE001
            continue
    result: dict[str, Any] = {"schema": PROVENANCE_SCHEMA, "event_count": len(lines), "events": lines}
    if build_elapsed_sec is not None:
        result["build_elapsed_sec"] = round(float(build_elapsed_sec), 3)
    result["verify"] = [to_jsonable(dict(entry)) for entry in (verify_log or ())]
    return result


__all__ = [
    "PROVENANCE_SCHEMA", "collect_run_info", "frame_statistics", "new_run_id", "summarize_result", "to_jsonable",
]
