"""[v3.23.737] Đọc lại MỌI câu đã dựng bằng mô hình thứ hai (PaddleOCR-VL) trên video vùng ROI → ``phim.qa.json.gz``.

Ảnh lấy từ video ROI (đúng ảnh gốc, chưa tiền xử lý, đọc nhanh vì nhỏ) thay vì giải mã lại phim gốc cho từng câu như
lượt soát câu nghi ngờ (mỗi câu mở và tua bộ giải mã một lần — quá chậm cho cả nghìn câu). Mỗi câu: 3 khung ở 20 % /
50 % / 80 % thời lượng; gom thành lô cho bộ nhận dạng (VL ghép 8 dòng một trang trên GPU).

Định dạng (``schema`` 1)::

    {"schema": 1, "model": "...", "frames_per_line": 3, "elapsed_sec": ..., "lines": [
        {"i": 1, "s": 3.006, "e": 6.846, "t": "câu đã dựng", "ts": [..3 mốc..], "vl": ["..", "..", ".."],
         "c": [0.99, ..]}, ...]}

``vl`` rỗng ở một mốc = mô hình không thấy chữ ở khung đó (câu thừa / sai mốc thời gian).

[v3.23.744] Ảnh đọc lại được CẮT SÁT hộp phụ đề (hộp chữ thô của lượt OCR ở khung gần mốc nhất), không còn cả khung video
ROI: bộ chuyển đổi chạy bộ dò chữ trên ảnh nhận được rồi nối mọi hộp → chữ trong cảnh lọt vào bản đọc ("喝酒去WARMHOME",
"封印" thay câu phụ đề). Đo 457 câu hai phim (57 câu bất đồng đã xem ảnh): PaddleOCR-VL cả khung đúng 8/57 → cắt sát
50/57; 400 câu ngẫu nhiên 398/400. Mỗi câu ghi ``box`` = khung cắt ở mốc giữa (toạ độ video ROI; ``null`` = cả khung).
"""

from __future__ import annotations

import bisect
import gzip
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.ocr_engine_port import (
    OcrEngineConfig,
    OcrEnginePort,
)
from subtitles_extractor.domain.ports.quality_audit_port import QualityAuditPort
from subtitles_extractor.domain.value_objects.ocr_model_names import (
    PADDLE_VL_DRAFT_RECOGNITION_MODEL,
    is_vl_recognizer,
)

logger = logging.getLogger(__name__)

#: Vị trí các khung đọc lại trong thời lượng mỗi câu.
AUDIT_FRAME_POSITIONS: Final[tuple[float, ...]] = (0.2, 0.5, 0.8)
#: Số ảnh mỗi lô gửi bộ nhận dạng.
AUDIT_BATCH: Final[int] = 24
#: [v3.23.744] Khung OCR thô dùng để lấy hộp phụ đề phải cách mốc đọc lại không quá chừng này (giây).
CROP_FRAME_TOLERANCE_SEC: Final[float] = 0.25
#: Lề quanh hộp phụ đề, theo tỉ lệ chiều cao dòng chữ (cộng 4 px) — đúng lề đã đo.
CROP_PAD_RATIO: Final[float] = 0.25

def _accepts_config(engine: object) -> bool:
    """Engine nhận ``infer_batch(..., config=...)`` (cấu hình theo lượt — v3.23.656)?"""
    import inspect

    method = getattr(engine, "infer_batch", None)
    try:
        return method is not None and "config" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


__all__ = [
    "AUDIT_FRAME_POSITIONS", "VlQualityAuditor", "audit_timestamps", "line_text", "strip_scene_latin",
    "subtitle_crop_rect",
]


def _is_han(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


_LATIN_DIGITS = re.compile(r"[A-Za-z0-9\uff10-\uff19\uff21-\uff3a\uff41-\uff5a]+")


def strip_scene_latin(reading: str, line: str) -> str:
    """[v3.23.744] Bỏ chữ Latin / số lạc vào bản đọc lại khi câu phụ đề KHÔNG có chữ Latin / số (hàm thuần).

    Chữ cảnh liền sát dòng phụ đề bị chính bộ dò gộp chung một hộp ("林娇娇这女人eet dream", "喝酒去WAR", "201717上签字"
    — 豪门Z外) nên cắt ảnh không tách được. Phần chữ Hán (thứ cần đối chiếu) giữ nguyên.
    """
    if _LATIN_DIGITS.search(line or "") or not _LATIN_DIGITS.search(reading or ""):
        return reading
    return "".join(_LATIN_DIGITS.sub("", reading).split())


def subtitle_crop_rect(
    boxes: Sequence[OcrTextBox], text: str, width: int, height: int, offset: tuple[int, int] = (0, 0),
) -> tuple[int, int, int, int] | None:
    """[v3.23.744] Khung cắt sát câu phụ đề ``text`` trong một khung (hàm thuần): ``(x0, y0, x1, y1)`` theo ảnh đích.

    * Hộp neo: hộp có chữ Hán trùng câu nhiều nhất (hộp chữ cảnh không trùng / ít trùng);
    * thêm các hộp có chữ Hán trùng câu nằm CÙNG DÒNG với hộp neo, hoặc dòng sát trên / dưới cao bằng và căn giữa cùng
      trục (phụ đề hai dòng) — chữ cảnh trùng chữ của câu ("封印" trong "修为封印") nhưng ở chỗ khác bị loại;
    * lề :data:`CROP_PAD_RATIO` × chiều cao dòng + 4 px; ``offset`` dời toạ độ hộp (ROI) sang ảnh đích (video ROI).

    ``None`` khi không có hộp nào trùng chữ câu.
    """
    wanted = {char for char in text if _is_han(char)}
    if not wanted:
        return None
    scored: list[tuple[int, OcrTextBox, tuple[int, int, int, int]]] = []
    for box in boxes:
        bbox = box.bounding_box
        if bbox is None:
            continue
        shared = sum(1 for char in box.text or "" if char in wanted)
        if shared:
            scored.append((shared, box, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
    if not scored:
        return None
    _score, _anchor, (ax0, ay0, ax1, ay1) = max(scored, key=lambda item: item[0])
    line_height = max(ay1 - ay0, 1)
    chosen = []
    for _shared, _box, (x0, y0, x1, y1) in scored:
        same_line = min(y1, ay1) - max(y0, ay0) > 0.4 * min(line_height, max(y1 - y0, 1))
        # Phụ đề hai dòng: dòng sát trên / dưới, cao xấp xỉ nhau, CĂN GIỮA cùng trục (chữ cảnh "封印" ngay trên dòng
        # "修为封印" lệch trục 162 px — phim 86 tập).
        stacked = (
            (0 <= y0 - ay1 <= 0.6 * line_height or 0 <= ay0 - y1 <= 0.6 * line_height)
            and abs((y1 - y0) - line_height) <= 0.3 * line_height
            and abs((x0 + x1) / 2 - (ax0 + ax1) / 2) <= 0.25 * max(x1 - x0, ax1 - ax0)
        )
        if same_line or stacked:
            chosen.append((x0, y0, x1, y1))
    x0 = min(rect[0] for rect in chosen)
    y0 = min(rect[1] for rect in chosen)
    x1 = max(rect[2] for rect in chosen)
    y1 = max(rect[3] for rect in chosen)
    pad = int(CROP_PAD_RATIO * line_height) + 4
    dx, dy = offset
    rect = (max(0, x0 + dx - pad), max(0, y0 + dy - pad), min(width, x1 + dx + pad), min(height, y1 + dy + pad))
    return rect if rect[2] - rect[0] >= 8 and rect[3] - rect[1] >= 8 else None


def audit_timestamps(start: float, end: float) -> list[float]:
    """Các mốc đọc lại của một câu (hàm thuần): 20 % / 50 % / 80 % thời lượng, trùng thì bỏ."""
    span = max(end - start, 0.0)
    out: list[float] = []
    for fraction in AUDIT_FRAME_POSITIONS:
        moment = round(start + span * fraction, 3)
        if not out or moment - out[-1] > 1e-3:
            out.append(moment)
    return out


def line_text(boxes: Sequence[object]) -> tuple[str, float]:
    """Chữ của một khung (hộp xếp trái → phải) và tin cậy trung bình (hàm thuần)."""
    ordered = sorted(
        boxes, key=lambda box: min((float(p[0]) for p in (getattr(box, "polygon", None) or ())), default=0.0),
    )
    texts = [str(getattr(box, "text", "") or "").strip() for box in ordered]
    confidences = [float(getattr(box, "confidence", 0.0)) for box in ordered]
    return "".join(texts), (sum(confidences) / len(confidences) if confidences else 0.0)


class VlQualityAuditor(QualityAuditPort):
    """Người đọc thứ hai: nhận dạng lại từng câu bằng cấu hình được truyền (PaddleOCR-VL, ảnh gốc)."""

    def __init__(self, ocr_engine: OcrEnginePort, *, mask_in_frame: bool = False) -> None:
        """``mask_in_frame``: [v3.23.748] giữ ảnh CỠ KHUNG, chỉ để lại vùng câu phụ đề (phần còn lại đen) thay vì cắt
        rời — cho bộ đọc có BỘ DÒ CHỮ chạy theo lô (PaddleOCR): mọi ảnh cùng cỡ, bộ dò làm việc đúng tỉ lệ của bước OCR
        chính. Cắt rời (mặc định) cho bộ đọc đọc thẳng cả ảnh (GLM-OCR)."""
        self._ocr_engine = ocr_engine
        self._mask_in_frame = mask_in_frame

    def audit(
        self, events: Sequence[SubtitleEvent], clip_path: Path, sidecar_path: Path, output_path: Path, *,
        config: OcrEngineConfig, progress: Callable[[float], bool] | None = None,
        frames: Sequence[OcrFrameResult] | None = None,
    ) -> Path | None:
        from subtitles_extractor.infrastructure.video.roi_clip_reader import RoiClipReader

        if not events or not clip_path.is_file() or not sidecar_path.is_file():
            return None
        if not _accepts_config(self._ocr_engine):
            # Engine không nhận cấu hình theo lượt → sẽ đọc bằng mô hình mặc định, không phải "người đọc thứ hai".
            logger.warning("Bộ nhận dạng không nhận cấu hình theo lượt — bỏ qua đọc lại kiểm định.")
            return None
        started = time.perf_counter()
        reader = RoiClipReader(clip_path, sidecar_path)
        boxed = sorted((frame for frame in frames or () if frame.text_boxes), key=lambda frame: frame.timestamp_sec)
        boxed_times = [frame.timestamp_sec for frame in boxed]
        offset = reader.box_offset if boxed else (0, 0)     # chỉ cần khi có hộp chữ để cắt
        lines: list[dict[str, Any]] = []
        pending: list[tuple[int, int, np.ndarray, float]] = []      # (câu, vị trí mốc, ảnh, mốc)
        cropped = whole = 0
        try:
            for number, event in enumerate(events, start=1):
                stamps = audit_timestamps(float(event.start_sec), float(event.end_sec))
                lines.append({
                    "i": number, "s": round(float(event.start_sec), 3), "e": round(float(event.end_sec), 3),
                    "t": str(event.text or ""), "ts": stamps, "vl": [""] * len(stamps), "c": [0.0] * len(stamps),
                    "box": None,
                })
                for slot, moment in enumerate(stamps):
                    image = reader.frame_at(moment)
                    if image is None:
                        continue
                    rect = self._crop_rect(boxed, boxed_times, moment, str(event.text or ""), image, offset)
                    if rect is not None:
                        x0, y0, x1, y1 = rect
                        if self._mask_in_frame:
                            # [v3.23.748] Cắt rời mỗi câu một cỡ → bộ dò lô 64 ảnh (máy chủ dự án) không ghép được; đệm
                            # theo ảnh lớn nhất lô đổi tỉ lệ thu-phóng của bộ dò → đo 97 câu: 50/57 → 46/57 (rơi "一",
                            # dính chữ cảnh). Giữ cỡ khung, che đen ngoài câu: cùng cỡ + đúng tỉ lệ bước OCR chính.
                            masked = np.zeros_like(image)
                            masked[y0:y1, x0:x1] = image[y0:y1, x0:x1]
                            image = masked
                        else:
                            image = np.ascontiguousarray(image[y0:y1, x0:x1])
                        cropped += 1
                    else:
                        whole += 1
                    if slot == len(stamps) // 2:
                        lines[-1]["box"] = list(rect) if rect is not None else None
                    pending.append((number - 1, slot, image, moment))
                if len(pending) >= AUDIT_BATCH:
                    self._recognize(pending, lines, config)
                    pending = []
                    if progress is not None and progress(number / len(events)) is False:
                        return None
            if pending:
                self._recognize(pending, lines, config)
        finally:
            reader.close()
        model_label = config.recognition_model_name
        vl_error = getattr(self._ocr_engine, "vl_error", None)
        if vl_error and is_vl_recognizer(model_label):
            # [v3.23.739] VL lỗi → bộ chuyển đổi đã lùi về PP-OCR: ghi đúng bộ đọc thật (trước: nhãn "PaddleOCR-VL" sai).
            model_label = f"{PADDLE_VL_DRAFT_RECOGNITION_MODEL} (PP-OCR — PaddleOCR-VL lỗi)"
            logger.warning("Gói kiểm định đọc bằng PP-OCR vì PaddleOCR-VL lỗi: %s", vl_error)
        logger.info("Gói kiểm định: %d ảnh cắt sát hộp phụ đề, %d ảnh cả khung (không có hộp chữ thô gần mốc).",
                    cropped, whole)
        payload = {
            "schema": 1, "input": "subtitle_box_crop", "model": model_label, "vl_error": vl_error or "", "frames_per_line": len(AUDIT_FRAME_POSITIONS),
            "elapsed_sec": round(time.perf_counter() - started, 1), "lines": lines,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(output_path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        logger.info("Đã đọc lại %d câu bằng %s trong %.0f s → %s.", len(lines), model_label,
                    payload["elapsed_sec"], output_path.name)
        return output_path

    @staticmethod
    def _crop_rect(
        boxed: Sequence[OcrFrameResult], boxed_times: Sequence[float], moment: float, text: str, image: np.ndarray,
        offset: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        """[v3.23.744] Khung cắt sát câu ở ``moment``: hộp chữ thô của khung OCR gần nhất (≤ ``CROP_FRAME_TOLERANCE_SEC``)."""
        if not boxed:
            return None
        position = bisect.bisect_left(boxed_times, moment)
        near = sorted(
            (index for index in (position - 1, position, position - 2, position + 1) if 0 <= index < len(boxed)),
            key=lambda index: abs(boxed_times[index] - moment),
        )
        for index in near:
            if abs(boxed_times[index] - moment) > CROP_FRAME_TOLERANCE_SEC:
                break
            rect = subtitle_crop_rect(boxed[index].text_boxes, text, image.shape[1], image.shape[0], offset)
            if rect is not None:
                return rect
        return None

    def _recognize(
        self, pending: list[tuple[int, int, np.ndarray, float]], lines: list[dict[str, Any]], config: OcrEngineConfig,
    ) -> None:
        images = [item[2] for item in pending]
        results = self._ocr_engine.infer_batch(  # type: ignore[call-arg]  # đã kiểm _accepts_config
            images, list(range(len(images))), [item[3] for item in pending], config=config,
        )
        for (line, slot, _image, _moment), result in zip(pending, results, strict=False):
            raw_text, confidence = line_text(result.text_boxes)
            text = strip_scene_latin(raw_text, lines[line]["t"])
            if text != raw_text:
                lines[line].setdefault("raw", [""] * len(lines[line]["vl"]))[slot] = raw_text
            lines[line]["vl"][slot] = text
            lines[line]["c"][slot] = round(confidence, 3)
