"""Grid search tham số PaddleOCR để tìm cấu hình tối ưu cho ứng dụng.

Yêu cầu môi trường:
    * paddleocr >= 2.9
    * paddlepaddle-gpu (khuyến nghị) hoặc paddlepaddle (CPU)
    * opencv-python, numpy, Levenshtein (đã có trong requirements.txt)

Cách dùng:
    python tools/grid_search_paddle_ocr.py \\
        --video path/to/video.mp4 \\
        --reference path/to/reference.srt \\
        --roi 95,880,531,99 \\
        --output-dir grid_results/ \\
        --preset oat

Mỗi combination tham số sẽ:
    1. Chạy PaddleOCR trên video (sample mỗi sample-step giây)
    2. Tạo file .seraw.json + chạy SubtitleBuilder
    3. Đo chất lượng (CER, detection recall, exact accuracy, runtime)
    4. Ghi kết quả vào CSV để so sánh

Presets:
    * ``oat`` (DEFAULT): One-At-A-Time — quét từng tham số từ baseline,
      ~22 configs, cho phép COMPARE TÁC ĐỘNG CỦA TỪNG THAM SỐ (15-40 phút).
    * ``minimal``: 8-12 combinations — quét nhanh trục cơ bản (10-30 phút).
    * ``standard``: 35+ combinations — vừa rộng vừa nhanh (1-2 giờ).
    * ``exhaustive``: 1000+ combinations — quét triệt để (qua đêm).

Tham số có ảnh hưởng LỚN nhất (theo phân tích thực nghiệm fulltest):
    * ``text_det_limit_side_len`` (TÁC ĐỘNG LỚN NHẤT)
        Mặc định cũ 736. Video dọc 720x1280: shortest=720 → KHÔNG scale →
        chữ '一' (gạch ngang) bị mất do quá nhỏ ở scale gốc. Tăng lên 960+
        bắt được nhiều chữ nhỏ.
    * ``text_det_unclip_ratio`` (Ảnh hưởng đến CHỮ BIÊN câu)
        Mặc định 1.5: region detection khít, có thể clip chữ cuối ('了').
        Tăng 2.0-2.5: ít clip hơn.
    * ``recognition_model_name``: server > mobile khoảng 5-10% CER cho chữ
      stylized/font đẹp, nhưng chậm 2-3x.
    * ``text_det_box_thresh``: thấp hơn 0.6 = bắt thêm dòng mờ nhưng tăng FP.

Output:
    * ``grid_results/<combo_hash>/<video>.seraw.json``
    * ``grid_results/<combo_hash>/<video>.srt``
    * ``grid_results/<combo_hash>/config.json``
    * ``grid_results/summary.csv``: bảng tổng hợp
    * Có thể tạo báo cáo Markdown bằng ``tools/report_grid_search.py``.

Đề xuất:
    Sau khi chạy, mở ``summary.csv`` xếp theo (exact_accuracy DESC, runtime_sec ASC)
    để tìm config thắng cuộc. So sánh giữa các config sai khác để xác định ảnh
    hưởng của TỪNG TỪ tham số.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ParameterCombination:
    """Một combo tham số cụ thể để test."""

    detection_model_name: str
    recognition_model_name: str
    text_det_thresh: float
    text_det_box_thresh: float
    text_det_unclip_ratio: float
    text_rec_score_thresh: float
    # [v2.25] Hai tham số sau ảnh hưởng RẤT LỚN đến tỷ lệ phát hiện chữ
    # nhỏ ('一', '了') vì điều khiển scale ảnh đầu vào detection. Mặc định
    # mới `text_det_limit_side_len=960` (thay vì 736) cho phụ đề video dọc
    # 720x1280 vì shortest side = 720 < 736 → ảnh KHÔNG được scale lên ở
    # config cũ → mất nhiều chữ nhỏ.
    text_det_limit_side_len: int
    text_det_limit_type: str  # "min" hoặc "max"
    sample_step_sec: float
    enable_mkldnn: bool
    text_recognition_batch_size: int
    preprocess_apply_clahe: bool
    preprocess_clahe_clip_limit: float
    preprocess_apply_sharpen: bool
    preprocess_upscale_target_height_px: int
    preprocess_add_white_border: bool
    preprocess_border_thickness_px: int

    def short_hash(self) -> str:
        repr_str = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha1(repr_str.encode("utf-8")).hexdigest()[:10]


@dataclass(frozen=True, slots=True)
class CombinationResult:
    combo: ParameterCombination
    total_reference_events: int
    detection_recall: float
    exact_text_accuracy: float
    mean_best_frame_cer: float
    p90_best_frame_cer: float
    total_frames_sampled: int
    frames_with_text: int
    garbage_frames_count: int
    mean_confidence: float
    ocr_runtime_sec: float

    def as_csv_row(self) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        flat.update(dataclasses.asdict(self.combo))
        flat["short_hash"] = self.combo.short_hash()
        flat["total_reference_events"] = self.total_reference_events
        flat["detection_recall"] = round(self.detection_recall, 4)
        flat["exact_text_accuracy"] = round(self.exact_text_accuracy, 4)
        flat["mean_best_frame_cer"] = round(self.mean_best_frame_cer, 4)
        flat["p90_best_frame_cer"] = round(self.p90_best_frame_cer, 4)
        flat["total_frames_sampled"] = self.total_frames_sampled
        flat["frames_with_text"] = self.frames_with_text
        flat["garbage_frames_count"] = self.garbage_frames_count
        flat["mean_confidence"] = round(self.mean_confidence, 4)
        flat["ocr_runtime_sec"] = round(self.ocr_runtime_sec, 2)
        return flat


# ---------------------------------------------------------------------------
# Preset search-space generators
# ---------------------------------------------------------------------------
def generate_minimal_search_space() -> list[ParameterCombination]:
    """8 combo nhanh — quét các trục quan trọng nhất.

    Quét 3 trục chính:
      * Detection threshold (det_thresh): 0.3 vs 0.2 (lower = recall cao hơn)
      * Recognition score threshold: 0.0 vs 0.4 (filter chữ yếu)
      * Sample step: 0.04s (25fps) vs 0.08s (12.5fps — tiết kiệm 50% thời gian)

    Kết quả: 2×2×2 = 8 combo.
    """
    combinations: list[ParameterCombination] = []
    for det_thresh in [0.3, 0.2]:
        for rec_score in [0.0, 0.4]:
            for sample_step in [0.04, 0.08]:
                combinations.append(
                    ParameterCombination(
                        detection_model_name="PP-OCRv5_mobile_det",
                        recognition_model_name="PP-OCRv5_mobile_rec",
                        text_det_thresh=det_thresh,
                        text_det_box_thresh=0.6,
                        text_det_unclip_ratio=1.5,
                        text_rec_score_thresh=rec_score,
                        text_det_limit_side_len=960,
                        text_det_limit_type="min",
                        sample_step_sec=sample_step,
                        enable_mkldnn=False,
                        text_recognition_batch_size=16,
                        preprocess_apply_clahe=True,
                        preprocess_clahe_clip_limit=3.0,
                        preprocess_apply_sharpen=False,
                        preprocess_upscale_target_height_px=96,
                        preprocess_add_white_border=True,
                        preprocess_border_thickness_px=8,
                    )
                )
    return combinations


def generate_standard_search_space() -> list[ParameterCombination]:
    """~35 combo — vừa rộng vừa nhanh."""
    combinations: list[ParameterCombination] = []
    # Trục 1: mobile vs server model (chỉ rec, det giữ mobile để tiết kiệm)
    for rec_model in ["PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"]:
        for det_thresh in [0.30, 0.20, 0.15]:
            for unclip_ratio in [1.5, 2.0, 2.5]:
                combinations.append(
                    ParameterCombination(
                        detection_model_name="PP-OCRv5_mobile_det",
                        recognition_model_name=rec_model,
                        text_det_thresh=det_thresh,
                        text_det_box_thresh=0.6,
                        text_det_unclip_ratio=unclip_ratio,
                        text_rec_score_thresh=0.0,
                        text_det_limit_side_len=960,
                        text_det_limit_type="min",
                        sample_step_sec=0.04,
                        enable_mkldnn=False,
                        text_recognition_batch_size=16,
                        preprocess_apply_clahe=True,
                        preprocess_clahe_clip_limit=3.0,
                        preprocess_apply_sharpen=False,
                        preprocess_upscale_target_height_px=96,
                        preprocess_add_white_border=True,
                        preprocess_border_thickness_px=8,
                    )
                )

    # Trục 2: server det + preprocessing aggressive (cho stylized fonts)
    for apply_sharpen in [False, True]:
        for clahe_clip in [3.0, 4.0]:
            combinations.append(
                ParameterCombination(
                    detection_model_name="PP-OCRv5_server_det",
                    recognition_model_name="PP-OCRv5_server_rec",
                    text_det_thresh=0.15,
                    text_det_box_thresh=0.5,
                    text_det_unclip_ratio=2.0,
                    text_rec_score_thresh=0.0,
                    text_det_limit_side_len=960,
                    text_det_limit_type="min",
                    sample_step_sec=0.04,
                    enable_mkldnn=False,
                    text_recognition_batch_size=8,
                    preprocess_apply_clahe=True,
                    preprocess_clahe_clip_limit=clahe_clip,
                    preprocess_apply_sharpen=apply_sharpen,
                    preprocess_upscale_target_height_px=128,
                    preprocess_add_white_border=True,
                    preprocess_border_thickness_px=12,
                )
            )

    # Trục 3: Sample step (perf vs quality tradeoff)
    for sample_step in [0.04, 0.06, 0.08, 0.10, 0.15]:
        combinations.append(
            ParameterCombination(
                detection_model_name="PP-OCRv5_mobile_det",
                recognition_model_name="PP-OCRv5_mobile_rec",
                text_det_thresh=0.30,
                text_det_box_thresh=0.6,
                text_det_unclip_ratio=1.5,
                text_rec_score_thresh=0.0,
                text_det_limit_side_len=960,
                text_det_limit_type="min",
                sample_step_sec=sample_step,
                enable_mkldnn=False,
                text_recognition_batch_size=16,
                preprocess_apply_clahe=True,
                preprocess_clahe_clip_limit=3.0,
                preprocess_apply_sharpen=False,
                preprocess_upscale_target_height_px=96,
                preprocess_add_white_border=True,
                preprocess_border_thickness_px=8,
            )
        )

    # [v2.25 NEW] Trục 4: text_det_limit_side_len — tác động lớn đến phát
    # hiện chữ nhỏ. Mặc định cũ 736 không scale ảnh 720x1280, mất nhiều chữ.
    for side_len_value in [736, 960, 1280, 1600]:
        combinations.append(
            ParameterCombination(
                detection_model_name="PP-OCRv5_mobile_det",
                recognition_model_name="PP-OCRv5_mobile_rec",
                text_det_thresh=0.30,
                text_det_box_thresh=0.6,
                text_det_unclip_ratio=1.5,
                text_rec_score_thresh=0.0,
                text_det_limit_side_len=side_len_value,
                text_det_limit_type="min",
                sample_step_sec=0.04,
                enable_mkldnn=False,
                text_recognition_batch_size=16,
                preprocess_apply_clahe=True,
                preprocess_clahe_clip_limit=3.0,
                preprocess_apply_sharpen=False,
                preprocess_upscale_target_height_px=96,
                preprocess_add_white_border=True,
                preprocess_border_thickness_px=8,
            )
        )

    return combinations


def generate_exhaustive_search_space() -> list[ParameterCombination]:
    """Quét rộng — ~100+ combo, chạy qua đêm."""
    grid: dict[str, list[Any]] = {
        "detection_model_name": ["PP-OCRv5_mobile_det", "PP-OCRv5_server_det"],
        "recognition_model_name": ["PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"],
        "text_det_thresh": [0.15, 0.20, 0.30],
        "text_det_box_thresh": [0.5, 0.6],
        "text_det_unclip_ratio": [1.5, 2.0, 2.5],
        "text_rec_score_thresh": [0.0],
        "text_det_limit_side_len": [736, 960, 1280],
        "text_det_limit_type": ["min"],
        "sample_step_sec": [0.04, 0.08],
        "enable_mkldnn": [False],
        "text_recognition_batch_size": [16],
        "preprocess_apply_clahe": [True, False],
        "preprocess_clahe_clip_limit": [3.0],
        "preprocess_apply_sharpen": [False, True],
        "preprocess_upscale_target_height_px": [96, 128],
        "preprocess_add_white_border": [True],
        "preprocess_border_thickness_px": [8],
    }
    combinations: list[ParameterCombination] = []
    field_names = list(grid.keys())
    for value_tuple in itertools.product(*[grid[k] for k in field_names]):
        kwargs = dict(zip(field_names, value_tuple, strict=True))
        combinations.append(ParameterCombination(**kwargs))
    return combinations


def generate_oat_search_space() -> list[ParameterCombination]:
    """One-At-A-Time — quét từng tham số riêng lẻ từ baseline.

    Strategy thông minh hơn full grid: thay vì tổ hợp ngẫu nhiên, bắt đầu
    từ baseline mặc định production và đổi TỪNG tham số một. Cho phép cô
    lập tác động của từng tham số đến CER/recall/runtime.

    ~22 configs:
        - 1 baseline
        - 4 variations text_det_limit_side_len (736, 960, 1280, 1600)
        - 4 variations text_det_box_thresh (0.4, 0.5, 0.7, 0.8)
        - 4 variations text_det_unclip_ratio (1.5, 1.8, 2.0, 2.5)
        - 3 variations text_det_thresh (0.15, 0.25, 0.4)
        - 3 variations text_rec_score_thresh (0.0, 0.3, 0.5)
        - 3 variations model size (mobile_det+server_rec, server+server)

    Returns:
        Danh sách ~22 ParameterCombination.
    """
    def make_base(**overrides: Any) -> ParameterCombination:
        defaults: dict[str, Any] = {
            "detection_model_name": "PP-OCRv5_mobile_det",
            "recognition_model_name": "PP-OCRv5_mobile_rec",
            "text_det_thresh": 0.30,
            "text_det_box_thresh": 0.6,
            "text_det_unclip_ratio": 1.5,
            "text_rec_score_thresh": 0.0,
            "text_det_limit_side_len": 960,
            "text_det_limit_type": "min",
            "sample_step_sec": 0.04,
            "enable_mkldnn": False,
            "text_recognition_batch_size": 16,
            "preprocess_apply_clahe": True,
            "preprocess_clahe_clip_limit": 3.0,
            "preprocess_apply_sharpen": False,
            "preprocess_upscale_target_height_px": 96,
            "preprocess_add_white_border": True,
            "preprocess_border_thickness_px": 8,
        }
        defaults.update(overrides)
        return ParameterCombination(**defaults)

    combinations: list[ParameterCombination] = [make_base()]
    for value in [736, 1280, 1600]:
        combinations.append(make_base(text_det_limit_side_len=value))
    for value in [0.4, 0.5, 0.7]:
        combinations.append(make_base(text_det_box_thresh=value))
    for value in [1.8, 2.0, 2.5]:
        combinations.append(make_base(text_det_unclip_ratio=value))
    for value in [0.15, 0.20, 0.40]:
        combinations.append(make_base(text_det_thresh=value))
    for value in [0.3, 0.5, 0.7]:
        combinations.append(make_base(text_rec_score_thresh=value))

    combinations.append(make_base(
        recognition_model_name="PP-OCRv5_server_rec",
    ))
    combinations.append(make_base(
        detection_model_name="PP-OCRv5_server_det",
    ))
    combinations.append(make_base(
        detection_model_name="PP-OCRv5_server_det",
        recognition_model_name="PP-OCRv5_server_rec",
    ))
    return combinations


# ---------------------------------------------------------------------------
# OCR runner — uses PaddleOCR directly
# ---------------------------------------------------------------------------
def _apply_preprocessing(
    image_rgb: np.ndarray,
    combo: ParameterCombination,
) -> np.ndarray:
    """Áp dụng preprocessing pipeline giống `image_filters.py`."""
    height = image_rgb.shape[0]

    if height < combo.preprocess_upscale_target_height_px:
        scale = combo.preprocess_upscale_target_height_px / max(1, height)
        scale = min(scale, 3.0)
        new_height = int(round(height * scale))
        new_width = int(round(image_rgb.shape[1] * scale))
        interpolation = cv2.INTER_LINEAR if scale > 2.0 else cv2.INTER_CUBIC
        image_rgb = cv2.resize(
            image_rgb, (new_width, new_height), interpolation=interpolation,
        )

    if combo.preprocess_apply_clahe and image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=combo.preprocess_clahe_clip_limit, tileGridSize=(8, 8),
        )
        l_enhanced = clahe.apply(l_chan)
        lab_enhanced = cv2.merge([l_enhanced, a_chan, b_chan])
        image_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    if combo.preprocess_apply_sharpen:
        blurred = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=1.0)
        image_rgb = cv2.addWeighted(image_rgb, 1.5, blurred, -0.5, 0)

    if combo.preprocess_add_white_border and combo.preprocess_border_thickness_px > 0:
        image_rgb = cv2.copyMakeBorder(
            image_rgb,
            top=combo.preprocess_border_thickness_px,
            bottom=combo.preprocess_border_thickness_px,
            left=combo.preprocess_border_thickness_px,
            right=combo.preprocess_border_thickness_px,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    return image_rgb


def run_ocr_for_combination(
    *,
    video_path: Path,
    roi_xywh: tuple[int, int, int, int],
    combo: ParameterCombination,
    output_seraw_json_path: Path,
) -> tuple[float, dict[str, Any]]:
    """Chạy PaddleOCR trên video với combo cụ thể, trả về (runtime_sec, meta_dict)."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "Không tìm thấy thư viện PaddleOCR. "
            "Cài đặt: pip install paddleocr paddlepaddle-gpu"
        ) from exc

    ocr_engine = PaddleOCR(
        device="gpu",
        lang="ch",
        text_detection_model_name=combo.detection_model_name,
        text_recognition_model_name=combo.recognition_model_name,
        text_recognition_batch_size=combo.text_recognition_batch_size,
        text_det_thresh=combo.text_det_thresh,
        text_det_box_thresh=combo.text_det_box_thresh,
        text_det_unclip_ratio=combo.text_det_unclip_ratio,
        text_rec_score_thresh=combo.text_rec_score_thresh,
        text_det_limit_side_len=combo.text_det_limit_side_len,
        text_det_limit_type=combo.text_det_limit_type,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=combo.enable_mkldnn,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    frames_per_second = capture.get(cv2.CAP_PROP_FPS) or 25.0
    sample_step_frames = max(1, int(round(combo.sample_step_sec * frames_per_second)))
    total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    roi_x, roi_y, roi_w, roi_h = roi_xywh

    frames_serialized: list[dict[str, Any]] = []
    frame_index = 0
    runtime_start = time.perf_counter()

    while True:
        ok, raw_frame_bgr = capture.read()
        if not ok:
            break

        if frame_index % sample_step_frames == 0:
            cropped_bgr = raw_frame_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            prepared_rgb = _apply_preprocessing(cropped_rgb, combo)
            prepared_bgr = cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2BGR)

            timestamp_sec = frame_index / frames_per_second

            try:
                raw_output = ocr_engine.predict(input=prepared_bgr)
            except RuntimeError as exc:
                logger.warning("OCR fail frame %d: %s", frame_index, exc)
                raw_output = None

            if raw_output is not None and hasattr(raw_output, "__iter__"):
                raw_list = list(raw_output)
                first = raw_list[0] if raw_list else None
            else:
                first = raw_output

            boxes_payload: list[dict[str, Any]] = []
            if first is not None:
                target = first.get("res", first) if isinstance(first, dict) else first
                texts = list(getattr(target, "get", lambda *a, **k: [])("rec_texts", []) or [])
                scores = list(target.get("rec_scores", []) or [])
                polys = list(target.get("rec_polys", target.get("dt_polys", [])) or [])
                for box_text, box_score, box_poly in zip(texts, scores, polys, strict=False):
                    polygon_int = [
                        [int(round(float(pt[0]))), int(round(float(pt[1])))]
                        for pt in box_poly
                    ]
                    boxes_payload.append({
                        "t": str(box_text).strip(),
                        "c": float(box_score),
                        "p": polygon_int,
                    })

            frames_serialized.append({
                "ts": timestamp_sec,
                "boxes": boxes_payload,
            })

        frame_index += 1

    capture.release()
    runtime_sec = time.perf_counter() - runtime_start

    meta_dict: dict[str, Any] = {
        "video_name": video_path.name,
        "video_duration_sec": (total_video_frames / frames_per_second) if frames_per_second else 0.0,
        "frame_count": total_video_frames,
        "sample_step_sec": combo.sample_step_sec,
        "detection_model": combo.detection_model_name,
        "recognition_model": combo.recognition_model_name,
        "score_threshold": combo.text_rec_score_thresh,
        "roi_xywh": list(roi_xywh),
        "app_version": "grid-search",
        "combo_hash": combo.short_hash(),
    }

    output_seraw_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_seraw_json_path.write_text(
        json.dumps({"meta": meta_dict, "frames": frames_serialized}, ensure_ascii=False),
        encoding="utf-8",
    )

    return runtime_sec, meta_dict


# ---------------------------------------------------------------------------
# Quality measurement — reuse logic of analyze_ocr_quality.py
# ---------------------------------------------------------------------------
def measure_combination_quality(
    seraw_json_path: Path,
    reference_srt_path: Path,
    runtime_sec: float,
    combo: ParameterCombination,
) -> CombinationResult:
    from analyze_ocr_quality import (  # type: ignore[import-not-found]
        analyze_ocr_quality,
        load_raw_ocr_snapshots,
        load_reference_srt,
    )

    reference_events = load_reference_srt(reference_srt_path)
    frame_snapshots = load_raw_ocr_snapshots(seraw_json_path)
    quality = analyze_ocr_quality(
        reference_events=reference_events,
        frame_snapshots=frame_snapshots,
    )
    return CombinationResult(
        combo=combo,
        total_reference_events=quality.total_reference_events,
        detection_recall=quality.detection_recall,
        exact_text_accuracy=quality.exact_text_accuracy,
        mean_best_frame_cer=quality.mean_min_cer,
        p90_best_frame_cer=quality.p90_min_cer,
        total_frames_sampled=quality.total_frames,
        frames_with_text=quality.frames_with_any_text,
        garbage_frames_count=quality.garbage_frames_count,
        mean_confidence=quality.mean_confidence,
        ocr_runtime_sec=runtime_sec,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def write_summary_csv(
    results: Iterable[CombinationResult],
    output_path: Path,
) -> None:
    rows = [r.as_csv_row() for r in results]
    if not rows:
        return
    field_names = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search PaddleOCR cho tìm cấu hình tối ưu.",
    )
    parser.add_argument("--video", type=Path, required=True, help="Đường dẫn video MP4")
    parser.add_argument("--reference", type=Path, required=True, help="Phụ đề chuẩn SRT")
    parser.add_argument(
        "--roi", type=str, required=True,
        help="ROI 'x,y,w,h' (vd '95,880,531,99' cho phụ đề dưới video 720x1280).",
    )
    parser.add_argument(
        "--preset", choices=["minimal", "standard", "exhaustive", "oat"],
        default="oat",
        help="Mức độ quét tham số (default: oat — one-at-a-time, ~22 configs).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("grid_results"),
        help="Thư mục output (default: ./grid_results).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Giới hạn số combo (debug).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    args = parse_cli_args()

    if not args.video.exists():
        print(f"LỖI: video không tồn tại: {args.video}", file=sys.stderr)
        return 1
    if not args.reference.exists():
        print(f"LỖI: reference SRT không tồn tại: {args.reference}", file=sys.stderr)
        return 1

    try:
        roi_xywh_tuple: tuple[int, int, int, int] = tuple(
            int(v) for v in args.roi.split(",")
        )  # type: ignore[assignment]
        if len(roi_xywh_tuple) != 4:
            raise ValueError
    except ValueError:
        print(f"LỖI: format ROI phải là 'x,y,w,h', nhận: {args.roi}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    space_generator = {
        "minimal": generate_minimal_search_space,
        "standard": generate_standard_search_space,
        "exhaustive": generate_exhaustive_search_space,
        "oat": generate_oat_search_space,
    }[args.preset]
    combinations = space_generator()
    if args.limit:
        combinations = combinations[: args.limit]

    print(f"[Grid Search] {len(combinations)} combo, preset='{args.preset}'")

    results: list[CombinationResult] = []
    for idx, combo in enumerate(combinations, start=1):
        combo_hash = combo.short_hash()
        combo_dir = args.output_dir / combo_hash
        combo_dir.mkdir(parents=True, exist_ok=True)
        seraw_json_path = combo_dir / f"{args.video.stem}.seraw.json"
        config_json_path = combo_dir / "config.json"
        config_json_path.write_text(
            json.dumps(dataclasses.asdict(combo), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[{idx}/{len(combinations)}] hash={combo_hash} "
              f"{combo.detection_model_name.split('_')[1]}+{combo.recognition_model_name.split('_')[1]} "
              f"det_thresh={combo.text_det_thresh} step={combo.sample_step_sec}s ...")

        try:
            runtime_sec, _ = run_ocr_for_combination(
                video_path=args.video,
                roi_xywh=roi_xywh_tuple,
                combo=combo,
                output_seraw_json_path=seraw_json_path,
            )
        except (RuntimeError, OSError) as exc:
            print(f"    LỖI: {exc}", file=sys.stderr)
            continue

        result = measure_combination_quality(
            seraw_json_path=seraw_json_path,
            reference_srt_path=args.reference,
            runtime_sec=runtime_sec,
            combo=combo,
        )
        results.append(result)
        print(f"    -> recall={result.detection_recall * 100:.1f}% "
              f"exact={result.exact_text_accuracy * 100:.1f}% "
              f"CER={result.mean_best_frame_cer:.4f} "
              f"runtime={result.ocr_runtime_sec:.1f}s")

    # Sắp theo (exact_text_accuracy DESC, runtime ASC).
    results.sort(key=lambda r: (-r.exact_text_accuracy, r.ocr_runtime_sec))
    summary_csv_path = args.output_dir / "summary.csv"
    write_summary_csv(results, summary_csv_path)
    print(f"\n✅ Xuất tổng kết: {summary_csv_path}")
    print("\n--- TOP 5 COMBO ---")
    for r in results[:5]:
        print(f"  hash={r.combo.short_hash()} | "
              f"exact={r.exact_text_accuracy * 100:.1f}% | "
              f"CER={r.mean_best_frame_cer:.4f} | "
              f"runtime={r.ocr_runtime_sec:.1f}s | "
              f"models={r.combo.detection_model_name.split('_')[1]}/"
              f"{r.combo.recognition_model_name.split('_')[1]} | "
              f"step={r.combo.sample_step_sec}s | "
              f"det_thresh={r.combo.text_det_thresh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
