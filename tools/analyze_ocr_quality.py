"""Phân tích chất lượng OCR RAW (`.seraw.json`) đối chiếu phụ đề chuẩn.

Mục đích: đo lường khả năng của TẦNG OCR (PaddleOCR + preprocessing +
config), TÁCH BIỆT khỏi tầng SubtitleBuilder. Trả lời:

    1. OCR engine có DETECT được câu phụ đề ở thời điểm đó không?
       → ``detection_recall``.
    2. Khi đã detect, OCR engine có ĐỌC ĐÚNG text không?
       → ``exact_text_accuracy`` (>= 1 frame match exact với ref text).
    3. Mức độ sai chữ trung bình.
       → ``per_event_min_cer`` (best frame match per event).
    4. Bao nhiêu frame "rác" (text không khớp câu chuẩn nào).
       → ``garbage_ratio``.

Tool này KHÔNG cần PaddleOCR — chỉ cần `.seraw.json` có sẵn.

Usage:
    python tools/analyze_ocr_quality.py path/to/file.seraw.json \\
        --reference path/to/reference.srt
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import Levenshtein


@dataclass(frozen=True, slots=True)
class SubtitleEvent:
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True, slots=True)
class OcrFrameSnapshot:
    timestamp_sec: float
    text_boxes: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class QualityReport:
    total_reference_events: int
    detected_events_count: int
    exactly_matched_events_count: int
    detection_recall: float
    exact_text_accuracy: float
    mean_min_cer: float
    median_min_cer: float
    p90_min_cer: float
    total_frames: int
    frames_with_any_text: int
    total_text_boxes: int
    garbage_frames_count: int
    garbage_ratio_among_text_frames: float
    mean_confidence: float
    median_confidence: float
    p25_confidence: float

    def render_report(self) -> str:
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  BÁO CÁO CHẤT LƯỢNG OCR RAW")
        lines.append("=" * 60)
        lines.append(f"  Tổng số câu phụ đề chuẩn        : {self.total_reference_events}")
        lines.append(f"  Câu được OCR DETECT             : {self.detected_events_count} "
                     f"({self.detection_recall * 100:.1f}%)")
        lines.append(f"  Câu được OCR ĐỌC ĐÚNG EXACT     : {self.exactly_matched_events_count} "
                     f"({self.exact_text_accuracy * 100:.1f}%)")
        lines.append("")
        lines.append(f"  Mean best-frame CER             : {self.mean_min_cer:.4f}")
        lines.append(f"  Median best-frame CER           : {self.median_min_cer:.4f}")
        lines.append(f"  P90 best-frame CER              : {self.p90_min_cer:.4f}")
        lines.append("")
        lines.append(f"  Tổng frames OCR sampled         : {self.total_frames}")
        lines.append(f"  Frames có >=1 text box          : {self.frames_with_any_text} "
                     f"({self.frames_with_any_text / max(1, self.total_frames) * 100:.1f}%)")
        lines.append(f"  Tổng số text boxes              : {self.total_text_boxes}")
        lines.append(f"  Frames rác (không khớp câu nào) : {self.garbage_frames_count} "
                     f"({self.garbage_ratio_among_text_frames * 100:.1f}%)")
        lines.append("")
        lines.append(f"  Mean confidence                 : {self.mean_confidence:.4f}")
        lines.append(f"  Median confidence               : {self.median_confidence:.4f}")
        lines.append(f"  P25 confidence                  : {self.p25_confidence:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


def parse_srt_to_seconds(timecode: str) -> float:
    hours, minutes, rest = timecode.split(":")
    seconds, milliseconds = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0


def load_reference_srt(path: Path) -> list[SubtitleEvent]:
    raw_content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    blocks = re.split(r"\n\n+", raw_content.strip())
    events: list[SubtitleEvent] = []
    for block in blocks:
        block_lines = block.strip().splitlines()
        if len(block_lines) < 3:
            continue
        match = re.search(
            r"(\d\d:\d\d:\d\d,\d{3}).*-->.*?(\d\d:\d\d:\d\d,\d{3})",
            block_lines[1],
        )
        if not match:
            continue
        events.append(
            SubtitleEvent(
                start_sec=parse_srt_to_seconds(match.group(1)),
                end_sec=parse_srt_to_seconds(match.group(2)),
                text=" ".join(block_lines[2:]).strip(),
            )
        )
    return events


def load_raw_ocr_snapshots(path: Path) -> list[OcrFrameSnapshot]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        OcrFrameSnapshot(
            timestamp_sec=float(frame["ts"]),
            text_boxes=tuple(
                (str(box["t"]).strip(), float(box["c"]))
                for box in frame.get("boxes", [])
            ),
        )
        for frame in data.get("frames", [])
    ]


def compute_character_error_rate(hypothesis_text: str, reference_text: str) -> float:
    """CER chuẩn ngành = edit_distance / max(1, len(reference))."""
    if not reference_text:
        return 1.0 if hypothesis_text else 0.0
    distance = Levenshtein.distance(hypothesis_text, reference_text)
    return distance / len(reference_text)


def analyze_ocr_quality(
    reference_events: list[SubtitleEvent],
    frame_snapshots: list[OcrFrameSnapshot],
    time_tolerance_sec: float = 0.10,
    min_cer_for_garbage: float = 0.50,
) -> QualityReport:
    """Tính chất lượng OCR raw.

    Args:
        reference_events: Phụ đề chuẩn.
        frame_snapshots: Danh sách frame OCR đã load.
        time_tolerance_sec: Mở rộng cửa sổ thời gian khi đối chiếu (mặc định
            +/- 0.10s — đủ cover 2-3 frame sample ở 25fps).
        min_cer_for_garbage: Frame có text với CER tốt nhất so với mọi câu
            ref vẫn >= ngưỡng này → coi là rác.

    Returns:
        QualityReport.
    """
    detected_events_count = 0
    exactly_matched_events_count = 0
    per_event_min_cer_list: list[float] = []
    confidence_values: list[float] = []
    total_text_boxes = 0
    frames_with_any_text = 0

    # Tập text reference đã chuẩn hóa (đối chiếu nhanh).
    reference_text_set: set[str] = {event.text for event in reference_events}

    for snapshot in frame_snapshots:
        if snapshot.text_boxes:
            frames_with_any_text += 1
            for _box_text, box_confidence in snapshot.text_boxes:
                total_text_boxes += 1
                confidence_values.append(box_confidence)

    for ref_event in reference_events:
        candidate_texts: list[tuple[str, float]] = []
        for snapshot in frame_snapshots:
            if (
                ref_event.start_sec - time_tolerance_sec
                <= snapshot.timestamp_sec
                <= ref_event.end_sec + time_tolerance_sec
            ):
                for box_text, box_confidence in snapshot.text_boxes:
                    if box_text:
                        candidate_texts.append((box_text, box_confidence))

        if not candidate_texts:
            per_event_min_cer_list.append(1.0)
            continue

        detected_events_count += 1
        # Tìm best CER trong vùng.
        per_event_min_cer = min(
            compute_character_error_rate(box_text, ref_event.text)
            for box_text, _ in candidate_texts
        )
        per_event_min_cer_list.append(per_event_min_cer)

        if per_event_min_cer == 0.0:
            exactly_matched_events_count += 1

    # Đếm frame rác: text trong frame không match câu ref nào (CER tốt nhất
    # so với mọi câu ref >= ngưỡng).
    garbage_frames_count = 0
    for snapshot in frame_snapshots:
        if not snapshot.text_boxes:
            continue
        is_garbage_frame = True
        for box_text, _ in snapshot.text_boxes:
            if box_text in reference_text_set:
                is_garbage_frame = False
                break
            min_cer_against_any_ref = min(
                compute_character_error_rate(box_text, ref_event.text)
                for ref_event in reference_events
            )
            if min_cer_against_any_ref < min_cer_for_garbage:
                is_garbage_frame = False
                break
        if is_garbage_frame:
            garbage_frames_count += 1

    total_reference_events = len(reference_events)
    sorted_cer_values = sorted(per_event_min_cer_list)

    return QualityReport(
        total_reference_events=total_reference_events,
        detected_events_count=detected_events_count,
        exactly_matched_events_count=exactly_matched_events_count,
        detection_recall=detected_events_count / max(1, total_reference_events),
        exact_text_accuracy=exactly_matched_events_count / max(1, total_reference_events),
        mean_min_cer=statistics.mean(per_event_min_cer_list) if per_event_min_cer_list else 0.0,
        median_min_cer=statistics.median(per_event_min_cer_list) if per_event_min_cer_list else 0.0,
        p90_min_cer=sorted_cer_values[int(0.9 * len(sorted_cer_values))] if sorted_cer_values else 0.0,
        total_frames=len(frame_snapshots),
        frames_with_any_text=frames_with_any_text,
        total_text_boxes=total_text_boxes,
        garbage_frames_count=garbage_frames_count,
        garbage_ratio_among_text_frames=garbage_frames_count / max(1, frames_with_any_text),
        mean_confidence=statistics.mean(confidence_values) if confidence_values else 0.0,
        median_confidence=statistics.median(confidence_values) if confidence_values else 0.0,
        p25_confidence=sorted(confidence_values)[len(confidence_values) // 4] if confidence_values else 0.0,
    )


def print_per_event_diagnosis(
    reference_events: list[SubtitleEvent],
    frame_snapshots: list[OcrFrameSnapshot],
    time_tolerance_sec: float = 0.10,
    max_problematic: int = 20,
) -> None:
    """In chẩn đoán per-event với CER > 0 (sắp xếp giảm dần)."""
    issues: list[tuple[float, str, str, float, int, float]] = []

    for ref_event in reference_events:
        candidates: list[tuple[str, float]] = []
        for snapshot in frame_snapshots:
            if (
                ref_event.start_sec - time_tolerance_sec
                <= snapshot.timestamp_sec
                <= ref_event.end_sec + time_tolerance_sec
            ):
                for box_text, box_confidence in snapshot.text_boxes:
                    if box_text:
                        candidates.append((box_text, box_confidence))

        if not candidates:
            issues.append((ref_event.start_sec, ref_event.text, "<MISSING>", 0.0, 0, 1.0))
            continue

        best_text, best_confidence = min(
            candidates,
            key=lambda item: compute_character_error_rate(item[0], ref_event.text),
        )
        best_cer = compute_character_error_rate(best_text, ref_event.text)
        if best_cer > 0:
            issues.append((ref_event.start_sec, ref_event.text, best_text, best_confidence, len(candidates), best_cer))

    issues.sort(key=lambda row: row[5], reverse=True)

    print(f"\n--- Top {min(max_problematic, len(issues))} câu OCR đọc SAI/MISS ---")
    for start, ref_text, best_text, conf, n_candidates, cer in issues[:max_problematic]:
        if best_text == "<MISSING>":
            print(f"  [{start:7.2f}s] MISS    | ref={ref_text!r}")
        else:
            print(
                f"  [{start:7.2f}s] CER={cer:.3f} | ref={ref_text!r:35s} "
                f"best_ocr={best_text!r:35s} (conf={conf:.3f}, {n_candidates} cands)"
            )


def run_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Phân tích chất lượng OCR raw từ .seraw.json đối chiếu phụ đề chuẩn.",
    )
    parser.add_argument("seraw_json_path", type=Path, help="Đường dẫn file .seraw.json")
    parser.add_argument(
        "-r", "--reference",
        type=Path, required=True,
        help="Đường dẫn file SRT phụ đề chuẩn",
    )
    parser.add_argument(
        "--tolerance",
        type=float, default=0.10,
        help="Mở rộng cửa sổ thời gian khi đối chiếu (giây, mặc định 0.10)",
    )
    parser.add_argument(
        "--garbage-cer",
        type=float, default=0.50,
        help="Ngưỡng CER để xác định frame rác (mặc định 0.50)",
    )
    parser.add_argument(
        "--top-problems",
        type=int, default=20,
        help="In top N câu sai (mặc định 20)",
    )
    args = parser.parse_args()

    if not args.seraw_json_path.exists():
        print(f"LỖI: không tìm thấy {args.seraw_json_path}", file=sys.stderr)
        return 1
    if not args.reference.exists():
        print(f"LỖI: không tìm thấy {args.reference}", file=sys.stderr)
        return 1

    reference_events = load_reference_srt(args.reference)
    frame_snapshots = load_raw_ocr_snapshots(args.seraw_json_path)

    report = analyze_ocr_quality(
        reference_events=reference_events,
        frame_snapshots=frame_snapshots,
        time_tolerance_sec=args.tolerance,
        min_cer_for_garbage=args.garbage_cer,
    )
    print(report.render_report())
    print_per_event_diagnosis(
        reference_events=reference_events,
        frame_snapshots=frame_snapshots,
        time_tolerance_sec=args.tolerance,
        max_problematic=args.top_problems,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
