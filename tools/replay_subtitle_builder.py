#!/usr/bin/env python3
"""Replay SubtitleBuilder từ dữ liệu OCR thô — test và so sánh chất lượng.

Mục đích:
    Chạy và đánh giá SubtitleBuilder với các tham số cấu hình khác nhau
    mà KHÔNG cần chạy lại OCR (tiết kiệm hàng phút xử lý GPU).

Cách dùng:
    # Chạy với config mặc định:
    python replay_subtitle_builder.py chinese_vid1.seraw.json

    # Chạy với config tuỳ chỉnh:
    python replay_subtitle_builder.py chinese_vid1.seraw.json \\
        --similarity-threshold 0.80 \\
        --merge-gap 0.50 \\
        --min-duration 0.15 \\
        --use-viterbi

    # So sánh với phụ đề chuẩn:
    python replay_subtitle_builder.py chinese_vid1.seraw.json \\
        --reference chinese_vid1_good.srt \\
        --similarity-threshold 0.75

    # Quét nhiều tham số tự động (grid search):
    python replay_subtitle_builder.py chinese_vid1.seraw.json \\
        --reference chinese_vid1_good.srt \\
        --grid-search

    # Xuất kết quả ra file SRT:
    python replay_subtitle_builder.py chinese_vid1.seraw.json \\
        --output out.srt \\
        --reference chinese_vid1_good.srt

Pipeline:
    1. Load OcrFrameResult từ .seraw.json
    2. Khởi tạo SubtitleBuilder với config được cung cấp
    3. Gọi builder.build(frames)
    4. [Tuỳ chọn] So sánh output với reference SRT và in báo cáo
    5. [Tuỳ chọn] Xuất file SRT

Replay SubtitleBuilder từ dữ liệu OCR thô — test và so sánh chất lượng.

Mục đích:
    Chạy và đánh giá SubtitleBuilder với các tham số cấu hình khác nhau
    mà KHÔNG cần chạy lại OCR (tiết kiệm hàng phút xử lý GPU).

[CẢI TIẾN QUAN TRỌNG]: Tự động truy xuất và nhúng `Roi` vào quá trình Build giả lập
nhằm kích hoạt màng lọc rác không gian (Spatial Outlier).
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


@dataclass
class SrtEvent:
    index: int
    start_sec: float
    end_sec: float
    text: str


def main() -> int:
    args = _parse_args()

    try:
        frames, meta = _load_raw_data(Path(args.raw_data))
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[LỖI] Không nạp được file dữ liệu thô: {exc}", file=sys.stderr)
        return 1

    print(_format_meta(meta, len(frames)))

    # 2. Khôi phục Vùng quét (ROI) từ Metadata
    roi = None
    if meta.roi_xywh:
        from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment
        # Bắt buộc Căn Giữa để tối ưu hóa màng lọc Rác khi chạy Offline Test
        roi = Roi(
            x=meta.roi_xywh[0], 
            y=meta.roi_xywh[1], 
            width=meta.roi_xywh[2], 
            height=meta.roi_xywh[3], 
            alignment=TextAlignment.CENTER
        )

    reference_events: list[SrtEvent] | None = None
    if args.reference:
        try:
            reference_events = _load_srt(Path(args.reference))
            print(f"Phụ đề chuẩn: {args.reference} ({len(reference_events)} câu)\n")
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"[CẢNH BÁO] Không nạp được phụ đề chuẩn: {exc}", file=sys.stderr)

    if args.grid_search:
        return _run_grid_search(frames, reference_events, args, roi)
    else:
        return _run_single(frames, reference_events, args, roi, verbose=True)


def _run_grid_search(
    frames: list,
    reference_events: list[SrtEvent] | None,
    args: argparse.Namespace,
    roi: object = None
) -> int:
    if reference_events is None:
        print("[LỖI] Grid search cần --reference.", file=sys.stderr)
        return 1

    grid = {
        "similarity_threshold":[0.70, 0.75, 0.80, 0.85],
        "merge_gap_sec":[0.40, 0.60, 0.80],
        "min_duration_sec":[0.10, 0.15, 0.20],
        "use_viterbi":          [False, True],
    }

    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(itertools.product(*param_values))

    print(f"Grid search: {len(combinations)} tổ hợp tham số\n")
    print(f"{'#':>4}  {'sim':>5} {'gap':>5} {'dur':>5} {'vit':>4}  "
          f"{'câu':>5} {'F1':>7} {'WER':>7} {'score':>8}")
    print("-" * 65)

    best_score = -1.0
    best_params: dict = {}
    best_result_str = ""

    for combo_idx, combo in enumerate(combinations, start=1):
        params = dict(zip(param_names, combo))
        builder_cfg = _build_config(**params)

        try:
            result_events = _run_builder(frames, builder_cfg, roi)
        except Exception as exc:  # noqa: BLE001
            print(f"[LỖI] combo #{combo_idx}: {exc}", file=sys.stderr)
            continue

        metrics = _compare(result_events, reference_events)
        composite = (
            metrics["subtitle_f1"] * 0.60
            + metrics["count_ratio"] * 0.20
            + metrics["timing_score"] * 0.20
        )

        row = (
            f"{combo_idx:>4}  {params['similarity_threshold']:>5.2f} "
            f"{params['merge_gap_sec']:>5.2f} {params['min_duration_sec']:>5.2f} "
            f"{'T' if params['use_viterbi'] else 'F':>4}  "
            f"{len(result_events):>5} {metrics['subtitle_f1']:>7.3f} "
            f"{metrics['wer']:>7.3f} {composite:>8.4f}"
        )
        print(row)

        if composite > best_score:
            best_score = composite
            best_params = params
            best_result_str = row

    print("\n" + "=" * 65)
    print(f"🏆 Tốt nhất (score={best_score:.4f}):")
    print(f"   {best_result_str}")
    print(f"\nLệnh chạy lại:")
    print(
        f"   python {Path(__file__).name} {args.raw_data}"
        + (f" --reference {args.reference}" if args.reference else "")
        + f" --similarity-threshold {best_params.get('similarity_threshold', 0.75)}"
        + f" --merge-gap {best_params.get('merge_gap_sec', 0.60)}"
        + f" --min-duration {best_params.get('min_duration_sec', 0.15)}"
        + (" --use-viterbi" if best_params.get("use_viterbi") else "")
    )
    return 0


def _run_single(
    frames: list,
    reference_events: list[SrtEvent] | None,
    args: argparse.Namespace,
    roi: object = None,
    verbose: bool = True,
) -> int:
    builder_cfg = _build_config(
        similarity_threshold=args.similarity_threshold,
        merge_gap_sec=args.merge_gap,
        min_duration_sec=args.min_duration,
        min_confidence=args.min_confidence,
        use_viterbi=args.use_viterbi,
        viterbi_open_penalty=args.viterbi_penalty,
        min_text_chars=args.min_text_chars,
    )

    import time
    t0 = time.perf_counter()
    try:
        result_events = _run_builder(frames, builder_cfg, roi)
    except Exception as exc:  # noqa: BLE001
        print(f"[LỖI] SubtitleBuilder thất bại: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - t0

    print(f"SubtitleBuilder: {len(result_events)} câu trong {elapsed:.3f}s")

    if verbose:
        print("\n--- 10 câu đầu ---")
        for ev in result_events[:10]:
            print(f"  {_sec_to_srt(ev.start_sec)} --> {_sec_to_srt(ev.end_sec)}  {ev.text}")
        if len(result_events) > 10:
            print(f"  ... ({len(result_events) - 10} câu nữa)")

    if reference_events is not None:
        print()
        _print_comparison(result_events, reference_events)

    if args.output:
        _write_srt(result_events, Path(args.output))
        print(f"\n✅ Đã xuất: {args.output}")

    return 0


def _compare(
    result: list,
    reference: list[SrtEvent],
) -> dict[str, float]:
    """So sánh kết quả build với reference SRT.

    v3.6+: Nâng cấp matching bằng text-similarity boost —
    nếu IOU < 0.30 nhưng text similarity >= 0.80 và overlap > 0,
    vẫn tính là khớp (bắt các timing-offset case).
    Sliding-window O((N+M)×k) thay vì O(N×M).
    """
    import bisect
    import rapidfuzz.fuzz as _fuzz

    if not reference:
        return {"subtitle_f1": 0.0, "count_ratio": 0.0, "wer": 1.0,
                "timing_score": 0.0, "missing": 0, "extra": len(result)}

    n_ref = len(reference)
    n_res = len(result)
    count_ratio = min(n_ref, n_res) / max(n_ref, n_res) if max(n_ref, n_res) > 0 else 0.0

    res_sorted = sorted(result, key=lambda e: e.start_sec)
    ref_sorted = sorted(reference, key=lambda e: e.start_sec)
    ref_starts = [e.start_sec for e in ref_sorted]

    _WINDOW_SEC = 5.0       # ±5s sliding window
    _IOU_HARD = 0.30        # hard IOU threshold (no text check needed)
    _IOU_SOFT = 0.05        # soft IOU (any overlap) + text similarity
    _TEXT_SIM_MIN = 0.75    # text similarity ngưỡng tối thiểu cho soft match

    matched_pairs: list[tuple[int, int]] = []
    used_ref: set[int] = set()

    for ri, rev in enumerate(res_sorted):
        lo = bisect.bisect_left(ref_starts, rev.start_sec - _WINDOW_SEC)
        hi = bisect.bisect_right(ref_starts, rev.end_sec + _WINDOW_SEC)
        best_score = 0.0
        best_ji = -1

        for ji in range(lo, hi):
            if ji in used_ref:
                continue
            jev = ref_sorted[ji]
            iou = _time_iou(rev.start_sec, rev.end_sec, jev.start_sec, jev.end_sec)

            if iou >= _IOU_HARD:
                # Hard match by timing alone
                score = iou + 1.0  # bias to prefer hard over soft
            elif iou >= _IOU_SOFT:
                # Soft match: require text similarity
                text_sim = _fuzz.ratio(
                    rev.text.replace(" ", ""),
                    jev.text.replace(" ", ""),
                ) / 100.0
                if text_sim >= _TEXT_SIM_MIN:
                    score = iou + text_sim * 0.5
                else:
                    continue
            else:
                continue

            if score > best_score:
                best_score = score
                best_ji = ji

        if best_ji >= 0:
            matched_pairs.append((ri, best_ji))
            used_ref.add(best_ji)

    n_matched = len(matched_pairs)
    precision = n_matched / n_res if n_res > 0 else 0.0
    recall = n_matched / n_ref if n_ref > 0 else 0.0
    subtitle_f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    all_ref_text = "".join(ev.text for ev in reference)
    all_res_text = "".join(ev.text for ev in result)
    wer = _cer(all_res_text, all_ref_text)
    timing_score = n_matched / n_ref if n_ref > 0 else 0.0

    return {
        "subtitle_f1": subtitle_f1,
        "count_ratio": count_ratio,
        "wer": wer,
        "timing_score": timing_score,
        "missing": n_ref - n_matched,
        "extra": n_res - n_matched,
    }


def _print_comparison(result: list, reference: list[SrtEvent]) -> None:
    metrics = _compare(result, reference)
    n_ref = len(reference)
    n_res = len(result)

    print("═" * 55)
    print("  SO SÁNH VỚI PHỤ ĐỀ CHUẨN")
    print("═" * 55)
    print(f"  Phụ đề chuẩn  : {n_ref:>4} câu")
    print(f"  Kết quả        : {n_res:>4} câu  "
          f"({n_res - n_ref:+d})")
    print(f"  Ghép đúng      : {n_ref - int(metrics['missing']):>4} câu "
          f"({(1 - metrics['missing'] / n_ref) * 100:.1f}%)")
    print(f"  Câu bị mất     : {int(metrics['missing']):>4} câu "
          f"(có trong chuẩn nhưng thiếu trong kết quả)")
    print(f"  Câu thừa       : {int(metrics['extra']):>4} câu "
          f"(không có trong chuẩn)")
    print()
    print(f"  F1 phụ đề      : {metrics['subtitle_f1']:.4f}  "
          f"{'(tốt)' if metrics['subtitle_f1'] >= 0.85 else '(cần cải thiện)'}")
    print(f"  CER text       : {metrics['wer']:.4f}  "
          f"(0=hoàn hảo, 1=tệ nhất)")
    print(f"  Timing score   : {metrics['timing_score']:.4f}")

    composite = (
        metrics["subtitle_f1"] * 0.60
        + metrics["count_ratio"] * 0.20
        + metrics["timing_score"] * 0.20
    )
    bar_len = int(composite * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    print(f"\n  Điểm tổng hợp  : {composite:.4f}  [{bar}]")
    print("═" * 55)


def _load_raw_data(path: Path) -> tuple[list, object]:
    from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import (
        load_raw_ocr,
    )
    return load_raw_ocr(path)


def _build_config(
    similarity_threshold: float = 0.75,
    merge_gap_sec: float = 0.60,
    min_duration_sec: float = 0.15,
    min_confidence: float = 0.50,
    use_viterbi: bool = False,
    viterbi_open_penalty: float = 0.35,
    min_text_chars: int = 2,
    **_: object,
) -> object:
    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        SubtitleBuilderConfig,
    )
    return SubtitleBuilderConfig(
        similarity_threshold=similarity_threshold,
        merge_gap_sec=merge_gap_sec,
        min_duration_sec=min_duration_sec,
        min_confidence=min_confidence,
        use_viterbi=use_viterbi,
        viterbi_open_penalty=viterbi_open_penalty,
        min_text_chars=min_text_chars,
    )


def _run_builder(frames: list, config: object, roi: object = None) -> list:
    from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
    builder = SubtitleBuilder(config)
    return builder.build(frames, roi=roi)


def _load_srt(path: Path) -> list[SrtEvent]:
    content = path.read_text(encoding="utf-8-sig")
    events: list[SrtEvent] =[]
    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        m = re.search(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1],
        )
        if not m:
            continue
        events.append(SrtEvent(
            index=int(lines[0].strip()) if lines[0].strip().isdigit() else 0,
            start_sec=_srt_to_sec(m.group(1)),
            end_sec=_srt_to_sec(m.group(2)),
            text=" ".join(lines[2:]).strip(),
        ))
    return events


def _write_srt(events: list, path: Path) -> None:
    lines: list[str] =[]
    for i, ev in enumerate(events, start=1):
        lines.append(str(i))
        lines.append(f"{_sec_to_srt(ev.start_sec)} --> {_sec_to_srt(ev.end_sec)}")
        lines.append(ev.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _sec_to_srt(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _srt_to_sec(ts: str) -> float:
    h, m, s_ms = ts.split(":")
    s, ms = s_ms.replace(",", ".").split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _time_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def _cer(hyp: str, ref: str) -> float:
    """Tính Character Error Rate — dùng rapidfuzz (C++) thay vì Python DP.

    Tối ưu v2.9: Python DP gốc O(|hyp|×|ref|) → timeout với chuỗi dài
    (test3: 30k×32k = 960M ops). rapidfuzz C++ SIMD nhanh hơn 100-200×.
    """
    if not ref:
        return 0.0 if not hyp else 1.0
    hyp_c = hyp.replace(" ", "")
    ref_c = ref.replace(" ", "")
    if not ref_c:
        return 1.0
    try:
        import rapidfuzz.distance.Levenshtein as _lev
        return min(1.0, _lev.normalized_distance(hyp_c, ref_c))
    except Exception:
        # Fallback chunked DP nếu rapidfuzz không available
        n, m = len(ref_c), len(hyp_c)
        prev = list(range(m + 1))
        for rc in ref_c:
            curr = [prev[0] + 1]
            for j, hc in enumerate(hyp_c, 1):
                curr.append(prev[j - 1] if rc == hc else 1 + min(prev[j], curr[-1], prev[j - 1]))
            prev = curr
        return min(1.0, prev[m] / n)


def _format_meta(meta: object, n_frames: int) -> str:
    lines =[
        "┌─ Thông tin dữ liệu thô ───────────────────────────┐",
        f"│  Video       : {meta.video_name}",
        f"│  Thời lượng  : {meta.video_duration_sec:.1f}s",
        f"│  Số frame    : {n_frames}",
        f"│  Sample step : {meta.sample_step_sec}s",
        f"│  Det model   : {meta.detection_model}",
        f"│  Rec model   : {meta.recognition_model}",
        f"│  Lưu lúc     : {meta.saved_at}",
        "└──────────────────────────────────────────────────┘",
    ]
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay SubtitleBuilder từ dữ liệu OCR thô (.seraw.json)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Ví dụ:
  # Chạy cơ bản:
  python replay_subtitle_builder.py data.seraw.json

  # So sánh với chuẩn:
  python replay_subtitle_builder.py data.seraw.json -r good.srt

  # Tuỳ chỉnh tham số:
  python replay_subtitle_builder.py data.seraw.json -r good.srt --sim 0.80 --gap 0.50

  # Grid search tự động:
  python replay_subtitle_builder.py data.seraw.json -r good.srt --grid-search
""",
    )
    p.add_argument("raw_data", help="Đường dẫn file .seraw.json hoặc .seraw.json.gz")
    p.add_argument("-r", "--reference", metavar="SRT",
                   help="File phụ đề chuẩn để so sánh chất lượng")
    p.add_argument("-o", "--output", metavar="SRT",
                   help="Xuất kết quả ra file SRT")

    p.add_argument("--sim", "--similarity-threshold", dest="similarity_threshold",
                   type=float, default=0.75, metavar="FLOAT",
                   help="Ngưỡng similarity để gộp frame (default: 0.75)")
    p.add_argument("--gap", "--merge-gap", dest="merge_gap",
                   type=float, default=0.60, metavar="SEC",
                   help="Khoảng cách thời gian tối đa để gộp (default: 0.60s)")
    p.add_argument("--min-duration", dest="min_duration",
                   type=float, default=0.15, metavar="SEC",
                   help="Thời lượng tối thiểu (default: 0.15s)")
    p.add_argument("--min-confidence", dest="min_confidence",
                   type=float, default=0.50, metavar="FLOAT",
                   help="Ngưỡng confidence OCR tối thiểu (default: 0.50)")
    p.add_argument("--min-text-chars", dest="min_text_chars",
                   type=int, default=2, metavar="N",
                   help="Số ký tự tối thiểu của câu (default: 2)")
    p.add_argument("--use-viterbi", dest="use_viterbi",
                   action="store_true", default=False,
                   help="Dùng thuật toán Viterbi thay vì Greedy")
    p.add_argument("--viterbi-penalty", dest="viterbi_penalty",
                   type=float, default=0.35, metavar="FLOAT",
                   help="Open penalty của Viterbi (default: 0.35)")

    p.add_argument("--grid-search", dest="grid_search",
                   action="store_true", default=False,
                   help="Quét tự động lưới tham số và tìm combo tốt nhất")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())