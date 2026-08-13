"""Hiệu chuẩn ĐỆ QUY thuật toán tự tách câu STT từ dữ liệu thô (.sestt.json).

Quy trình:
    1. Đọc dữ liệu thô WhisperX (.sestt.json) + (tuỳ chọn) SRT mẫu chuẩn.
    2. Chạy thuật toán tách câu với NHIỀU bộ tham số (grid search).
    3. So với SRT mẫu bằng các chỉ số: số cue, độ dài TB, độ lệch số cue, F1 ranh
       giới thời gian (boundary), CER ghép văn bản.
    4. In bảng xếp hạng để chọn tham số tối ưu.

Dùng:
    python tools/calibrate_stt_split.py data.sestt.json --reference good.srt
    python tools/calibrate_stt_split.py data.sestt.json   # chỉ thống kê, không so
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig  # noqa: E402
from subtitles_extractor.infrastructure.stt.whisperx_adapter import WhisperXAdapter  # noqa: E402


@dataclass
class CueStat:
    count: int
    avg_dur: float
    max_dur: float
    avg_chars: float


def _load_segments(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("segments", payload if isinstance(payload, list) else [])


def _parse_srt_cues(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8-sig")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        m = re.search(
            r"(\d\d):(\d\d):(\d\d)[,.](\d+)\s*-->\s*(\d\d):(\d\d):(\d\d)[,.](\d+)", lines[1]
        )
        if not m:
            continue
        g = list(map(int, m.groups()))
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        cues.append((start, end, "".join(lines[2:])))
    return cues


def _stat_from_events(events) -> CueStat:
    if not events:
        return CueStat(0, 0.0, 0.0, 0.0)
    durs = [e.interval.end_sec - e.interval.start_sec for e in events]
    chars = [len(e.text) for e in events]
    return CueStat(len(events), sum(durs) / len(durs), max(durs), sum(chars) / len(chars))


def _boundary_f1(events, reference: list[tuple[float, float, str]], tol: float = 0.5) -> float:
    """F1 của ranh giới bắt đầu cue (khớp trong dung sai tol giây)."""
    if not events or not reference:
        return 0.0
    pred_starts = [e.interval.start_sec for e in events]
    ref_starts = [r[0] for r in reference]
    matched = 0
    used = set()
    for ps in pred_starts:
        for i, rs in enumerate(ref_starts):
            if i in used:
                continue
            if abs(ps - rs) <= tol:
                matched += 1
                used.add(i)
                break
    precision = matched / len(pred_starts)
    recall = matched / len(ref_starts)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hiệu chuẩn thuật toán tách câu STT.")
    parser.add_argument("raw_path", type=Path, help="File .sestt.json dữ liệu thô.")
    parser.add_argument("--reference", type=Path, default=None, help="SRT mẫu chuẩn để so.")
    args = parser.parse_args()

    segments = _load_segments(args.raw_path)
    has_words = sum(1 for s in segments if s.get("words"))
    print(f"Đọc {len(segments)} segment, {has_words} có word-level.")
    reference = _parse_srt_cues(args.reference) if args.reference else None
    if reference:
        ref_stat_durs = [e - s for s, e, _ in reference]
        print(
            f"SRT mẫu: {len(reference)} cue, dài TB="
            f"{sum(ref_stat_durs)/len(ref_stat_durs):.2f}s"
        )

    # Grid search các tham số.
    gaps = [0.25, 0.4, 0.6]
    max_chars_list = [12, 16, 20]
    max_durs = [4.0, 5.0, 6.0]

    results = []
    for gap, max_chars, max_dur in product(gaps, max_chars_list, max_durs):
        cfg = TranscriptionConfig(
            split_gap_sec=gap, max_chars_per_cue=max_chars, max_cue_duration_sec=max_dur,
        )
        events = WhisperXAdapter._segments_to_events(segments, cfg)
        stat = _stat_from_events(events)
        f1 = _boundary_f1(events, reference) if reference else 0.0
        count_diff = abs(stat.count - len(reference)) if reference else 0
        results.append((gap, max_chars, max_dur, stat, f1, count_diff))

    # Xếp hạng: nếu có reference → ưu tiên F1 cao + lệch số cue thấp.
    if reference:
        results.sort(key=lambda r: (-r[4], r[5]))
    else:
        results.sort(key=lambda r: r[3].avg_dur)

    print("\n gap | chars | maxdur | #cue | dàiTB |  F1  | lệch#")
    print("-" * 58)
    for gap, mc, md, stat, f1, cdiff in results[:12]:
        print(
            f"{gap:4.2f} | {mc:5d} | {md:6.1f} | {stat.count:4d} | "
            f"{stat.avg_dur:5.2f} | {f1:.3f} | {cdiff:4d}"
        )
    if reference and results:
        best = results[0]
        print(
            f"\n→ Tham số tốt nhất: gap={best[0]}, max_chars={best[1]}, "
            f"max_dur={best[2]} (F1={best[4]:.3f}, lệch {best[5]} cue)."
        )


if __name__ == "__main__":
    main()
