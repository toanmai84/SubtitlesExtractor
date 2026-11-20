"""[v3.23.742] Báo cáo chất lượng một lượt "Xuất dữ liệu thô OCR" (giai đoạn 0 kế hoạch 4.0) — không cần dò mắt từng dòng.

    python tools/quality_report.py <thư mục chứa phim.seraw.json.gz, phim.qa.json.gz, phim.roi.mp4, phim.roi.json> [--timing N]

Đo trên kết quả ghi trong seraw (``meta.result``):

1. **Nội dung**: câu mà PaddleOCR-VL (gói kiểm định) đọc lại ≥ 2/3 khung giống hệt → đồng thuận; còn lại → danh sách bất
   đồng cần xem ảnh (VL đọc cả dòng nên hay dính chữ trong cảnh / ký hiệu: bất đồng KHÔNG có nghĩa ứng dụng sai).
2. **Câu sót (mức bộ dò)**: khung có hộp chữ dạng phụ đề (tin cậy cao, đúng dải dọc đo từ phim) mà không câu nào phủ.
3. **Câu sót (mức điểm ảnh)**: quét video ROI mỗi 0,2 s, tỉ lệ điểm sáng có viền tối (nét chữ) trong dải phụ đề; đoạn ≥ 0,6 s
   vượt ngưỡng (phân vị 10 % của khung có câu) mà không câu nào phủ → ứng viên câu sót (bộ dò không thấy chữ).
4. **Câu thừa**: câu mà VL không thấy chữ ở cả 3 khung, hoặc điểm ảnh gần như không có nét chữ.
5. **Thời gian** (``--timing N``, mặc định 200): mẫu N câu; mặt nạ nét chữ khung giữa câu làm mẫu, đoạn liên tục quanh đó
   còn ≥ 60 % nét chữ mẫu = lúc chữ thật hiện / tắt; lệch so với mốc câu tính theo khung (mốc cuối = lúc chữ tắt).

Ghi ``quality_report.json`` + in tóm tắt. Đo thật phim nối 86 tập (lượt 3.23.741): 3634/3663 đồng thuận, 29 bất đồng (ảnh:
ứng dụng sai 1 — "腌攒"), 0 sót mức bộ dò, 1 sót mức điểm ảnh ("诶" 5660 s), 0 thừa, thời gian 298/298 đúng ±1 khung.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import random
import statistics as st
from pathlib import Path

import numpy as np


def _load(folder: Path) -> tuple[dict, dict, Path, dict]:
    seraw = json.load(gzip.open(next(folder.glob("*.seraw.json.gz"))))
    qa_path = next(folder.glob("*.qa.json.gz"), None)
    qa = json.load(gzip.open(qa_path)) if qa_path else {"lines": [], "model": ""}
    clip = next(folder.glob("*.roi.mp4"))
    side = json.loads(clip.with_suffix(".json").read_text(encoding="utf-8"))
    return seraw, qa, clip, side


def _norm(text: str) -> str:
    return "".join((text or "").split())


def _is_han(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text or "")


def text_mask(gray: np.ndarray, rows: tuple[int, int]) -> np.ndarray:
    """Mặt nạ nét chữ phụ đề: điểm sáng (≥ 200) có điểm tối (≤ 80) cách 2–3 px theo ngang/dọc (viền chữ)."""
    band = gray[rows[0]:rows[1], :].astype(np.int16)
    bright, dark = band >= 200, band <= 80
    near = np.zeros_like(dark)
    for k in (2, 3):
        near[:, k:] |= dark[:, :-k]
        near[:, :-k] |= dark[:, k:]
        near[k:, :] |= dark[:-k, :]
        near[:-k, :] |= dark[k:, :]
    return bright & near


def main() -> None:  # noqa: C901, PLR0915 — kịch bản đo nối tiếp
    import av

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--timing", type=int, default=200)
    args = parser.parse_args()
    seraw, qa, clip, side = _load(args.folder)
    frames = seraw["frames"]
    events = sorted((e["s"], e["e"], e["t"]) for e in seraw["meta"]["result"]["events"])
    starts = [s for s, _, _ in events]
    start_offset = float(side.get("source_start_sec") or 0.0)

    def covered(t: float, pad: float = 0.05) -> bool:
        i = bisect.bisect_right(starts, t + pad) - 1
        return i >= 0 and events[i][0] - pad <= t <= events[i][1] + pad

    report: dict = {"events": len(events), "qa_model": qa.get("model", "")}
    # 1. nội dung
    disagree = [line for line in qa["lines"]
                if any(line["vl"]) and sum(_norm(v) == _norm(line["t"]) for v in line["vl"]) < 2]
    blind = [line for line in qa["lines"] if not any(line["vl"])]
    report["content"] = {"agree": len(qa["lines"]) - len(disagree) - len(blind), "disagree": disagree,
                         "vl_sees_nothing": blind}
    # dải phụ đề đo từ phim
    hi = [b for f in frames for b in f["boxes"] if b["c"] >= 0.95 and _is_han(b["t"])]
    y0 = st.median(min(p[1] for p in b["p"]) for b in hi)
    y1 = st.median(max(p[1] for p in b["p"]) for b in hi)
    cx = st.median((min(p[0] for p in b["p"]) + max(p[0] for p in b["p"])) / 2 for b in hi)
    h = y1 - y0

    def subtitle_box(b: dict) -> bool:
        ys = [p[1] for p in b["p"]]
        xs = [p[0] for p in b["p"]]
        return (b["c"] >= 0.85 and abs(min(ys) - y0) < 0.35 * h and abs(max(ys) - y1) < 0.35 * h
                and abs((min(xs) + max(xs)) / 2 - cx) < 80 and _is_han(b["t"]))

    # 2. sót mức bộ dò
    runs, cur = [], None
    for f in frames:
        hit = [b for b in f["boxes"] if subtitle_box(b)]
        if hit and not covered(f["ts"]):
            cur = cur or [f["ts"], f["ts"], 0, "".join(b["t"] for b in hit)]
            cur[1], cur[2] = f["ts"], cur[2] + 1
        elif cur:
            runs.append(cur)
            cur = None
    report["missed_detector"] = runs
    # 3 + 4. quét điểm ảnh
    rows = (max(0, int(y0 - 0.3 * h)), int(y1 + 0.3 * h) + 1)
    pix: list[tuple[float, float]] = []
    with av.open(str(clip)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        nxt = 0.0
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base) + start_offset
            if t < nxt:
                continue
            nxt = t + 0.2
            pix.append((t, float(text_mask(frame.to_ndarray(format="gray"), rows).mean())))
    times = [t for t, _ in pix]
    line_level = sorted(
        st.median(vals) for s, e, _t in events
        if (vals := [r for _, r in pix[bisect.bisect_left(times, s):bisect.bisect_right(times, e)]])
    )
    # [v3.23.743] Ngưỡng = ½ mức nét chữ của câu yếu nhất (phân vị 1 % trung vị theo CÂU). Trước: phân vị 10 % theo
    # KHUNG (gồm cả khung chữ đang mờ dần) → phim 豪门Z外 ngưỡng ~0,0002, 50/50 ứng viên là cảnh không có chữ, trong khi
    # câu một chữ thật thấp nhất 0,0044.
    threshold = 0.5 * line_level[len(line_level) // 100] if line_level else 0.004
    pruns, cur = [], None
    for t, r in pix:
        if r >= threshold and not covered(t, 0.12):
            cur = cur or [t, t, 0, 0.0]
            cur[1], cur[2], cur[3] = t, cur[2] + 1, max(cur[3], r)
        elif cur:
            pruns.append(cur)
            cur = None
    report["pixel_threshold"] = threshold
    report["missed_pixels"] = sorted((x for x in pruns if x[2] >= 3), key=lambda x: -x[3])[:50]
    weak = []
    for s, e, text in events:
        vals = [r for _, r in pix[bisect.bisect_left(times, s):bisect.bisect_right(times, e)]]
        if vals and st.median(vals) < threshold * 0.25:
            weak.append((s, e, text))
    report["weak_pixel_lines"] = weak
    # 5. thời gian
    random.seed(7)
    sample = sorted(random.sample(events, min(args.timing, len(events))))
    errs_s, errs_e = [], []
    outliers: list[dict[str, object]] = []
    with av.open(str(clip)) as container:
        stream = container.streams.video[0]
        fd = 1 / float(stream.average_rate or 25)
        for s, e, _text in sample:
            container.seek(int(max(0.0, s - 0.5 - start_offset - 1.0) / float(stream.time_base)), stream=stream)
            fr = []
            for frame in container.decode(stream):
                t = float(frame.pts * stream.time_base) + start_offset
                if t < s - 0.5:
                    continue
                if t > e + 0.5:
                    break
                fr.append((t, text_mask(frame.to_ndarray(format="gray"), rows)))
            if len(fr) < 3:
                continue
            k = min(range(len(fr)), key=lambda i: abs(fr[i][0] - (s + e) / 2))
            ref = fr[k][1]
            if ref.sum() < 400:
                continue

            def kept(m: np.ndarray, ref: np.ndarray = ref) -> float:
                # [v3.23.743] Độ phủ (phần nét chữ mẫu còn trên khung) thay IoU: nền sáng đổi sau chữ làm PHẦN HỢP tăng
                # gấp đôi (豪门Z外 "门没关": 2746 → 5088 điểm, IoU 0,97 → 0,51) dù chữ vẫn nguyên → báo nhầm lệch 7,6 khung.
                return int((m & ref).sum()) / int(ref.sum())

            i0, i1 = k, k
            while i0 > 0 and kept(fr[i0 - 1][1]) >= 0.6:
                i0 -= 1
            while i1 + 1 < len(fr) and kept(fr[i1 + 1][1]) >= 0.6:
                i1 += 1
            if i0 == 0 or i1 == len(fr) - 1:
                continue
            errs_s.append((s - fr[i0][0]) / fd)
            errs_e.append((e - (fr[i1][0] + fd)) / fd)
            if abs(errs_s[-1]) > 1.01 or abs(errs_e[-1]) > 1.01:
                outliers.append({"s": s, "e": e, "t": _text, "start_frames": round(errs_s[-1], 1),
                                 "end_frames": round(errs_e[-1], 1)})
    report["timing"] = {
        "measured": len(errs_s),
        "start_within_1_frame": sum(abs(v) <= 1.01 for v in errs_s) / max(1, len(errs_s)),
        "end_within_1_frame": sum(abs(v) <= 1.01 for v in errs_e) / max(1, len(errs_e)),
        "outliers": outliers,
    }
    out = args.folder / "quality_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    c = report["content"]
    print(f"Câu: {len(events)} | VL ({report['qa_model']}) đồng thuận {c['agree']} | bất đồng {len(c['disagree'])} | "
          f"VL không thấy chữ {len(c['vl_sees_nothing'])}")
    print(f"Sót mức bộ dò: {len(runs)} đoạn | ứng viên sót mức điểm ảnh: {len(report['missed_pixels'])} | "
          f"câu ít nét chữ: {len(weak)}")
    tm = report["timing"]
    print(f"Thời gian ({tm['measured']} câu): đầu ±1 khung {tm['start_within_1_frame']:.1%}, "
          f"cuối ±1 khung {tm['end_within_1_frame']:.1%}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
