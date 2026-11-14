#!/usr/bin/env python3
"""[v3.23.717] Thử sức khâu dựng phụ đề: gây nhiễu GIẢ ĐỊNH lên OCR thô THẬT rồi dựng lại.

Mục đích: đón đầu lỗi OCR chưa gặp. Lấy một tệp ``.seraw.json(.gz)`` của lượt chạy thật, áp một kịch bản nhiễu (OCR hụt
một lần đọc, hộp chữ phình, nhiễu cảnh, đọc sai một chữ, phim độ phân giải khác, mốc thời gian lớn…), dựng lại bằng ĐÚNG
cấu hình dựng ghi trong tệp (``meta.run.builder``; tệp cũ thì dùng mặc định) và ghi SRT để so với đáp án.

Nhiễu tác động lên LẦN ĐỌC (khung được OCR + các bản chép "cùng điểm ảnh" của nó), giữ nguyên cờ cùng điểm ảnh — đúng
cách dữ liệu thật hình thành. Gây nhiễu từng khung riêng lẻ mà giữ cờ sẽ tạo dữ liệu không thể có thật.

Cách dùng::

    python tools/stress_subtitle_builder.py phim.seraw.json.gz ra.srt dropout:0.03 --seed 1
    python tools/stress_subtitle_builder.py phim.seraw.json.gz ra.srt --list

Nhóm BẤT BIẾN (kết quả phải y như ``none``): ``scale:<hệ số>``, ``tshift:<giây>``, ``shuffle``, ``fps30``,
``dupts:<tỉ lệ>``. Nhóm NHIỄU: ``dropout``, ``framedrop``, ``framedrop2``, ``lowconf``, ``tall``, ``jitter:<px>``,
``noise``, ``subst``, ``edgecut``, ``nok``, ``halfstep:<n>``. [v3.23.718] Thêm: ``split`` (hộp câu bị tách đôi), ``punct``
(dấu câu thừa), ``trad`` (chữ phồn lẻ tẻ), ``watermark`` (logo cố định), ``clip:<px>`` (vùng chữ cắt chân), ``vfr:<ms>``
(mốc rung), ``gap`` (mất khung), ``yshift:<px>`` / ``xshift:<px>`` (nửa sau phim đổi vị trí phụ đề).

Đầu vào có thể là bản nạp sẵn ``(frames, meta)`` dạng ``.pkl``; biến môi trường ``SUBEXT_SRC`` chọn mã nguồn của bản
khác để so trước / sau.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pickle
import random
import sys
from pathlib import Path

# [v3.23.718] ``SUBEXT_SRC`` chạy cùng kịch bản trên mã của bản khác (so sánh trước / sau một thay đổi).
sys.path.insert(0, os.environ.get("SUBEXT_SRC") or str(Path(__file__).resolve().parent.parent / "src"))

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.composition.builder_config import app_default_builder_config
from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import (
    SubtitleBuilder,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi
from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import load_raw_ocr

SCENARIOS = (
    "none",
    "scale",
    "tshift",
    "shuffle",
    "nok",
    "dropout",
    "framedrop",
    "framedrop2",
    "lowconf",
    "tall",
    "jitter",
    "noise",
    "subst",
    "edgecut",
    "halfstep",
    "fps30",
    "dupts",
    "split",
    "punct",
    "trad",
    "watermark",
    "clip",
    "vfr",
    "gap",
    "yshift",
    "xshift",
)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument(
    "raw", nargs="?", help="Tệp .seraw.json hoặc .seraw.json.gz của lượt chạy thật"
)
parser.add_argument("output", nargs="?", help="Tệp SRT ghi kết quả")
parser.add_argument(
    "scenario", nargs="?", default="none", help="Kịch bản[:tham số], vd dropout:0.03"
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--list", action="store_true", help="Liệt kê kịch bản rồi thoát")
options = parser.parse_args()
if options.list or not options.raw or not options.output:
    print("\n".join(SCENARIOS))
    raise SystemExit(0)

name, _, raw_arg = options.scenario.partition(":")
arg = float(raw_arg) if raw_arg else None
# [v3.23.726] Kịch bản cần tham số mà không có: trước đây sập "TypeError: '<' not supported ... 'NoneType'" giữa chừng.
_NO_ARG_SCENARIOS = frozenset({"none", "shuffle", "nok", "fps30", "watermark"})
if arg is None and name not in _NO_ARG_SCENARIOS:
    raise SystemExit(f"Kịch bản '{name}' cần tham số, ví dụ '{name}:0.03' (xem docstring của công cụ).")

if options.raw.endswith(".pkl"):  # bản nạp sẵn (frames, meta) để chạy nhiều kịch bản nhanh
    with open(options.raw, "rb") as handle:
        frames, meta = pickle.load(handle)
else:
    frames, meta = load_raw_ocr(Path(options.raw))
rng = random.Random(options.seed)
names = {field.name for field in dataclasses.fields(SubtitleBuilderConfig)}
recorded = (getattr(meta, "run", None) or {}).get("builder")
cfg = (
    {key: value for key, value in recorded.items() if key in names}
    if recorded
    # [v3.23.720] Không ghi cấu hình lượt chạy → MẶC ĐỊNH CỦA ỨNG DỤNG (trước: mặc định DTO, lệch ứng dụng 5 trường).
    else dataclasses.asdict(app_default_builder_config(sample_step_sec=meta.sample_step_sec))
)
x, y, w, h = meta.roi_xywh
tshift = 0.0


def box(b, text=None, conf=None, poly=None):
    return OcrTextBox(
        text=b.text if text is None else text,
        confidence=Confidence(float(b.confidence) if conf is None else conf),
        polygon=b.polygon if poly is None else poly,
    )


def fr(f, boxes=None, ts=None, k=None):
    return OcrFrameResult(
        frame_index=f.frame_index,
        timestamp_sec=f.timestamp_sec if ts is None else ts,
        text_boxes=list(f.text_boxes) if boxes is None else boxes,
        continues_previous=f.continues_previous if k is None else k,
    )


def unk(fs):  # bỏ cờ cùng điểm ảnh ở khung sau khung bị đổi (khung đổi = lần đọc mới)
    return fs


# Một LẦN ĐỌC = khung được OCR + các khung sau nó mang cờ cùng điểm ảnh VÀ cùng nội dung (bản chép). Nhiễu giả định
# tác động lên lần đọc (mọi bản chép đổi theo), cờ cùng điểm ảnh giữ nguyên — đúng cách dữ liệu thật hình thành.
def sig(f):
    return tuple(
        (b.text, round(float(b.confidence), 4), tuple(map(tuple, b.polygon)))
        for b in f.text_boxes
    )


def reads(fs):
    out = []
    for f in fs:
        if out and f.continues_previous and sig(f) == sig(out[-1][0]):
            out[-1].append(f)
        else:
            out.append([f])
    return out


def per_read(fn, prob, only_text=True):
    out = []
    for run in reads(frames):
        if (run[0].text_boxes or not only_text) and rng.random() < prob:
            boxes = fn(run[0])
            out.extend(fr(f, list(boxes)) for f in run)
        else:
            out.extend(run)
    return out


if name == "none":
    pass
elif name == "scale":  # phim độ phân giải khác: mọi toạ độ × arg
    s = arg
    frames = [
        fr(
            f,
            [
                box(b, poly=[(px * s, py * s) for px, py in b.polygon])
                for b in f.text_boxes
            ],
        )
        for f in frames
    ]
    x, y, w, h = round(x * s), round(y * s), round(w * s), round(h * s)
    for k in ("y_clustering_tolerance_min_px", "alignment_tolerance_min_px"):
        cfg[k] = cfg[k] * s
elif name == "tshift":  # phim rất dài / mốc lớn: cộng arg giây
    tshift = arg
    frames = [fr(f, ts=f.timestamp_sec + arg) for f in frames]
elif name == "shuffle":
    frames = list(frames)
    rng.shuffle(frames)
elif name == "nok":  # dữ liệu không có cờ cùng điểm ảnh (bộ giải mã khác)
    frames = [fr(f, k=False) for f in frames]
elif name == "dropout":  # arg phần LẦN ĐỌC có chữ trả về rỗng (OCR hụt)
    frames = per_read(lambda f: [], arg)
elif (
    name == "framedrop"
):  # arg phần KHUNG có chữ rỗng, không còn cờ (bộ lọc khung coi là hình khác, OCR hụt)
    frames = [
        fr(f, [], k=False) if f.text_boxes and rng.random() < arg else f for f in frames
    ]
elif name == "framedrop2":  # như trên nhưng hụt 2 khung liền
    out = list(frames)
    i = 0
    while i < len(out) - 1:
        if out[i].text_boxes and rng.random() < arg:
            out[i] = fr(out[i], [], k=False)
            out[i + 1] = fr(out[i + 1], [], k=False)
            i += 6
        else:
            i += 1
    frames = out
elif name == "lowconf":  # arg phần lần đọc tụt tin cậy xuống 0,45–0,7
    frames = per_read(
        lambda f: [box(b, conf=rng.uniform(0.45, 0.7)) for b in f.text_boxes], arg
    )
elif name == "tall":  # arg phần lần đọc: mọi hộp cao hết vùng chữ

    def tb(b):
        xs = [p[0] for p in b.polygon]
        return box(b, poly=[(min(xs), 0), (max(xs), 0), (max(xs), h), (min(xs), h)])

    frames = per_read(lambda f: [tb(b) for b in f.text_boxes], arg)
elif name == "jitter":  # MỌI lần đọc: hộp rung ± arg px mỗi cạnh

    def j(b):
        xs = [p[0] for p in b.polygon]
        ys = [p[1] for p in b.polygon]
        x0, x1, y0, y1 = (
            min(xs) + rng.uniform(-arg, arg),
            max(xs) + rng.uniform(-arg, arg),
            min(ys) + rng.uniform(-arg, arg),
            max(ys) + rng.uniform(-arg, arg),
        )
        return box(b, poly=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    frames = per_read(lambda f: [j(b) for b in f.text_boxes], 1.1)
elif (
    name == "noise"
):  # arg phần lần đọc (kể cả lần đọc rỗng) có thêm 1–2 hộp nhiễu cảnh ở vị trí bất kỳ
    pool = [
        "1",
        "7",
        "N",
        "V",
        "II",
        "2>",
        "山",
        "公",
        "小",
        "上",
        "ABC",
        "川",
        "三",
        "人",
    ]

    def nb():
        bw, bh = rng.uniform(15, 140), rng.uniform(14, h)
        bx, by = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        return OcrTextBox(
            text=rng.choice(pool),
            confidence=Confidence(rng.uniform(0.4, 0.95)),
            polygon=[(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)],
        )

    frames = per_read(
        lambda f: list(f.text_boxes) + [nb() for _ in range(rng.randint(1, 2))],
        arg,
        only_text=False,
    )
elif name == "subst":  # arg phần lần đọc câu ≥ 3 chữ: một chữ bị đọc thành chữ khác
    pool = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会"

    def sb(b):
        t = b.text
        if len(t) < 3:
            return b
        i = rng.randrange(len(t))
        return box(
            b, text=t[:i] + rng.choice(pool) + t[i + 1 :], conf=rng.uniform(0.7, 0.95)
        )

    frames = per_read(lambda f: [sb(b) for b in f.text_boxes], arg)
elif (
    name == "edgecut"
):  # lần đọc ĐẦU / CUỐI của mỗi lượt chữ đọc thiếu chữ cuối (hiện dần / mờ dần), tin cậy thấp
    rs = reads(frames)
    out = []

    def txt(run):
        return "|".join(b.text for b in run[0].text_boxes)

    for i, run in enumerate(rs):
        f = run[0]
        ok = len(f.text_boxes) == 1 and len(f.text_boxes[0].text) >= 3
        edge = ok and (
            (i == 0 or txt(rs[i - 1]) != txt(run))
            or (i + 1 == len(rs) or txt(rs[i + 1]) != txt(run))
        )
        if edge and len(run) <= 3 and rng.random() < arg:
            b = f.text_boxes[0]
            nbx = [box(b, text=b.text[:-1], conf=rng.uniform(0.5, 0.8))]
            out.extend(fr(g, nbx) for g in run)
        else:
            out.extend(run)
    frames = out
elif name == "halfstep":  # lấy mẫu thưa: giữ 1 trên arg khung
    frames = [fr(f, k=False) for f in frames[:: int(arg)]]
elif name == "fps30":  # co thời gian như phim 30 fps (mọi mốc × 25/30)
    frames = [fr(f, ts=f.timestamp_sec * 25 / 30) for f in frames]
elif name == "dupts":  # arg phần khung bị lặp (cùng mốc, cùng nội dung)
    out = []
    for f in frames:
        out.append(f)
        if rng.random() < arg:
            out.append(f)
    frames = out
elif name == "split":  # arg phần lần đọc: hộp ≥ 4 chữ bị bộ khoanh chữ tách đôi (khe 0,2–0,6 chữ, như có dấu cách)
    def sp(f):
        out_boxes = []
        for b in f.text_boxes:
            text = b.text
            xs = [p[0] for p in b.polygon]
            ys = [p[1] for p in b.polygon]
            if len(text) < 4:
                out_boxes.append(b)
                continue
            cut = rng.randint(1, len(text) - 1)
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            pitch = (x1 - x0) / len(text)
            gap = pitch * rng.uniform(0.2, 0.6)
            mid = x0 + pitch * cut
            out_boxes.append(box(b, text=text[:cut], poly=[(x0, y0), (mid - gap / 2, y0), (mid - gap / 2, y1), (x0, y1)]))
            out_boxes.append(box(b, text=text[cut:], poly=[(mid + gap / 2, y0), (x1, y0), (x1, y1), (mid + gap / 2, y1)]))
        return out_boxes
    frames = per_read(sp, arg)
elif name == "punct":  # arg phần lần đọc: OCR thêm dấu câu (cuối hoặc đầu câu)
    marks = ["，", "。", "！", "？", "…", "、", "!", "?", ",", "."]
    def pu(f):
        if len(f.text_boxes) != 1:
            return f.text_boxes
        b = f.text_boxes[0]
        mark = rng.choice(marks)
        return [box(b, text=(b.text + mark) if rng.random() < 0.8 else ("—" + b.text))]
    frames = per_read(pu, arg)
elif name == "trad":  # arg phần lần đọc: một chữ giản thể bị đọc thành phồn thể
    import zhconv
    def tr(f):
        out_boxes = []
        for b in f.text_boxes:
            spots = [i for i, ch in enumerate(b.text) if zhconv.convert(ch, "zh-tw") != ch]
            if not spots:
                out_boxes.append(b)
                continue
            i = rng.choice(spots)
            out_boxes.append(box(b, text=b.text[:i] + zhconv.convert(b.text[i], "zh-tw") + b.text[i + 1 :]))
        return out_boxes
    frames = per_read(tr, arg)
elif name == "watermark":  # MỌI khung: logo chữ Latin cố định ở góc vùng chữ (kênh / tài khoản đăng phim)
    wm = OcrTextBox(text="@DuanJu88", confidence=Confidence(0.93), polygon=[(8, 6), (150, 6), (150, 30), (8, 30)])
    frames = [fr(f, list(f.text_boxes) + [wm]) for f in frames]
elif name == "clip":  # vùng chữ vẽ hụt: mọi hộp bị cắt chân arg px (mép dưới vùng chữ cắt vào chữ)
    def cl(b):
        xs = [p[0] for p in b.polygon]
        ys = [p[1] for p in b.polygon]
        bottom = min(max(ys), h - arg)
        top = min(min(ys), bottom - 4)
        return box(b, poly=[(min(xs), top), (max(xs), top), (max(xs), bottom), (min(xs), bottom)])
    frames = [fr(f, [cl(b) for b in f.text_boxes]) for f in frames]
elif name == "vfr":  # mốc khung rung ± arg ms (phim VFR / mốc PTS không đều), thứ tự giữ nguyên
    out, last = [], -1.0
    for f in frames:
        ts = max(last + 0.001, f.timestamp_sec + rng.uniform(-arg, arg) / 1000.0)
        out.append(fr(f, ts=round(ts, 4)))
        last = ts
    frames = out
elif name == "gap":  # arg phần khung biến mất hẳn (bộ giải mã bỏ khung, lượt chạy tiếp từ checkpoint)
    frames = [f for f in frames if rng.random() >= arg]
elif name in ("yshift", "xshift"):  # nửa sau phim (tập khác khi nối phim) phụ đề dời vị trí arg px
    half = frames[len(frames) // 2].timestamp_sec
    dx, dy = (arg, 0.0) if name == "xshift" else (0.0, -arg)
    def mv(b):
        return box(b, poly=[(px + dx, py + dy) for px, py in b.polygon])
    frames = [fr(f, [mv(b) for b in f.text_boxes]) if f.timestamp_sec >= half else f for f in frames]
else:
    raise SystemExit("Kịch bản lạ: " + name + " (xem --list)")
events = SubtitleBuilder(SubtitleBuilderConfig(**cfg)).build(
    frames, roi=Roi(x=x, y=y, width=w, height=h)
)


def stamp(seconds: float) -> str:
    if name == "fps30":
        seconds = seconds * 30 / 25
    elif tshift:
        seconds -= tshift
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


Path(options.output).write_text(
    "".join(
        f"{index}\n{stamp(event.start_sec)} --> {stamp(event.end_sec)}\n{event.text}\n\n"
        for index, event in enumerate(events, 1)
    ),
    encoding="utf-8",
)
print(f"{options.scenario} (seed {options.seed}): {len(events)} câu → {options.output}")
