"""[v3.23.736] Phim "合欢宗从照顾师弟道侣开始修仙" (80 tập nối, 2 giờ 49 phút, PP-OCRv6 small, bản 3.23.735).

Lượt chạy thật trên máy chủ dự án (seraw ``test data/hh_v3_23_735``), kiểm từng điểm bằng ảnh video vùng ROI:

* lượt soát câu bằng PaddleOCR-VL đổi SAI 2/3 câu nó đổi: "谨遵师兄吩咐" → "分咐" (VL 3/3 đọc mất bộ 口; "吩咐" có ở câu
  khác trong phim, "分咐" thì không) và "接下来" → "接下來" (VL ra chữ PHỒN THỂ trong phim giản thể);
* "阵" (37:24, dòng dưới của phụ đề hai dòng) đọc xen kẽ "阵" | "降" | "阵" (tin cậy ≈ 0,6) → khâu dựng chỉ giữ mảnh cuối
  0,72 s, mất 0,88 s đầu.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import fields
from pathlib import Path

import pytest

from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
    accept_vl_reread,
    arbitrate_disputed_verdict,
)
from subtitles_extractor.application.services.subtitle_pipeline.read_fusion import (
    relabel_single_char_sandwiches,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence

DATA = Path(__file__).resolve().parents[3] / "test data" / "hh_v3_23_735"
CORPUS = ["你要是没什么吩咐的话", "接下来的日子", "我们走吧", "谨遵师兄吩咐"]


# ── lượt soát câu bằng VL ────────────────────────────────────────────────────────────────────────────────────────────
def test_vl_traditional_reading_is_not_a_correction() -> None:
    assert accept_vl_reread("接下来", ["接下来", "接下來", "接下來"], CORPUS) is None


def test_vl_change_against_film_corpus_is_rejected() -> None:
    assert accept_vl_reread("谨遵师兄吩咐", ["谨遵师兄分咐"] * 3, CORPUS) is None
    assert arbitrate_disputed_verdict("谨遵师兄吩咐", "谨遵师兄分咐", CORPUS) == "谨遵师兄吩咐"


def test_vl_real_corrections_still_accepted() -> None:
    # Ngữ liệu không nghiêng về bản đang có → vẫn nhận bản VL (các ca đúng đã đo ở 3.23.733).
    assert accept_vl_reread("霍", ["嚯", "嚯", "嚯"], CORPUS) == "嚯"
    assert accept_vl_reread("你和他的那些腌攒事", ["你和他的那些腌臜事"] * 3, ["我的事再腌臜"]) == "你和他的那些腌臜事"
    assert arbitrate_disputed_verdict("有志者事竞成", "有志者事竟成", CORPUS) == "有志者事竟成"
    # Không truyền ngữ liệu: như trước.
    assert accept_vl_reread("谨遵师兄吩咐", ["谨遵师兄分咐"] * 3) == "谨遵师兄分咐"


# ── chữ đơn đọc xen kẽ A|B|A ─────────────────────────────────────────────────────────────────────────────────────────
def _frame(ts: float, text: str | None, x0: int = 320, conf: float = 0.62) -> OcrFrameResult:
    boxes = [] if text is None else [OcrTextBox(text=text, confidence=Confidence(conf),
                                                 polygon=[(x0, 0), (x0 + 80, 0), (x0 + 80, 100), (x0, 100)])]
    return OcrFrameResult(round(ts / 0.04), ts, boxes)


def _texts(frames: list[OcrFrameResult]) -> list[str | None]:
    return [f.text_boxes[0].text if f.text_boxes else None for f in frames]


def test_single_char_sandwich_relabelled() -> None:
    seq = ["阵"] * 12 + ["降"] * 10 + ["阵"] * 18
    frames = [_frame(2244.02 + i * 0.04, t) for i, t in enumerate(seq)]
    out, changed = relabel_single_char_sandwiches(frames)
    assert changed == 10 and set(_texts(out)) == {"阵"}


@pytest.mark.parametrize("seq", [
    ["好"] * 10 + ["嗯"] * 10 + [None] * 3 + ["好"] * 10,      # có khoảng trống → câu thật khác nhau
    ["好"] * 10 + ["嗯"] * 20 + ["好"] * 10,                   # đoạn giữa dài 0,8 s → câu thật
    ["好"] * 10 + ["嗯"] * 10 + ["行"] * 10,                   # hai đầu khác chữ
])
def test_real_single_char_lines_untouched(seq: list[str | None]) -> None:
    frames = [_frame(100 + i * 0.04, t) for i, t in enumerate(seq)]
    out, changed = relabel_single_char_sandwiches(frames)
    assert changed == 0 and _texts(out) == seq


def test_sandwich_needs_same_position() -> None:
    seq = [("阵", 320)] * 10 + [("降", 520)] * 10 + [("阵", 320)] * 10
    frames = [_frame(100 + i * 0.04, t, x0=x) for i, (t, x) in enumerate(seq)]
    assert relabel_single_char_sandwiches(frames)[1] == 0


# ── phát lại lượt chạy thật ──────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def replayed():  # noqa: ANN201
    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        SubtitleBuilderConfig,
    )
    from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import (
        SubtitleBuilder,
    )
    from subtitles_extractor.domain.value_objects.roi import Roi
    from subtitles_extractor.infrastructure.serializers.raw_ocr_serializer import (
        load_raw_ocr,
    )

    raw = DATA / "hh.seraw.json.gz"
    if not raw.is_file():
        pytest.skip("thiếu dữ liệu test data/hh_v3_23_735")
    meta_json = json.load(gzip.open(raw))["meta"]
    names = {f.name for f in fields(SubtitleBuilderConfig)}
    cfg = SubtitleBuilderConfig(**{k: v for k, v in meta_json["run"]["builder"].items() if k in names})
    frames, meta = load_raw_ocr(raw)
    return SubtitleBuilder(cfg).build(frames, roi=Roi(*meta.roi_xywh))


def test_real_run_keeps_whole_zhen(replayed) -> None:  # noqa: ANN001
    zhen = [e for e in replayed if e.text == "阵" and 2243 < e.start_sec < 2246]
    assert len(zhen) == 1 and zhen[0].start_sec == pytest.approx(2244.02, abs=0.05)
    assert zhen[0].end_sec == pytest.approx(2245.62, abs=0.05)


def test_real_run_count(replayed) -> None:  # noqa: ANN001
    assert len(replayed) == 4278


def test_roi_clips_present() -> None:
    clips = sorted((DATA / "clips").glob("*.roi.mp4"))
    if not clips:
        pytest.skip("thiếu dữ liệu test data/hh_v3_23_735")
    assert len(clips) == 5 and all(clip.with_suffix(".json").is_file() for clip in clips)
