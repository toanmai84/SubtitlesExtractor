"""[v3.23.719] Bất biến của khâu dựng (``tools/stress_subtitle_builder.py``) vỡ trên OCR thật 豪门Z外 (1080 px, chữ 86 px).

1. ``dupts``: khung lặp (cùng mốc, cùng nội dung) được đếm hai lần → lệch phiếu bầu khi hai bản đọc gần hoà nhau
   ("混蛋龌龊的家伙" thành "混蛋 龌龊的家伙").
2. ``scale``: bộ lọc mật độ Y dùng bin 5 px / sàn 10 px cố định → cùng một phim ở độ phân giải khác giữ / bỏ khác nhau
   (×0,5 giữ rác Latin "TOUTS" sát mép trên vùng chữ).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.outlier_detection import filter_y_position_outliers
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import drop_duplicate_samples
from subtitles_extractor.application.services.subtitle_pipeline.spatial_filters import (
    filter_cross_frame_spatial_outliers,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi


def _box(text: str, x0: float, y0: float, x1: float, y1: float, conf: float = 0.99) -> OcrTextBox:
    return OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _frame(ts: float, boxes: list[OcrTextBox]) -> OcrFrameResult:
    return OcrFrameResult(frame_index=round(ts * 25), timestamp_sec=ts, text_boxes=boxes)


# ── 1. Khung lặp ────────────────────────────────────────────────────────────────────────────────────────────────


def test_duplicate_samples_are_dropped_but_conflicting_same_time_frames_are_kept() -> None:
    a = _frame(1.0, [_box("你好", 10, 10, 90, 60)])
    a_copy = _frame(1.0, [_box("你好", 10, 10, 90, 60)])
    b_other = _frame(1.0, [_box("你们", 10, 10, 90, 60)])
    c = _frame(1.04, [_box("你好", 10, 10, 90, 60)])
    kept, dropped = drop_duplicate_samples([c, a, a_copy, b_other, a_copy])
    assert dropped == 2
    assert [frame.timestamp_sec for frame in kept] == [1.0, 1.0, 1.04]
    assert [frame.text_boxes[0].text for frame in kept] == ["你好", "你们", "你好"]


def test_no_duplicates_returns_sorted_frames_untouched() -> None:
    frames = [_frame(round(0.04 * i, 2), [_box("好", 0, 0, 50, 50)]) for i in range(5, 0, -1)]
    kept, dropped = drop_duplicate_samples(frames)
    assert dropped == 0
    assert [frame.timestamp_sec for frame in kept] == sorted(frame.timestamp_sec for frame in frames)


_ROI = Roi(x=0, y=774, width=720, height=136)
_LINE_BOX = (222, 65, 498, 133)


def _tie_film() -> list[OcrFrameResult]:
    """Vài câu thường + một câu có hai bản đọc gần hoà phiếu (kèm / không kèm dấu cách), xen kẽ như dữ liệu thật."""
    frames: list[OcrFrameResult] = []
    clock = 0.0
    lines = ("修炼一途的", "自炼气始吧", "经筑基之后", "寻求飞升啊", "检测到宿主", "即将面临了")
    for line in lines:
        for _ in range(30):
            clock = round(clock + 0.04, 2)
            frames.append(_frame(clock, [_box(line, *_LINE_BOX, conf=1.0)]))
        for _ in range(8):
            clock = round(clock + 0.04, 2)
            frames.append(_frame(clock, []))
    for index in range(24):
        clock = round(clock + 0.04, 2)
        text = "混蛋 龌龊的家伙" if index % 2 else "混蛋龌龊的家伙"
        frames.append(_frame(clock, [_box(text, *_LINE_BOX, conf=0.85)]))
    for _ in range(10):
        clock = round(clock + 0.04, 2)
        frames.append(_frame(clock, []))
    return frames


def _events(frames: list[OcrFrameResult]) -> list[tuple[float, float, str]]:
    config = SubtitleBuilderConfig(sample_step_sec=0.04)
    return [
        (round(event.start_sec, 3), round(event.end_sec, 3), event.text)
        for event in SubtitleBuilder(config).build(frames, roi=_ROI)
    ]


def test_duplicated_frames_do_not_change_the_built_subtitles() -> None:
    frames = _tie_film()
    reference = _events(frames)
    rng = random.Random(7)
    for _ in range(5):
        duplicated: list[OcrFrameResult] = []
        for frame in frames:
            duplicated.append(frame)
            if rng.random() < 0.15:
                duplicated.append(frame)
        assert _events(duplicated) == reference
    # Lặp đúng các khung "kèm dấu cách" (kịch bản xấu nhất cho phiếu bầu): kết quả vẫn y như cũ.
    worst = [copy for frame in frames for copy in ([frame, frame] if " " in frame.joined_text else [frame])]
    assert _events(worst) == reference


#: 3 s OCR thật 豪门Z外 (2841–2844 s) sau kịch bản ``dupts:0.02`` (seed 1): có các khung bị lặp.
_DUPTS_FIXTURE = Path(__file__).resolve().parents[3] / "test data" / "haomen_dupts_2841s.json"


def _fixture_frames() -> list[OcrFrameResult]:
    rows = json.loads(_DUPTS_FIXTURE.read_text(encoding="utf-8"))
    return [
        OcrFrameResult(
            frame_index=round(ts * 24), timestamp_sec=ts, continues_previous=bool(same),
            text_boxes=[OcrTextBox(text=t, confidence=Confidence(c), polygon=[tuple(p) for p in poly]) for t, c, poly in boxes],
        )
        for ts, same, boxes in rows
    ]


def test_real_duplicated_frames_do_not_flip_the_reading() -> None:
    frames = _fixture_frames()
    assert len(frames) > len({frame.timestamp_sec for frame in frames})      # dữ liệu có khung lặp thật
    roi = Roi(x=0, y=1444, width=1080, height=130)
    config = SubtitleBuilderConfig(sample_step_sec=0.04, merge_gap_sec=0.6)

    def texts(sequence: list[OcrFrameResult]) -> list[str]:
        return [event.text for event in SubtitleBuilder(config).build(sequence, roi=roi)]

    unique: list[OcrFrameResult] = []
    for frame in frames:
        if not unique or frame.timestamp_sec != unique[-1].timestamp_sec:
            unique.append(frame)
    assert "混蛋龌龊的家伙" in texts(unique)
    assert texts(frames) == texts(unique)      # mã 3.23.718: "混蛋 龌龊的家伙"


# ── 2. Bộ lọc mật độ Y theo cỡ chữ ──────────────────────────────────────────────────────────────────────────────


def _layout(scale: float) -> list[OcrFrameResult]:
    """Phụ đề thật của 豪门Z外 (chữ cao 81–86 px, rộng ~71 px) + rác Latin "TOUTS" sát mép trên ở vài khung."""
    frames = []
    for index in range(400):
        ts = round(index * 0.04, 2)
        boxes = [_box("等阿姨做好了再叫你", 221 * scale, 5 * scale, 862 * scale, 86 * scale)]   # hộp thật 797,44 s
        if index % 40 == 0:
            boxes.append(_box('TOUTS"', 614 * scale, -8 * scale, 680 * scale, 53 * scale, conf=0.86))
        frames.append(_frame(ts, boxes))
    return frames


def _kept_texts(frames: list[OcrFrameResult], scale: float) -> list[tuple[str, ...]]:
    roi = Roi(x=0, y=round(1444 * scale), width=round(1080 * scale), height=round(130 * scale))
    return [tuple(box.text for box in frame.text_boxes) for frame in filter_cross_frame_spatial_outliers(frames, roi)]


def test_density_filter_gives_the_same_result_at_any_resolution() -> None:
    reference = _kept_texts(_layout(1.0), 1.0)
    assert all('TOUTS"' not in texts for texts in reference)      # rác bị bỏ ở độ phân giải gốc
    for scale in (0.5, 0.75, 1.5, 2.0):
        assert _kept_texts(_layout(scale), scale) == reference, scale


def test_pixel_scale_one_keeps_the_legacy_behaviour() -> None:
    centers = [50.0] * 200 + [30.0] * 3
    assert filter_y_position_outliers(centers) == filter_y_position_outliers(centers, pixel_scale=1.0)
    # Cùng dữ liệu thu nhỏ một nửa: bin 2,5 px → mặt nạ y như bản gốc.
    assert filter_y_position_outliers([c / 2 for c in centers], pixel_scale=0.5) == filter_y_position_outliers(centers)


def test_clipped_glyph_bottoms_do_not_narrow_the_density_band() -> None:
    """Vùng chữ cắt chân 25 px (thử sức ``clip:25`` trên 86 tập): chiều cao hộp co 67 → 46 px nhưng bề rộng một chữ
    giữ 58 px. Thước đo theo chiều cao làm dải giữ hẹp lại và bỏ các lần đọc "哼" có hộp cao hơn (39–111) → câu "哼"
    3825,8 s kết thúc sớm 0,28 s. Thước đo theo bề rộng một chữ giữ đúng."""
    frames = []
    for index in range(1200):
        ts = round(index * 0.04, 2)
        if 600 <= index < 620:
            boxes = [_box("哼", 320, 39, 403, 111, conf=0.74)]          # một câu "哼" thật (20 khung), hộp cao hơn
        elif index % 50 < 40:
            boxes = [_box("修炼一途的经筑基之后", 70, 65, 70 + 10 * 58, 111)]       # chân chữ bị cắt: cao 46 px
        else:
            boxes = []
        frames.append(_frame(ts, boxes))
    roi = Roi(x=0, y=774, width=720, height=111)
    kept = {text for frame in filter_cross_frame_spatial_outliers(frames, roi) for text in (b.text for b in frame.text_boxes)}
    assert "哼" in kept
