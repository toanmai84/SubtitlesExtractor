"""[v3.23.713] Vòng tự soát phân xử câu có hai bản đọc mâu thuẫn bằng nhiều lần đọc lại trên ảnh gốc.

Đo trên 915 khung thật của phim mẫu (model small): không tiền xử lý đọc đúng 909, cấu hình viền + làm nét + tương
phản 1,2 đọc đúng 896 và làm rơi chữ "一" đầu câu ở 7/35 câu (không tiền xử lý: 1/35). Sáu câu "đọc lẫn chữ" của
các lượt chạy thật đều ra đúng khi đọc lại 12–24 khung bằng ảnh gốc rồi bỏ phiếu.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.ocr_suspicion import SUSPECT_READS_DISAGREE, ocr_suspicions
from subtitles_extractor.application.services.subtitle_pipeline.geometry_check import (
    OCR_NOTE_DISPUTED,
    accept_disputed_reread,
)


class TestAcceptDisputedReread:
    def test_real_votes_pick_the_right_reading(self) -> None:
        # số phiếu đo thật trên khung hình (ảnh gốc, model small)
        assert accept_disputed_reread("做这等腌攒之事", ["做这等腌臜之事"] * 12 + ["做这等腌攒之事"] * 5) == "做这等腌臜之事"
        assert accept_disputed_reread("这不是痴蛤蟆吗", ["这不是癞蛤蟆吗"] * 15 + ["这不是痴蛤蟆吗"]) == "这不是癞蛤蟆吗"
        assert accept_disputed_reread("我跟秦执师兄清清自白的", ["我跟秦执师兄清清白白的"] * 20 + ["我跟秦执师兄清清自白的"]) \
            == "我跟秦执师兄清清白白的"

    def test_current_text_can_be_confirmed(self) -> None:
        assert accept_disputed_reread("把你狠狠踩在脚下", ["把你狠狠踩在脚下"] * 17 + ["把你狠狼踩在脚下"] * 7) == "把你狠狠踩在脚下"

    def test_side_garbage_around_the_line_is_ignored(self) -> None:
        rereads = ["27下山探风吗", "下山探风吗CHN", "下山探风吗", "下山探风鸣"]
        assert accept_disputed_reread("下山探风鸣", rereads) == "下山探风吗"

    @pytest.mark.parametrize("rereads", [
        [],                                                        # không đọc lại được
        ["做这等腌臜之事"] * 2,                                      # quá ít phiếu
        ["做这等腌臜之事"] * 4 + ["做这等腌攒之事"] * 3,              # hơn chưa đủ 2 phiếu
        ["你说什么呢"] * 9,                                          # câu khác hẳn
        ["做这等腌臜之事"] * 3 + ["做这等腌攒之事"] * 2 + ["做这等腌赞之事"] * 2,   # chưa quá bán
    ])
    def test_weak_evidence_keeps_the_flag(self, rereads: list[str]) -> None:
        assert accept_disputed_reread("做这等腌攒之事", rereads) is None

    def test_whitespace_and_empty_current(self) -> None:
        assert accept_disputed_reread("  ", ["一"] * 5) is None
        assert accept_disputed_reread("下山 探风鸣", ["下山探风吗"] * 5) == "下山探风吗"


def test_disputed_note_reaches_the_editor_filter() -> None:
    events = [SimpleNamespace(text="做这等腌攒之事", ocr_notes=(OCR_NOTE_DISPUTED,), frame_count=30, confidence=0.96,
                              start_sec=0.0, end_sec=1.4)]
    assert ocr_suspicions(events) == {0: [SUSPECT_READS_DISAGREE]}


def test_ui_strings_exist_for_the_new_reason() -> None:
    import json
    from pathlib import Path

    import subtitles_extractor

    data = Path(subtitles_extractor.__file__).parent / "data"
    for name in ("strings_vi.json", "strings_en.json"):
        text = (data / name).read_text(encoding="utf-8")
        assert '"ed_suspect_reads_disagree"' in text and json.loads(text)


def test_builder_flags_disputed_line_separately_from_restored() -> None:
    """Câu gộp từ hai bản đọc lẫn chữ mang cờ "hai bản đọc mâu thuẫn" (để vòng tự soát đọc lại), không phải "chữ khôi phục"."""
    import dataclasses

    from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
    from subtitles_extractor.application.services.subtitle_pipeline.orchestrator import SubtitleBuilder
    from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
    from subtitles_extractor.domain.value_objects.confidence import Confidence
    from subtitles_extractor.domain.value_objects.roi import Roi

    def frame(ts: float, text: str = "", cont: bool = False) -> OcrFrameResult:
        boxes = [OcrTextBox(text=text, confidence=Confidence(0.97), polygon=[(128, 70), (593, 70), (593, 129), (128, 129)])] if text else []
        return OcrFrameResult(int(ts * 25), ts, boxes, continues_previous=cont)

    frames = [frame(0.04 * i) for i in range(5)]
    for text, count in (("把你狠狠踩在脚下", 12), ("把你狠狼踩在脚下", 10), ("把你狠狠踩在脚下", 11), ("把你狠狼踩在脚下", 6)):
        for _ in range(count):
            frames.append(frame(0.04 * len(frames), text, cont=len(frames) > 5))
    frames += [frame(0.04 * (len(frames) + i)) for i in range(5)]
    config = dataclasses.replace(SubtitleBuilderConfig(), sample_step_sec=0.04)
    events = SubtitleBuilder(config).build(frames, roi=Roi(x=0, y=774, width=720, height=136))
    assert [event.text for event in events] == ["把你狠狠踩在脚下"]
    assert OCR_NOTE_DISPUTED in events[0].ocr_notes and "restored_char" not in events[0].ocr_notes


def test_one_way_same_pixel_disagreement_is_flagged_but_not_fused() -> None:
    """Tập 47 (lượt thật): "腌攒" đọc ba lần rồi "腌臜" một lần trên CÙNG điểm ảnh — không xen kẽ nên không gộp,
    nhưng trước bản này câu sai cũng không được gắn cờ → vòng soát không bao giờ đọc lại."""
    from subtitles_extractor.application.services.subtitle_pipeline.read_fusion import fuse_same_pixel_reads
    from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
    from subtitles_extractor.domain.value_objects.confidence import Confidence

    def frame(index: int, text: str, conf: float) -> OcrFrameResult:
        box = OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(159, 68), (562, 68), (562, 132), (159, 132)])
        return OcrFrameResult(frame_index=index, timestamp_sec=index * 0.04, text_boxes=[box],
                              continues_previous=index > 0)

    frames = [frame(i, "做这等腌攒之事", 0.96) for i in range(34)] + [frame(i, "做这等腌臜之事", 0.90) for i in (34, 35)]
    fused, fixed, conflicts = fuse_same_pixel_reads(frames, center_x=360.0, char_pitch=55.0, text_height=64.0)
    assert fixed == 0
    assert ["".join(b.text for b in f.text_boxes) for f in fused] == ["做这等腌攒之事"] * 34 + ["做这等腌臜之事"] * 2
    assert conflicts == {"做这等腌攒之事", "做这等腌臜之事"}
