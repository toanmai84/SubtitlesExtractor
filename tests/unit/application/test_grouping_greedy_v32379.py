"""[v3.23.79] Test lõi gộp greedy ``grouping.group_using_greedy`` + khởi tạo group.

Greedy grouping là thuật toán gộp frame OCR thành câu phụ đề (mặc định
``use_viterbi=False``, đạt F1 tương đương Viterbi nhưng nhanh gấp 5-10 lần). Phủ các nhánh
quyết định XÁC ĐỊNH:
- Đầu vào rỗng -> không có group.
- Text GIỐNG nhau, gần thời gian (gap nhỏ hơn merge_gap) -> GỘP một group.
- Cách nhau quá ``merge_gap_sec`` -> TÁCH group mới.
"""

from __future__ import annotations

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_pipeline.grouping import (
    create_initial_frame_group,
    group_using_greedy,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence


def _frame(frame_index: int, ts: float, text: str, conf: float = 0.92) -> OcrFrameResult:
    box = OcrTextBox(
        text=text,
        confidence=Confidence(conf),
        polygon=[(0, 0), (200, 0), (200, 24), (0, 24)],
    )
    return OcrFrameResult(frame_index=frame_index, timestamp_sec=ts, text_boxes=[box])


_CONFIG = SubtitleBuilderConfig()


def test_empty_frames_yields_no_groups() -> None:
    assert group_using_greedy([], _CONFIG) == []


def test_create_initial_frame_group_single_frame() -> None:
    group = create_initial_frame_group(_frame(0, 1.5, "Xin chào"), _CONFIG)
    assert group.reconstructed_text == "Xin chào"
    assert group.start_timestamp_sec == 1.5
    assert group.end_timestamp_sec == 1.5
    assert group.total_frames_count == 1


def test_identical_text_close_in_time_merges_into_one_group() -> None:
    # gap 0.10s <= merge_gap (0.60) và < ngưỡng lặp (0.35) -> gộp.
    frames = [_frame(0, 0.0, "Xin chào"), _frame(1, 0.10, "Xin chào")]
    groups = group_using_greedy(frames, _CONFIG)
    assert len(groups) == 1
    assert groups[0].total_frames_count == 2


def test_large_time_gap_splits_into_two_groups() -> None:
    # gap 1.0s > merge_gap (0.60) -> mở group mới dù text giống.
    frames = [_frame(0, 0.0, "Xin chào"), _frame(1, 1.0, "Xin chào")]
    groups = group_using_greedy(frames, _CONFIG)
    assert len(groups) == 2


def test_three_frames_one_gap_produces_two_groups() -> None:
    frames = [
        _frame(0, 0.0, "Câu một"),
        _frame(1, 0.05, "Câu một"),  # gộp với frame 0
        _frame(2, 2.0, "Câu hai"),   # gap lớn -> group mới
    ]
    groups = group_using_greedy(frames, _CONFIG)
    assert len(groups) == 2
    assert groups[0].total_frames_count == 2
