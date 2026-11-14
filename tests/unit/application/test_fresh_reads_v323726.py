"""[v3.23.726] Bỏ phiếu theo LẦN ĐỌC OCR thật, không theo số khung (khung chép kết quả không phải bằng chứng mới).

Lượt chạy thật 3.23.725 (phim nối 86 tập, NVDEC, chỉ phóng to): 3657/3659 câu đáp án đúng, sai hai câu:
* "做这等腌臜之事": trên cùng điểm ảnh đọc 5 lần — 臜 0,931 / 攒 0,968 / 臜 0,963 / 攒 0,961 / 臜 0,914 (臜 3, 攒 2), nhưng
  hai lần "攒" rơi đầu hai đợt chép 11 khung → 21 khung 攒 / 15 khung 臜 → "腌攒".
* "哼": 哼 0,569 (2 khung) / 亨 0,508 (11 khung) / 哼 0,634 (2 khung) — cả ba dưới 0,80 nên bước gộp chuỗi bỏ qua → "亨".
Phát lại đúng seraw đó: hai câu về đáp án, 3661 câu còn lại trùng khít.
"""

from __future__ import annotations

from subtitles_extractor.application.services.subtitle_pipeline.read_fusion import (
    fresh_read_flags,
    fuse_same_pixel_reads,
)
from subtitles_extractor.application.services.subtitle_pipeline.voting import vote_best_text_rover
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.value_objects.confidence import Confidence

CENTER, PITCH, HEIGHT = 360.0, 55.3, 67.0


def _line(text: str, conf: float, shift: int = 0) -> list[OcrTextBox]:
    width = PITCH * (len(text) + 0.35)
    left, right = int(CENTER - width / 2) + shift, int(CENTER + width / 2) + shift
    return [OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(left, 66), (right, 66), (right, 133), (left, 133)])]


def _chain(reads: list[tuple[str, float, int]], start: float = 0.0) -> list[OcrFrameResult]:
    """Mỗi phần tử = một LẦN ĐỌC (chữ, tin cậy, số khung chép); mọi khung sau khung đầu cùng điểm ảnh."""
    frames: list[OcrFrameResult] = []
    for number, (text, conf, count) in enumerate(reads):
        boxes = _line(text, conf, shift=number % 2)       # lần đọc lại: hộp hơi khác (như dữ liệu thật)
        for _ in range(count):
            frames.append(OcrFrameResult(len(frames), round(start + 0.04 * len(frames), 3), list(boxes),
                                         continues_previous=bool(frames)))
    return frames


def _texts(frames: list[OcrFrameResult]) -> list[str]:
    return ["".join(box.text for box in frame.text_boxes) for frame in frames]


#: Dữ liệu thật 4996,92–4998,32 s (lượt 3.23.725).
_YAN = [("做这等腌臜之事", 0.93085, 2), ("做这等腌攒之事", 0.967805, 10), ("做这等腌臜之事", 0.963256, 11),
        ("做这等腌攒之事", 0.961124, 11), ("做这等腌臜之事", 0.913749, 2)]
#: Dữ liệu thật 1756,60–1757,16 s.
_HENG = [("哼", 0.569435, 2), ("亨", 0.50783, 11), ("哼", 0.633771, 2)]


def test_fresh_read_flags_mark_copies_only() -> None:
    frames = _chain(_HENG)
    assert fresh_read_flags(frames) == [True, False, True] + [False] * 10 + [True, False]


def test_legacy_raw_without_continuation_flag_counts_every_frame() -> None:
    frames = [OcrFrameResult(i, 0.04 * i, _line("哼", 0.6)) for i in range(5)]   # seraw < 3.23.711
    assert fresh_read_flags(frames) == [True] * 5


def test_identical_boxes_without_continuation_are_separate_reads() -> None:
    frames = _chain(_HENG)
    frames[1] = OcrFrameResult(1, 0.04, list(frames[0].text_boxes), continues_previous=False)
    assert fresh_read_flags(frames)[1] is True


def test_substitution_is_decided_by_reads_not_by_copied_frames() -> None:
    fused, _fixed, conflicts = fuse_same_pixel_reads(_chain(_YAN), CENTER, PITCH, HEIGHT)
    assert set(_texts(fused)) == {"做这等腌臜之事"}           # mã cũ: "腌攒" (21 khung so với 15)
    assert "做这等腌臜之事" in conflicts                       # vẫn gắn cờ nghi ngờ


def test_weak_alternating_reads_on_same_pixels_are_one_line() -> None:
    fused, fixed, conflicts = fuse_same_pixel_reads(_chain(_HENG), CENTER, PITCH, HEIGHT)
    assert set(_texts(fused)) == {"哼"} and fixed == 11       # mã cũ: để nguyên → câu "亨"
    assert "哼" in conflicts


def test_weak_one_directional_change_is_left_alone() -> None:
    frames = _chain([("哼", 0.57, 2), ("亨", 0.51, 11)])       # A → B một chiều: có thể là hai câu thật
    fused, fixed, _ = fuse_same_pixel_reads(frames, CENTER, PITCH, HEIGHT)
    assert fixed == 0 and _texts(fused) == _texts(frames)


def test_weak_reads_need_subtitle_geometry_and_minimum_confidence() -> None:
    off_center = _chain(_HENG)
    off_center = [OcrFrameResult(f.frame_index, f.timestamp_sec,
                                 [OcrTextBox(b.text, b.confidence, [(x + 200, y) for x, y in b.polygon]) for b in f.text_boxes],
                                 continues_previous=f.continues_previous) for f in off_center]
    assert fuse_same_pixel_reads(off_center, CENTER, PITCH, HEIGHT)[1] == 0     # vật trong cảnh lệch tâm
    garbage = _chain([("哼", 0.35, 2), ("亨", 0.30, 5), ("哼", 0.36, 2)])
    assert fuse_same_pixel_reads(garbage, CENTER, PITCH, HEIGHT)[1] == 0        # dưới 0,40: rác
    assert fuse_same_pixel_reads(_chain(_HENG), None, None, None)[1] == 0        # không biết hình học phim


def test_weak_reads_do_not_spread_into_empty_frames() -> None:
    frames = _chain(_HENG)
    frames.insert(5, OcrFrameResult(99, 0.2, [], continues_previous=True))
    fused, _, _ = fuse_same_pixel_reads(frames, CENTER, PITCH, HEIGHT)
    assert _texts(fused)[5] == ""


def test_rover_votes_on_reads_not_copies() -> None:
    assert vote_best_text_rover(_chain(_HENG), 0.3, 15.0) == "哼"           # mã cũ: "亨" (11 khung)
    legacy = [OcrFrameResult(i, 0.04 * i, f.text_boxes) for i, f in enumerate(_chain(_HENG))]
    assert vote_best_text_rover(legacy, 0.3, 15.0) == "亨"                   # dữ liệu cũ: giữ cách đếm cũ
