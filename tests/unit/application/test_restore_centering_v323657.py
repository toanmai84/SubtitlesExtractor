"""[v3.23.657] B3: khôi phục ký tự OCR làm rơi phải khớp vị trí CĂN GIỮA của phụ đề.

Ca thật (tập 8): 18/20 khung đọc ``小小教训你们一下吧`` (hộp chữ tâm 301 — trùng tâm ROI 301,5), 2 khung đọc
``一小小教训你们一下吧`` (hộp lệch tâm 18 px, chỉ rộng thêm 0,58 chữ: một vệt nhiễu bên trái). Bộ khôi phục "一" (ngưỡng
thấp vì "一" thật hay bị đọc rơi) vẫn chèn → câu sai. Phụ đề căn giữa: chữ thật làm hộp CÂN quanh tâm, nhiễu thì không.
"""

from __future__ import annotations

from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    centering_reference,
    restore_dropped_yi_prefix,
)

WITHOUT = "小小教训你们一下吧"
WITH = "一" + WITHOUT
CANDS = [(WITHOUT, 0.99)] * 18 + [(WITH, 0.98)] * 2


def test_off_centre_restoration_is_rejected() -> None:
    geometry = {WITHOUT: (301.0, 515.0), WITH: (283.5, 549.0)}       # số đo thật tập 8
    with centering_reference(301.5):
        assert restore_dropped_yi_prefix(WITHOUT, CANDS, geometry=geometry) == WITHOUT


def test_same_box_restoration_is_kept() -> None:
    # OCR đọc rơi "一" nhưng hộp vẫn bao đủ chữ: hai cách đọc cùng hộp → vẫn khôi phục.
    geometry = {WITHOUT: (301.0, 570.0), WITH: (301.0, 570.0)}
    with centering_reference(301.5):
        assert restore_dropped_yi_prefix(WITHOUT, CANDS, geometry=geometry) == WITH


def test_detector_missed_first_char_is_kept() -> None:
    # Bộ dò bỏ sót chữ đầu: hộp không có chữ lệch sang phải nửa chữ, hộp có chữ thì căn giữa → khôi phục.
    geometry = {WITHOUT: (330.0, 515.0), WITH: (301.5, 572.0)}
    with centering_reference(301.5):
        assert restore_dropped_yi_prefix(WITHOUT, CANDS, geometry=geometry) == WITH


def test_without_reference_behaviour_unchanged() -> None:
    geometry = {WITHOUT: (301.0, 515.0), WITH: (283.5, 549.0)}
    assert restore_dropped_yi_prefix(WITHOUT, CANDS, geometry=geometry) == WITH  # không biết tâm (ROI lệch/không ROI)
    assert restore_dropped_yi_prefix(WITHOUT, CANDS) == WITH


def _frames(roi_w: int = 603):
    from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
    from subtitles_extractor.domain.value_objects.confidence import Confidence

    frames = []
    for i in range(20):
        noisy = i in (9, 11)                                   # 2/20 khung đọc lẫn vệt nhiễu bên trái thành "一"
        text, left = (WITH, 9) if noisy else (WITHOUT, 43)
        box = OcrTextBox(text=text, confidence=Confidence(0.98 if noisy else 0.99),
                         polygon=[(left, 30), (558, 30), (558, 90), (left, 90)])
        frames.append(OcrFrameResult(frame_index=i, timestamp_sec=20.0 + 0.04 * i, text_boxes=[box]))
    return frames


def test_builder_rejects_off_centre_restoration_on_centred_roi() -> None:
    from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
    from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
    from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment

    centred = Roi(x=57, y=819, width=603, height=119, alignment=TextAlignment.CENTER)
    events = SubtitleBuilder(SubtitleBuilderConfig()).build(_frames(), roi=centred)
    assert [e.text for e in events] == [WITHOUT]
    left = Roi(x=57, y=819, width=603, height=119, alignment=TextAlignment.LEFT)
    assert [e.text for e in SubtitleBuilder(SubtitleBuilderConfig()).build(_frames(), roi=left)] == [WITH]   # không căn giữa: như cũ


def test_vertical_roi_is_not_checked_for_centring() -> None:
    """Rà soát độc lập: phụ đề DỌC thì bề rộng hộp ≈ một chữ — luật căn giữa ngang không áp dụng được."""
    from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
    from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
    from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment, TextOrientation

    vertical = Roi(x=57, y=819, width=603, height=119, alignment=TextAlignment.CENTER,
                   orientation=TextOrientation.VERTICAL)
    left = Roi(x=57, y=819, width=603, height=119, alignment=TextAlignment.LEFT)
    expected = [e.text for e in SubtitleBuilder(SubtitleBuilderConfig()).build(_frames(), roi=left)]
    assert [e.text for e in SubtitleBuilder(SubtitleBuilderConfig()).build(_frames(), roi=vertical)] == expected
