"""[v3.23.701] Lỗi tìm ra khi tự quét mọi khung hình phim 86 tập làm đáp án và đối chiếu ảnh thật.

1. "叶然" đúng giữa bị loại vì hộp vật thể "二" (chồng dọc) bị gom chung dòng → dòng lệch tâm.
2. Câu cuối phim đặt cao hơn mọi câu ("我真的是冤枉的", tin cậy 1,0 ở 65 khung) bị bộ lọc Y toàn phim loại.
3. Bảng phồn → giản thiếu chữ (遙, 蕭…) mà model OCR server đọc ra ở phim giản thể.
"""

from __future__ import annotations

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.application.services.subtitle_pipeline.constants import HAN_TRADITIONAL_TO_SIMPLIFIED
from subtitles_extractor.application.services.subtitle_pipeline.han_t2s_table import OPENCC_TRADITIONAL_TO_SIMPLIFIED
from subtitles_extractor.application.services.subtitle_pipeline.spatial_filters import (
    clean_spatial_outliers,
    filter_cross_frame_spatial_outliers,
)
from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    normalize_traditional_to_simplified,
    unify_minority_traditional,
)
from subtitles_extractor.domain.entities.ocr_frame_result import OcrFrameResult, OcrTextBox
from subtitles_extractor.domain.value_objects.confidence import Confidence
from subtitles_extractor.domain.value_objects.roi import Roi, TextAlignment

ROI = Roi(x=34, y=773, width=645, height=148, alignment=TextAlignment.CENTER)   # ROI lượt chạy thật của chủ dự án


def _box(text: str, x0: int, y0: int, x1: int, y1: int, conf: float = 1.0) -> OcrTextBox:
    return OcrTextBox(text=text, confidence=Confidence(conf), polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


class TestMergedLineRescue:
    def test_centered_subtitle_survives_a_scene_box_on_the_same_line(self) -> None:
        """Toạ độ thật (khung 4585,8 s): "二" 124–169 × 22–95, "叶然" 235–355 × 54–130."""
        frame = OcrFrameResult(0, 4585.8, [_box("二", 124, 22, 169, 95, 0.96), _box("叶然", 235, 54, 355, 130)])
        kept = clean_spatial_outliers(frame, ROI, SubtitleBuilderConfig())
        assert [b.text for b in kept.text_boxes] == ["叶然"]

    def test_valid_line_is_untouched_and_fully_offcentre_line_still_dropped(self) -> None:
        ok = OcrFrameResult(0, 1.0, [_box("人呢", 234, 53, 351, 126)])
        assert [b.text for b in clean_spatial_outliers(ok, ROI, SubtitleBuilderConfig()).text_boxes] == ["人呢"]
        off = OcrFrameResult(0, 1.0, [_box("招", 2, 50, 40, 120, 0.9), _box("牌", 45, 52, 85, 121, 0.9)])
        assert clean_spatial_outliers(off, ROI, SubtitleBuilderConfig()).text_boxes == []


def _film(extra: list[OcrFrameResult]) -> list[OcrFrameResult]:
    base = [OcrFrameResult(i, i * 0.03, [_box("人呢" if i % 2 else "你要去哪", 220, 54, 420, 126)]) for i in range(10000)]   # đủ lớn để cụm 65 khung < 1 % (như phim thật)
    return base + extra


class TestOffbandLineRescue:
    def test_stable_alone_high_line_is_kept(self) -> None:
        """Câu cuối phim: hộp y −1→57 (chạm mép trên ROI), một mình trên màn hình, 65 khung."""
        tail = [OcrFrameResult(20000 + k, 400 + k * 0.03, [_box("我真的是冤枉的", 108, 0, 475, 57)]) for k in range(65)]
        kept = filter_cross_frame_spatial_outliers(_film(tail), roi=ROI)
        assert sum(1 for f in kept for b in f.text_boxes if b.text == "我真的是冤枉的") == 65

    def test_scene_text_next_to_a_subtitle_short_runs_and_single_chars_are_not_rescued(self) -> None:
        beside = [OcrFrameResult(20000 + k, 400 + k * 0.03, [_box("天霄峰", 200, 0, 420, 57), _box("人呢", 220, 54, 420, 126)])
                  for k in range(40)]
        short = [OcrFrameResult(21000 + k, 500 + k * 0.03, [_box("天霄峰", 200, 0, 420, 57)]) for k in range(5)]
        single = [OcrFrameResult(22000 + k, 600 + k * 0.03, [_box("终", 300, 0, 360, 57)]) for k in range(40)]
        kept = filter_cross_frame_spatial_outliers(_film(beside + short + single), roi=ROI)
        texts = [b.text for f in kept for b in f.text_boxes]
        assert "天霄峰" not in texts and "终" not in texts


class TestTraditionalTable:
    def test_server_model_traditional_reads_are_normalised(self) -> None:
        assert normalize_traditional_to_simplified("離金丹一步之遙 蕭炎竟是我自己 夠了") == "离金丹一步之遥 萧炎竟是我自己 够了"

    def test_ambiguous_or_simplified_valid_characters_are_excluded(self) -> None:
        for ch in "乾於徵覆藉瞭夥著干后":
            assert ch not in OPENCC_TRADITIONAL_TO_SIMPLIFIED
        assert not set(OPENCC_TRADITIONAL_TO_SIMPLIFIED) & set(OPENCC_TRADITIONAL_TO_SIMPLIFIED.values())
        assert len(HAN_TRADITIONAL_TO_SIMPLIFIED) > 2500

    def test_minority_rule_uses_the_bigger_table(self) -> None:
        texts = ["萧炎来了"] * 200 + ["蕭炎竟是我自己"]
        assert unify_minority_traditional(texts) == {200: "萧炎竟是我自己"}
