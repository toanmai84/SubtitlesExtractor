"""[v3.23.616] Viết hoa chữ đầu dòng mở câu mới sau dòng đã kết câu.

Chuẩn đoán 14/10 21:43 (phim "天骄女友退婚" tập 1, bản dịch cuối): dòng 59 "Từ hôn?!" rồi
dòng 60 "sắp chết bỗng giật mình ngồi dậy,". Trên mọi file chuẩn đoán dịch thật (65 phim,
gần 180 nghìn dòng) lưới sửa 198 dòng; soát 40 dòng chọn ngẫu nhiên đều đúng chính tả.
"""

from __future__ import annotations

from subtitles_extractor.application.services.sentence_case_guard import (
    capitalize_sentence_starts,
    supports_sentence_case,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.presentation.view_models.translate_page_view_model import (
    _capitalize_retranslated,
)
from tests.integration.fake_translator import FakeSubtitleTranslator


class TestGuard:
    def test_real_line_after_exclamation(self) -> None:
        texts = [
            "Từ hôn?!",
            "sắp chết bỗng giật mình ngồi dậy,",
            "Tiêu Viêm lại chính là ta?!",
        ]
        assert capitalize_sentence_starts(texts) == {
            1: "Sắp chết bỗng giật mình ngồi dậy,"
        }

    def test_real_corpus_cases(self) -> None:
        texts = ["Ấy da.", "sống còn chẳng bằng con chó nhà tao.", "Xưởng trưởng!",
                 "chú thấy đỡ hơn chút nào chưa?", "đường dài lắm chông gai,"]
        assert capitalize_sentence_starts(texts) == {
            1: "Sống còn chẳng bằng con chó nhà tao.",
            3: "Chú thấy đỡ hơn chút nào chưa?",
            4: "Đường dài lắm chông gai,",
        }

    def test_continuations_are_untouched(self) -> None:
        texts = [
            "Ta ra khỏi sơn môn,", "chẳng phải bị",  # câu còn tiếp
            "thì hôn sự hai ta...", "sư huynh nói gì?",  # câu bỏ lửng
            "Khoan đã…", "hôm nay tiểu sư muội",
        ]
        assert capitalize_sentence_starts(texts) == {}

    def test_lines_not_starting_with_a_letter_are_untouched(self) -> None:
        texts = ["Bao nhiêu tiền?", "20 vạn!", "Hả?", "[thở dài] haiz.", "Ừ.",
                 "...phối hợp với người đó.", "Đi.", "- đi đâu?"]
        assert capitalize_sentence_starts(texts) == {}

    def test_brand_names_written_in_camel_case_are_kept(self) -> None:
        assert capitalize_sentence_starts(["Mua gì?", "iPhone mới nhất."]) == {}

    def test_abbreviation_is_not_a_sentence_end(self) -> None:
        assert capitalize_sentence_starts(["Gạo, muối, v.v.", "đủ cả rồi"]) == {}
        assert capitalize_sentence_starts(["Tới TP.", "hồ chí minh"]) == {}

    def test_empty_line_does_not_hide_the_previous_sentence(self) -> None:
        assert capitalize_sentence_starts(["Đi thôi.", "", "nhanh lên!"]) == {
            2: "Nhanh lên!"
        }

    def test_leading_space_is_preserved(self) -> None:
        assert capitalize_sentence_starts(["Đi.", "  ừ"]) == {1: "  Ừ"}

    def test_turkish_target_is_skipped(self) -> None:
        assert supports_sentence_case("Vietnamese")
        assert supports_sentence_case("English")
        assert not supports_sentence_case("Turkish")


def _events(texts: list[str]) -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=i + 1, text=text, interval=TimeInterval(float(i), i + 0.9))
        for i, text in enumerate(texts)
    ]


def _run(contiguous: bool) -> list[str]:
    source = ["退婚", "垂死病中惊坐起", "萧炎竟是我自己"]
    translator = FakeSubtitleTranslator(dictionary={
        "退婚": "Từ hôn?!",
        "垂死病中惊坐起": "sắp chết bỗng giật mình ngồi dậy,",
        "萧炎竟是我自己": "Tiêu Viêm lại chính là ta?!",
    })
    request = TranslateSubtitlesRequest(
        events=_events(source),
        stages=[TranslationStageConfig(
            kind=TranslationStageKind.LITERAL, model_name="fake", batch_size=50
        )],
        context=TranslationContext(target_lang="Vietnamese"),
        contiguous=contiguous,
    )
    response = TranslateSubtitlesUseCase(translator=translator).execute(request)
    return [event.text for event in response.events]


class TestUseCase:
    def test_final_translation_is_capitalized(self) -> None:
        assert _run(contiguous=True)[1] == "Sắp chết bỗng giật mình ngồi dậy,"

    def test_subset_retranslation_is_left_to_the_caller(self) -> None:
        """Dòng kề trong tập con không phải dòng kề trong phim."""
        assert _run(contiguous=False)[1] == "sắp chết bỗng giật mình ngồi dậy,"


class TestRetranslateMerge:
    def test_uses_real_neighbour_and_only_touches_retranslated_lines(self) -> None:
        current = ["Từ hôn?!", "cũ,", "Đi.", "ừ"]
        merged = _capitalize_retranslated(
            current, {1: "sắp chết bỗng giật mình ngồi dậy,"}, "Vietnamese"
        )
        assert merged == {1: "Sắp chết bỗng giật mình ngồi dậy,"}  # dòng 3 "ừ" giữ nguyên

    def test_continuation_of_real_neighbour_is_kept(self) -> None:
        current = ["Ta ra khỏi sơn môn,", "cũ"]
        assert _capitalize_retranslated(current, {1: "chẳng phải bị"}, "Vietnamese") == {
            1: "chẳng phải bị"
        }
