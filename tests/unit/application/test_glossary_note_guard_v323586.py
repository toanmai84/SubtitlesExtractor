"""[v3.23.586] Chú thích của bảng thuật ngữ không được nằm trong lời thoại.

Đo trên 50 chuẩn đoán dịch thật: chú thích trong ngoặc của bảng thuật ngữ bị chép nguyên
văn vào 459 dòng dịch thô (47/50 lượt chạy) và còn tới bản cuối ở 26 dòng của 15 lượt —
"tôi gọi 120 (cấp cứu y tế) cho cô.", "chiếc xe Mercedes G-Class (Đại G) này…". Khâu
thuyết minh đọc cả phần trong ngoặc.
"""

# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.glossary_note_guard import (
    find_note_leaks,
    glossary_note_phrases,
    split_glossary_note,
    strip_leaked_notes,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _batch_glossary_block,
)

# Các mục CHÉP NGUYÊN từ bảng thuật ngữ của những lượt chạy thật.
GLOSSARY = "\n".join([
    "光天化日 = thanh thiên bạch nhật (giữa ban ngày ban mặt)",
    "賣淫嫖妓 = mua bán dâm (mãi dâm mua dâm)",
    "死相 = đồ chết tiệt (câu mắng yêu)",
    "二十萬 = hai mươi vạn (200.000 tệ)",
    "大G = xe Mercedes G-Class (Đại G)",
    "120 = 120 (cấp cứu y tế)",
    "试管 = thụ tinh ống nghiệm (IVF)",
    "百万起步 = tiền triệu (tệ) trở lên",
    "越狗 = quân Việt (từ ngữ miệt thị thời chiến)",
    "諒山 = Lạng Sơn",
    "東龍派出所 = Đồn công an Đông Long",
])

# (câu gốc, bản dịch thật bị chèn chú thích, bản đúng)
REAL_LEAKS = [
    (
        "光天化日，賣淫嫖妓",
        "Thanh thiên bạch nhật (giữa ban ngày ban mặt), mua bán dâm (mãi dâm mua dâm),",
        "Thanh thiên bạch nhật, mua bán dâm,",
    ),
    ("死相", "Đồ chết tiệt (câu mắng yêu)", "Đồ chết tiệt"),
    (
        "這裡是二十萬，我老婆呢",
        "Đây là hai mươi vạn (200.000 tệ), vợ tôi đâu?",
        "Đây là hai mươi vạn, vợ tôi đâu?",
    ),
    (
        "这辆大G是你的？",
        "chiếc xe Mercedes G-Class (Đại G) này là của ai cơ?",
        "chiếc xe Mercedes G-Class này là của ai cơ?",
    ),
    ("我给你打幺二零", "tôi gọi 120 (cấp cứu y tế) cho cô.", "tôi gọi 120 cho cô."),
    (
        "孩子是试管的",
        "Đứa bé sinh ra bằng thụ tinh ống nghiệm (IVF),",
        "Đứa bé sinh ra bằng thụ tinh ống nghiệm,",
    ),
    ("百万起步?", "tiền triệu (tệ) trở lên à?", "tiền triệu trở lên à?"),
    # Model lặp lại chính cụm vừa viết thay cho chú thích.
    (
        "個二個還沒見到越狗的影子",
        "Một đám chưa kịp nhìn thấy bóng dáng quân Việt (quân Việt) đâu,",
        "Một đám chưa kịp nhìn thấy bóng dáng quân Việt đâu,",
    ),
    (
        "還幾乎死在諒山呢",
        "Suýt chút nữa bỏ mạng ở Lạng Sơn (Lạng Sơn) đấy.",
        "Suýt chút nữa bỏ mạng ở Lạng Sơn đấy.",
    ),
]


class TestSplitGlossaryNote:
    @pytest.mark.parametrize(
        ("translation", "expected"),
        [
            (
                "thanh thiên bạch nhật (giữa ban ngày ban mặt)",
                ("thanh thiên bạch nhật", "giữa ban ngày ban mặt"),
            ),
            ("tiền triệu (tệ) trở lên", ("tiền triệu trở lên", "tệ")),
            ("Đồn công an Đông Long", ("Đồn công an Đông Long", "")),
            ("pháo đôi / pháo liên hoàn", ("pháo đôi / pháo liên hoàn", "")),
            # Chỉ có phần trong ngoặc: không còn gì để dùng → giữ nguyên.
            ("(tiếng lóng)", ("(tiếng lóng)", "")),
            ("", ("", "")),
        ],
    )
    def test_split(self, translation: str, expected: tuple[str, str]) -> None:
        assert split_glossary_note(translation) == expected


class TestStripLeakedNotes:
    @pytest.mark.parametrize(("source", "leaked", "clean"), REAL_LEAKS)
    def test_real_leaks_are_removed(self, source: str, leaked: str, clean: str) -> None:
        phrases = glossary_note_phrases(GLOSSARY)
        assert strip_leaked_notes(leaked, source, phrases)[0] == clean

    def test_clean_line_is_returned_untouched(self) -> None:
        phrases = glossary_note_phrases(GLOSSARY)
        text = "Đây là hai mươi vạn, vợ tôi đâu?"
        assert strip_leaked_notes(text, "這裡是二十萬，我老婆呢", phrases) == (text, ())

    def test_other_parentheticals_are_kept(self) -> None:
        """Chỉ dẫn "(Thở dốc)" không phải chú thích thuật ngữ — không đụng tới."""
        phrases = glossary_note_phrases(GLOSSARY)
        assert strip_leaked_notes("(Thở dốc)", "L上上", phrases) == ("(Thở dốc)", ())
        text = "Pha! (Nhổ nước bọt)"
        assert strip_leaked_notes(text, "我呸！", phrases) == (text, ())

    def test_source_with_parentheses_is_left_alone(self) -> None:
        phrases = glossary_note_phrases(GLOSSARY)
        text = "Gọi 120 (cấp cứu y tế) đi."
        assert strip_leaked_notes(text, "打120（急救）吧", phrases) == (text, ())

    def test_line_that_is_only_a_note_is_kept(self) -> None:
        phrases = glossary_note_phrases(GLOSSARY)
        assert strip_leaked_notes("(Đại G)", "大G", phrases) == ("(Đại G)", ())

    def test_part_of_a_word_is_not_a_repetition(self) -> None:
        text = "Ở tận Lào Cai (ai) cơ."
        assert strip_leaked_notes(text, "在老街", frozenset()) == (text, ())

    def test_leaks_are_reported_per_line(self) -> None:
        sources = {7: "死相", 8: "走吧"}
        translations = {7: "Đồ chết tiệt (câu mắng yêu)", 8: "Đi thôi."}
        [leak] = find_note_leaks(sources, translations, GLOSSARY)
        assert (leak.index, leak.notes, leak.text_after) == (
            7, ("câu mắng yêu",), "Đồ chết tiệt"
        )


class TestBatchTermBlock:
    def test_must_translate_as_carries_no_note(self) -> None:
        """Nguyên nhân gốc: chú thích nằm ngay trong chỉ thị "must_translate_as"."""
        batch = [SimpleNamespace(text="光天化日，賣淫嫖妓"), SimpleNamespace(text="死相")]
        assert _batch_glossary_block(batch, GLOSSARY) == [
            {"term": "光天化日", "must_translate_as": "thanh thiên bạch nhật"},
            {"term": "賣淫嫖妓", "must_translate_as": "mua bán dâm"},
            {"term": "死相", "must_translate_as": "đồ chết tiệt"},
        ]

    def test_entry_without_note_is_unchanged(self) -> None:
        batch = [SimpleNamespace(text="東龍派出所到了")]
        assert _batch_glossary_block(batch, GLOSSARY) == [
            {"term": "東龍派出所", "must_translate_as": "Đồn công an Đông Long"}
        ]


class _LeakyTranslator:
    """Bộ dịch giả: chép chú thích vào lời thoại như model thật đã làm."""

    def __init__(self, outputs: dict[str, str]) -> None:
        self._outputs = outputs

    def is_available(self) -> bool:
        return True

    def translate_stage(
        self, *, input_lines: list[TranslationLine], **_kwargs: object
    ) -> list[TranslationLine]:
        return [
            TranslationLine(
                index=line.index, start_ms=line.start_ms, end_ms=line.end_ms,
                text=self._outputs[line.original_text or line.text],
                original_text=line.original_text or line.text,
            )
            for line in input_lines
        ]


class TestPipeline:
    def test_notes_are_stripped_from_the_stage_output(self) -> None:
        cases = REAL_LEAKS[:3]
        translator = _LeakyTranslator({source: leaked for source, leaked, _ in cases})
        request = TranslateSubtitlesRequest(
            events=[
                SubtitleEvent(index=i, text=source, interval=TimeInterval(i, i + 0.9))
                for i, (source, _leaked, _clean) in enumerate(cases, start=1)
            ],
            stages=[
                TranslationStageConfig(
                    kind=TranslationStageKind.LITERAL, model_name="m", batch_size=10
                )
            ],
            context=TranslationContext(target_lang="Vietnamese", glossary=GLOSSARY),
        )
        response = TranslateSubtitlesUseCase(translator=translator).execute(request)
        assert [event.text for event in response.events] == [clean for *_, clean in cases]
        events = [
            event for event in get_process_log().snapshot()["events"]
            if event["event"] == "glossary_notes_stripped"
        ]
        assert events[0]["lines"] == [1, 2, 3]
