"""[v3.23.587] Mục thuật ngữ ``A / B``: chọn MỘT phương án, không chép cả hai vào thoại.

Đo trên 51 chuẩn đoán dịch thật (10 079 mục thuật ngữ):

* 3 084 mục có vế dịch ``A / B``; khối nhắc theo lô đưa nguyên chuỗi vào
  ``must_translate_as`` → 99 dòng dịch thô ở 28 lượt chạy chép cả hai phương án, 3 dòng
  còn tới bản cuối (lượt mới nhất: "ngày mai đến đồn công an xã/phường.").
* 98 mục có vế gốc ``A / B`` (hai cách viết); khối nhắc tìm nguyên chuỗi ấy trong câu gốc
  nên 228 dòng chứa các thuật ngữ này không hề được nhắc.
"""

# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.glossary_note_guard import find_note_leaks
from subtitles_extractor.application.services.glossary_options import (
    TermRendering,
    term_renderings,
    translation_options,
)
from subtitles_extractor.application.services.roster_format import (
    cjk_substitution_map,
    substitute_known_cjk,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _batch_glossary_block,
)

# Các mục CHÉP NGUYÊN từ bảng thuật ngữ của những lượt chạy thật.
GLOSSARY = "\n".join([
    "派出所 = đồn công an xã/phường",
    "衝動 = bốc đồng / manh động",
    "人渣 = cặn bã / loài súc sinh",
    "造次 = làm càn / vô lễ",
    "甄先生 = ông Chân / anh Chân",
    "犯賤 = đê tiện / không biết điều / rước nhục vào thân",
    "借一步说话 = nói chuyện riêng / ra chỗ vắng nói chuyện",
    "急性白血病 / 白血病 = bệnh bạch cầu cấp tính / bệnh bạch cầu",
    "老山 / 魯山 = Lão Sơn",
    "炸药 / 炸弹 = chất nổ / bom",
    "绿毛龟 / 绿帽子 = rùa lông xanh / kẻ bị cắm sừng / cắm sừng",
    "雙響炮 / 連環炮 = pháo đôi / pháo liên hoàn (tiếng lóng hai cha con cùng chơi)",
    "東龍派出所 = Đồn công an Đông Long",
    "惹麻煩 = gây chuyện / chuốc rắc rối",
])

# Mục DỰNG THÊM để kiểm biên (kho chưa có mục nào viết liền mà KHÔNG phải hai phương án).
NOT_A_CHOICE = "\n".join([
    "待命 = túc trực 24/24",
    "百万年薪 = mức lương một triệu tệ/năm",
])

# (câu gốc, bản dịch thật chép cả hai phương án, bản đúng)
REAL_LEAKS = [
    (
        "你說甚麼你别衝動",
        "Mày nói gì thế, đừng bốc đồng / manh động chứ.",
        "Mày nói gì thế, đừng bốc đồng chứ.",
    ),
    ("人渣！", "Cặn bã / loài súc sinh!", "Cặn bã!"),
    ("不要，甄先生，不要", "Đừng, ông Chân / anh Chân, đừng.", "Đừng, ông Chân, đừng."),
    (
        "能否借一步说话？",
        "Có thể nói chuyện riêng / ra chỗ vắng nói chuyện một chút không?",
        "Có thể nói chuyện riêng một chút không?",
    ),
    # Bản CUỐI của lượt mới nhất — vế dịch viết liền "xã/phường".
    (
        "叫所有男村民明天去派出所",
        "Bảo tất cả nam giới trong làng ngày mai đến đồn công an xã/phường.",
        "Bảo tất cả nam giới trong làng ngày mai đến đồn công an xã.",
    ),
    (
        "本派出所女公安",
        "nữ công an của đồn công an xã/phường",
        "nữ công an của đồn công an xã",
    ),
    # Thuật ngữ nằm ở dòng KẾ BÊN (lô dịch nhiều dòng) — bản cuối của một lượt khác.
    ("还要当着我和沈总的面", "còn định làm càn / vô lễ", "còn định làm càn"),
    # Model viết sát dấu gạch chéo.
    ("誰來惹麻煩", "Ai gây chuyện/chuốc rắc rối cơ chứ.", "Ai gây chuyện cơ chứ."),
]

# Dấu "/" KHÔNG phải hai phương án thuật ngữ — bản cuối thật, phải giữ nguyên.
REAL_LEGITIMATE = [
    ("二十四小时待命", "phải túc trực 24/24."),
    ("我愿意出百万年薪", "tôi sẵn sàng trả mức lương một triệu tệ/năm"),
    ("好谢谢啊", "Vâng, cảm ơn anh/chị nhiều lắm."),
]


class TestTranslationOptions:
    @pytest.mark.parametrize(
        ("translation", "expected"),
        [
            ("bốc đồng / manh động", ("bốc đồng", "manh động")),
            (
                "đê tiện / không biết điều / rước nhục vào thân",
                ("đê tiện", "không biết điều", "rước nhục vào thân"),
            ),
            # Chú thích trong ngoặc không phải phương án dùng trong lời thoại.
            (
                "pháo đôi / pháo liên hoàn (tiếng lóng hai cha con cùng chơi)",
                ("pháo đôi", "pháo liên hoàn"),
            ),
            (
                "phá từ đường (tuyệt tự tuyệt tôn / hủy hoại hạ bộ)",
                ("phá từ đường",),
            ),
            ("đồn công an xã/phường", ("đồn công an xã", "đồn công an phường")),
            ("Đồn công an Đông Long", ("Đồn công an Đông Long",)),
            ("", ()),
        ],
    )
    def test_options(self, translation: str, expected: tuple[str, ...]) -> None:
        assert translation_options(translation) == expected

    @pytest.mark.parametrize(
        "translation",
        ["túc trực 24/24", "mức lương một triệu tệ/năm", "nhạc AC/DC", "km/h", "50/50"],
    )
    def test_slash_that_is_not_a_choice_is_kept(self, translation: str) -> None:
        assert translation_options(translation) == (translation,)


class TestTermRenderings:
    def test_source_variants_share_one_translation(self) -> None:
        assert term_renderings("老山 / 魯山 = Lão Sơn") == [
            TermRendering("老山", ("Lão Sơn",)),
            TermRendering("魯山", ("Lão Sơn",)),
        ]

    def test_equal_counts_pair_up_in_order(self) -> None:
        assert term_renderings("炸药 / 炸弹 = chất nổ / bom") == [
            TermRendering("炸药", ("chất nổ",)),
            TermRendering("炸弹", ("bom",)),
        ]

    def test_unequal_counts_give_every_variant_every_option(self) -> None:
        options = ("rùa lông xanh", "kẻ bị cắm sừng", "cắm sừng")
        entry = "绿毛龟 / 绿帽子 = rùa lông xanh / kẻ bị cắm sừng / cắm sừng"
        assert term_renderings(entry) == [
            TermRendering("绿毛龟", options),
            TermRendering("绿帽子", options),
        ]


def _block(*sources: str) -> list[dict[str, object]]:
    batch = [SimpleNamespace(text=text) for text in sources]
    return _batch_glossary_block(batch, GLOSSARY)


class TestBatchTermBlock:
    def test_alternatives_are_separate_values_not_one_slashed_string(self) -> None:
        """Nguyên nhân gốc: "must_translate_as": "bốc đồng / manh động".

        [v3.23.589] Phương án đầu giữ khoá "must_translate_as" (xem
        ``test_glossary_reminder_v323589``); các phương án còn lại tách riêng.
        """
        assert _block("你說甚麼你别衝動") == [
            {
                "term": "衝動",
                "must_translate_as": "bốc đồng",
                "also_acceptable": ["manh động"],
            }
        ]

    def test_tight_slash_is_expanded(self) -> None:
        assert _block("本派出所女公安") == [
            {
                "term": "派出所",
                "must_translate_as": "đồn công an xã",
                "also_acceptable": ["đồn công an phường"],
            }
        ]

    def test_source_variant_is_found(self) -> None:
        """Trước đây tìm nguyên chuỗi "老山 / 魯山" trong câu gốc → không bao giờ khớp."""
        for spelling in ("魯山", "老山"):
            assert _block(f"我在{spelling}打過仗") == [
                {"term": spelling, "must_translate_as": "Lão Sơn"}
            ]

    def test_paired_variant_gets_its_own_translation(self) -> None:
        assert _block("这个是炸弹") == [{"term": "炸弹", "must_translate_as": "bom"}]

    def test_single_translation_is_unchanged(self) -> None:
        assert _block("東龍派出所到了")[-1] == {
            "term": "東龍派出所", "must_translate_as": "Đồn công an Đông Long"
        }

    def test_slash_that_is_not_a_choice_stays_one_translation(self) -> None:
        batch = [SimpleNamespace(text="二十四小时待命"), SimpleNamespace(text="百万年薪")]
        assert _batch_glossary_block(batch, NOT_A_CHOICE) == [
            {"term": "待命", "must_translate_as": "túc trực 24/24"},
            {"term": "百万年薪", "must_translate_as": "mức lương một triệu tệ/năm"},
        ]


def _fixed(source: str, translation: str) -> str:
    glossary = f"{GLOSSARY}\n{NOT_A_CHOICE}"
    leaks = find_note_leaks({1: source}, {1: translation}, glossary)
    return leaks[0].text_after if leaks else translation


class TestJoinedOptionsAreReduced:
    @pytest.mark.parametrize(("source", "leaked", "clean"), REAL_LEAKS)
    def test_real_leaks_keep_the_first_option(
        self, source: str, leaked: str, clean: str
    ) -> None:
        assert _fixed(source, leaked) == clean

    @pytest.mark.parametrize(("source", "translation"), REAL_LEGITIMATE)
    def test_other_slashes_are_kept(self, source: str, translation: str) -> None:
        assert _fixed(source, translation) == translation

    def test_source_with_a_slash_is_left_alone(self) -> None:
        text = "Chọn bốc đồng / manh động đi."
        assert _fixed("衝動/冷靜", text) == text

    def test_option_followed_by_an_unrelated_phrase_is_kept(self) -> None:
        """Chỉ gộp khi HAI vế đều là phương án của cùng một mục."""
        text = "Đừng bốc đồng / hãy bình tĩnh."
        assert _fixed("你别衝動", text) == text

    def test_capital_letter_of_the_line_is_kept(self) -> None:
        assert _fixed("衝動", "Bốc đồng / manh động quá!") == "Bốc đồng quá!"

    def test_note_and_joined_options_in_one_line(self) -> None:
        leaked = "Chơi pháo đôi / pháo liên hoàn (tiếng lóng hai cha con cùng chơi) à?"
        assert _fixed("玩雙響炮呀", leaked) == "Chơi pháo đôi à?"

    def test_dropped_option_is_reported(self) -> None:
        [leak] = find_note_leaks({7: "人渣！"}, {7: "Cặn bã / loài súc sinh!"}, GLOSSARY)
        assert (leak.index, leak.notes) == (7, ("/ loài súc sinh",))


class TestResidualCjkSubstitution:
    """Bản đồ thay chữ Hán còn sót dùng CÙNG cách tách hai vế với khối nhắc theo lô."""

    # Mục thật: dấu "/" nằm TRONG chú thích.
    NOTED = "卸货 = dỡ hàng (sinh con / lâm bồn)"

    def test_note_with_a_slash_does_not_leak_half_a_parenthesis(self) -> None:
        mapping = cjk_substitution_map("", self.NOTED)
        assert mapping["卸货"] == "dỡ hàng"  # trước đây "dỡ hàng (sinh con"
        assert substitute_known_cjk("Sớm 卸货 rồi,", mapping) == "Sớm dỡ hàng rồi,"

    def test_every_source_variant_is_a_key(self) -> None:
        mapping = cjk_substitution_map("", GLOSSARY)
        assert (mapping["老山"], mapping["魯山"]) == ("Lão Sơn", "Lão Sơn")
        assert (mapping["炸药"], mapping["炸弹"]) == ("chất nổ", "bom")
        assert "老山 / 魯山" not in mapping

    def test_first_option_is_used(self) -> None:
        mapping = cjk_substitution_map("", GLOSSARY)
        assert mapping["衝動"] == "bốc đồng"
        assert mapping["派出所"] == "đồn công an xã"

    def test_slash_that_is_not_a_choice_is_kept_whole(self) -> None:
        mapping = cjk_substitution_map("", NOT_A_CHOICE)
        assert mapping["待命"] == "túc trực 24/24"  # trước đây "túc trực 24"

    def test_translation_that_is_only_a_note_is_skipped(self) -> None:
        assert "嗯哼" not in cjk_substitution_map("", "嗯哼 = (tiếng ậm ừ)")


class TestPipeline:
    def test_stage_output_is_cleaned_and_recorded(self) -> None:
        cases = [REAL_LEAKS[0], REAL_LEAKS[4], REAL_LEGITIMATE[0]]
        sources = [
            TranslationLine(index=i, start_ms=i, end_ms=i + 1, text=row[0])
            for i, row in enumerate(cases, start=1)
        ]
        output = [
            TranslationLine(
                index=i, start_ms=i, end_ms=i + 1, text=row[1], original_text=row[0]
            )
            for i, row in enumerate(cases, start=1)
        ]
        get_process_log().clear()
        cleaned = TranslateSubtitlesUseCase._strip_glossary_notes(
            output, sources, GLOSSARY, "LITERAL"
        )
        assert [line.text for line in cleaned] == [
            REAL_LEAKS[0][2], REAL_LEAKS[4][2], REAL_LEGITIMATE[0][1]
        ]
        events = [
            event for event in get_process_log().snapshot()["events"]
            if event["event"] == "glossary_notes_stripped"
        ]
        assert events[-1]["lines"] == [1, 2]
