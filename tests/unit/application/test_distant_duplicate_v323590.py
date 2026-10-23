"""[v3.23.590] Hồi quy: hai dòng CÁCH XA nhau mang cùng bản dịch dù câu gốc khác hẳn.

Chuẩn đoán thật v3.23.589 (phim "灭门惨案II"): dòng 1141 "一二三" ra "Một chiếc thuyền
nan lướt nhẹ trôi" — nguyên văn bản dịch của dòng 1165 "一葉輕舟去" cách đó 24 dòng. Sai
từ bản dịch thô tới bản cuối, không cờ chuẩn đoán nào báo.

Đo trên bản cuối của 54 lượt dịch thật: 87 cặp như vậy ở 36 lượt. Có cặp một dòng mang
lời dòng kia ("宋" ra "Liễu đại tiểu thư"), có cặp là hai câu đồng nghĩa dịch giống nhau
là đúng ("彻底好了" / "已经痊愈了"). Nên khâu dọn cuối dịch lại riêng các dòng đó và chỉ
nhận dòng nào ra KHÁC HẲN bản cũ.

Mọi dòng dưới đây chép nguyên văn; cột "dịch lại" là bản dịch thô của CHÍNH dòng đó ở một
lượt chạy khác của cùng phim (riêng 1129-1130 là bản dịch lại thật của lượt v3.23.589).
"""

# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.application.services.distant_duplicate_guard import (
    DISTANT_MAX_GAP,
    find_distant_duplicates,
    retranslations_to_take,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)

# Lượt v3.23.589: (dòng, câu gốc, bản địa hoá trước khâu dọn cuối, dịch lại).
BOAT_SONG = [
    (1129, "你死得瞑目吧", "Mày chết đi cho nhắm mắt.",
     "Mày nhắm mắt xuôi tay được rồi đấy."),
    (1130, "你死得瞑目吧你死得瞑目了吧", "Mày chết đi cho nhắm mắt.",
     "Mày chết cho nhắm mắt đi, mày chết cho nhắm mắt đi."),
    (1139, "這小子不死，我們麻煩可大了",
     "Thằng nhãi này mà không chết, chúng ta phiền to đấy.",
     "Thằng nhóc này không chết, chúng ta gặp rắc rối lớn rồi."),
    (1140, "反正都是死，就讓他先死", "Đằng nào cũng chết, chi bằng để nó chết trước.",
     "Dù sao cũng phải chết, cứ để nó chết trước đi."),
    (1141, "一二三", "Một chiếc thuyền nan lướt nhẹ trôi", "Một hai ba"),
    (1142, "你有甚麼打算", "Anh có dự tính gì không?", "Em có dự tính gì?"),
    (1163, "一定是有人想頂了種男的罪",
     "nhất định là có kẻ muốn nhận tội thay gã mượn giống.",
     "chắc chắn là có kẻ muốn nhận tội thay gã mượn giống đó."),
    (1164, "你以為會嗎", "Đồng chí nghĩ vậy sao?", "Anh nghĩ vậy sao?"),
    (1165, "一葉輕舟去", "Một chiếc thuyền nan lướt nhẹ trôi",
     "Một chiếc thuyền nan lướt nhẹ trôi"),
    (1166, "别玩嘛", "Đừng đùa nữa.", "Đừng đùa nữa."),
]

# Phim "傻子落水" (3902 dòng), bản cuối của lượt v3.23.579; dịch lại = bản thô v3.23.569.
INTERRUPTED_NAME = [
    (443, "宋钱弟弟", "Tống Tiền đệ,", "Tống Tiền em à,"),
    (463, "嫂子", "Tống Tiền đệ,", "Chị à..."),
    (2974, "柳大小姐", "Liễu đại tiểu thư", "Liễu đại tiểu thư"),
    (2977, "宋", "Liễu đại tiểu thư...", "Tống..."),
    (3003, "那个柳大小姐", "Liễu đại tiểu thư,", "Đại tiểu thư nhà họ Liễu kia,"),
]

# Cùng phim, bản cuối của lượt v3.23.549: hai câu gốc ĐỒNG NGHĨA, dịch giống nhau là đúng.
SYNONYMS = [
    (1743, "彻底好了", "đã khỏi hẳn rồi.", "khỏi hẳn rồi."),
    (1779, "已经痊愈了", "đã khỏi hẳn rồi.", "của cô đã khỏi hẳn rồi."),
]

Row = tuple[int, str, str, str]


def _columns(rows: list[Row]) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    sources = {row[0]: row[1] for row in rows}
    translations = {row[0]: row[2] for row in rows}
    retranslated = {row[0]: row[3] for row in rows}
    return sources, translations, retranslated


class TestFindDistantDuplicates:
    def test_real_case(self) -> None:
        sources, translations, _ = _columns(BOAT_SONG)
        assert find_distant_duplicates(sources, translations) == [(1141, 1165)]

    def test_adjacent_duplicates_belong_to_the_other_guard(self) -> None:
        sources, translations, _ = _columns(BOAT_SONG)
        pairs = find_distant_duplicates(sources, translations)
        assert not any({1129, 1130} & set(pair) for pair in pairs)

    def test_three_lines_give_two_chained_pairs(self) -> None:
        """"柳大小姐" / "那个柳大小姐" là cùng một câu gốc nên không ghép với nhau."""
        sources, translations, _ = _columns(INTERRUPTED_NAME)
        assert find_distant_duplicates(sources, translations) == [
            (443, 463), (2974, 2977), (2977, 3003),
        ]

    def test_synonymous_sources_are_still_reported(self) -> None:
        """Bộ dò không biết hai câu đồng nghĩa — phần đó do bản dịch lại quyết định."""
        sources, translations, _ = _columns(SYNONYMS)
        assert find_distant_duplicates(sources, translations) == [(1743, 1779)]

    def test_same_source_repeated_is_not_a_pair(self) -> None:
        sources = {1: "柳大小姐", 2: "走吧", 3: "柳大小姐"}
        translations = {1: "Liễu đại tiểu thư,", 2: "Đi thôi.", 3: "Liễu đại tiểu thư."}
        assert find_distant_duplicates(sources, translations) == []

    def test_short_translations_are_ignored(self) -> None:
        sources = {1: "好了", 2: "走吧", 3: "行"}
        translations = {1: "Được rồi.", 2: "Đi thôi.", 3: "Được rồi."}
        assert find_distant_duplicates(sources, translations) == []

    @pytest.mark.parametrize(("gap", "found"), [(DISTANT_MAX_GAP, True),
                                                (DISTANT_MAX_GAP + 1, False)])
    def test_only_lines_of_the_same_stretch_are_paired(
        self, gap: int, found: bool
    ) -> None:
        sources = {1141: "一二三", 1141 + gap: "一葉輕舟去"}
        translations = dict.fromkeys(sources, "Một chiếc thuyền nan lướt nhẹ trôi")
        pairs = find_distant_duplicates(sources, translations)
        assert pairs == ([(1141, 1141 + gap)] if found else [])

    def test_latin_script_sources_are_compared_by_word(self) -> None:
        sources = {1: "One two three", 2: "Let's go", 3: "A small boat drifts away"}
        translations = {
            1: "Một chiếc thuyền nan trôi", 2: "Đi thôi.", 3: "Một chiếc thuyền nan trôi",
        }
        assert find_distant_duplicates(sources, translations) == [(1, 3)]


class TestRetranslationsToTake:
    def test_real_case_only_the_stray_line_changes(self) -> None:
        _, translations, retranslated = _columns(BOAT_SONG)
        taken = retranslations_to_take([(1141, 1165)], translations, retranslated)
        assert taken == {1141: "Một hai ba"}

    def test_interrupted_name_gets_its_own_words(self) -> None:
        sources, translations, retranslated = _columns(INTERRUPTED_NAME)
        pairs = find_distant_duplicates(sources, translations)
        assert retranslations_to_take(pairs, translations, retranslated) == {
            463: "Chị à...",
            2977: "Tống...",
        }

    def test_synonymous_lines_keep_their_polished_translation(self) -> None:
        _, translations, retranslated = _columns(SYNONYMS)
        assert retranslations_to_take([(1743, 1779)], translations, retranslated) == {}

    def test_missing_retranslation_changes_nothing(self) -> None:
        _, translations, _ = _columns(BOAT_SONG)
        assert retranslations_to_take([(1141, 1165)], translations, {}) == {}
        assert retranslations_to_take([(1141, 1165)], translations, {1141: "  "}) == {}

    def test_pair_that_would_stay_identical_is_left_alone(self) -> None:
        """Dịch lại mà hai dòng vẫn ra CÙNG một câu mới: không có gì chắc hơn bản cũ."""
        _, translations, _ = _columns(BOAT_SONG)
        again = {1141: "Thuyền ai lướt sóng", 1165: "Thuyền ai lướt sóng."}
        assert retranslations_to_take([(1141, 1165)], translations, again) == {}


def _lines(rows: list[Row], column: int) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=row[0], start_ms=row[0] * 1000, end_ms=row[0] * 1000 + 900,
            text=row[column], original_text=row[1],
        )
        for row in rows
    ]


class _Retranslator:
    """Bộ dịch giả: trả cột "dịch lại" cho đúng các dòng được hỏi."""

    def __init__(self, rows: list[Row], error: Exception | None = None) -> None:
        self.answer = {row[0]: row[3] for row in rows}
        self.error = error
        self.calls: list[list[int]] = []
        self.attach_video: list[bool] = []

    def translate_stage(
        self, *, stage: Any, source_lines: list[TranslationLine], **kwargs: Any
    ) -> list[TranslationLine]:
        assert stage.kind is TranslationStageKind.LITERAL
        self.calls.append([line.index for line in source_lines])
        self.attach_video.append(kwargs["attach_video"])
        if self.error is not None:
            raise self.error
        return [replace(line, text=self.answer[line.index]) for line in source_lines]


def _request(with_literal: bool = True) -> TranslateSubtitlesRequest:
    kinds = [TranslationStageKind.LOCALIZE]
    if with_literal:
        kinds.insert(0, TranslationStageKind.LITERAL)
    return TranslateSubtitlesRequest(
        events=[],
        stages=[TranslationStageConfig(kind=kind, model_name="m") for kind in kinds],
        context=TranslationContext(target_lang="Vietnamese"),
    )


def _cleanup(
    rows: list[Row], retranslator: _Retranslator, with_literal: bool = True
) -> dict[int, str]:
    use_case = TranslateSubtitlesUseCase(translator=retranslator)  # type: ignore[arg-type]
    result = use_case._cleanup_duplicate_adjacent(
        _lines(rows, 2), _lines(rows, 1), _request(with_literal), None
    )
    return {line.index: line.text for line in result}


def _events(name: str) -> list[dict[str, Any]]:
    return [e for e in get_process_log().snapshot()["events"] if e["event"] == name]


@pytest.fixture(autouse=True)
def _fresh_log() -> None:
    get_process_log().clear()


WITHOUT_ADJACENT = [row for row in BOAT_SONG if row[0] not in (1129, 1130)]


class TestFinalCleanup:
    def test_stray_line_is_retranslated_and_its_twin_kept(self) -> None:
        cleaned = _cleanup(WITHOUT_ADJACENT, _Retranslator(WITHOUT_ADJACENT))
        assert cleaned[1141] == "Một hai ba"
        assert cleaned[1165] == "Một chiếc thuyền nan lướt nhẹ trôi"

    def test_no_other_line_is_touched(self) -> None:
        _, translations, _ = _columns(WITHOUT_ADJACENT)
        cleaned = _cleanup(WITHOUT_ADJACENT, _Retranslator(WITHOUT_ADJACENT))
        assert {i for i in cleaned if cleaned[i] != translations[i]} == {1141}

    def test_one_text_only_request_for_just_the_pair(self) -> None:
        retranslator = _Retranslator(WITHOUT_ADJACENT)
        _cleanup(WITHOUT_ADJACENT, retranslator)
        assert retranslator.calls == [[1141, 1165]]
        assert retranslator.attach_video == [False]

    def test_shares_the_request_with_adjacent_duplicates(self) -> None:
        """Lượt thật có cả nhóm trùng liền kề 1129-1130: vẫn chỉ MỘT lượt gọi."""
        retranslator = _Retranslator(BOAT_SONG)
        cleaned = _cleanup(BOAT_SONG, retranslator)
        assert retranslator.calls == [[1129, 1130, 1141, 1165]]
        assert cleaned[1129] == "Mày nhắm mắt xuôi tay được rồi đấy."
        assert cleaned[1130] == "Mày chết cho nhắm mắt đi, mày chết cho nhắm mắt đi."
        assert cleaned[1141] == "Một hai ba"

    def test_synonymous_pair_costs_a_request_but_changes_nothing(self) -> None:
        retranslator = _Retranslator(SYNONYMS)
        _, translations, _ = _columns(SYNONYMS)
        assert _cleanup(SYNONYMS, retranslator) == translations
        assert len(retranslator.calls) == 1

    def test_clean_translation_costs_no_request(self) -> None:
        rows = [row for row in WITHOUT_ADJACENT if row[0] != 1141]
        retranslator = _Retranslator(rows)
        _, translations, _ = _columns(rows)
        assert _cleanup(rows, retranslator) == translations
        assert retranslator.calls == []

    def test_failed_retranslation_keeps_everything(self) -> None:
        retranslator = _Retranslator(
            WITHOUT_ADJACENT, error=SubtitleTranslationError("503")
        )
        _, translations, _ = _columns(WITHOUT_ADJACENT)
        assert _cleanup(WITHOUT_ADJACENT, retranslator) == translations

    def test_without_a_literal_stage_nothing_is_retranslated(self) -> None:
        retranslator = _Retranslator(WITHOUT_ADJACENT)
        _, translations, _ = _columns(WITHOUT_ADJACENT)
        cleaned = _cleanup(WITHOUT_ADJACENT, retranslator, with_literal=False)
        assert cleaned == translations and retranslator.calls == []


class TestProcessLog:
    def test_pairs_and_replaced_lines_are_recorded(self) -> None:
        _cleanup(WITHOUT_ADJACENT, _Retranslator(WITHOUT_ADJACENT))
        assert _events("distant_duplicates_retranslated") == [{
            "event": "distant_duplicates_retranslated",
            "pairs": [[1141, 1165]],
            "replaced": [1141],
        }]
        assert _events("repeated_lines_retranslated") == []

    def test_adjacent_and_distant_are_recorded_separately(self) -> None:
        _cleanup(BOAT_SONG, _Retranslator(BOAT_SONG))
        [adjacent] = _events("repeated_lines_retranslated")
        assert adjacent["groups"] == [[1129, 1130]] == adjacent["fixed"]
        [distant] = _events("distant_duplicates_retranslated")
        assert distant["pairs"] == [[1141, 1165]]


class TestDiagnosticsFlag:
    @staticmethod
    def _flags(rows: list[Row], column: int) -> dict[str, Any]:
        sources = [
            SimpleNamespace(index=r[0], text=r[1], start_sec=r[0], end_sec=r[0] + 0.9)
            for r in rows
        ]
        translated = [
            SimpleNamespace(index=r[0], text=r[column], start_sec=r[0],
                            end_sec=r[0] + 0.9)
            for r in rows
        ]
        return detect_quality_flags(sources, translated)

    def test_real_final_is_flagged(self) -> None:
        flags = self._flags(WITHOUT_ADJACENT, 2)
        assert flags["duplicate_distant"] == [{"indices": [1141, 1165]}]
        assert flags["counts"]["duplicate_distant"] == 1
        assert "duplicate_distant" in flags["note"]

    def test_cleaned_final_is_not(self) -> None:
        rows = [
            (row[0], row[1], "Một hai ba" if row[0] == 1141 else row[2], row[3])
            for row in WITHOUT_ADJACENT
        ]
        flags = self._flags(rows, 2)
        assert flags["duplicate_distant"] == []
        assert flags["counts"]["duplicate_distant"] == 0
