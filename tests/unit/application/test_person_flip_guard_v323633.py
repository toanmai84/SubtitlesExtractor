# ruff: noqa: RUF001 — chữ Hán/dấu câu fullwidth thuộc câu gốc chép từ phim
"""[v3.23.633] Lưới đảo ngôi: 你 (người nghe) bị khâu sau biến thành "ta"/"chúng ta".

Ca thật (chấm mù tập 1 phim mẫu, nhóm đệ tử giễu Diệp Nhiên, dòng 72/74/75).
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.person_flip_guard import (
    find_person_flips,
    flipped_person,
    source_person,
)


class TestSourcePerson:
    @pytest.mark.parametrize(("source", "expected"), [
        ("当年你是天纵之资", 2),
        ("才会给你们缔结婚约", 2),
        ("您请坐", 2),
        ("我飞升个鬼", 1),
        ("咱们还见什么外", 1),
        ("你给我滚一边去吧你", None),     # có cả 你 lẫn 我
        ("修炼一途", None),
    ])
    def test_cases(self, source: str, expected: int | None) -> None:
        assert source_person(source) == expected


class TestFlippedPerson:
    @pytest.mark.parametrize("translation", [
        "Năm xưa ta là thiên tài hiếm có,",
        "mới hứa hôn cho hai ta.",
        "Giờ tôi vẫn chỉ là một",
    ])
    def test_second_person_source_turned_first(self, translation: str) -> None:
        assert flipped_person("当年你是天纵之资", translation)

    @pytest.mark.parametrize("translation", [
        "Năm xưa ngươi có tư chất thiên tài",
        "Năm xưa huynh là thiên tài",              # huynh nhập nhằng → không tính
        "Thiên tài hiếm có,",                      # bỏ đại từ → không có bằng chứng
        "Ngươi nói ta nghe, năm xưa ra sao",       # có cả hai ngôi
        "Người ta bảo năm xưa là thiên tài",       # "người ta" là ngôi ba
    ])
    def test_not_a_flip(self, translation: str) -> None:
        assert not flipped_person("当年你是天纵之资", translation)

    def test_first_person_source_turned_second(self) -> None:
        assert flipped_person("我飞升个鬼", "Ngươi đi mà phi thăng")
        assert not flipped_person("我飞升个鬼", "Ta phi thăng cái quái gì")

    def test_source_without_person_mark_never_flips(self) -> None:
        assert not flipped_person("修炼一途", "ta tu luyện")


class TestFindPersonFlips:
    SOURCES = {1: "当年你是天纵之资", 2: "才会给你们缔结婚约", 3: "你现在还是一个", 4: "为你寻来的新药"}

    def test_real_case_reverts_only_lines_the_later_stage_broke(self) -> None:
        previous = {1: "Năm xưa ngươi có tư chất thiên tài", 2: "mới cho các ngươi đính hôn",
                    3: "Giờ ngươi vẫn chỉ là một", 4: "Thuốc mới tìm cho con"}
        current = {1: "Năm xưa ta là thiên tài hiếm có,", 2: "mới hứa hôn cho hai ta.",
                   3: "Giờ ngươi vẫn chỉ là một,", 4: "Thuốc mới tìm cho con,"}
        assert find_person_flips(self.SOURCES, previous, current) == {1, 2}

    def test_not_reverted_when_previous_stage_was_already_flipped(self) -> None:
        previous = {1: "Năm xưa ta là thiên tài"}
        current = {1: "Năm xưa ta là thiên tài hiếm có,"}
        assert find_person_flips(self.SOURCES, previous, current) == set()

    def test_unchanged_or_unknown_lines_ignored(self) -> None:
        previous = {1: "Năm xưa ta là thiên tài"}
        assert find_person_flips(self.SOURCES, previous, previous) == set()
        assert find_person_flips(self.SOURCES, {}, {1: "Ta là thiên tài"}) == set()
        assert find_person_flips({}, previous, {1: "Ta ở đây"}) == set()


class TestUseCaseStep:
    def _lines(self, texts: dict[int, str], originals: dict[int, str]):
        from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine

        return [
            TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=t,
                            original_text=originals[i])
            for i, t in texts.items()
        ]

    def test_reverts_flipped_lines_only_and_records_event(self) -> None:
        from subtitles_extractor.application.services.translation_process_log import (
            get_process_log,
        )
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )

        get_process_log().clear()
        sources = {1: "当年你是天纵之资", 2: "才会给你们缔结婚约", 3: "修炼一途"}
        source_lines = self._lines(sources, sources)
        previous = self._lines({1: "Năm xưa ngươi có tư chất", 2: "mới cho các ngươi đính hôn",
                                3: "Con đường tu luyện"}, sources)
        current = self._lines({1: "Năm xưa ta là thiên tài", 2: "mới cho các ngươi đính ước",
                               3: "Con đường tu luyện,"}, sources)
        use_case = TranslateSubtitlesUseCase(translator=object())
        result = use_case._revert_person_flips(current, previous, source_lines, "Giai đoạn 3")
        assert {line.index: line.text for line in result} == {
            1: "Năm xưa ngươi có tư chất",       # lùi về bản trước
            2: "mới cho các ngươi đính ước",     # không đảo ngôi → giữ bản mới
            3: "Con đường tu luyện,",
        }
        events = [e for e in get_process_log().events_since(0) if e["event"] == "person_flip_revert"]
        assert events and events[-1]["lines"] == [1]

    def test_returns_same_list_when_nothing_to_revert(self) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )

        sources = {1: "当年你是天纵之资"}
        lines = self._lines({1: "Năm xưa ngươi có tư chất"}, sources)
        use_case = TranslateSubtitlesUseCase(translator=object())
        assert use_case._revert_person_flips(lines, lines, lines, "x") is lines
