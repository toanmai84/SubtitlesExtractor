"""[v3.23.565] Hồi quy: lưới xưng hô ép sửa theo cue GÁN NHẦM người nói.

Chuẩn đoán thật v3.23.564, dòng 1827-1829 — Tống Dao kể tội nhà họ Từ trong MỘT câu nối
bằng dấu phẩy: "Đánh bị thương anh tôi trước đây, / đánh gãy chân mẹ tôi, / còn ép tôi
phải bỏ học cấp ba,". Cue ghi dòng đầu là 宋瑶 nhưng hai dòng sau là 安雪漫; lưới xưng hô
tin cue và ép theo quy tắc của An Tuyết Mạn ("em") → cùng một câu thành "anh tôi … mẹ em
… ép em". Trên cả phim có 124 câu nhiều dòng mà cue đổi người nói giữa câu.
"""

from __future__ import annotations

import json

from subtitles_extractor.application.services.address_rule_guard import (
    find_address_rule_fixes,
    find_address_rule_violations,
    unstable_cue_lines,
)

CHARACTERS = (
    "宋瑶 (Tống Dao) [nữ, ~18] — em gái Tống Tiền\n"
    "安雪漫 (An Tuyết Mạn) [nữ, ~24] — bạn gái cũ; xưng hô: với Từ Tuấn Hào tự xưng 'em' "
    "gọi 'anh'\n"
    "徐俊豪 (Từ Tuấn Hào) [nam, ~25] — phản diện\n"
)


def _cues(pairs: dict[int, tuple[str, str]]) -> str:
    return json.dumps(
        [{"id": i, "spk": spk, "to": to} for i, (spk, to) in pairs.items()],
        ensure_ascii=False,
    )


class TestUnstableCueLines:
    def test_speaker_change_inside_one_sentence_is_unstable(self) -> None:
        translations = {
            1: "Đánh bị thương anh tôi trước đây,",
            2: "đánh gãy chân mẹ tôi,",
            3: "còn ép tôi phải bỏ học cấp ba.",
            4: "Anh nói gì cơ?",
        }
        cues = _cues(
            {
                1: ("宋瑶", "安雪漫"),
                2: ("安雪漫", "徐俊豪"),
                3: ("安雪漫", "徐俊豪"),
                4: ("徐俊豪", "安雪漫"),
            }
        )
        assert unstable_cue_lines(translations, cues) == {1, 2, 3}

    def test_speaker_change_after_full_stop_is_normal_dialogue(self) -> None:
        translations = {1: "Em đi đây.", 2: "Ừ, đi cẩn thận nhé!"}
        cues = _cues({1: ("宋瑶", "宋钱"), 2: ("宋钱", "宋瑶")})
        assert unstable_cue_lines(translations, cues) == set()

    def test_ellipsis_continues_the_sentence(self) -> None:
        translations = {1: "Đây chính là...", 2: "mảnh nguyên thần."}
        cues = _cues({1: ("宋钱", "红莲女帝"), 2: ("春花", "宋钱")})
        assert unstable_cue_lines(translations, cues) == {1, 2}

    def test_non_adjacent_indices_do_not_join(self) -> None:
        translations = {1: "Đánh bị thương anh tôi,", 5: "mẹ tôi"}
        cues = _cues({1: ("宋瑶", "徐俊豪"), 5: ("安雪漫", "徐俊豪")})
        assert unstable_cue_lines(translations, cues) == set()

    def test_consistent_speaker_across_sentence_is_stable(self) -> None:
        translations = {1: "Ngươi không những hết ngốc,", 2: "mà còn thiên hạ vô địch."}
        cues = _cues({1: ("红莲女帝", "宋钱"), 2: ("红莲女帝", "宋钱")})
        assert unstable_cue_lines(translations, cues) == set()

    def test_empty_cues(self) -> None:
        assert unstable_cue_lines({1: "a,"}, "") == set()


def _scene_with_real_case() -> tuple[dict[int, str], dict[int, str], str]:
    """Cảnh An Tuyết Mạn nói với Từ Tuấn Hào (xưng 'em'), chèn đúng ca thật 1827-1829."""
    sources = {i: "我知道" for i in range(1820, 1827)}
    translations = {i: "em biết mà." for i in range(1820, 1827)}
    pairs = {i: ("安雪漫", "徐俊豪") for i in range(1820, 1827)}
    sources.update({1827: "之前打伤我哥", 1828: "打断我妈的腿", 1829: "还逼我高中辍学"})
    translations.update(
        {
            1827: "Đánh bị thương anh tôi trước đây,",
            1828: "đánh gãy chân mẹ tôi,",
            1829: "còn ép tôi phải bỏ học cấp ba.",
        }
    )
    pairs.update(
        {1827: ("宋瑶", "安雪漫"), 1828: ("安雪漫", "徐俊豪"), 1829: ("安雪漫", "徐俊豪")}
    )
    for i in range(1830, 1834):
        sources[i] = "我知道"
        translations[i] = "em biết mà."
        pairs[i] = ("安雪漫", "徐俊豪")
    return sources, translations, _cues(pairs)


class TestGuardSkipsUnstableLines:
    def test_real_case_1828_is_not_forced(self) -> None:
        sources, translations, cues = _scene_with_real_case()
        fixes = find_address_rule_fixes(sources, translations, cues, CHARACTERS)
        assert {fix.index for fix in fixes} & {1828, 1829} == set()

    def test_violation_explains_why(self) -> None:
        sources, translations, cues = _scene_with_real_case()
        reasons = {
            v.index: v.reason
            for v in find_address_rule_violations(sources, translations, cues, CHARACTERS)
        }
        assert "gán nhầm người nói" in reasons[1829]

    def test_stable_line_in_same_scene_is_still_fixed(self) -> None:
        sources, translations, cues = _scene_with_real_case()
        translations[1831] = "tôi biết mà."
        fixes = find_address_rule_fixes(sources, translations, cues, CHARACTERS)
        assert [(f.index, f.text_after) for f in fixes] == [(1831, "em biết mà.")]
