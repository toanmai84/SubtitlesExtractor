"""[v3.23.555] Hồi quy: chuẩn hoá xưng hô theo ĐA SỐ TRONG CẢNH + hai chốt chống báo nhầm.

Chuẩn đoán thật v3.23.554 (phim 3902 dòng): lưới ép được 8 dòng, nhưng hai nhóm lớn
nhất vẫn lọt:

1. ``张依贝↔宋钱`` — hai cách xưng bị TRỘN trong cùng một cảnh ('em' 12 dòng / 'tôi' 4
   dòng). Chế độ quy tắc bó tay vì roster cho hai phương án ('anh/tôi'), mà roster cho
   phép hai cách KHÔNG có nghĩa là được trộn trong một cảnh.
2. ``宋钱→徐俊豪`` dòng 2908-2951 — cảnh đối đầu, phải xưng 'tao' nhưng cả 6 dòng ra 'tôi'
   nên chốt "cả cảnh đều lệch" tưởng nhầm là phim đổi giọng. Thực ra các cảnh SAU vẫn
   'tao' — phim không đổi gì, chỉ là model dịch ẩu một cảnh.

Kèm hai ca báo lệch OAN đã sửa: 'mẹ' (dòng 559) và từ ghép "anh trai" (dòng 3298).
"""

from __future__ import annotations

import json

from subtitles_extractor.application.services.address_rule_guard import (
    allowed_self_forms,
    find_address_rule_fixes,
    find_address_rule_violations,
)

# Roster rút gọn từ phim thật: 宋钱 với 张依贝 có HAI phương án, với Từ Tuấn Hào chỉ một.
CHARACTERS = (
    "宋钱 (Tống Tiền) [nam, ~22] — nhân vật chính; xưng hô: với Trương Y Bối tự xưng "
    "'anh/tôi' gọi 'em'; với Từ Tuấn Hào tự xưng 'tao' gọi 'mày'; với Hứa Đào Hoa tự "
    "xưng 'em' gọi 'chị Đào Hoa'\n"
    "张依贝 (Trương Y Bối) [nữ, ~20] — hàng xóm; xưng hô: với Tống Tiền tự xưng 'em' "
    "gọi 'anh'\n"
    "王麻子 (Vương Ma Tử) [nam, ~40] — côn đồ trong làng; xưng hô: với Tống Tiền tự "
    "xưng 'tao' gọi 'mày'\n"
    "徐俊豪 (Từ Tuấn Hào) [nam, ~25] — phản diện\n"
    "许桃花 (Hứa Đào Hoa) [nữ, ~25] — góa phụ\n"
    "宋母 (Mẹ Tống) [nữ, ~50] — mẹ Tống Tiền; xưng hô: với Tống Dao tự xưng 'mẹ' "
    "gọi 'con'\n"
    "宋瑶 (Tống Dao) [nữ, ~18] — em gái; xưng hô: với Mẹ Tống tự xưng 'con' gọi 'mẹ'\n"
)


def _cues(entries: list[tuple[int, str, str]]) -> str:
    return json.dumps(
        [{"id": index, "spk": spk, "to": to} for index, spk, to in entries],
        ensure_ascii=False,
    )


def _pair_cues(
    speaker: str, addressee: str, indices: list[int]
) -> list[tuple[int, str, str]]:
    return [(index, speaker, addressee) for index in indices]


class TestSceneMajority:
    """Ca thật ``宋钱→张依贝``: roster cho HAI phương án nên chế độ quy tắc bó tay.

    Trong một cảnh vẫn chỉ được dùng MỘT cách: 'anh' 8 dòng / 'tôi' 4 dòng là trôi.
    """

    @staticmethod
    def _scene() -> tuple[dict[int, str], dict[int, str], str]:
        indices = list(range(580, 596))
        sources = {i: "我知道了" for i in indices}
        translations = {i: "anh biết rồi" for i in indices}
        for i in (583, 585, 587, 589):
            translations[i] = "tôi biết rồi"
        cues = _cues(_pair_cues("宋钱", "张依贝", indices))
        return sources, translations, cues

    def test_minority_is_normalised_to_majority(self) -> None:
        sources, translations, cues = self._scene()
        fixes = find_address_rule_fixes(sources, translations, cues, CHARACTERS)
        assert [f.index for f in fixes] == [583, 585, 587, 589]
        assert all(f.text_after == "anh biết rồi" for f in fixes)

    def test_majority_outside_roster_options_is_not_trusted(self) -> None:
        """Ca thật v3.23.548: côn đồ quát 'câm miệng cho TAO' suýt thành 'tôi'."""
        indices = list(range(1300, 1312))
        sources = {i: "我说了" for i in indices}
        translations = {i: "tôi nói rồi" for i in indices}
        for i in (1305, 1309):
            translations[i] = "tao nói rồi"
        cues = _cues(_pair_cues("王麻子", "宋钱", indices))
        assert find_address_rule_fixes(sources, translations, cues, CHARACTERS) == []

    def test_switch_at_end_of_scene_is_kept(self) -> None:
        """Thiểu số nằm TRỌN ở cuối cảnh = đổi giọng có chủ ý, không phải trôi."""
        indices = list(range(580, 596))
        sources = {i: "我知道了" for i in indices}
        translations = {i: "anh biết rồi" for i in indices}
        for i in (592, 593, 594, 595):
            translations[i] = "tôi biết rồi"
        cues = _cues(_pair_cues("宋钱", "张依贝", indices))
        assert find_address_rule_fixes(sources, translations, cues, CHARACTERS) == []

    def test_short_scene_is_not_enough_evidence(self) -> None:
        indices = list(range(580, 585))
        sources = {i: "我知道了" for i in indices}
        translations = {i: "anh biết rồi" for i in indices}
        translations[582] = "tôi biết rồi"
        cues = _cues(_pair_cues("宋钱", "张依贝", indices))
        assert find_address_rule_fixes(sources, translations, cues, CHARACTERS) == []

    def test_no_clear_majority_means_no_change(self) -> None:
        indices = list(range(580, 592))
        sources = {i: "我知道了" for i in indices}
        translations = {
            i: ("anh biết rồi" if step % 2 else "tôi biết rồi")
            for step, i in enumerate(indices)
        }
        cues = _cues(_pair_cues("宋钱", "张依贝", indices))
        assert find_address_rule_fixes(sources, translations, cues, CHARACTERS) == []


class TestRuleReturnsLater:
    """Cảnh lệch HOÀN TOÀN: sửa hay không tuỳ phim có quay lại quy tắc hay không."""

    @staticmethod
    def _film(later_form: str) -> tuple[dict[int, str], dict[int, str], str]:
        confrontation = list(range(2900, 2906))
        later = list(range(3050, 3056))  # cách xa > 40 dòng nên là CẢNH KHÁC
        sources = {i: "我今天来" for i in confrontation + later}
        translations = {i: "tôi đến đây hôm nay" for i in confrontation}
        translations.update({i: f"{later_form} đến đây hôm nay" for i in later})
        cues = _cues(_pair_cues("宋钱", "徐俊豪", confrontation + later))
        return sources, translations, cues

    def test_scene_is_fixed_when_film_returns_to_the_rule(self) -> None:
        """Ca thật 2908-2951: cảnh đối đầu ra 'tôi', các cảnh SAU vẫn 'tao'."""
        sources, translations, cues = self._film("tao")
        fixes = find_address_rule_fixes(sources, translations, cues, CHARACTERS)
        assert [f.index for f in fixes] == list(range(2900, 2906))
        assert all(f.pronoun_expected == "tao" for f in fixes)

    def test_permanent_switch_is_left_alone(self) -> None:
        """Ca thật 3815: quan hệ đổi hẳn, phim không quay lại — roster đã lỗi thời."""
        sources, translations, cues = self._film("tôi")
        assert find_address_rule_fixes(sources, translations, cues, CHARACTERS) == []

    def test_last_scene_of_a_pair_is_never_forced(self) -> None:
        indices = list(range(3880, 3886))
        sources = {i: "有我在啊" for i in indices}
        translations = {i: "có anh ở đây mà" for i in indices}
        cues = _cues(_pair_cues("宋钱", "许桃花", indices))
        assert find_address_rule_fixes(sources, translations, cues, CHARACTERS) == []


class TestFalsePositivesRemoved:
    def test_kinship_self_form_is_recognised(self) -> None:
        """Ca thật dòng 559: "Mẹ thực sự đứng lên được rồi con ơi!" xưng ĐÚNG là 'mẹ'."""
        indices = list(range(555, 565))
        sources = {i: "我真的站起来了" for i in indices}
        translations = {i: "Mẹ thực sự đứng lên được rồi con ơi!" for i in indices}
        cues = _cues(_pair_cues("宋母", "宋瑶", indices))
        assert find_address_rule_violations(sources, translations, cues, CHARACTERS) == []

    def test_compound_noun_is_not_a_pronoun(self) -> None:
        """Ca thật dòng 3298: "anh trai đi dự" — danh từ ghép, không phải xưng hô."""
        indices = list(range(3295, 3305))
        sources = {i: "我哥去参加" for i in indices}
        translations = {i: "anh trai đi dự" for i in indices}
        cues = _cues(_pair_cues("宋瑶", "宋母", indices))
        assert find_address_rule_violations(sources, translations, cues, CHARACTERS) == []


class TestAllowedForms:
    def test_both_options_are_allowed(self) -> None:
        assert allowed_self_forms("tự xưng 'anh/tôi' gọi 'em'") == {"anh", "tôi"}

    def test_phase_rule_allows_both_phases(self) -> None:
        rule = "trước xưng 'tôi' gọi 'cô Liễu' (sau xưng 'em' gọi 'chị Mộ Vân')"
        assert allowed_self_forms(rule) == {"tôi", "em"}

    def test_title_is_not_a_pronoun(self) -> None:
        assert allowed_self_forms("tự xưng 'Tống thần y' gọi 'quý khách'") == set()
