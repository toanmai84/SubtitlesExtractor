"""[v3.23.568] Hồi quy: quy tắc xưng hô gọi người nghe bằng TÊN GỐC bị bỏ sót toàn bộ.

Chuẩn đoán thật v3.23.567 (phân tích bằng gemini-3.6-flash, dịch bằng 3.5-flash-lite):
bảng nhân vật ghi quy tắc kiểu ``với 柳慕云 tự xưng 'tôi' gọi 'chị Mộ Vân'`` — người
nghe là TÊN GỐC. Bộ tra cũ chỉ tra theo phiên âm nên đọc được **6/58** mệnh đề (20 cặp,
6 người nói): khâu dịch không nhận quy tắc từng dòng, lưới xưng hô không kiểm được gì.
Bản dịch ra ``张依贝`` với Tống Tiền "tôi" 8 lần / "em" 8 lần trong CÙNG cảnh dù bảng
chốt "em". Tra cả tên gốc: 57/58 mệnh đề, 164 cặp, 34 người nói; lưới sửa đúng 47 dòng.

Kèm lỗi thứ hai: mệnh đề nhiều người nghe ("với Từ Tuấn Hào và An Tuyết Mạn tự xưng 'tôi'
gọi 'các người'") bị gắn cho người ĐẦU và quy tắc dính nguyên đuôi "và An Tuyết Mạn…".
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from subtitles_extractor.application.services.address_rule_guard import (
    find_address_rule_fixes,
)
from subtitles_extractor.application.services.context_diagnostics import (
    build_context_diagnostics,
)
from subtitles_extractor.application.services.roster_format import (
    address_clauses,
    address_rule_map,
    lookup_address_rule,
)

REAL_ROSTER = (
    "宋钱 / 宋神医 (Tống Tiền) [nam, ~23] — nhân vật chính. Xưng hô: với 柳慕云 tự xưng "
    "'tôi' gọi 'chị Mộ Vân' (danh xưng: Liễu tiểu thư); với 徐俊豪 tự xưng 'tao' gọi "
    "'mày' (danh xưng: ông); với 宋母 tự xưng 'con' gọi 'mẹ'; với 张依贝 tự xưng 'anh' "
    "gọi 'em'\n"
    "宋母 (Mẹ Tống) [nữ, ~50] — mẹ. Xưng hô: với 宋钱 và 宋瑶 tự xưng 'mẹ' gọi 'con'.\n"
    "宋瑶 (Tống Dao) [nữ, ~19] — em gái. Xưng hô: với 宋钱 tự xưng 'em' gọi 'anh'; "
    "với Từ Tuấn Hào và An Tuyết Mạn tự xưng 'tôi' gọi 'các người'\n"
    "徐俊豪 (Từ Tuấn Hào) [nam, ~25] — phản diện. Xưng hô: với 宋钱 trước khi bị sa thải "
    "tự xưng 'tao' gọi 'mày', sau đó tự xưng 'cháu' gọi 'ông'\n"
    "安雪漫 (An Tuyết Mạn) [nữ, ~23] — bạn gái cũ\n"
    "张依贝 (Trương Y Bối) [nữ, ~23] — bạn học. Xưng hô: với 宋钱 tự xưng 'em' gọi 'anh "
    "Tống Tiền' / 'sư phụ'; với 柳慕云 / 赵可可 tự xưng 'tôi' gọi 'cô'.\n"
    "柳慕云 (Liễu Mộ Vân) [nữ, ~25] — tổng giám đốc\n"
    "赵可可 / 可可 (Triệu Khả Khả) [nữ, ~22] — trợ lý"
)


def _cues(pairs: dict[int, tuple[str, str]]) -> str:
    return json.dumps(
        [{"id": index, "spk": spk, "to": to} for index, (spk, to) in pairs.items()],
        ensure_ascii=False,
    )


class TestOriginalNameListener:
    def test_rule_naming_listener_by_original_name_is_read(self) -> None:
        rules = address_rule_map(REAL_ROSTER)
        assert rules[("张依贝", "宋钱")].startswith("tự xưng 'em' gọi 'anh Tống Tiền'")
        assert rules[("宋钱", "柳慕云")].startswith("tự xưng 'tôi' gọi 'chị Mộ Vân'")
        assert rules[("宋钱", "宋母")] == "tự xưng 'con' gọi 'mẹ'"

    def test_cue_label_with_aliases_finds_the_rule(self) -> None:
        rules = address_rule_map(REAL_ROSTER)
        rule = lookup_address_rule(rules, "张依贝", "宋钱 / 宋神医")
        assert rule.startswith("tự xưng 'em'")

    def test_condition_after_original_name_is_kept(self) -> None:
        rule = address_rule_map(REAL_ROSTER)[("徐俊豪", "宋钱")]
        assert rule.startswith("trước khi bị sa thải tự xưng 'tao' gọi 'mày'")
        assert "sau đó tự xưng 'cháu'" in rule

    def test_every_clause_of_the_real_roster_is_resolved(self) -> None:
        clauses = address_clauses(REAL_ROSTER)
        assert clauses and all(clause.listeners for clause in clauses)


class TestGroupListener:
    def test_singular_group_rule_applies_to_each_member(self) -> None:
        rules = address_rule_map(REAL_ROSTER)
        assert rules[("宋母", "宋钱")] == "tự xưng 'mẹ' gọi 'con'"
        assert rules[("宋母", "宋瑶")] == "tự xưng 'mẹ' gọi 'con'"
        assert rules[("张依贝", "赵可可")] == "tự xưng 'tôi' gọi 'cô'"

    def test_plural_group_rule_is_not_given_to_one_person(self) -> None:
        rules = address_rule_map(REAL_ROSTER)
        assert ("宋瑶", "徐俊豪") not in rules
        assert ("宋瑶", "安雪漫") not in rules

    def test_group_tail_does_not_leak_into_the_rule(self) -> None:
        roster = (
            "宋母 (Mẹ Tống) — mẹ. Xưng hô: với Tống Tiền và Tống Dao tự xưng 'mẹ' "
            "gọi 'con'\n"
            "宋钱 (Tống Tiền) — con\n宋瑶 (Tống Dao) — con gái"
        )
        assert address_rule_map(roster)[("宋母", "宋钱")] == "tự xưng 'mẹ' gọi 'con'"

    def test_pair_rule_beats_group_rule(self) -> None:
        roster = (
            "宋母 (Mẹ Tống) — mẹ. Xưng hô: với 宋钱 và 宋瑶 tự xưng 'tôi' gọi 'con'; "
            "với 宋钱 tự xưng 'mẹ' gọi 'con trai'\n"
            "宋钱 (Tống Tiền) — con\n宋瑶 (Tống Dao) — con"
        )
        rules = address_rule_map(roster)
        assert rules[("宋母", "宋钱")] == "tự xưng 'mẹ' gọi 'con trai'"
        assert rules[("宋母", "宋瑶")] == "tự xưng 'tôi' gọi 'con'"

    def test_chinese_conjunction_is_split(self) -> None:
        roster = (
            "宋母 (Mẹ Tống) — mẹ. Xưng hô: với 宋钱和宋瑶 tự xưng 'mẹ' gọi 'con'\n"
            "宋钱 (Tống Tiền) — con\n宋瑶 (Tống Dao) — con"
        )
        rules = address_rule_map(roster)
        assert ("宋母", "宋钱") in rules and ("宋母", "宋瑶") in rules


class TestGuardSeesTheRule:
    def test_real_scene_is_fixed(self) -> None:
        """Ca thật 593-604: 张依贝 nói với Tống Tiền, bảng chốt 'em', dịch lẫn 'tôi'."""
        pairs = {i: ("张依贝", "宋钱 / 宋神医") for i in range(593, 605)}
        sources = {i: "我手镯不小心落河里了" for i in pairs}
        translations = {i: "Vòng tay của em rơi xuống nước," for i in pairs}
        translations[593] = "Vòng tay của tôi rơi xuống nước,"
        translations[604] = "Tôi lên bờ rồi,"
        fixes = find_address_rule_fixes(sources, translations, _cues(pairs), REAL_ROSTER)
        assert [(f.index, f.text_after) for f in fixes] == [
            (593, "Vòng tay của em rơi xuống nước,"),
            (604, "Em lên bờ rồi,"),
        ]


class TestAddressPayload:
    def test_translation_batch_carries_the_rule(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (  # noqa: E501
            GeminiSubtitleTranslator,
        )

        translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        translator._active_characters = REAL_ROSTER
        batch = [SimpleNamespace(index=1, speaker="张依贝", addressee="宋钱 / 宋神医")]
        payload = [{"line_no": 1}]
        translator._attach_address_rules_once(payload, batch)
        assert payload[0]["address"].startswith("tự xưng 'em'")


class TestDiagnostics:
    @staticmethod
    def _report(characters: str) -> dict:
        context = SimpleNamespace(
            target_lang="Vietnamese",
            source_lang="zh",
            overview="Tóm tắt.",
            characters=characters,
            glossary="",
            visual_cues="",
            enable_tags=False,
            include_desc=True,
        )
        return build_context_diagnostics(context=context, source_events=[])

    def test_resolution_stats_are_reported(self) -> None:
        stats = self._report(REAL_ROSTER)["roster"]["address_rules"]
        assert stats["clauses"] == stats["resolved"] >= 8
        assert stats["unresolved_listeners"] == []

    def test_mostly_unresolved_rules_are_flagged(self) -> None:
        roster = "宋钱 (Tống Tiền) — chính. Xưng hô: " + "; ".join(
            f"với Người Lạ {i} tự xưng 'tôi' gọi 'anh'" for i in range(6)
        )
        report = self._report(roster)
        codes = {flag["code"] for flag in report["quality_flags"]}
        assert "address_rules_unresolved" in codes

    def test_resolved_roster_is_not_flagged(self) -> None:
        codes = {flag["code"] for flag in self._report(REAL_ROSTER)["quality_flags"]}
        assert "address_rules_unresolved" not in codes
