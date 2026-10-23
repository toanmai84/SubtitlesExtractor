"""[v3.23.551] Hồi quy cho chuẩn đoán thật (phim 3902 dòng, v3.23.550).

1. Mở lại dự án hiện bản dịch CŨ: giai đoạn 'final' chỉ được ghi khi sửa tay nhưng lại được ƯU TIÊN
   khi nạp lại — file thật có 'final' từ 3 ngày trước, chỉ khớp 29% bản vừa dịch.
2. Xưng hô trôi TRONG CÙNG CẢNH dù roster đã chốt rõ (``张依贝`` với Tống Tiền: "tự xưng 'em' gọi
   'anh'", vậy mà dòng 601-655 ra "tôi/cậu"). Bộ đo cũ gộp cả phim nên vừa báo nhầm đổi giọng
   hợp lý giữa các cảnh, vừa bỏ sót trôi thật giữa hai đại từ tự xưng.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from types import SimpleNamespace

from subtitles_extractor.application.services.roster_format import (
    address_rule_map,
    lookup_address_rule,
)
from subtitles_extractor.application.services.translation_diagnostics import address_drift


@dataclass
class _Event:
    index: int
    text: str


def _cues(pairs: dict[int, tuple[str, str]]) -> str:
    return json.dumps([{"id": i, "spk": s, "to": t} for i, (s, t) in pairs.items()])


# ── 1. 'final' phải là bản vừa dịch ──────────────────────────────────────────
class TestFinalStagePersisted:
    def test_full_translation_overwrites_final(self) -> None:
        from subtitles_extractor.presentation.view_models.translate_page_view_model import (
            TranslatePageViewModel,
        )

        source = inspect.getsource(TranslatePageViewModel._on_worker_finished)
        stages_at = source.index("self._persist_stage_outputs()")
        final_at = source.index("self._persist_current_result()")
        assert final_at > stages_at, "Phải ghi 'final' SAU khi lưu các giai đoạn"


# ── 2a. Bộ đo xưng hô theo vai + theo cảnh ───────────────────────────────────
class TestAddressDrift:
    def test_drift_within_scene_between_self_pronouns(self) -> None:
        """Ca thật: cùng cảnh, tự xưng "em" rồi "tôi"."""
        sources = {
            1: "我手镯掉了", 2: "我去捡", 3: "我上岸了", 4: "我记得",
            5: "我冷", 6: "我好累",
        }
        texts = {
            1: "Vòng tay của em rơi rồi.", 2: "Em đi nhặt.", 3: "Tôi lên bờ rồi.",
            4: "Tôi nhớ là…", 5: "Em lạnh quá.", 6: "Tôi mệt quá.",
        }
        cues = _cues({i: ("张依贝", "宋钱") for i in sources})
        findings = address_drift(cues, sources, [_Event(i, t) for i, t in texts.items()])
        assert len(findings) == 1
        assert findings[0]["role"] == "self"
        assert set(findings[0]["variants"]) == {"em", "tôi"}

    def test_switch_between_scenes_is_not_drift(self) -> None:
        """Ca thật 王麻子: ngoài suối "tao/mày", nửa đêm "anh/em" — hai cảnh khác nhau."""
        sources = {1: "我来陪你", 2: "我告诉你", 3: "我不走",
                   200: "我知道你没睡", 201: "我就知道", 202: "我等你"}
        texts = {1: "Tao đến chơi với mày.", 2: "Tao nói cho mày biết.", 3: "Tao không đi.",
                 200: "Anh biết em chưa ngủ.", 201: "Anh biết ngay mà.", 202: "Anh đợi em."}
        cues = _cues({i: ("王麻子", "许桃花") for i in sources})
        assert address_drift(cues, sources, [_Event(i, t) for i, t in texts.items()]) == []

    def test_third_person_and_plural_are_ignored(self) -> None:
        """"cậu ấy" là ngôi ba, "chúng ta"/我们 là số nhiều — không phải tự xưng/gọi."""
        sources = {1: "我给他烧纸", 2: "我怕他", 3: "我们家最棒", 4: "我们走",
                   5: "我说了", 6: "我知道"}
        texts = {1: "Tôi đốt vàng mã cho cậu ấy.", 2: "Tôi sợ cậu ấy.",
                 3: "Nhà chúng ta giỏi nhất.", 4: "Chúng ta đi.",
                 5: "Tôi nói rồi.", 6: "Tôi biết."}
        cues = _cues({i: ("许桃花", "王麻子") for i in sources})
        assert address_drift(cues, sources, [_Event(i, t) for i, t in texts.items()]) == []

    def test_address_role_uses_you_lines(self) -> None:
        sources = {1: "你好厉害", 2: "你别走", 3: "你看", 4: "你过来", 5: "你说", 6: "你坐"}
        texts = {1: "Anh giỏi quá.", 2: "Anh đừng đi.", 3: "Cậu xem.",
                 4: "Cậu qua đây.", 5: "Anh nói đi.", 6: "Cậu ngồi đi."}
        cues = _cues({i: ("张依贝", "宋钱") for i in sources})
        findings = address_drift(cues, sources, [_Event(i, t) for i, t in texts.items()])
        assert findings and findings[0]["role"] == "address"
        assert findings[0]["examples"]["anh"] == 1

    def test_single_stray_variant_is_noise(self) -> None:
        sources = {i: "我去" for i in range(1, 7)}
        texts = {i: "Em đi." for i in range(1, 6)}
        texts[6] = "Tôi đi."
        cues = _cues({i: ("张依贝", "宋钱") for i in sources})
        assert address_drift(cues, sources, [_Event(i, t) for i, t in texts.items()]) == []


# ── 2b. Tra sẵn quy tắc xưng hô ──────────────────────────────────────────────
ROSTER = (
    "张依贝 (Trương Y Bối) [nữ, ~22] — hoa khôi; xưng hô: với Tống Tiền tự xưng 'em' gọi "
    "'anh Tống Tiền'; với Xuân Hoa tự xưng 'em' gọi 'chị dâu Xuân Hoa'\n"
    "宋钱 (Tống Tiền) [nam, ~22] — nhân vật chính; xưng hô: với Trương Y Bối tự xưng 'anh' "
    "gọi 'em'\n"
    "向城 / 向大师 (Hướng Thành / Hướng Đại Sư) [nam, ~45] — võ sư; xưng hô: với Tống Tiền "
    "trước khi thua tự xưng 'ta' gọi 'ngươi', sau khi thua xưng 'tôi' gọi 'Tống tiên sinh'\n"
    "柳杰 / 柳河 (Liễu Kiệt / Liễu Hà) [nam] — anh họ; xưng hô: với Tống Tiền tự xưng 'tao' "
    "gọi 'mày'\n"
    "春花 (Xuân Hoa) [nữ] — chị dâu"
)


class TestAddressRuleMap:
    def test_basic_pair_resolved_by_original_names(self) -> None:
        rules = address_rule_map(ROSTER)
        assert rules[("张依贝", "宋钱")] == "tự xưng 'em' gọi 'anh Tống Tiền'"
        assert rules[("宋钱", "张依贝")] == "tự xưng 'anh' gọi 'em'"

    def test_phase_conditioned_rule_kept_whole(self) -> None:
        """Ca thật: "với Tống Tiền TRƯỚC KHI THUA tự xưng…" — tên người nghe dính điều kiện."""
        rule = address_rule_map(ROSTER)[("向城", "宋钱")]
        assert rule.startswith("trước khi thua tự xưng 'ta'")
        assert "sau khi thua xưng 'tôi'" in rule

    def test_full_alias_string_is_a_key(self) -> None:
        assert ("向城 / 向大师", "宋钱") in address_rule_map(ROSTER)

    def test_lookup_tolerates_alias_order(self) -> None:
        """Cues ghi "柳河 / 柳杰", roster ghi "柳杰 / 柳河"."""
        rules = address_rule_map(ROSTER)
        assert lookup_address_rule(rules, "柳河 / 柳杰", "宋钱") == "tự xưng 'tao' gọi 'mày'"

    def test_self_pair_returns_nothing(self) -> None:
        """Cues gán nhầm nói với chính mình (độc thoại) — không có quy tắc."""
        assert lookup_address_rule(address_rule_map(ROSTER), "宋钱", "宋钱") == ""

    def test_character_without_rule(self) -> None:
        assert lookup_address_rule(address_rule_map(ROSTER), "春花", "宋钱") == ""


class TestAddressRuleInPayload:
    def _translator(self):
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )

        translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        translator._active_characters = ROSTER
        return translator

    def _line(self, index: int, speaker: str, addressee: str):
        return SimpleNamespace(index=index, speaker=speaker, addressee=addressee)

    def test_rule_attached_once_per_pair_per_batch(self) -> None:
        """Chỉ dòng ĐẦU mỗi cặp được gắn — gắn mọi dòng tốn thêm ~11% ký tự prompt."""
        translator = self._translator()
        batch = [
            self._line(1, "张依贝", "宋钱"),
            self._line(2, "张依贝", "宋钱"),
            self._line(3, "宋钱", "张依贝"),
            self._line(4, "张依贝", "宋钱"),
        ]
        payload = [{"line_no": line.index} for line in batch]
        translator._attach_address_rules_once(payload, batch)
        assert [("address" in item) for item in payload] == [True, False, True, False]
        assert payload[0]["address"] == "tự xưng 'em' gọi 'anh Tống Tiền'"

    def test_lines_without_speaker_untouched(self) -> None:
        translator = self._translator()
        batch = [self._line(1, "", "宋钱"), self._line(2, "张依贝", "")]
        payload = [{"line_no": 1}, {"line_no": 2}]
        translator._attach_address_rules_once(payload, batch)
        assert all("address" not in item for item in payload)

    def test_context_payload_does_not_carry_rules(self) -> None:
        """Dòng ngữ cảnh (không cần dịch) không được gắn quy tắc — tốn token vô ích."""
        source = inspect.getsource(
            self._translator().__class__._line_to_payload
        )
        assert "_attach_address_rule" not in source

    def test_prompt_explains_address_field(self) -> None:
        import subtitles_extractor.infrastructure.translation.gemini_translation_adapter as m

        source = inspect.getsource(m)
        assert "TRƯỜNG 'address' là quy tắc xưng hô ĐÃ TRA SẴN" in source
        assert "GIỮ cách mới cho HẾT cảnh đó" in source
