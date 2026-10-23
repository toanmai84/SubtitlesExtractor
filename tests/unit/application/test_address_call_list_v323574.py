"""[v3.23.574] Hồi quy: xưng hô ghi kiểu "tự xưng 'S', gọi <Tên> 'A'" → 0 quy tắc.

Chuẩn đoán thật phim "灭门惨案II借种" (phân tích bằng gemini-3.8-flash, 17 nhân vật):
mọi nhân vật đều ghi "Cách xưng hô: tự xưng 'em', gọi Chân Kiếm Thanh 'Kiếm ca/anh',
gọi Chu Chính Quân 'A Quân'." thay vì "với <Tên> tự xưng … gọi …". Bộ đọc cũ chỉ hiểu
dạng sau nên ``address_rules.clauses`` = 0: khâu dịch không được gắn quy tắc xưng hô
nào, lưới xưng hô và số liệu chuẩn đoán mù hoàn toàn. Cùng kiểu viết (và "tự xưng là
'x'", "với X gọi là 'y'") có ở 4/49 bảng nhân vật thật khác.

Kèm theo: giai đoạn tinh chỉnh cùng phim đẩy lệch dòng 706-709 (dòng 709 mang câu bịa
"Cô không phải nói muốn xin giống sao?…") mà bộ dò bỏ sót vì dòng 707 bị cắt lại câu
nằm GIỮA hai dòng bằng chứng mạnh.
"""

from __future__ import annotations

import json
from dataclasses import replace
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
from subtitles_extractor.application.services.stage_shift_guard import find_stage_shifts
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)

# Nguyên văn các dòng của bảng nhân vật thật (rút gọn phần mô tả).
FILM_ROSTER = "\n".join([
    "龍傳人 (Long Truyền Nhân) [nam, ~40] — Trưởng đồn công an Đông Long. Cách xưng hô: "
    "tự xưng 'tôi/anh', gọi cấp dưới 'cô/cậu', gọi dân làng bằng tên riêng hoặc "
    "'ông/bà'.",
    "余小蓮 (Dư Tiểu Liên) [nữ, ~25] — Nữ công an tại đồn Đông Long. Xưng hô: tự xưng "
    "'tôi/em', gọi Long Truyền Nhân là 'Trưởng đồn'.",
    "秦可秀 (Tần Khả Tú) [nữ, ~25] — Vợ Chân Kiếm Thanh. Cách xưng hô: tự xưng 'em', gọi "
    "Chân Kiếm Thanh 'Kiếm ca/anh', gọi Chu Chính Quân 'A Quân'.",
    "洪震 (Hồng Chấn) [nam, trung niên] — Tên ác ôn. Cách xưng hô: tự xưng 'tao/ba', gọi "
    "Hồng Chí Quyền 'con'.",
    "朱正軍 (Chu Chính Quân) [nam, ~30] — Cựu chiến binh. Cách xưng hô: tự xưng "
    "'tôi/em', gọi Chân Kiếm Thanh 'Kiếm ca/anh', gọi Tần Khả Tú 'chị dâu/A Tú'.",
    "甄劍聲 (Chân Kiếm Thanh) [nam, ~30] — Chồng Tần Khả Tú. Cách xưng hô: tự xưng "
    "'anh/tôi', gọi Tần Khả Tú 'A Tú/em', gọi Chu Chính Quân 'A Quân/chú'.",
    "洪志權 (Hồng Chí Quyền) [nam, ~20] — Con trai Hồng Chấn. Cách xưng hô: gọi "
    "Hồng Chấn 'ba/lão già'.",
    "醫生 (Bác sĩ) [nam, trung niên] — Bác sĩ thị trấn. Xưng hô: tự xưng 'tôi'.",
])


def _rules() -> dict[tuple[str, str], str]:
    return address_rule_map(FILM_ROSTER)


class TestCallListFormat:
    def test_every_listed_listener_becomes_a_clause(self) -> None:
        clauses = address_clauses(FILM_ROSTER)
        assert len(clauses) == 11
        assert sum(1 for clause in clauses if clause.listeners) == 9

    def test_shared_self_form_is_attached_to_each_listener(self) -> None:
        rules = _rules()
        assert rules[("秦可秀", "甄劍聲")] == "tự xưng 'em' gọi 'Kiếm ca/anh'"
        assert rules[("秦可秀", "朱正軍")] == "tự xưng 'em' gọi 'A Quân'"
        assert rules[("洪震", "洪志權")] == "tự xưng 'tao/ba' gọi 'con'"

    def test_address_without_self_form(self) -> None:
        assert _rules()[("洪志權", "洪震")] == "gọi 'ba/lão già'"
        assert _rules()[("余小蓮", "龍傳人")] == "tự xưng 'tôi/em' gọi 'Trưởng đồn'"

    def test_group_listeners_stay_unresolved(self) -> None:
        clauses = address_clauses(FILM_ROSTER)
        unresolved = [c.raw_listener for c in clauses if not c.listeners]
        assert unresolved == ["cấp dưới", "dân làng bằng tên riêng hoặc"]

    def test_self_form_alone_is_not_a_clause(self) -> None:
        assert not any(c.speakers[0] == "醫生" for c in address_clauses(FILM_ROSTER))


class TestLaForms:
    def test_xung_la_in_the_canonical_format(self) -> None:
        """Bảng cũ cùng phim: "với Tần Khả Tú tự xưng là 'tôi/anh' gọi 'em/…'"."""
        roster = FILM_ROSTER + (
            "\n阿積 (A Tích) [nam, ~25] — Nhân viên kỹ thuật. Xưng hô: với Tần Khả Tú tự "
            "xưng là 'tôi' gọi là 'chị'"
        )
        assert address_rule_map(roster)[("阿積", "秦可秀")] == "tự xưng 'tôi' gọi 'chị'"

    def test_with_listener_but_no_self_form(self) -> None:
        """Bảng nhân vật thật (phim khác): "với Tống Tiền gọi là 'Tống thần y'"."""
        roster = (
            "宋钱 (Tống Tiền) [nam, ~23] — nhân vật chính\n"
            "孙院长 (Viện trưởng Tôn) [nam, ~55] — Viện trưởng; xưng hô: với Liễu Mộ Vân "
            "xưng 'tôi' gọi 'Liễu tiểu thư', với Tống Tiền gọi là 'Tống thần y'\n"
            "柳慕云 (Liễu Mộ Vân) [nữ, ~25] — tổng giám đốc"
        )
        rules = address_rule_map(roster)
        assert rules[("孙院长", "宋钱")] == "gọi 'Tống thần y'"
        assert rules[("孙院长", "柳慕云")] == "xưng 'tôi' gọi 'Liễu tiểu thư'"

    def test_bare_la_is_never_a_listener(self) -> None:
        roster = "宋钱 (Tống Tiền) [nam] — chính; xưng hô: gọi là 'Tống ca'"
        assert address_clauses(roster) == []


class TestCanonicalFormatIsUnchanged:
    def test_canonical_rules(self) -> None:
        roster = (
            "宋钱 (Tống Tiền) [nam, ~23] — nhân vật chính. Xưng hô: với 许桃花 tự xưng "
            "'em' gọi 'chị' (danh xưng: chị Đào Hoa); với 张依贝 tự xưng 'anh' gọi 'em'\n"
            "许桃花 (Hứa Đào Hoa) [nữ, ~24] — chị dâu\n"
            "张依贝 (Trương Y Bối) [nữ, ~23] — bạn học"
        )
        rules = address_rule_map(roster)
        assert rules[("宋钱", "许桃花")] == (
            "tự xưng 'em' gọi 'chị' (danh xưng: chị Đào Hoa)"
        )
        assert rules[("宋钱", "张依贝")] == "tự xưng 'anh' gọi 'em'"
        assert len(address_clauses(roster)) == 2


class TestDownstream:
    def test_translation_batch_carries_the_rule(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (  # noqa: E501
            GeminiSubtitleTranslator,
        )

        translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        translator._active_characters = FILM_ROSTER
        batch = [SimpleNamespace(index=1, speaker="朱正軍", addressee="秦可秀")]
        payload = [{"line_no": 1}]
        translator._attach_address_rules_once(payload, batch)
        assert payload[0]["address"] == "tự xưng 'tôi/em' gọi 'chị dâu/A Tú'"

    def test_scene_majority_fix_now_happens(self) -> None:
        """Ca thật dòng 605: cả cảnh Chân Kiếm Thanh xưng "anh" với Chu Chính Quân."""
        scene = {
            563: ("阿軍我們在你那裡又吃又住", "A Quân, vợ chồng anh ở chỗ cậu ăn ở ké."),
            566: ("當是我們兩夫婦送給你的", "Xem như quà của hai vợ chồng anh tặng."),
            605: ("坐呀\uff01我沒事哩", "Ngồi đi! Tôi vẫn ổn mà."),
            633: (
                "到了最後\uff0c衹剩下我一個", "Đến cuối cùng, chỉ còn lại một mình anh."
            ),
            634: ("可惜我生不如死", "Đáng tiếc là anh sống không bằng chết."),
            636: ("阿軍可否幫我一個忙\uff1f", "A Quân có thể giúp anh một việc không?"),
            641: ("那次我雖然檢回了性命", "Lần đó tuy anh nhặt lại được tính mạng,"),
            645: ("已經對我的田產虎視眈眈", "đã nhòm ngó điền sản của anh rồi."),
            646: ("如果我沒有兒子", "Nếu anh không có con trai,"),
            647: ("我的田產就會守不住了", "điền sản của anh sẽ không giữ được nữa."),
        }
        cues = json.dumps(
            [{"id": i, "spk": "甄劍聲", "to": "朱正軍", "cue": ""} for i in scene],
            ensure_ascii=False,
        )
        sources = {i: row[0] for i, row in scene.items()}
        translations = {i: row[1] for i, row in scene.items()}
        fixes = find_address_rule_fixes(sources, translations, cues, FILM_ROSTER)
        assert [(f.index, f.text_after) for f in fixes] == [
            (605, "Ngồi đi! Anh vẫn ổn mà.")
        ]

    def test_diagnostics_count_the_rules(self) -> None:
        context = SimpleNamespace(
            target_lang="Vietnamese", source_lang="zh", overview="Tóm tắt.",
            characters=FILM_ROSTER, glossary="", visual_cues="",
            enable_tags=False, include_desc=True,
        )
        report = build_context_diagnostics(context=context, source_events=[])
        stats = report["roster"]["address_rules"]
        assert (stats["clauses"], stats["resolved"]) == (11, 9)

    def test_lookup_by_cue_labels(self) -> None:
        assert lookup_address_rule(_rules(), "甄劍聲", "秦可秀") == (
            "tự xưng 'anh/tôi' gọi 'A Tú/em'"
        )


# Nguyên văn phim này, dòng 700-712: (dòng, câu gốc, dịch thô, tinh chỉnh).
SHIFT_ROWS = [
    (700, "蠻配你呀", "rất hợp với em.", "rất hợp với em."),
    (701, "這玉鐲子這麼漂亮", "Vòng ngọc này đẹp như vậy,", "Vòng ngọc này đẹp như vậy,"),
    (702, "不如留下送給你的愛人吧", "chi bằng anh giữ lại để tặng người yêu của mình đi.",
     "chi bằng anh giữ lại để tặng người yêu của mình đi."),
    (703, "就是這個原因\uff0c我才送給你", "Chính vì lý do này nên anh mới tặng em.",
     "Chính vì lý do này nên anh mới tặng em."),
    (704, "我愛劍哥劍哥也很愛我", "Em yêu Kiếm ca, Kiếm ca cũng rất yêu em.",
     "Em yêu Kiếm ca, Kiếm ca cũng rất yêu em."),
    (705, "這玉鐲子你送給别人吧", "Vòng ngọc này anh mang tặng người khác đi.",
     "Vòng ngọc này anh mang tặng người khác đi."),
    (706, "劍哥\uff0c劍哥", "Kiếm ca, Kiếm ca...",
     "Kiếm ca, tôi nào điểm nào thua kém anh ta chứ,"),
    (707, "我有甚麼比不上他呢", "Tôi có điểm nào thua kém anh ta chứ?",
     "còn kịp không thể so sánh,"),
    (708, "感情是不能比較的", "Tình cảm là thứ không thể so sánh.",
     "điểm này đã không thể so sánh với Kiếm ca rồi."),
    (709, "就是這一點你已經比不上劍哥",
     "Chính điểm này thôi anh đã không thể sánh bằng Kiếm ca rồi.",
     "Cô không phải nói muốn xin giống sao? Một lần là không đủ đâu."),
    (710, "雖然你不喜歡我", "Cho dù em không thích anh,", "Cho dù em không thích anh,"),
    (711, "但我真的喜歡你", "nhưng anh thật sự thích em.", "nhưng anh thật sự thích em."),
    (712, "不要呀不要\uff01", "Đừng mà, không!", "Đừng mà, không!"),
]
SHIFT_SOURCE = {row[0]: row[1] for row in SHIFT_ROWS}
SHIFT_LITERAL = {row[0]: row[2] for row in SHIFT_ROWS}
SHIFT_STYLE = {row[0]: row[3] for row in SHIFT_ROWS}


class TestShiftWithWeakLineInside:
    def test_real_shift_is_found(self) -> None:
        shifts = find_stage_shifts(SHIFT_LITERAL, SHIFT_STYLE)
        assert [(s.from_index, s.to_index, s.offset, s.lines) for s in shifts] == [
            (706, 708, 1, 3)
        ]

    def test_stage_is_reverted_including_the_invented_line(self) -> None:
        def _lines(texts: dict[int, str]) -> list[TranslationLine]:
            return [
                TranslationLine(
                    index=i, start_ms=i * 1000, end_ms=i * 1000 + 900,
                    text=texts[i], original_text=SHIFT_SOURCE[i],
                )
                for i in sorted(texts)
            ]

        class _Referee:
            def translate_stage(self, *, stage, source_lines, **_kwargs):
                assert stage.kind is TranslationStageKind.LITERAL
                return [replace(ln, text=SHIFT_LITERAL[ln.index]) for ln in source_lines]

        request = TranslateSubtitlesRequest(
            events=[],
            stages=[
                TranslationStageConfig(kind=kind, model_name="m")
                for kind in (TranslationStageKind.LITERAL, TranslationStageKind.STYLE)
            ],
            context=TranslationContext(target_lang="Vietnamese"),
        )
        use_case = TranslateSubtitlesUseCase(translator=_Referee())  # type: ignore[arg-type]
        result = use_case._repair_stage_shifts(
            _lines(SHIFT_STYLE), _lines(SHIFT_LITERAL), _lines(SHIFT_SOURCE), request,
            None, "Phong cách",
        )
        fixed = {line.index: line.text for line in result}
        for index in range(706, 710):
            assert fixed[index] == SHIFT_LITERAL[index]
