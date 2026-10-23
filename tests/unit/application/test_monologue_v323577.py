"""[v3.23.577] Hồi quy: dòng ĐỘC THOẠI bị dịch như đang nói với ai đó.

Chuẩn đoán thật phim "傻子落水…" (3902 dòng, v3.23.576): khâu phân tích ghi độc thoại
nội tâm bằng nhãn cues "người nói = người nghe" (``宋钱 → 宋钱``). Khâu dịch nhận
``speaker: 宋钱, addressee: 宋钱`` — không hiểu là tự nói với mình — nên ở MỌI lần dịch
phim này, dòng 125-126 ra "Nhưng em có truyền thừa của Thần Tiên tỷ tỷ, trên đời này có
bệnh gì làm khó được em?" (nhân vật chính đang một mình), dòng 21 Hứa Đào Hoa nghĩ thầm
"nhưng chị thủ tiết hai năm". Đo trên 31 lần dịch: 33% dòng độc thoại đo được tự xưng bằng
từ vai vế.

Không phải mọi nhãn tự-nói đều là độc thoại: lượt 900-909 là đàn em Ngô Nhị Cẩu la hét
("你他妈想造反啊", "狗哥我控制不住了") nhưng bị ghi ``吴二狗 → 吴二狗``. Dữ liệu dưới đây
là nguyên văn các ca thật.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from subtitles_extractor.application.services.monologue_guard import (
    MONOLOGUE_ADDRESSEE,
    MONOLOGUE_RULE,
    is_self_addressed,
    monologue_self_forms,
    self_addressed_runs,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    VisualCue,
)

ROSTER = "\n".join([
    "宋钱 (Tống Tiền) [nam, ~23] — nhân vật chính; xưng hô: với Hồng Liên Nữ Đế tự xưng "
    "'em' gọi 'Thần Tiên tỷ tỷ', với Thiết Trụ tự xưng 'em' gọi 'anh Thiết Trụ'",
    "许桃花 (Hứa Đào Hoa) [nữ, ~26] — góa phụ; xưng hô: với Tống Tiền tự xưng 'chị' gọi "
    "'chú em'",
    "铁柱 (Thiết Trụ) [nam, ~30] — người trong thôn",
    "吴二狗 / 土狗 / 狗哥 (Ngô Nhị Cẩu / Thổ Cẩu / Cẩu ca) [nam, ~35] — kẻ bảo kê chợ",
])

# (dòng, người nói, người nghe, câu gốc, bản dịch v3.23.576)
ROWS = [
    (17, "宋钱", "许桃花", "好呀", "Hay quá."),
    (18, "许桃花", "许桃花", "这傻小子", "Thằng ngốc này"),
    (19, "许桃花", "许桃花", "长得可真结实", "trông vạm vỡ phết."),
    (20, "许桃花", "许桃花", "虽然有些趁人之危", "Dù hơi thừa cơ người gặp nạn,"),
    (21, "许桃花", "许桃花", "但我守寡两年", "nhưng chị thủ tiết hai năm,"),
    (22, "许桃花", "许桃花", "实在忍不住了", "thực sự không nhịn nổi nữa."),
    (84, "宋钱", "宋钱", "这一切都是真的", "Hóa ra tất cả chuyện này là thật."),
    (85, "宋钱", "宋钱", "我要集齐六块", "Mình phải thu thập đủ sáu"),
    (86, "宋钱", "宋钱", "元神碎片", "mảnh nguyên thần."),
    (88, "宋钱", "宋钱", "才能让我和神仙姐姐", "mới có thể giúp em và Thần Tiên tỷ tỷ"),
    (91, "宋钱", "宋钱", "我还得先救桃花嫂子", "mình vẫn phải đi cứu chị Đào Hoa trước."),
    (123, "宋钱", "宋钱", "没想到铁柱哥", "Không ngờ anh Thiết Trụ"),
    (124, "宋钱", "宋钱", "还有这个病", "lại mắc căn bệnh này."),
    (125, "宋钱", "宋钱", "但我有神仙姐姐的传承",
     "Nhưng em có truyền thừa của Thần Tiên tỷ tỷ,"),
    (126, "宋钱", "宋钱", "世上哪有病能难得到我\uff1f",
     "trên đời này có bệnh gì làm khó được em?"),
    (127, "宋钱", "铁柱", "铁柱哥", "Anh Thiết Trụ,"),
    (899, "", "", "居然踹了他一脚", "lại quay sang đá hắn một cước?"),
    (900, "吴二狗", "吴二狗", "你他妈想造反啊\uff01", "Chúng mày muốn phản à!"),
    (901, "吴二狗", "吴二狗", "狗哥我控制不住了",
     "Cẩu ca, tôi kiểm soát không nổi nữa rồi!"),
    (903, "吴二狗", "吴二狗", "我控制不住了", "Tôi kiểm soát không nổi nữa!"),
    (916, "宋钱", "张依贝", "得让他们长点记性", "Phải cho chúng nó nhớ đời mới được."),
    (917, "吴二狗", "吴二狗", "我们是人渣", "Chúng tôi là cặn bã,"),
    (918, "吴二狗", "吴二狗", "我们有罪", "chúng tôi có tội,"),
    (1391, "许桃花", "许桃花", "什么", "Cái gì?"),
    (1392, "许桃花", "许桃花", "我没听错吧", "Tôi không nghe nhầm chứ?"),
    (1393, "许桃花", "许桃花", "吴二狗居然自称土狗",
     "Ngô Nhị Cẩu đường đường lại tự nhận mình là Thổ Cẩu."),
]
SOURCES = {row[0]: row[3] for row in ROWS}
FINAL = {row[0]: row[4] for row in ROWS}
CUES = [VisualCue(line_no=r[0], speaker=r[1], addressee=r[2]) for r in ROWS if r[1]]


def _runs() -> dict[int, tuple[tuple[int, ...], bool]]:
    return {
        run.lines[0]: (run.lines, run.monologue)
        for run in self_addressed_runs(CUES, SOURCES, ROSTER)
    }


class TestSelfAddressedRuns:
    def test_inner_thoughts_are_monologues(self) -> None:
        runs = _runs()
        assert runs[18] == ((18, 19, 20, 21, 22), True)
        assert runs[84] == ((84, 85, 86, 88, 91, 123, 124, 125, 126), True)

    def test_run_spoken_to_someone_is_not_a_monologue(self) -> None:
        """Đàn em la hét "你他妈想造反啊", "狗哥我控制不住了" dưới nhãn 吴二狗→吴二狗."""
        assert _runs()[900] == ((900, 901, 903), False)

    def test_leading_vocative_alone_marks_a_dialogue(self) -> None:
        cues = [VisualCue(line_no=1, speaker="吴二狗", addressee="吴二狗")]
        runs = self_addressed_runs(cues, {1: "狗哥我控制不住了"}, ROSTER)
        assert [run.monologue for run in runs] == [False]

    def test_group_self_reference_is_not_a_monologue(self) -> None:
        assert _runs()[917] == ((917, 918), False)

    def test_name_as_subject_is_still_a_monologue(self) -> None:
        """"吴二狗居然自称土狗" — tên làm chủ ngữ, không phải tiếng gọi."""
        assert _runs()[1391] == ((1391, 1392, 1393), True)

    def test_line_without_self_label_breaks_the_run(self) -> None:
        assert 127 not in {i for lines, _ in _runs().values() for i in lines}

    def test_aliases(self) -> None:
        assert is_self_addressed("宋钱 / 宋", "宋钱")
        assert not is_self_addressed("宋钱", "许桃花")
        assert not is_self_addressed("", "")


class TestMonologueSelfForms:
    def test_real_kin_self_forms_are_reported(self) -> None:
        found = {
            item["speaker"]: item
            for item in monologue_self_forms(CUES, SOURCES, FINAL, ROSTER)
        }
        assert found["宋钱"]["lines"] == [88, 125, 126]
        assert found["宋钱"]["variants"] == {"mình": 2, "em": 3}
        assert found["许桃花"]["lines"] == [21]

    def test_kin_word_before_a_name_is_third_person(self) -> None:
        """91: "mình vẫn phải đi cứu chị Đào Hoa" — "chị" là người được nhắc tới."""
        found = monologue_self_forms(CUES, SOURCES, FINAL, ROSTER)
        assert all(91 not in item["lines"] for item in found)

    def test_dialogue_runs_are_not_measured(self) -> None:
        found = monologue_self_forms(CUES, SOURCES, FINAL, ROSTER)
        assert "吴二狗" not in {item["speaker"] for item in found}


def _line(row: tuple) -> TranslationLine:
    index, speaker, addressee, source, _ = row
    return TranslationLine(
        index=index, start_ms=index * 1000, end_ms=index * 1000 + 900, text=source,
        speaker=speaker, addressee=addressee,
    )


class TestUseCaseSeparatesMonologues:
    def test_misread_run_loses_its_listener_label(self) -> None:
        get_process_log().clear()
        lines = [_line(row) for row in ROWS]
        result = {
            line.index: line
            for line in TranslateSubtitlesUseCase._separate_monologues(lines, ROSTER)
        }
        assert [result[i].addressee for i in (900, 901, 903, 917, 918)] == [""] * 5
        assert result[125].addressee == "宋钱"  # độc thoại: giữ để khâu dịch nhận ra
        assert result[17].addressee == "许桃花"  # thoại thường: không đụng
        events = [e for e in get_process_log().snapshot()["events"]
                  if e["event"] == "monologue_lines"]
        assert events[-1]["cleared"] == [900, 901, 903, 917, 918]
        assert events[-1]["monologue"] == 17

    def test_no_self_labels_returns_the_same_list(self) -> None:
        lines = [_line(ROWS[0])]
        assert TranslateSubtitlesUseCase._separate_monologues(lines, ROSTER) is lines


class TestTranslationPayload:
    def _translator(self):
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (  # noqa: E501
            GeminiSubtitleTranslator,
        )

        translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        translator._active_characters = ROSTER
        return translator

    def test_monologue_lines_say_there_is_no_listener(self) -> None:
        batch = [
            SimpleNamespace(index=124, speaker="宋钱", addressee="宋钱"),
            SimpleNamespace(index=125, speaker="宋钱", addressee="宋钱"),
            SimpleNamespace(index=127, speaker="宋钱", addressee="铁柱"),
        ]
        payload = [{"line_no": 124, "addressee": "宋钱"},
                   {"line_no": 125, "addressee": "宋钱"},
                   {"line_no": 127, "addressee": "铁柱"}]
        self._translator()._attach_address_rules_once(payload, batch)
        assert payload[0]["addressee"] == MONOLOGUE_ADDRESSEE
        assert payload[0]["address"] == MONOLOGUE_RULE
        assert payload[1] == {"line_no": 125, "addressee": MONOLOGUE_ADDRESSEE}
        assert payload[2]["address"] == "tự xưng 'em' gọi 'anh Thiết Trụ'"

    def test_rule_forbids_kin_self_reference(self) -> None:
        assert "KHÔNG tự xưng anh/em/chị/con/cháu" in MONOLOGUE_RULE


class _Event:
    def __init__(self, index: int, text: str) -> None:
        self.index = index
        self.text = text


class TestDiagnostics:
    def _flags(self) -> dict:
        cues = json.dumps(
            [{"id": r[0], "spk": r[1], "to": r[2], "cue": ""} for r in ROWS if r[1]],
            ensure_ascii=False,
        )
        return detect_quality_flags(
            [_Event(i, s) for i, s in SOURCES.items()],
            [_Event(i, t) for i, t in FINAL.items()],
            characters=ROSTER, visual_cues=cues,
        )

    def test_monologue_kin_self_is_counted(self) -> None:
        flags = self._flags()
        assert flags["counts"]["monologue_kin_self"] == 4
        assert "monologue_self_forms" in flags["note"]

    def test_self_pair_is_not_an_address_pair(self) -> None:
        """Cờ theo CẶP không còn liệt kê 宋钱 → 宋钱 (trước đây: "chị 6, em 5, ta 3…")."""
        flags = self._flags()
        pairs = {(f["speaker"], f["addressee"]) for f in flags["address_inconsistency"]}
        drift = {(f["speaker"], f["addressee"]) for f in flags["address_drift"]}
        assert ("宋钱", "宋钱") not in pairs | drift

    def test_real_run_no_longer_reports_the_self_pair(self) -> None:
        from subtitles_extractor.application.services.translation_diagnostics import (
            address_pair_stats,
        )

        cues = [
            {"id": 200 + n, "spk": "宋钱", "to": "宋钱", "cue": ""} for n in range(12)
        ]
        texts = ["em", "chị", "ta", "tôi"] * 3
        events = [_Event(200 + n, f"{word} nghĩ") for n, word in enumerate(texts)]
        raw = json.dumps(cues, ensure_ascii=False)
        flags, examined = address_pair_stats(raw, events)
        # [v3.23.586] Hàm đo cặp cũng tự bỏ độc thoại (trước đây chỉ detect_quality_flags
        # lọc, nên ai gọi thẳng hàm này vẫn nhận cặp ảo 宋钱 → 宋钱).
        assert (flags, examined) == ([], 0)
        full = detect_quality_flags(
            [_Event(200 + n, "我想") for n in range(12)], events,
            characters=ROSTER, visual_cues=raw,
        )
        assert full["address_inconsistency"] == []


class TestPipeline:
    def test_translator_receives_separated_labels(self) -> None:
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesRequest,
        )
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            TranslationContext,
            TranslationStageConfig,
            TranslationStageKind,
        )
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        received: dict[int, str] = {}
        # Pipeline đánh số dòng theo vị trí (1..N) — cues cũng dùng số vị trí.
        position = {row[0]: pos for pos, row in enumerate(ROWS, start=1)}

        class _Translator:
            def translate_stage(self, *, input_lines, **_kwargs):
                received.update({line.index: line.addressee for line in input_lines})
                return [replace(line, text="bản dịch") for line in input_lines]

        events = [
            SubtitleEvent(index=pos, text=row[3], interval=TimeInterval(pos, pos + 0.9))
            for pos, row in enumerate(ROWS, start=1)
        ]
        cues = json.dumps(
            [{"id": position[r[0]], "spk": r[1], "to": r[2], "cue": ""}
             for r in ROWS if r[1]],
            ensure_ascii=False,
        )
        request = TranslateSubtitlesRequest(
            events=events,
            stages=[TranslationStageConfig(
                kind=TranslationStageKind.LITERAL, model_name="m"
            )],
            context=TranslationContext(
                target_lang="Vietnamese", characters=ROSTER, visual_cues=cues
            ),
            video_refs=("video",),
            enable_visual_cues=True,
        )
        use_case = TranslateSubtitlesUseCase(translator=_Translator())  # type: ignore[arg-type]
        use_case.execute(request)
        assert received[position[125]] == "宋钱"
        assert received[position[900]] == ""
        assert received[position[17]] == "许桃花"


class TestParticleOnlyDuplicates:
    """Kèm theo: hai dòng gốc chỉ khác trợ từ dịch giống hệt là ĐÚNG — không dịch lại."""

    def test_real_pair_173_174_is_not_a_duplicate(self) -> None:
        from subtitles_extractor.application.services.line_echo_guard import (
            repeated_line_groups,
        )

        sources = {173: "怎么是你", 174: "怎么是你啊", 175: "看啊", 176: "看"}
        translations = {173: "Sao lại là chú em thế?", 174: "Sao lại là chú em thế?",
                        175: "Nhìn này!", 176: "Nhìn này!"}
        assert repeated_line_groups(sources, translations) == []

    def test_different_content_is_still_a_duplicate(self) -> None:
        from subtitles_extractor.application.services.line_echo_guard import (
            repeated_line_groups,
        )

        sources = {1860: "连大家的工资都", 1861: "发不了"}
        translations = {1860: "không phát nổi.", 1861: "không phát nổi."}
        assert repeated_line_groups(sources, translations) == [(1860, 1861)]
