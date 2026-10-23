"""[v3.23.583] Giai đoạn tinh chỉnh làm hỏng xưng hô đúng của bản trước → khôi phục.

Chuẩn đoán thật v3.23.582 (phim "he thong sung the"): bản dịch thô để Tống Hân Sảng nói
với bạn thân Trương Mẫn "tớ … cậu" (đúng vai ngang hàng; roster 'mình'/'cậu'), giai đoạn
tinh chỉnh đổi 6 dòng giữa cảnh sang "chị … em". Lâm Dật nói với Trương Mẫn: bản thô "em"
(đúng roster), tinh chỉnh đổi 9 dòng thành "tôi".
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from subtitles_extractor.application.services.address_regression_guard import (
    allowed_address_forms,
    find_address_regressions,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    address_inconsistencies,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)

SONG = "宋欣爽 / 小爽"
ZHANG = "张敏 / 小敏"
LIN = "林逸"

# Ba dòng bảng nhân vật THẬT của lần phân tích đó (rút gọn phần mô tả).
CHARACTERS = "\n".join([
    "宋欣爽 / 小爽 (Tống Hân Sảng / Tiểu Sảng) [nữ, ~27] — chị dâu của Lâm Dật; "
    "xưng hô: với Lâm Dật tự xưng 'chị' gọi 'Tiểu Dật'; với Trương Mẫn tự xưng 'mình' "
    "gọi 'cậu'; với Tần Nguyệt tự xưng 'tôi' gọi 'cô'",
    "张敏 / 小敏 (Trương Mẫn / Tiểu Mẫn) [nữ, ~27] — bạn thân y tá của Tống Hân Sảng; "
    "xưng hô: với Tống Hân Sảng tự xưng 'mình' gọi 'cậu'; với Lâm Dật tự xưng 'chị' "
    "gọi 'Lâm Dật'",
    "林逸 (Lâm Dật) [nam, ~24] — nhân vật chính; xưng hô: với Tống Hân Sảng tự xưng "
    "'em' gọi 'chị dâu'; với Trương Mẫn tự xưng 'em' gọi 'chị Mẫn'",
])

# (dòng, người nói, người nghe, câu gốc, bản dịch thô, bản tinh chỉnh) — dữ liệu thật.
ROWS = [
    (2842, LIN, ZHANG, "相信我", "Tin em đi,", "Tin tôi đi,"),
    (2847, LIN, ZHANG, "你相信我", "chị tin em đi,", "chị tin tôi đi,"),
    (3009, ZHANG, SONG, "他", "Cậu ấy...", "Nó..."),
    (3010, SONG, ZHANG, "小逸知道了我的病情", "Tiểu Dật mà biết bệnh tình của tớ,",
     "Nếu Tiểu Dật biết bệnh tình của chị,"),
    (3015, SONG, ZHANG, "想给我治病啊", "muốn chữa bệnh cho tớ không?",
     "định dùng để chữa bệnh cho chị à?"),
    (3042, SONG, ZHANG, "不能浪费在我身上", "không thể lãng phí lên người tớ được.",
     "không thể lãng phí lên người chị được."),
    (3043, SONG, ZHANG, "小敏", "Tiểu Mẫn,", "Tiểu Mẫn,"),
    (3044, SONG, ZHANG, "我求你了", "tớ cầu xin cậu đấy,", "chị cầu xin em đấy,"),
    (3045, SONG, ZHANG, "你快带我去找他", "cậu mau đưa tớ đi tìm cậu ấy.",
     "em mau đưa chị đi tìm thằng bé."),
    (858, LIN, SONG, "你放心", "Chị xem,", "Em đảm bảo,"),
]


def _inputs(rows=ROWS) -> tuple[dict, dict, dict, str]:
    sources = {row[0]: row[3] for row in rows}
    previous = {row[0]: row[4] for row in rows}
    current = {row[0]: row[5] for row in rows}
    cues = json.dumps(
        [{"id": row[0], "spk": row[1], "to": row[2], "cue": ""} for row in rows],
        ensure_ascii=False,
    )
    return sources, previous, current, cues


def _fixed() -> dict[int, str]:
    sources, previous, current, cues = _inputs()
    return {
        item.index: item.text_after
        for item in find_address_regressions(sources, previous, current, cues, CHARACTERS)
    }


class TestRealScene:
    def test_friends_get_their_peer_pronouns_back(self) -> None:
        fixed = _fixed()
        assert fixed[3015] == "định dùng để chữa bệnh cho tớ à?"
        assert fixed[3042] == "không thể lãng phí lên người tớ được."
        assert fixed[3044] == "tớ cầu xin cậu đấy,"
        assert fixed[3045] == "cậu mau đưa tớ đi tìm thằng bé."

    def test_lin_yi_keeps_em_with_zhang_min(self) -> None:
        fixed = _fixed()
        assert fixed[2842] == "Tin em đi,"
        assert fixed[2847] == "chị tin em đi,"

    def test_rephrased_line_where_the_pronoun_changed_role_is_left_alone(self) -> None:
        # "你放心": bản trước 'chị' là người nghe; bản mới "Em đảm bảo" — 'em' là người
        # nói.
        assert 858 not in _fixed()

    def test_unstable_label_is_left_alone(self) -> None:
        # Dòng trước "Nó..." chưa hết câu mà nhãn đổi người nói → không đủ chắc.
        assert 3010 not in _fixed()

    def test_lines_without_pronoun_change_are_untouched(self) -> None:
        assert 3043 not in _fixed()
        assert 3009 not in _fixed()


def _one(previous_text: str, current_text: str, source: str = "我求你了") -> str | None:
    rows = [(3044, SONG, ZHANG, source, previous_text, current_text)]
    sources, previous, current, cues = _inputs(rows)
    found = find_address_regressions(sources, previous, current, cues, CHARACTERS)
    return found[0].text_after if found else None


class TestSafetyChecks:
    def test_previous_version_must_follow_the_roster(self) -> None:
        assert _one("Tôi xin cậu đấy,", "Chị xin em đấy,") is None

    def test_peer_forms_are_interchangeable(self) -> None:
        # Roster 'mình', bản dịch 'tớ' → 'mình': cùng vai ngang hàng, không phải lỗi.
        assert _one("tớ cầu xin cậu đấy,", "mình cầu xin cậu đấy,") is None

    def test_half_restoration_is_refused(self) -> None:
        # 'em' không đứng cùng chỗ với 'cậu' → không đặt lại được hết → không sửa gì.
        assert _one("tớ cầu xin cậu đấy,", "chị cầu xin đó, nhé em") is None

    def test_pronoun_after_kin_word_is_left_alone(self) -> None:
        assert _one("Mẹ tớ đang đợi.", "Mẹ chị đang đợi.", source="我妈还等着") is None

    def test_plural_source_is_skipped(self) -> None:
        assert _one("tớ với cậu đi.", "chị với em đi.", source="我们走吧") is None

    def test_third_person_occurrence_does_not_block(self) -> None:
        assert _one(
            "cậu đưa tớ đi tìm cậu ấy.", "em đưa chị đi tìm thằng bé.",
            source="你带我去找他",
        ) == "cậu đưa tớ đi tìm thằng bé."


class TestAddressForms:
    def test_head_word_of_each_option(self) -> None:
        assert allowed_address_forms("tự xưng 'chị' gọi 'em trai'") == {"em"}
        assert allowed_address_forms("tự xưng 'em' gọi 'chị Mẫn'") == {"chị"}
        assert allowed_address_forms("tự xưng 'em' gọi 'anh / Tống Tiền'") == {"anh"}

    def test_names_and_titles_are_not_pronouns(self) -> None:
        assert allowed_address_forms("tự xưng 'tôi' gọi 'Lâm tiên sinh'") == set()


class TestInconsistencyIgnoresKinNouns:
    def test_real_doctor_scene_is_not_flagged(self) -> None:
        # 林逸 → 主治医生 (dòng thật): "Bác sĩ!" và "chị dâu tôi" từng bị đếm là các biến
        # thể xưng hô 'bác' và 'chị' của cặp này.
        lines = {
            2653: "Bác sĩ!", 2654: "Bác sĩ ơi!", 2656: "Bác sĩ!", 2659: "Bác sĩ mau lên!",
            2660: "Mau khám cho chị dâu tôi!", 2661: "Chị dâu tôi rốt cuộc bị sao vậy?",
            2673: "Ông nói câu này có ý gì?", 2679: "Hết bao nhiêu, tôi quét mã trả anh.",
            2692: "Cái gì, lời này của ông có ý gì?", 2707: "Lời này của ông có ý gì?",
            2708: "Ý ông là bảo chị dâu tôi",
        }
        cues = json.dumps(
            [{"id": index, "spk": LIN, "to": "主治医生"} for index in lines],
            ensure_ascii=False,
        )
        events = [
            SimpleNamespace(index=index, text=text) for index, text in lines.items()
        ]
        assert address_inconsistencies(cues, events) == []


class TestPipeline:
    """Cả lượt dịch hai giai đoạn: bản cuối giữ xưng hô đúng của bản dịch thô."""

    def test_execute_restores_the_refinement_regressions(self) -> None:
        from dataclasses import replace

        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesRequest,
            TranslateSubtitlesUseCase,
        )
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            TranslationContext,
            TranslationStageConfig,
            TranslationStageKind,
        )
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        # Dòng thật 3042-3045 (liền nhau, cùng người nói) rồi dòng 858 — pipeline đánh số
        # lại theo vị trí 1..5 nên Visual Cues cũng theo vị trí.
        rows = [row for index in (3042, 3043, 3044, 3045, 858)
                for row in ROWS if row[0] == index]
        literal = {pos: row[4] for pos, row in enumerate(rows, start=1)}
        style = {pos: row[5] for pos, row in enumerate(rows, start=1)}

        # [v3.23.592] Lưới dòng lẻ xin một lượt dịch THÔ làm trọng tài cho dòng bị viết
        # lại khác hẳn (dòng 858). Câu trả lời thật cho "你放心" ở cả bảy lượt dịch phim
        # này là "Chị yên tâm," / "Chị cứ yên tâm," — không phải bản trước "Chị xem,".
        referee = {**literal, 5: "Chị yên tâm,"}

        class _Pipeline:
            def __init__(self) -> None:
                self.literal_calls = 0

            def translate_stage(self, *, stage, input_lines, **_kwargs):
                texts = style
                if stage.kind is TranslationStageKind.LITERAL:
                    self.literal_calls += 1
                    texts = literal if self.literal_calls == 1 else referee
                return [replace(line, text=texts[line.index]) for line in input_lines]

        events = [
            SubtitleEvent(index=pos, text=row[3], interval=TimeInterval(pos, pos + 0.9))
            for pos, row in enumerate(rows, start=1)
        ]
        cues = json.dumps(
            [
                {"id": pos, "spk": row[1], "to": row[2], "cue": ""}
                for pos, row in enumerate(rows, start=1)
            ],
            ensure_ascii=False,
        )
        request = TranslateSubtitlesRequest(
            events=events,
            stages=[
                TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m"),
                TranslationStageConfig(kind=TranslationStageKind.STYLE, model_name="m"),
            ],
            context=TranslationContext(
                target_lang="Vietnamese", characters=CHARACTERS, visual_cues=cues
            ),
        )
        use_case = TranslateSubtitlesUseCase(translator=_Pipeline())  # type: ignore[arg-type]
        response = use_case.execute(request)
        final = [event.text for event in response.events]
        assert final == [
            "không thể lãng phí lên người tớ được.",
            "Tiểu Mẫn,",
            "tớ cầu xin cậu đấy,",
            "cậu mau đưa tớ đi tìm thằng bé.",
            "Em đảm bảo,",
        ]
        events_logged = get_process_log().snapshot()["events"]
        restored = [
            event for event in events_logged
            if event["event"] == "address_regression_restored"
        ]
        assert restored and restored[-1]["label"]
