"""[v3.23.591] Hai chỗ khiến file chuẩn đoán nói SAI hoặc nói THIẾU.

1. Bảng nhân vật thật v3.23.590 ghi ``龍傳人 … với bác sĩ tự xưng 'tôi' gọi 'ông/bác
   sĩ'``. "bác sĩ" là nghề, nhưng bộ đếm đại từ lấy chữ đầu "bác" nên thấy hai đại từ
   ("ông", "bác") và báo ``roster_ambiguous_address`` kèm lời khuyên "nên phân tích lại
   ngữ cảnh" — một lượt phân tích lại tốn 17-20 request của model hạn mức 20 request/ngày,
   cho một bảng đang đúng.
2. File chuẩn đoán dịch không ghi giai đoạn nào có đính video. Bản dịch khác hẳn giữa hai
   chế độ (model dịch theo lời nghe được — xem ``test_video_role_rule_v323591``), mà muốn
   biết phải mở nhật ký ứng dụng đếm từng lời gọi "N phần video".
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from subtitles_extractor.application.services.roster_format import (
    _pronoun_count,
    ambiguous_address_rules,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

# ── Bảng nhân vật thật v3.23.590 (phim "灭门惨案II"), chép nguyên văn ───────────────────
CHIEF_ROW = (
    "龍傳人 (Long Truyền Nhân) [nam, ~35] — Trưởng đồn kiêm Đội trưởng công an Đông "
    "Long, tính tình khoác lác, đố kỵ và háo sắc nhưng bản chất lương thiện, giàu lòng "
    "trắc ẩn; cuối cùng quyết định che chở tha bổng cho Tần Khả Tú; xưng hô: với Tần "
    "Khả Tú tự xưng 'tôi' gọi 'chị' (danh xưng: Chân đại tẩu), với thuộc cấp (Ái Quốc, "
    "Công an béo) tự xưng 'tôi' gọi 'cậu', với Dư Tiểu Liên tự xưng 'anh' gọi 'em', "
    "với Chu Chính Quân tự xưng 'tôi' gọi 'anh', với dân làng tự xưng 'tôi' gọi 'các "
    "người/bà con', với bác sĩ tự xưng 'tôi' gọi 'ông/bác sĩ', với Lão Biểu tự xưng "
    "'em' gọi 'anh'"
)
DOCTOR_ROW = (
    "醫生 (Bác sĩ) [nam, ~50] — bác sĩ trạm xá trấn Đông Long, tính cách bộc trực, "
    "khám thai cho Khả Tú rồi phát hiện cô nhiễm giang mai giai đoạn một và khuyên cô "
    "phá thai; xưng hô: với Chân Kiếm Sanh tự xưng 'tôi' gọi 'anh', với Tần Khả Tú tự "
    "xưng 'tôi' gọi 'cô', với Long Truyền Nhân tự xưng 'tôi' gọi 'đồng chí công an'"
)
# Bảng nhân vật thật v3.23.585 (cùng phim, bản phân tích khác): mơ hồ THẬT, vẫn phải báo.
KIEM_ROW_585 = (
    "甄劍聲 / 劍哥 (Chân Kiếm Thanh / Kiếm ca) [nam, ~35] — Chồng của Tần Khả Tú, cựu "
    "chiến binh miền Bắc bị di chứng chiến tranh mất khả năng sinh sản, tính tình điềm "
    "đạm, sẵn sàng nhờ bạn mượn giống để duy trì hương hỏa (nhân vật trong hồi tưởng); "
    "xưng hô: với Tần Khả Tú tự xưng 'anh' gọi 'em/A Tú', với Chu Chính Quân tự xưng "
    "'anh' gọi 'chú/A Quân', với Hồng Chấn xưng 'tôi' gọi 'anh/ông'"
)
HONG_CHAN_ROW_585 = (
    "洪震 (Hồng Chấn) [nam, ~50] — Tên ác bá côn đồ dâm ô trong làng, chuyên ăn vạ và "
    "ức hiếp người khác, cha của Hồng Chí Quyền; xưng hô: với vợ con tự xưng 'tao' gọi "
    "'mày', với người ngoài tự xưng 'tao' gọi 'mày', với Chân Kiếm Thanh (lúc bị đánh) "
    "tự xưng 'tôi' gọi 'Chân tiên sinh'"
)


class TestDoctorIsATitleNotAPronoun:
    def test_real_row_is_no_longer_flagged(self) -> None:
        assert ambiguous_address_rules(f"{CHIEF_ROW}\n{DOCTOR_ROW}") == []

    @pytest.mark.parametrize(
        "forms",
        ["ông/bác sĩ", "bác sĩ/ông", "anh/ông chủ", "các người/bà con", "bác sĩ"],
    )
    def test_title_next_to_one_pronoun_is_one_pronoun(self, forms: str) -> None:
        assert _pronoun_count(forms) <= 1

    def test_title_does_not_hide_two_real_pronouns(self) -> None:
        assert _pronoun_count("ông/anh/bác sĩ") == 2

    def test_real_ambiguity_is_still_flagged(self) -> None:
        assert ambiguous_address_rules(f"{KIEM_ROW_585}\n{HONG_CHAN_ROW_585}") == [
            "甄劍聲 / 劍哥 → Hồng Chấn: xưng 'tôi' gọi 'anh/ông'"
        ]

    @pytest.mark.parametrize("forms", ["bác/ông", "bác Chu/ông", "chú em/anh"])
    def test_kin_word_used_as_address_still_counts(self, forms: str) -> None:
        """"bác" đứng riêng hoặc trước TÊN là đại từ; chỉ "bác sĩ" mới là nghề."""
        assert _pronoun_count(forms) == 2

    def test_same_head_options_are_still_one(self) -> None:
        """Điều v3.23.586 đã sửa vẫn giữ."""
        assert _pronoun_count("ba/lão ba") == 1
        assert _pronoun_count("chị/chị dâu") == 1


def _events(count: int = 4) -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=i, text=f"句{i}", interval=TimeInterval(i, i + 0.9))
        for i in range(1, count + 1)
    ]


class _Translator:
    """Bộ dịch giả: ghi lại cờ đính video từng giai đoạn nhận được."""

    def __init__(self) -> None:
        self.attach: dict[str, bool] = {}

    def translate_stage(
        self,
        *,
        stage: TranslationStageConfig,
        input_lines: list[TranslationLine],
        attach_video: bool = False,
        **_kwargs: Any,
    ) -> list[TranslationLine]:
        self.attach[stage.kind.value] = attach_video
        kind = stage.kind.value
        return [replace(line, text=f"dịch {kind} {line.index}") for line in input_lines]


def _completed(
    attach: frozenset[TranslationStageKind], video_refs: tuple[Any, ...]
) -> dict[str, dict[str, Any]]:
    get_process_log().clear()
    request = TranslateSubtitlesRequest(
        events=_events(),
        stages=[
            TranslationStageConfig(kind=kind, model_name="gemini-3.5-flash-lite")
            for kind in (
                TranslationStageKind.LITERAL,
                TranslationStageKind.STYLE,
                TranslationStageKind.LOCALIZE,
            )
        ],
        context=TranslationContext(target_lang="Vietnamese"),
        video_refs=video_refs,
        attach_video_stages=attach,
    )
    TranslateSubtitlesUseCase(translator=_Translator()).execute(request)  # type: ignore[arg-type]
    return {
        event["stage"]: event
        for event in get_process_log().snapshot()["events"]
        if event["event"] == "stage_completed"
    }


ALL_STAGES = frozenset({
    TranslationStageKind.LITERAL,
    TranslationStageKind.STYLE,
    TranslationStageKind.LOCALIZE,
})


class TestStageCompletedSaysWhetherVideoWasAttached:
    def test_real_setup_every_stage_is_marked(self) -> None:
        """Cài đặt của lượt chạy thật: đính video ở cả ba giai đoạn."""
        completed = _completed(ALL_STAGES, ("files/SEG0",))
        assert {stage: event["attach_video"] for stage, event in completed.items()} == {
            "literal": True,
            "style": True,
            "localize": True,
        }

    def test_only_selected_stages_are_marked(self) -> None:
        completed = _completed(
            frozenset({TranslationStageKind.LITERAL}), ("files/SEG0",)
        )
        assert {stage: event["attach_video"] for stage, event in completed.items()} == {
            "literal": True,
            "style": False,
            "localize": False,
        }

    def test_no_uploaded_video_means_no_video(self) -> None:
        """Bật đính video mà không có đoạn nào tải lên thì lô gửi đi không có video."""
        completed = _completed(ALL_STAGES, ())
        assert [event["attach_video"] for event in completed.values()] == [False] * 3

    def test_other_fields_are_kept(self) -> None:
        event = _completed(ALL_STAGES, ("files/SEG0",))["style"]
        assert event["model"] == "gemini-3.5-flash-lite"
        assert event["lines"] == 4

    def test_direct_call_defaults_to_no_video(self) -> None:
        get_process_log().clear()
        stage = TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        TranslateSubtitlesUseCase._record_stage_completed(stage, "Dịch thô", [])
        assert get_process_log().snapshot()["events"][-1]["attach_video"] is False
