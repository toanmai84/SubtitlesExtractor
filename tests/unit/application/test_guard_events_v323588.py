"""[v3.23.588] Việc các lưới tất định đã làm phải nằm trong file chuẩn đoán dịch.

Lượt chạy thật v3.23.587: lưới số đổi năm dòng ĐÚNG sang lối đọc khác, khâu dọn dịch lại
hai dòng còn chữ Hán (12, 265), adapter hoàn nguyên hai dòng bị rớt nội dung (80, 116) —
cả ba chỉ có trong nhật ký ứng dụng. ``process_log`` của file chuẩn đoán chỉ ghi 5 loại
sự kiện, nên muốn biết bản cuối đã bị lưới nào sửa thì phải xin thêm ``app_log.txt``.
"""

# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim

from __future__ import annotations

from typing import Any

import pytest

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
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _line(index: int, text: str, source: str = "") -> TranslationLine:
    return TranslationLine(
        index=index, start_ms=index, end_ms=index + 1, text=text, original_text=source
    )


def _pairs(
    rows: dict[int, tuple[str, str]],
) -> tuple[list[TranslationLine], list[TranslationLine]]:
    sources = [_line(index, source) for index, (source, _text) in rows.items()]
    output = [_line(index, text, source) for index, (source, text) in rows.items()]
    return sources, output


def _events(name: str) -> list[dict[str, Any]]:
    return [e for e in get_process_log().snapshot()["events"] if e["event"] == name]


@pytest.fixture(autouse=True)
def _fresh_log() -> None:
    get_process_log().clear()


class TestNumberGuardIsRecorded:
    def test_real_case_records_line_reason_and_both_readings(self) -> None:
        sources, output = _pairs({
            788: ("你給我二十萬", "Mày đưa tao hai mươi vạn."),
            789: ("有二十萬我兩父子", "Có hai mươi vạn, hai cha con tao"),
            966: ("這裡是二十萬，我老婆呢", "Đây là hai trăm nghìn, vợ tôi đâu?"),
        })
        TranslateSubtitlesUseCase._enforce_number_consistency(output, sources, "LITERAL")
        assert _events("numbers_normalised") == [{
            "event": "numbers_normalised",
            "label": "LITERAL",
            "fixes": [{
                "index": 966, "reason": "lệch lối đọc",
                "before": "hai trăm nghìn", "after": "hai mươi vạn",
            }],
        }]

    def test_nothing_to_fix_records_nothing(self) -> None:
        sources, output = _pairs({788: ("你給我二十萬", "Mày đưa tao hai mươi vạn.")})
        TranslateSubtitlesUseCase._enforce_number_consistency(output, sources, "LITERAL")
        assert _events("numbers_normalised") == []


class TestNameRegressionIsRecorded:
    def test_reverted_line_is_recorded(self) -> None:
        """Ca thật v3.23.551: tinh chỉnh đổi "chị dâu Đào Hoa" thành "vợ Đào Hoa"."""
        source = "桃花嫂子的元神碎片"
        previous = [_line(7, "mảnh nguyên thần của chị dâu Đào Hoa", source)]
        current = [_line(7, "mảnh nguyên thần của vợ Đào Hoa", source)]
        fixed = TranslateSubtitlesUseCase._revert_name_regressions(
            current, previous, [_line(7, source)], "桃花嫂子 = chị dâu Đào Hoa", "STYLE"
        )
        assert fixed[0].text == previous[0].text
        assert _events("name_regressions_reverted") == [
            {"event": "name_regressions_reverted", "label": "STYLE", "lines": [7]}
        ]


class _Retranslator:
    """Bộ dịch giả cho khâu dọn: trả bản dịch lại đã sạch chữ Hán."""

    def __init__(self, clean: dict[int, str]) -> None:
        self._clean = clean

    def is_available(self) -> bool:
        return True

    def translate_stage(
        self, *, input_lines: list[TranslationLine], **_kwargs: object
    ) -> list[TranslationLine]:
        return [
            _line(line.index, self._clean[line.index], line.text) for line in input_lines
        ]


# Hai dòng thật của lượt này: chữ Hán sót qua cả ba giai đoạn, khâu dọn cuối dịch lại.
RESIDUAL = {
    12: ("我大丈夫有志，志難伸", "Đại trượng夫 mang chí lớn mà không gặp thời."),
    265: ("黑漆漆的，伸手不見五指", "Tối tăm mịt mờ,伸手不见五指."),
}
RETRANSLATED = {
    12: "Ta đường đường đại trượng phu có chí lớn, nhưng chí khó tỏ.",
    265: "Tối đen như mực, đưa tay không thấy năm ngón.",
}


def _request(sources: list[TranslationLine]) -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[
            SubtitleEvent(index=line.index, text=line.text, interval=TimeInterval(1, 2))
            for line in sources
        ],
        stages=[
            TranslationStageConfig(
                kind=TranslationStageKind.LITERAL, model_name="m", batch_size=10
            )
        ],
        context=TranslationContext(target_lang="Vietnamese"),
    )


class TestResidualCjkCleanupIsRecorded:
    def test_retranslated_lines_are_recorded(self) -> None:
        sources, output = _pairs(RESIDUAL)
        use_case = TranslateSubtitlesUseCase(translator=_Retranslator(RETRANSLATED))
        cleaned = use_case._cleanup_residual_cjk(output, sources, _request(sources), None)
        assert [line.text for line in cleaned] == list(RETRANSLATED.values())
        assert _events("residual_cjk_cleaned") == [{
            "event": "residual_cjk_cleaned", "how": "retranslation",
            "lines": [12, 265], "still_residual": [],
        }]

    def test_line_the_retry_could_not_clean_is_listed(self) -> None:
        sources, output = _pairs(RESIDUAL)
        stubborn = {**RETRANSLATED, 265: RESIDUAL[265][1]}
        use_case = TranslateSubtitlesUseCase(translator=_Retranslator(stubborn))
        use_case._cleanup_residual_cjk(output, sources, _request(sources), None)
        [event] = _events("residual_cjk_cleaned")
        assert (event["lines"], event["still_residual"]) == ([12], [265])

    def test_clean_output_records_nothing(self) -> None:
        sources, output = _pairs({1: ("走吧", "Đi thôi.")})
        use_case = TranslateSubtitlesUseCase(translator=_Retranslator({}))
        use_case._cleanup_residual_cjk(output, sources, _request(sources), None)
        assert _events("residual_cjk_cleaned") == []


class TestStageRegressionsAreRecorded:
    def test_reinserted_cjk_and_emptied_lines(self) -> None:
        previous = [_line(1, "Đại trượng phu có chí lớn."), _line(2, "Sao ông già này…")]
        current = [_line(1, "Đại trượng夫 có chí lớn."), _line(2, "")]
        use_case = TranslateSubtitlesUseCase(translator=_Retranslator({}))
        fixed = use_case._revert_cjk_regressions(current, previous, "STYLE")
        assert [line.text for line in fixed] == [line.text for line in previous]
        assert _events("cjk_regressions_reverted")[0]["lines"] == [1]
        assert _events("emptied_lines_reverted")[0]["lines"] == [2]


class TestContentDropRevertIsRecorded:
    def test_reverted_line_is_recorded(self) -> None:
        """Ca thật của v3.23.225: tinh chỉnh rút "Nên cháu mới theo mọi người" còn "Nên
        cháu hùa," — dòng 84 được hoàn nguyên; nay việc đó có trong file chuẩn đoán."""
        source, good = "你今天来这里做什么", "Hôm nay cháu tới đây làm gì thế hả cháu"
        previous = [_line(i, good, source) for i in range(1, 30)]
        current = [_line(i, good, source) for i in range(1, 30)]
        previous.append(_line(84, "Nên cháu mới theo mọi người", "所以我才跟着大家"))
        current.append(_line(84, "Nên cháu hùa,", "所以我才跟着大家"))
        revert = GeminiSubtitleTranslator._revert_stage_regressions
        fixed = revert(previous, current, True)
        assert fixed[-1].text == "Nên cháu mới theo mọi người"
        assert _events("content_drops_reverted") == [
            {"event": "content_drops_reverted", "lines": [84]}
        ]
