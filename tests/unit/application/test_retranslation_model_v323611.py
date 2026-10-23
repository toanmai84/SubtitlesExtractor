# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.611] Lượt dịch lại phụ phải dùng model CÒN hạn mức ngày.

Ca thật v3.23.609 (phim "灭门惨案II", nhật ký ``ae83b20d``): người dùng dịch thô bằng
``gemini-3.7-flash`` tới hết hạn mức, đổi giai đoạn phong cách sang ``gemini-3.6-flash``
rồi chạy tiếp. Trọng tài dòng lẻ vẫn gọi model dịch thô::

    Giai đoạn 3: Tinh chỉnh phong cách: dịch trọng tài cho 4 dòng lẻ bị viết lại thất
    bại (Model 'gemini-3.7-flash' đã chạm giới hạn 20 request/ngày ...) — giữ nguyên.

Cùng lỗi ở mọi lượt dịch lại phụ dùng giai đoạn dịch thô: lưới dải lệch, sửa khối lệch,
dọn CJK sót, dọn dòng trùng.
"""

from __future__ import annotations

from typing import Any

import pytest

from subtitles_extractor.application.services.retranslation_model import (
    choose_retranslation_model,
)
from subtitles_extractor.application.services.translation_process_log import (
    get_process_log,
)
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

EXHAUSTED = "gemini-3.7-flash"
AVAILABLE = "gemini-3.6-flash"


class _QuotaError(RuntimeError):
    """Giống lỗi adapter ném khi model đã chạm hạn mức ngày."""


class _Translator:
    """Bộ dịch giả: model hết hạn mức thì ném lỗi, model còn hạn mức dịch theo bảng."""

    def __init__(self, answers: dict[int, str], exhausted: set[str]) -> None:
        self.answers = answers
        self.exhausted = exhausted
        self.models: list[str] = []
        self.stages: list[TranslationStageConfig] = []

    def has_any_daily_quota(self, model_name: str) -> bool:
        return model_name not in self.exhausted

    def translate_stage(
        self, *, stage: TranslationStageConfig, input_lines: list[TranslationLine],
        **_kw: Any,
    ) -> list[TranslationLine]:
        self.models.append(stage.model_name)
        self.stages.append(stage)
        if stage.model_name in self.exhausted:
            raise _QuotaError(
                f"Model '{stage.model_name}' đã chạm giới hạn 20 request/ngày"
            )
        return [
            TranslationLine(
                index=line.index, start_ms=line.start_ms, end_ms=line.end_ms,
                text=self.answers.get(line.index, line.text),
                original_text=line.original_text,
            )
            for line in input_lines
        ]


def _stages(literal_model: str = EXHAUSTED) -> list[TranslationStageConfig]:
    return [
        TranslationStageConfig(
            kind=TranslationStageKind.LITERAL, model_name=literal_model,
            temperature=0.1, enable_thinking=True, thinking_level="low",
        ),
        TranslationStageConfig(
            kind=TranslationStageKind.STYLE, model_name=AVAILABLE, temperature=0.7
        ),
        TranslationStageConfig(
            kind=TranslationStageKind.LOCALIZE, model_name=AVAILABLE, temperature=0.5
        ),
    ]


def _request(literal_model: str = EXHAUSTED) -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[],
        stages=_stages(literal_model),
        context=TranslationContext(target_lang="Vietnamese", source_lang="zh"),
    )


# Dòng 10 "搔你胳肢窩": bản thô đúng, giai đoạn sau viết thành lời nghe được trong video.
ROWS: list[tuple[int, str, str, str]] = [
    (9, "一下子就騎在你的頭上", "lập tức trèo lên đầu mày.", "lập tức trèo lên đầu mày."),
    (10, "搔你胳肢窩", "Cù nách mày.", "Cười chết được."),
    (11, "可惜我龍游淺水遭蝦欺", "Chỉ tiếc rồng mắc cạn,", "Chỉ tiếc rồng mắc cạn,"),
    (12, "來，跟我來", "Nào, theo tôi.", "Nào, theo tôi."),
]


def _lines(column: int) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=row[0], start_ms=row[0] * 1000, end_ms=row[0] * 1000 + 900,
            text=row[column], original_text=row[1],
        )
        for row in ROWS
    ]


@pytest.fixture(autouse=True)
def _fresh_log() -> None:
    get_process_log().clear()


class TestChooseRetranslationModel:
    def test_keeps_literal_model_while_it_has_quota(self) -> None:
        assert choose_retranslation_model("a", ["b"], lambda _m: True) == "a"

    def test_falls_back_in_order(self) -> None:
        assert choose_retranslation_model(
            "a", ["b", "c"], lambda model: model == "c"
        ) == "c"

    def test_skips_blank_and_duplicate_names(self) -> None:
        asked: list[str] = []

        def has_quota(model: str) -> bool:
            asked.append(model)
            return False

        assert choose_retranslation_model("a", ["", "a", " b ", "b"], has_quota) == "a"
        assert asked == ["a", "b"]

    def test_no_model_left_keeps_literal_model(self) -> None:
        assert choose_retranslation_model("a", ["b"], lambda _m: False) == "a"


class TestLoneRewriteArbiterModel:
    def test_real_case_arbiter_uses_model_with_quota(self) -> None:
        translator = _Translator({10: "Cù nách mày."}, exhausted={EXHAUSTED})
        use_case = TranslateSubtitlesUseCase(translator=translator)
        result = use_case._revert_lone_rewrites(
            _lines(3), _lines(2), _lines(1), _request(), None,
            "Giai đoạn 3: Tinh chỉnh phong cách",
        )
        assert translator.models == [AVAILABLE]
        assert {line.index: line.text for line in result}[10] == "Cù nách mày."

    def test_arbiter_keeps_literal_settings_except_model(self) -> None:
        translator = _Translator({10: "Cù nách mày."}, exhausted={EXHAUSTED})
        use_case = TranslateSubtitlesUseCase(translator=translator)
        use_case._revert_lone_rewrites(
            _lines(3), _lines(2), _lines(1), _request(), None, "Giai đoạn 3"
        )
        stage = translator.stages[0]
        assert stage.kind is TranslationStageKind.LITERAL
        assert stage.temperature == 0.1
        assert stage.enable_thinking is True

    def test_active_stage_model_is_tried_first(self) -> None:
        """Model của giai đoạn đang chạy vừa trả lời được nên được ưu tiên."""
        translator = _Translator({10: "Cù nách mày."}, exhausted={EXHAUSTED})
        stages = [
            *_stages(),
        ]
        stages[2] = TranslationStageConfig(
            kind=TranslationStageKind.LOCALIZE, model_name="gemini-3.5-flash-lite"
        )
        request = TranslateSubtitlesRequest(
            events=[], stages=stages,
            context=TranslationContext(target_lang="Vietnamese", source_lang="zh"),
        )
        use_case = TranslateSubtitlesUseCase(translator=translator)
        use_case._revert_lone_rewrites(
            _lines(3), _lines(2), _lines(1), request, None, "Giai đoạn 3",
            active_stage=stages[1],
        )
        assert translator.models == [AVAILABLE]

    def test_literal_model_with_quota_is_unchanged(self) -> None:
        translator = _Translator({10: "Cù nách mày."}, exhausted=set())
        use_case = TranslateSubtitlesUseCase(translator=translator)
        use_case._revert_lone_rewrites(
            _lines(3), _lines(2), _lines(1), _request(), None, "Giai đoạn 3"
        )
        assert translator.models == [EXHAUSTED]

    def test_switch_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        translator = _Translator({10: "Cù nách mày."}, exhausted={EXHAUSTED})
        use_case = TranslateSubtitlesUseCase(translator=translator)
        with caplog.at_level("INFO"):
            use_case._revert_lone_rewrites(
                _lines(3), _lines(2), _lines(1), _request(), None, "Giai đoạn 3"
            )
        assert any(
            EXHAUSTED in record.getMessage() and AVAILABLE in record.getMessage()
            for record in caplog.records
        )


class TestOtherRetranslationsUseModelWithQuota:
    def test_residual_cjk_cleanup(self) -> None:
        translator = _Translator({10: "Cù nách mày."}, exhausted={EXHAUSTED})
        use_case = TranslateSubtitlesUseCase(translator=translator)
        current = [
            TranslationLine(index=10, start_ms=0, end_ms=900, text="Cù 胳肢窩 mày.",
                            original_text="搔你胳肢窩"),
        ]
        source = [
            TranslationLine(index=10, start_ms=0, end_ms=900, text="搔你胳肢窩",
                            original_text="搔你胳肢窩"),
        ]
        result = use_case._cleanup_residual_cjk(current, source, _request(), None)
        assert translator.models == [AVAILABLE]
        assert result[0].text == "Cù nách mày."

    def test_duplicate_adjacent_cleanup(self) -> None:
        translator = _Translator(
            {1: "Đến cả lương của mọi người", 2: "cũng không phát nổi."},
            exhausted={EXHAUSTED},
        )
        use_case = TranslateSubtitlesUseCase(translator=translator)
        current = [
            TranslationLine(index=1, start_ms=0, end_ms=900,
                            text="không phát nổi lương.", original_text="连大家的工资都"),
            TranslationLine(index=2, start_ms=1000, end_ms=1900,
                            text="không phát nổi lương.", original_text="发不了"),
        ]
        source = [
            TranslationLine(index=1, start_ms=0, end_ms=900, text="连大家的工资都",
                            original_text="连大家的工资都"),
            TranslationLine(index=2, start_ms=1000, end_ms=1900, text="发不了",
                            original_text="发不了"),
        ]
        use_case._cleanup_duplicate_adjacent(current, source, _request(), None)
        assert translator.models == [AVAILABLE]


class TestPipelineWiring:
    """Cả lượt dịch: trọng tài giai đoạn phong cách dùng model của chính giai đoạn đó."""

    def test_execute_arbiter_uses_running_stage_model(self) -> None:
        from dataclasses import replace

        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        class _Pipeline:
            """Dịch thô dùng hết hạn mức của model đó; giai đoạn sau viết lại dòng 10."""

            def __init__(self) -> None:
                self.literal_done = False
                self.subset_models: list[str] = []

            def has_any_daily_quota(self, model_name: str) -> bool:
                return not (self.literal_done and model_name == EXHAUSTED)

            def translate_stage(self, *, stage, source_lines, input_lines, **_kw):
                full = len(source_lines) == len(ROWS)
                if not full:
                    self.subset_models.append(stage.model_name)
                    if not self.has_any_daily_quota(stage.model_name):
                        raise _QuotaError(f"Model '{stage.model_name}' hết hạn mức")
                if stage.kind is TranslationStageKind.LOCALIZE:
                    return list(input_lines)
                column = 2 if stage.kind is TranslationStageKind.LITERAL else 3
                # Use case đánh số dòng lại từ 1 theo thứ tự phát.
                texts = {pos: row[column] for pos, row in enumerate(ROWS, start=1)}
                if stage.kind is TranslationStageKind.LITERAL and full:
                    self.literal_done = True
                return [replace(line, text=texts[line.index]) for line in input_lines]

        events = [
            SubtitleEvent(
                index=row[0], text=row[1],
                interval=TimeInterval(row[0], row[0] + 0.9),
            )
            for row in ROWS
        ]
        pipeline = _Pipeline()
        # Giai đoạn cuối dùng model khác: thứ tự dự phòng theo danh sách giai đoạn sẽ
        # chọn nó, nên chỉ đúng khi use case truyền giai đoạn ĐANG CHẠY xuống trọng tài.
        stages = _stages()
        stages[2] = replace(stages[2], model_name="gemini-3.5-flash-lite")
        request = TranslateSubtitlesRequest(
            events=events,
            stages=stages,
            context=TranslationContext(target_lang="Vietnamese", source_lang="zh"),
        )
        response = TranslateSubtitlesUseCase(translator=pipeline).execute(request)
        final = [event.text for event in response.events]
        assert pipeline.subset_models == [AVAILABLE]
        assert final[1] == "Cù nách mày."
