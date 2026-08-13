"""Test tích hợp: pipeline dịch đa giai đoạn end-to-end (use case thật + fake translator).

Kiểm chứng các thành phần phối hợp đúng: thứ tự giai đoạn, bảo toàn số dòng/chỉ số, và kết
quả tích luỹ ``stage_outputs`` dùng cho tính năng so sánh giai đoạn.
"""

from __future__ import annotations

import pytest

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
from tests.integration.fake_translator import FakeSubtitleTranslator


def _events() -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=1, text="你好", interval=TimeInterval(0.0, 1.0)),
        SubtitleEvent(index=2, text="再见", interval=TimeInterval(1.0, 2.0)),
        SubtitleEvent(index=3, text="谢谢", interval=TimeInterval(2.0, 3.0)),
    ]


def _stage(kind: TranslationStageKind) -> TranslationStageConfig:
    return TranslationStageConfig(kind=kind, model_name="fake-model", batch_size=50)


def test_single_stage_pipeline() -> None:
    translator = FakeSubtitleTranslator()
    use_case = TranslateSubtitlesUseCase(translator=translator)
    request = TranslateSubtitlesRequest(
        events=_events(),
        stages=[_stage(TranslationStageKind.LITERAL)],
        context=TranslationContext(target_lang="Vietnamese"),
    )
    response = use_case.execute(request)

    assert len(response.events) == 3
    # Số dòng & index được bảo toàn.
    assert [e.index for e in response.events] == [1, 2, 3]
    # Giai đoạn LITERAL đã chạy (tiền tố nhận biết).
    assert all(e.text.startswith("[literal] ") for e in response.events)
    assert response.completed_stages == [TranslationStageKind.LITERAL]


def test_multi_stage_runs_in_order() -> None:
    translator = FakeSubtitleTranslator()
    use_case = TranslateSubtitlesUseCase(translator=translator)
    request = TranslateSubtitlesRequest(
        events=_events(),
        stages=[
            _stage(TranslationStageKind.LITERAL),
            _stage(TranslationStageKind.STYLE),
            _stage(TranslationStageKind.LOCALIZE),
        ],
        context=TranslationContext(target_lang="Vietnamese"),
    )
    response = use_case.execute(request)

    # Đúng thứ tự pipeline.
    assert translator.translate_stage_calls == ["literal", "style", "localize"]
    assert response.completed_stages == [
        TranslationStageKind.LITERAL,
        TranslationStageKind.STYLE,
        TranslationStageKind.LOCALIZE,
    ]
    # Giai đoạn sau xử lý trên đầu ra giai đoạn trước → tiền tố lồng nhau.
    final_text = response.events[0].text
    assert "[localize] " in final_text and "[style] " in final_text


def test_stage_outputs_accumulated_for_comparison() -> None:
    translator = FakeSubtitleTranslator()
    use_case = TranslateSubtitlesUseCase(translator=translator)
    request = TranslateSubtitlesRequest(
        events=_events(),
        stages=[
            _stage(TranslationStageKind.LITERAL),
            _stage(TranslationStageKind.STYLE),
        ],
        context=TranslationContext(target_lang="Vietnamese"),
    )
    response = use_case.execute(request)

    # stage_outputs giữ kết quả của TỪNG giai đoạn (cho tính năng so sánh).
    assert set(response.stage_outputs.keys()) == {
        TranslationStageKind.LITERAL, TranslationStageKind.STYLE,
    }
    literal_out = response.stage_outputs[TranslationStageKind.LITERAL]
    assert len(literal_out) == 3
    # Đầu ra LITERAL chỉ có 1 lớp tiền tố, STYLE có 2 lớp.
    assert literal_out[0].text.startswith("[literal] ")
    style_out = response.stage_outputs[TranslationStageKind.STYLE]
    assert style_out[0].text.startswith("[style] [literal] ")


def test_consistent_dictionary_translation() -> None:
    # Mô phỏng bản dịch nhất quán (như Translation Memory): câu trong từ điển dịch cố định.
    translator = FakeSubtitleTranslator(
        dictionary={"你好": "Xin chào", "再见": "Tạm biệt", "谢谢": "Cảm ơn"}
    )
    use_case = TranslateSubtitlesUseCase(translator=translator)
    request = TranslateSubtitlesRequest(
        events=_events(),
        stages=[_stage(TranslationStageKind.LITERAL)],
        context=TranslationContext(target_lang="Vietnamese"),
    )
    response = use_case.execute(request)
    assert [e.text for e in response.events] == ["Xin chào", "Tạm biệt", "Cảm ơn"]


def test_empty_events_raises() -> None:
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        SubtitleTranslationError,
    )
    translator = FakeSubtitleTranslator()
    use_case = TranslateSubtitlesUseCase(translator=translator)
    request = TranslateSubtitlesRequest(
        events=[],
        stages=[_stage(TranslationStageKind.LITERAL)],
        context=TranslationContext(target_lang="Vietnamese"),
    )
    # Use case từ chối dịch khi không có phụ đề (hành vi bảo vệ).
    with pytest.raises(SubtitleTranslationError):
        use_case.execute(request)
