"""[v3.23.134] Test: giai đoạn RESUME từ checkpoint vẫn điền stage_outputs.

Trước fix: stage được resume KHÔNG ghi vào stage_outputs → màn so sánh giai đoạn (cần >=2
giai đoạn) bị thiếu giai đoạn đã resume sau khi tiếp tục dịch (huỷ/treo rồi chạy lại).
"""

from __future__ import annotations

from pathlib import Path

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
    ]


def _stage(kind: TranslationStageKind) -> TranslationStageConfig:
    return TranslationStageConfig(kind=kind, model_name="fake-model", batch_size=50)


def _request() -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=_events(),
        stages=[
            _stage(TranslationStageKind.LITERAL),
            _stage(TranslationStageKind.STYLE),
        ],
        context=TranslationContext(target_lang="Vietnamese"),
    )


def test_resumed_stage_populates_stage_outputs(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"

    # Lần 1: chạy đầy đủ, lưu checkpoint (rồi xoá khi xong) — nên ta CHẶN ở giữa bằng
    # cách chạy chỉ tới khi có checkpoint. Đơn giản hơn: dùng translator ném lỗi ở STYLE
    # để checkpoint LITERAL còn lại, rồi chạy lại lần 2.
    class _FailOnStyle(FakeSubtitleTranslator):
        def translate_stage(self, *, stage, **kw):  # type: ignore[no-untyped-def]
            if stage.kind is TranslationStageKind.STYLE:
                raise RuntimeError("Mô phỏng treo ở STYLE")
            return super().translate_stage(stage=stage, **kw)

    import contextlib

    uc1 = TranslateSubtitlesUseCase(translator=_FailOnStyle(), checkpoint_dir=ckpt)
    with contextlib.suppress(RuntimeError):
        uc1.execute(_request())  # checkpoint LITERAL được lưu trước khi treo

    # Lần 2: chạy lại — LITERAL resume từ checkpoint, STYLE chạy thật.
    uc2 = TranslateSubtitlesUseCase(
        translator=FakeSubtitleTranslator(), checkpoint_dir=ckpt
    )
    response = uc2.execute(_request())

    # CẢ HAI giai đoạn phải có mặt trong stage_outputs (kể cả LITERAL đã resume).
    assert TranslationStageKind.LITERAL in response.stage_outputs
    assert TranslationStageKind.STYLE in response.stage_outputs
    assert len(response.stage_outputs) == 2  # đủ 2 -> màn so sánh hoạt động
    # Giai đoạn resume giữ đúng số dòng.
    assert len(response.stage_outputs[TranslationStageKind.LITERAL]) == 2
