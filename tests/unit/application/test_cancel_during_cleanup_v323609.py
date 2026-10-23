"""[v3.23.609] Bấm Huỷ trong các bước dịch lại phụ (dọn dòng trùng, dọn chữ Hán sót).

Ba bước dịch lại phụ bắt chung ``SubtitleTranslationError`` — mà lỗi "người dùng huỷ"
(``TranslationCancelledError``) là lớp con của nó. Bấm Huỷ đúng lúc đó thì lỗi bị nuốt:
lượt dịch chạy tiếp tới cuối và báo "Dịch hoàn tất" như chưa từng bấm Huỷ. Với bước sửa
khối lệch (giữa giai đoạn), giai đoạn còn được LƯU vào bộ nhớ đệm khi chưa sửa xong — lần
sau dùng lại nguyên bản chưa sửa.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationCancelledError,
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

LITERAL = TranslationStageKind.LITERAL


class _CancelOnRetry:
    """Lô đầu dịch ra ``outputs``; lượt gọi kế (bước dịch lại phụ) = bấm Huỷ."""

    def __init__(self, outputs: dict[int, str]) -> None:
        self.outputs = outputs
        self.cancelled = False
        self.calls = 0

    def __call__(self, *, batch, **_kwargs):
        self.calls += 1
        if self.calls > 1:
            self.cancelled = True
            raise TranslationCancelledError("Người dùng đã huỷ tiến trình dịch.")
        return [replace(line, text=self.outputs[line.index]) for line in batch]


def _run(tmp_path: Path, outputs: dict[int, str]) -> tuple[_CancelOnRetry, object]:
    batches = _CancelOnRetry(outputs)
    adapter = GeminiSubtitleTranslator(api_key="KEY")
    adapter._ensure_available = lambda: None  # type: ignore[method-assign]
    adapter._translate_single_batch = batches  # type: ignore[method-assign]
    request = TranslateSubtitlesRequest(
        events=[
            SubtitleEvent(index=i, text=text, interval=TimeInterval(i, i + 0.9))
            for i, text in enumerate(("你好", "我们走", "快跑", "他来了"), start=1)
        ],
        stages=[
            TranslationStageConfig(kind=LITERAL, model_name="gemini-3.8-flash",
                                   batch_size=10)
        ],
        context=TranslationContext(target_lang="Vietnamese", source_lang="zh"),
    )
    use_case = TranslateSubtitlesUseCase(translator=adapter, checkpoint_dir=tmp_path)
    return batches, (use_case, request)


def test_cancel_while_cleaning_duplicate_lines_stops_the_run(tmp_path: Path) -> None:
    batches, (use_case, request) = _run(
        tmp_path, {1: "Xin chào", 2: "Chạy mau", 3: "Chạy mau", 4: "Anh ta tới rồi"}
    )
    with pytest.raises(TranslationCancelledError):
        use_case.execute(request, cancel_cb=lambda: batches.cancelled)


def test_cancel_while_cleaning_leftover_hanzi_stops_the_run(tmp_path: Path) -> None:
    batches, (use_case, request) = _run(
        tmp_path, {1: "Xin chào", 2: "Ta đi", 3: "快跑", 4: "Anh ta tới rồi"}
    )
    with pytest.raises(TranslationCancelledError):
        use_case.execute(request, cancel_cb=lambda: batches.cancelled)


def test_interruption_that_is_not_a_user_cancel_is_still_tolerated(
    tmp_path: Path,
) -> None:
    """Lỗi 'đã huỷ' do mạng (người dùng KHÔNG bấm Huỷ) ở bước phụ: bỏ qua bước đó."""
    _batches, (use_case, request) = _run(
        tmp_path, {1: "Xin chào", 2: "Chạy mau", 3: "Chạy mau", 4: "Anh ta tới rồi"}
    )
    response = use_case.execute(request, cancel_cb=lambda: False)
    assert [event.text for event in response.events][:2] == ["Xin chào", "Chạy mau"]
