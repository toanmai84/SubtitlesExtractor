"""[v3.23.609] Bấm Dịch lại khi mọi giai đoạn đã có trong bộ nhớ đệm: không nén/tải video.

Từ v3.23.608 bộ nhớ đệm được giữ sau khi dịch xong, nên lượt dịch lại (vd sau khi sửa
tay giai đoạn cuối) thường không cần gọi model nào. Worker vẫn nén + tải toàn bộ video
(nhật ký thật: 17 đoạn, hơn 5 phút) và còn chặn hẳn lượt dịch nếu model đính video đã
hết hạn mức ngày — dù chẳng cần gọi model.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

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
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

LITERAL = TranslationStageKind.LITERAL
STYLE = TranslationStageKind.STYLE


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _VideoProvider:
    def __init__(self) -> None:
        self.prepared = 0

    def set_active_key(self, _key: str) -> None:
        return None

    def prepare_and_upload(self, *_args: object, **_kwargs: object) -> list:
        self.prepared += 1
        return []


def _use_case(tmp_path: Path, calls: list[int], *, quota: bool = True):
    adapter = GeminiSubtitleTranslator(api_key="KEY")
    adapter._ensure_available = lambda: None  # type: ignore[method-assign]

    def _batch(*, batch, start_idx, **_kwargs):
        calls.append(start_idx)
        return [replace(line, text=f"dịch {line.index}") for line in batch]

    adapter._translate_single_batch = _batch  # type: ignore[method-assign]
    use_case = TranslateSubtitlesUseCase(translator=adapter, checkpoint_dir=tmp_path)
    use_case.has_any_daily_quota = lambda _model: quota  # type: ignore[method-assign]
    use_case.ensure_viable_key = lambda *_a, **_k: None  # type: ignore[method-assign]
    return use_case


def _request() -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[
            SubtitleEvent(index=i, text=f"句{i}", interval=TimeInterval(i, i + 0.9))
            for i in range(1, 7)
        ],
        stages=[
            TranslationStageConfig(kind=kind, model_name="gemini-3.8-flash", batch_size=2)
            for kind in (LITERAL, STYLE)
        ],
        context=TranslationContext(target_lang="Vietnamese", source_lang="zh"),
    )


def _run_worker(use_case, provider, request) -> tuple[list, list[str]]:
    from subtitles_extractor.presentation.workers.translate_worker import TranslateWorker

    worker = TranslateWorker(
        use_case=use_case, request=request, video_provider=provider,
        video_path="phim.mp4", attach_video_stages=frozenset({LITERAL}),
    )
    finished: list = []
    failed: list[str] = []
    worker.finished_ok.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.run()
    return finished, failed


def test_fully_cached_rerun_needs_no_video_and_no_quota(
    app: QApplication, tmp_path: Path
) -> None:
    calls: list[int] = []
    first = _use_case(tmp_path, calls)
    first.execute(replace(_request(), attach_video_stages=frozenset({LITERAL})))
    calls.clear()

    provider = _VideoProvider()
    exhausted = _use_case(tmp_path, calls, quota=False)
    finished, failed = _run_worker(exhausted, provider, _request())
    assert failed == []
    assert provider.prepared == 0
    assert calls == []
    assert finished and finished[0].reused_stages == (LITERAL, STYLE)


def test_rerun_that_needs_a_stage_still_prepares_the_video(
    app: QApplication, tmp_path: Path
) -> None:
    calls: list[int] = []
    _use_case(tmp_path, calls).execute(
        replace(_request(), attach_video_stages=frozenset({LITERAL}))
    )
    provider = _VideoProvider()
    changed = replace(
        _request(),
        stages=[
            _request().stages[0],
            replace(_request().stages[1], temperature=0.9),
        ],
    )
    finished, failed = _run_worker(_use_case(tmp_path, calls), provider, changed)
    assert failed == []
    assert provider.prepared == 1
    assert finished and finished[0].reused_stages == (LITERAL,)


def test_quota_check_uses_a_stage_that_will_actually_run(
    app: QApplication, tmp_path: Path
) -> None:
    """Dịch thô (3.8-flash, đã hết hạn mức) có sẵn trong bộ nhớ đệm; chỉ tinh chỉnh
    (3.7-flash, còn hạn mức) phải dịch. Trước đây worker xét hạn mức theo model của giai
    đoạn đính video ĐẦU TIÊN (dịch thô) → báo "mọi key hết quota" và chặn cả lượt."""
    calls: list[int] = []
    both_attached = frozenset({LITERAL, STYLE})
    _use_case(tmp_path, calls).execute(
        replace(_request(), attach_video_stages=both_attached)
    )
    stages = [
        _request().stages[0],
        replace(_request().stages[1], model_name="gemini-3.7-flash", temperature=0.9),
    ]
    use_case = _use_case(tmp_path, calls)
    use_case.has_any_daily_quota = (  # type: ignore[method-assign]
        lambda model: model != "gemini-3.8-flash"
    )
    from subtitles_extractor.presentation.workers.translate_worker import TranslateWorker

    provider = _VideoProvider()
    worker = TranslateWorker(
        use_case=use_case, request=replace(_request(), stages=stages),
        video_provider=provider, video_path="phim.mp4",
        attach_video_stages=both_attached,
    )
    failed: list[str] = []
    finished: list = []
    worker.failed.connect(failed.append)
    worker.finished_ok.connect(finished.append)
    worker.run()
    assert failed == []
    assert provider.prepared == 1
    assert finished and finished[0].reused_stages == (LITERAL,)


def test_cache_check_failure_falls_back_to_preparing_the_video(
    app: QApplication, tmp_path: Path
) -> None:
    calls: list[int] = []
    use_case = _use_case(tmp_path, calls)

    def _broken(_request):
        raise OSError("ổ đĩa không đọc được")

    use_case.stages_needing_model = _broken  # type: ignore[method-assign]
    provider = _VideoProvider()
    finished, failed = _run_worker(use_case, provider, _request())
    assert failed == []
    assert provider.prepared == 1
    assert finished
