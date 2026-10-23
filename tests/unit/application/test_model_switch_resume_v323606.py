"""[v3.23.606] Đổi model vì hết hạn mức ngày không được vứt các giai đoạn đã dịch xong.

Nhật ký thật 12/10 (灭门惨案II, v3.23.605, 1246 dòng, ba lần chạy liên tiếp):

* 05:50 — cả ba giai đoạn gemini-3.8-flash: dịch thô xong (22 lượt gọi, 43 phút), giai
  đoạn tinh chỉnh hết hạn mức ở lượt gọi đầu.
* 06:57 — đổi sang gemini-3.7-flash: "Bắt đầu Giai đoạn 2: Dịch thô" — dịch thô LẠI từ
  đầu (khoá checkpoint chứa tên model), tinh chỉnh được 11 lô thì hết hạn mức.
* 07:30 — đổi sang gemini-3.5-flash-lite: lại dịch thô và tinh chỉnh từ đầu; bản cuối
  hoàn toàn của model yếu nhất (240 dòng vượt ngân sách đọc, 80 dòng TTS đọc > 1,6x).

Mô phỏng đúng chuỗi đó với hạn mức thu nhỏ (lô 2 dòng, 12 dòng gốc thật 196-207).
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from subtitles_extractor.application.services.model_capability import (
    model_capability,
    result_reusable_for,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
    _compute_checkpoint_key,
    _compute_legacy_checkpoint_key,
    _StageCheckpoint,
    context_fingerprint,
    stage_input_fingerprint,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    QuotaExhaustedError,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

LITERAL = TranslationStageKind.LITERAL
STYLE = TranslationStageKind.STYLE
LOCALIZE = TranslationStageKind.LOCALIZE

STRONG = "gemini-3.8-flash"
MIDDLE = "gemini-3.6-flash"  # [v3.23.622] 3.7-flash là bí danh của 3.8 → dùng 3.6 làm hạng giữa
LITE = "gemini-3.5-flash-lite"

#: Dòng 196-207 thật của phim (câu gốc + mốc thời gian, giây).
REAL_SOURCES = [
    (816.0, 816.8, "不好意思"),
    (828.2, 829.933, "所長，怎麼這芒果是爛的"),  # noqa: RUF001 — chữ thật
    (831.933, 833.8, "不是爛的就不要浪費啦"),
    (840.867, 841.667, "芒果不是爛的剛才我."),
    (841.733, 841.799, "慢慢吃吧"),
    (841.8, 842.8, "所長，有發現呀"),  # noqa: RUF001 — chữ thật
    (844.733, 845.733, "這瓶精液"),
    (845.8, 847.733, "跟女死者身上發現的完全吻合"),
    (848.4, 849.6, "不過她有梅毒"),
    (850.0, 851.2, "你看看這份報告"),
    (852.0, 853.0, "我知道了"),
    (853.5, 854.9, "馬上去查"),
]
BATCH = 2
BATCHES_PER_STAGE = len(REAL_SOURCES) // BATCH


class _DailyQuota:
    """Thay bước gọi API một lô: mỗi model chỉ có ``budget[model]`` lượt mỗi "ngày"."""

    def __init__(self, budget: dict[str, int]) -> None:
        self.budget = dict(budget)
        self.calls: list[tuple[str, bool, int]] = []
        self._lock = threading.Lock()

    def __call__(self, *, batch, start_idx, model_name, is_literal=False, **_kwargs):
        with self._lock:
            if self.budget.get(model_name, 0) <= 0:
                raise QuotaExhaustedError(f"Model '{model_name}' hết hạn mức ngày.")
            self.budget[model_name] -= 1
            self.calls.append((model_name, is_literal, start_idx))
        return [replace(line, text=f"dịch {line.index}") for line in batch]

    def literal_calls(self) -> list[tuple[str, bool, int]]:
        return [call for call in self.calls if call[1]]


def _adapter(quota: _DailyQuota) -> GeminiSubtitleTranslator:
    adapter = GeminiSubtitleTranslator(api_key="KEY")
    adapter._ensure_available = lambda: None  # type: ignore[method-assign]
    adapter._translate_single_batch = quota  # type: ignore[method-assign]
    return adapter


def _request(model: str) -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[
            SubtitleEvent(index=i, text=text, interval=TimeInterval(start, end))
            for i, (start, end, text) in enumerate(REAL_SOURCES, start=1)
        ],
        stages=[
            TranslationStageConfig(kind=kind, model_name=model, batch_size=BATCH)
            for kind in (LITERAL, STYLE, LOCALIZE)
        ],
        context=TranslationContext(target_lang="Vietnamese", source_lang="zh"),
    )


def _run(tmp_path: Path, model: str, quota: _DailyQuota) -> bool:
    """Chạy một lượt dịch; True nếu xong, False nếu dừng vì hết hạn mức."""
    use_case = TranslateSubtitlesUseCase(
        translator=_adapter(quota), checkpoint_dir=tmp_path
    )
    try:
        use_case.execute(_request(model))
    except QuotaExhaustedError:
        return False
    return True


class TestMorningOfTwelfthOctober:
    def test_downgrades_keep_every_finished_batch(self, tmp_path: Path) -> None:
        # Lần 1: 3.8-flash đủ cho dịch thô, hết hạn mức ngay lô tinh chỉnh đầu tiên.
        first = _DailyQuota({STRONG: BATCHES_PER_STAGE})
        assert not _run(tmp_path, STRONG, first)
        assert len(first.literal_calls()) == BATCHES_PER_STAGE

        # Lần 2: 3.7-flash — bản dịch thô của 3.8-flash được giữ, hạn mức dành cho tinh
        # chỉnh; bản địa hoá được 2 lô thì hết.
        second = _DailyQuota({MIDDLE: BATCHES_PER_STAGE + 2})
        assert not _run(tmp_path, MIDDLE, second)
        assert second.literal_calls() == []
        assert len(second.calls) == BATCHES_PER_STAGE + 2

        # Lần 3: flash-lite chỉ còn phải dịch các lô bản địa hoá còn thiếu.
        third = _DailyQuota({LITE: 100})
        assert _run(tmp_path, LITE, third)
        assert third.calls == [
            (LITE, False, start) for start in range(2 * BATCH, len(REAL_SOURCES), BATCH)
        ]

    def test_upgrading_the_model_translates_again(self, tmp_path: Path) -> None:
        lite_run = _DailyQuota({LITE: BATCHES_PER_STAGE})
        assert not _run(tmp_path, LITE, lite_run)
        strong_run = _DailyQuota({STRONG: 100})
        assert _run(tmp_path, STRONG, strong_run)
        # Bản dịch thô của flash-lite KHÔNG được dùng thay cho 3.8-flash.
        assert len(strong_run.literal_calls()) == BATCHES_PER_STAGE

    def test_partial_stage_of_a_weaker_model_is_not_kept_after_upgrade(
        self, tmp_path: Path
    ) -> None:
        lite_run = _DailyQuota({LITE: BATCHES_PER_STAGE + 2})
        assert not _run(tmp_path, LITE, lite_run)  # dịch thô xong + 2 lô tinh chỉnh
        middle_run = _DailyQuota({MIDDLE: 100})
        assert _run(tmp_path, MIDDLE, middle_run)
        assert len(middle_run.calls) == 3 * BATCHES_PER_STAGE

    def test_file_left_by_v323605_is_adopted(self, tmp_path: Path) -> None:
        """Checkpoint lần chạy dở do bản cũ ghi (khoá chứa cấu hình) vẫn được nối tiếp."""
        request = _request(STRONG)
        legacy = _StageCheckpoint(
            tmp_path,
            _compute_legacy_checkpoint_key(request),
            context_fingerprint(request.context),
        )
        sources = TranslateSubtitlesUseCase._events_to_lines(request.events)
        literal = [replace(line, text=f"dịch {line.index}") for line in sources]
        legacy.save_stage(LITERAL, literal)
        style_input = stage_input_fingerprint(STYLE, literal)
        legacy.add_partial(STYLE, style_input, literal[:BATCH])

        quota = _DailyQuota({STRONG: 100})
        assert _run(tmp_path, STRONG, quota)
        assert quota.literal_calls() == []
        # Tinh chỉnh chỉ dịch các lô còn thiếu, bản địa hoá dịch đủ.
        assert len(quota.calls) == 2 * BATCHES_PER_STAGE - 1
        # [v3.23.608] File khoá cũ đã được nhận rồi xoá; bộ nhớ đệm khoá mới được GIỮ sau
        # khi dịch xong (để sửa/dùng lại từng giai đoạn).
        assert [path.name for path in tmp_path.glob("*.json")] == [
            f"{_compute_checkpoint_key(request)}.json"
        ]

    def test_new_checkpoint_records_model_of_each_stage(self, tmp_path: Path) -> None:
        assert not _run(tmp_path, STRONG, _DailyQuota({STRONG: BATCHES_PER_STAGE}))
        saved = json.loads(
            (tmp_path / f"{_compute_checkpoint_key(_request(LITE))}.json").read_text(
                encoding="utf-8"
            )
        )
        assert saved["stage_runs"][LITERAL.value]["model"] == STRONG


class TestModelCapability:
    @pytest.mark.parametrize(
        ("stored", "requested", "reusable"),
        [
            (STRONG, MIDDLE, True),
            (STRONG, LITE, True),
            (MIDDLE, LITE, True),
            (LITE, STRONG, False),
            (MIDDLE, STRONG, False),
            ("gemini-3.5-flash", "gemini-3.8-flash-lite", True),
            ("gemini-3.8-flash-lite", "gemini-3.5-flash", False),
            (STRONG, STRONG, True),
            ("gemini-flash-latest", MIDDLE, False),
            ("gpt-5", MIDDLE, False),
        ],
    )
    def test_reuse_only_when_stored_model_is_not_weaker(
        self, stored: str, requested: str, reusable: bool
    ) -> None:
        assert result_reusable_for(stored, requested) is reusable

    def test_lite_ranks_below_flash_of_any_generation(self) -> None:
        lite = model_capability("gemini-3.8-flash-lite")
        flash = model_capability("gemini-3.5-flash")
        assert lite is not None and flash is not None
        assert lite < flash


class TestClearCacheButton:
    def test_clearing_also_removes_the_file_left_by_v323605(self, tmp_path: Path) -> None:
        """Nút "Xoá bộ nhớ đệm dịch" phải xoá cả file khoá cũ — nếu không, lần dịch tới
        lại nhận nó về và "xoá" không có tác dụng.

        [v3.23.608] Gọi qua ViewModel thật (khoá tính đúng như lúc bấm Dịch).
        """
        from subtitles_extractor.composition.container import ApplicationContainer
        from subtitles_extractor.infrastructure.settings.json_file_repository import (
            JsonFileRepository,
        )
        from subtitles_extractor.presentation.view_models.translate_page_view_model import (
            TranslatePageViewModel,
        )

        request = _request(STRONG)
        container = ApplicationContainer(
            settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
            i18n_data_dir=tmp_path,
            user_data_dir=tmp_path,
        )
        checkpoint_dir = container.stage_checkpoint_dir
        sources = TranslateSubtitlesUseCase._events_to_lines(request.events)
        keys = (_compute_checkpoint_key(request), _compute_legacy_checkpoint_key(request))
        for key in keys:
            _StageCheckpoint(checkpoint_dir, key).save_stage(LITERAL, sources)
        view_model = TranslatePageViewModel(container)
        view_model.set_source_events(list(request.events))
        messages: list[str] = []
        view_model.status_message.connect(messages.append)
        removed = view_model.clear_translation_checkpoint(
            list(request.stages), request.context
        )
        assert removed == 2
        assert list(checkpoint_dir.glob("*.json")) == []
        assert messages
