"""[v3.23.149] Test dịch SONG SONG theo đợt (wave) + dự đoán quota khi song song.

Kiểm các bất biến quan trọng:
- Kết quả ghép ĐÚNG THỨ TỰ batch bất kể thứ tự hoàn thành (luồng nhanh/chậm xen kẽ).
- ``history_before`` neo theo ĐỢT đã chốt: đợt 1 lịch sử rỗng, đợt 2 thấy dòng đợt 1.
- ``_effective_parallel`` tự hạ theo RPM limit của model (dự đoán, không mở luồng thừa).
- Lỗi ở một batch trong đợt -> raise (không nuốt), đúng loại lỗi gốc.
"""

from __future__ import annotations

import time

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    SubtitleTranslationError,
)


def _lines(n: int) -> list[TranslationLine]:
    return [
        TranslationLine(
            index=i + 1, start_ms=i * 1000, end_ms=i * 1000 + 900, text=f"src {i + 1}"
        )
        for i in range(n)
    ]


def _stage() -> TranslationStageConfig:
    return TranslationStageConfig(
        kind=TranslationStageKind.LITERAL, model_name="gemini-3.1-flash-lite",
        batch_size=2, context_size=2,
    )


def _adapter(parallel: int) -> GeminiSubtitleTranslator:
    return GeminiSubtitleTranslator(api_key="KEY_A", parallel_batches=parallel)


def _run_parallel(adapter, lines, stage, histories=None):
    """Chạy _run_batches_parallel với _translate_single_batch giả (đánh dấu batch)."""

    def fake_batch(*, batch, history_before, start_idx, **_kw):
        if histories is not None:
            histories[start_idx] = list(history_before)
        # Batch đầu ngủ lâu hơn để đảo thứ tự hoàn thành trong đợt.
        time.sleep(0.05 if start_idx == 0 else 0.005)
        from dataclasses import replace
        return [replace(ln, text=f"vi {ln.index}") for ln in batch]

    adapter._translate_single_batch = fake_batch  # type: ignore[method-assign]
    return adapter._run_batches_parallel(
        parallel=2, input_lines=lines, source_lines=lines, batch_size=2,
        ctx_size=2, total_batches=(len(lines) + 1) // 2, stage=stage,
        config=None, is_literal=True, is_preprocess=False,
        use_dual_payload=False, progress_cb=None, cancel_cb=None,
    )


def test_parallel_output_order_preserved() -> None:
    adapter = _adapter(2)
    lines = _lines(8)  # 4 batch, 2 đợt x 2 luồng
    output = _run_parallel(adapter, lines, _stage())
    assert [ln.index for ln in output] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [ln.text for ln in output] == [f"vi {i}" for i in range(1, 9)]


def test_history_anchored_per_wave() -> None:
    adapter = _adapter(2)
    lines = _lines(8)
    histories: dict[int, list] = {}
    _run_parallel(adapter, lines, _stage(), histories)
    # Đợt 1 (batch start 0 và 2): lịch sử RỖNG (chưa có đợt nào chốt).
    assert histories[0] == [] and histories[2] == []
    # Đợt 2 (batch start 4 và 6): lịch sử = đuôi output đợt 1 (ctx_size=2 -> dòng 3,4).
    for start in (4, 6):
        nos = [item.get("line_no") for item in histories[start]]
        assert nos == [3, 4]


def test_effective_parallel_capped_by_rpm() -> None:
    quota = GeminiQuotaManager(
        rate_limits={"gemini-3.5-flash": RateLimit(rpm=3, tpm=250_000, rpd=20)}
    )
    adapter = GeminiSubtitleTranslator(
        api_key="KEY_A", quota_manager=quota, parallel_batches=4
    )
    # RPM=3 < yêu cầu 4 -> hạ về 3 (không mở luồng thừa).
    assert adapter._effective_parallel(4, "gemini-3.5-flash") == 3


def test_effective_parallel_no_quota_manager_keeps_request() -> None:
    adapter = _adapter(3)
    assert adapter._effective_parallel(3, "gemini-3.1-flash-lite") == 3
    assert adapter._effective_parallel(1, "gemini-3.1-flash-lite") == 1


def test_parallel_error_propagates() -> None:
    adapter = _adapter(2)
    lines = _lines(4)

    def failing_batch(*, batch, start_idx, **_kw):
        if start_idx == 2:
            raise SubtitleTranslationError("batch hỏng")
        from dataclasses import replace
        return [replace(ln, text=f"vi {ln.index}") for ln in batch]

    adapter._translate_single_batch = failing_batch  # type: ignore[method-assign]
    try:
        adapter._run_batches_parallel(
            parallel=2, input_lines=lines, source_lines=lines, batch_size=2,
            ctx_size=0, total_batches=2, stage=_stage(), config=None,
            is_literal=True, is_preprocess=False, use_dual_payload=False,
            progress_cb=None, cancel_cb=None,
        )
        raise AssertionError("Phải raise SubtitleTranslationError")
    except SubtitleTranslationError:
        pass


def test_state_lock_exists_and_reentrant() -> None:
    adapter = _adapter(2)
    with adapter._state_lock:  # noqa: SIM117 — lồng CHỦ ĐÍCH để kiểm RLock reentrant
        with adapter._state_lock:
            assert True
