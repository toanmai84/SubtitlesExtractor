"""[v3.23.608] Câu chữ của màn "Quản lý giai đoạn" và thông báo sau lượt dịch."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta, timezone

from subtitles_extractor.application.services.stage_cache import (
    StageAction,
    StageCacheEntry,
    StageReason,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationStageKind,
)
from subtitles_extractor.presentation.view_models.stage_cache_texts import (
    describe_cache_reuse,
    describe_next_run,
    describe_stored,
    format_saved_time,
)

LITERAL = TranslationStageKind.LITERAL
STYLE = TranslationStageKind.STYLE

BASE = StageCacheEntry(
    kind=STYLE,
    label="Tinh chỉnh",
    enabled=True,
    stored=True,
    line_count=3902,
    stored_model="gemini-3.5-flash-lite",
    requested_model="gemini-3.8-flash",
    completed_at="2026-10-13T00:49:00+00:00",
    edited_lines=0,
    partial_lines=0,
    partial_model="",
    action=StageAction.REUSE,
    reason=StageReason.NONE,
    reusable_lines=3902,
    total_lines=3902,
)


def test_next_run_texts() -> None:
    assert describe_next_run(BASE) == "Dùng lại nguyên, không gọi API"
    assert describe_next_run(replace(BASE, edited_lines=2)) == (
        "Dùng lại nguyên, không gọi API · gồm 2 dòng sửa tay"
    )
    weaker = replace(
        BASE, action=StageAction.RETRANSLATE, reason=StageReason.WEAKER_MODEL,
        edited_lines=4,
    )
    assert describe_next_run(weaker) == (
        "Dịch lại — bản lưu dịch bằng gemini-3.5-flash-lite, yếu hơn gemini-3.8-flash "
        "đang chọn · áp lại 4 dòng sửa tay"
    )
    partial = replace(
        BASE, action=StageAction.PARTIAL, reason=StageReason.INPUT_CHANGED,
        reusable_lines=3896,
    )
    assert describe_next_run(partial) == (
        "Giữ 3896/3902 dòng, chỉ dịch lại các lô còn thiếu — đầu vào đã đổi vì giai "
        "đoạn trước được sửa hoặc dịch lại"
    )
    unknown = replace(
        partial, reason=StageReason.UPSTREAM_PENDING, reusable_lines=-1
    )
    assert "các lô có dòng đổi" in describe_next_run(unknown)
    skipped = replace(BASE, action=StageAction.SKIPPED, edited_lines=2)
    assert describe_next_run(skipped) == (
        "Không chạy (giai đoạn đang tắt) — bản lưu vẫn được giữ"
    )
    assert describe_next_run(replace(BASE, action=StageAction.ABSENT)) == "—"
    assert describe_next_run(
        replace(BASE, action=StageAction.TRANSLATE, reason=StageReason.NOT_CACHED)
    ) == "Dịch mới"


def test_stored_texts() -> None:
    assert describe_stored(BASE) == "3902 dòng"
    assert describe_stored(replace(BASE, partial_lines=120)) == (
        "3902 dòng · lần dịch dở: 120 dòng"
    )
    empty = replace(BASE, stored=False, line_count=0)
    assert describe_stored(empty) == "Chưa có"
    assert describe_stored(replace(empty, partial_lines=360)) == (
        "Đang dở: 360/3902 dòng"
    )
    assert describe_stored(replace(empty, edited_lines=3)).startswith("Chỉ còn 3 dòng")


def test_saved_time_in_local_zone() -> None:
    vietnam = timezone(timedelta(hours=7))
    assert format_saved_time(BASE.completed_at, vietnam) == "13/10 07:49"
    assert format_saved_time("", vietnam) == "—"
    assert format_saved_time("không phải ngày", vietnam) == "—"


def test_reuse_summary_after_a_run() -> None:
    assert describe_cache_reuse((), {}, {}) == ""
    assert describe_cache_reuse(
        (LITERAL,), {STYLE: 3896}, {LITERAL: 4}
    ) == (
        "Dùng lại nguyên: Dịch thô. Chỉ dịch lại phần đổi: Tinh chỉnh (giữ 3896 dòng). "
        "Áp lại dòng sửa tay: Dịch thô 4."
    )
