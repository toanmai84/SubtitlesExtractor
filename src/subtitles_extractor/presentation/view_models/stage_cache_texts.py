"""[v3.23.608] Câu chữ hiển thị cho màn "Quản lý giai đoạn" (hàm thuần, không Qt).

Tách khỏi widget để kiểm thử được và để trang dịch dùng chung (thông báo sau khi dịch).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, tzinfo

from subtitles_extractor.application.services.stage_cache import (
    STAGE_LABELS,
    StageAction,
    StageCacheEntry,
    StageReason,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationStageKind,
)

__all__ = [
    "describe_cache_reuse",
    "describe_next_run",
    "describe_reason",
    "describe_stored",
    "format_saved_time",
]

_REASON_TEXTS: dict[StageReason, str] = {
    StageReason.CONFIG_CHANGED: (
        "cấu hình giai đoạn đã đổi: phong cách, nhiệt độ, cỡ lô, mức suy nghĩ hoặc đính "
        "video"
    ),
    StageReason.INPUT_CHANGED: "đầu vào đã đổi vì giai đoạn trước được sửa hoặc dịch lại",
    StageReason.CONTEXT_CHANGED: "bảng nhân vật/thuật ngữ đã đổi so với lúc lưu",
    StageReason.UPSTREAM_PENDING: "giai đoạn trước sẽ dịch lại",
    StageReason.NO_RUN_INFO: "bản lưu của phiên bản cũ, không rõ điều kiện dịch",
    StageReason.UNVERIFIED_INPUT: "bản lưu của phiên bản cũ, không rõ đầu vào",
    StageReason.LINE_COUNT_CHANGED: "số dòng phụ đề đã đổi",
}


def describe_reason(entry: StageCacheEntry) -> str:
    """Lý do giai đoạn không được dùng lại nguyên (rỗng nếu không có lý do để nói)."""
    if entry.reason is StageReason.WEAKER_MODEL:
        return (
            f"bản lưu dịch bằng {entry.stored_model}, yếu hơn {entry.requested_model} "
            "đang chọn"
        )
    return _REASON_TEXTS.get(entry.reason, "")


def describe_next_run(entry: StageCacheEntry) -> str:
    """Việc lần bấm "Dịch" tới sẽ làm với giai đoạn (theo cấu hình đang chọn).

    Args:
        entry: Trạng thái giai đoạn.

    Returns:
        Câu ngắn cho cột "Lần dịch tới".
    """
    reason = describe_reason(entry)
    if entry.action is StageAction.REUSE:
        text = "Dùng lại nguyên, không gọi API"
    elif entry.action is StageAction.PARTIAL:
        if entry.reusable_lines >= 0:
            text = (
                f"Giữ {entry.reusable_lines}/{entry.total_lines} dòng, chỉ dịch lại các "
                "lô còn thiếu"
            )
        else:
            text = "Giữ các dòng có đầu vào không đổi, chỉ dịch lại các lô có dòng đổi"
        if reason:
            text += f" — {reason}"
    elif entry.action is StageAction.RETRANSLATE:
        text = f"Dịch lại — {reason}" if reason else "Dịch lại"
    elif entry.action is StageAction.TRANSLATE:
        text = f"Dịch mới — {reason}" if reason else "Dịch mới"
    elif entry.action is StageAction.SKIPPED:
        text = "Không chạy (giai đoạn đang tắt) — bản lưu vẫn được giữ"
    else:
        return "—"
    if entry.edited_lines and entry.action is StageAction.REUSE:
        text += f" · gồm {entry.edited_lines} dòng sửa tay"
    elif entry.edited_lines and entry.action is not StageAction.SKIPPED:
        text += f" · áp lại {entry.edited_lines} dòng sửa tay"
    return text


def describe_stored(entry: StageCacheEntry) -> str:
    """Tình trạng bản lưu của giai đoạn (cột "Bản đã lưu")."""
    if entry.stored:
        text = f"{entry.line_count} dòng"
        if entry.partial_lines:
            text += f" · lần dịch dở: {entry.partial_lines} dòng"
        return text
    if entry.partial_lines:
        return f"Đang dở: {entry.partial_lines}/{entry.total_lines} dòng"
    if entry.edited_lines:
        return f"Chỉ còn {entry.edited_lines} dòng sửa tay (chờ dịch lại)"
    return "Chưa có"


def format_saved_time(iso_text: str, local_tz: tzinfo | None = None) -> str:
    """Thời điểm lưu (ISO, UTC) → "dd/mm HH:MM" theo giờ máy; rỗng thì "—".

    Args:
        iso_text: Thời điểm dạng ISO.
        local_tz: Múi giờ hiển thị; ``None`` = múi giờ của máy.

    Returns:
        Chuỗi hiển thị.
    """
    if not iso_text:
        return "—"
    try:
        moment = datetime.fromisoformat(iso_text)
    except ValueError:
        return "—"
    return moment.astimezone(local_tz).strftime("%d/%m %H:%M")


def describe_cache_reuse(
    reused_stages: Sequence[TranslationStageKind],
    carried_lines: Mapping[TranslationStageKind, int],
    reapplied_edits: Mapping[TranslationStageKind, int],
) -> str:
    """Tóm tắt phần lấy từ bộ nhớ đệm của một lượt dịch (rỗng nếu không dùng gì).

    Args:
        reused_stages: Giai đoạn dùng lại nguyên.
        carried_lines: Số dòng giữ lại ở giai đoạn chỉ dịch lại phần đổi.
        reapplied_edits: Số dòng sửa tay đã áp lại, theo giai đoạn.

    Returns:
        Vd "Dùng lại nguyên: Dịch thô. Chỉ dịch lại phần đổi: Tinh chỉnh (giữ 3896
        dòng). Áp lại dòng sửa tay: Dịch thô 4."
    """
    parts: list[str] = []
    if reused_stages:
        parts.append(
            "Dùng lại nguyên: "
            + ", ".join(STAGE_LABELS[kind] for kind in reused_stages)
        )
    if carried_lines:
        parts.append(
            "Chỉ dịch lại phần đổi: "
            + ", ".join(
                f"{STAGE_LABELS[kind]} (giữ {count} dòng)"
                for kind, count in carried_lines.items()
            )
        )
    if reapplied_edits:
        parts.append(
            "Áp lại dòng sửa tay: "
            + ", ".join(
                f"{STAGE_LABELS[kind]} {count}" for kind, count in reapplied_edits.items()
            )
        )
    return ". ".join(parts) + ("." if parts else "")
