"""[v3.23.610] Cấu hình suy nghĩ (thinking) gửi Gemini — kể cả khi người dùng TẮT nó.

Trước bản này, bỏ chọn "Thinking" thì không gửi cấu hình nào và model dùng mức MẶC ĐỊNH
của nó. Theo tài liệu Gemini API (trang Thinking, 10/2026), mặc định đó là:

* gemini-3.8-flash, 3.6-flash: Medium; gemini-3.1-flash-lite, 3-flash, các bản Pro: High;
* Gemini 2.5 Flash/Pro: suy nghĩ động (bật).

Tức là "tắt" lại suy nghĩ NHIỀU hơn mức "Thấp" người dùng chọn khi bật — chậm hơn, tốn
token suy nghĩ (tính như token đầu ra, vào cả TPM). Cách xử lý học từ
gemini-srt-translator (``_get_translate_config``):

* Gemini 3 trở đi không tắt hẳn được → gửi mức thấp nhất model nhận: ``minimal``; các
  model không có mức này (bản Pro, gemini-3.8-flash theo tài liệu) dùng ``low``.
* Gemini 2.5 Pro không tắt được → ngân sách tối thiểu 128; Gemini 2.5 Flash/Flash-Lite tắt
  bằng ngân sách 0.
* Ngân sách khi bật được kẹp vào khoảng model nhận (Pro 128 tới 32768, Flash tối đa
  24576) thay vì để API trả 400.

Các hàm ở đây là hàm thuần.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from subtitles_extractor.domain.value_objects.model_alias import canonical_model_id
from subtitles_extractor.infrastructure.gemini_model_family import (
    gemini_model_version,
    uses_thinking_level,
)

__all__ = [
    "LOWEST_COMMON_LEVEL",
    "MINIMAL_LEVEL",
    "THINKING_LEVELS",
    "ThinkingSettings",
    "resolve_thinking",
    "supports_minimal_level",
]

#: Các mức ``thinking_level`` của Gemini 3, từ thấp tới cao.
THINKING_LEVELS = ("minimal", "low", "medium", "high")
MINIMAL_LEVEL = "minimal"
#: Mức thấp nhất mọi model văn bản Gemini 3 đều nhận.
LOWEST_COMMON_LEVEL = "low"

#: Thế hệ đầu tiên có suy nghĩ (2.5); thế hệ trước không nhận ``thinking_config``.
_THINKING_FROM = (2, 5)
_PRO_MIN_BUDGET = 128
_PRO_MAX_BUDGET = 32768
_FLASH_MAX_BUDGET = 24576


@dataclass(frozen=True)
class ThinkingSettings:
    """Phần ``thinking_config`` sẽ gửi.

    Attributes:
        level: ``thinking_level`` (Gemini 3+), ``None`` nếu không gửi.
        budget: ``thinking_budget`` (Gemini 2.5), ``None`` = để model tự quyết (động).
    """

    level: str | None = None
    budget: int | None = None

    def as_kwargs(self) -> dict[str, Any]:
        """Tham số cho ``types.ThinkingConfig``."""
        kwargs: dict[str, Any] = {}
        if self.level is not None:
            kwargs["thinking_level"] = self.level
        if self.budget is not None:
            kwargs["thinking_budget"] = self.budget
        return kwargs


def _is_pro(model_name: str) -> bool:
    return "pro" in re.split(r"[-_.]", (model_name or "").lower())


def supports_minimal_level(model_name: str) -> bool:
    """Model có (theo hiểu biết hiện có) nhận mức ``minimal`` không.

    Args:
        model_name: Tên model Gemini 3+.

    Returns:
        ``False`` với bản Pro và các model tài liệu không liệt kê ``minimal``.
    """
    name = canonical_model_id(model_name)
    if _is_pro(name):
        return False
    # Tài liệu thinking.md (10/2026): ``minimal`` là "Not supported (error)" trên 3.8 Flash và
    # Gemini 4 (không tắt được suy nghĩ). Bí danh 3.7-flash đã được đổi sang 3.8 ở trên.
    if re.match(r"^gemini-3\.8-flash(?!-lite)", name):
        return False
    version = gemini_model_version(name)
    if version is not None and version[0] >= 4 and "lite" not in name:
        return False
    return True


def _level_for(model_name: str, requested: str, minimal_rejected: bool) -> str:
    level = (requested or "").lower()
    if level not in THINKING_LEVELS:
        level = LOWEST_COMMON_LEVEL
    if level == MINIMAL_LEVEL and (
        minimal_rejected or not supports_minimal_level(model_name)
    ):
        level = LOWEST_COMMON_LEVEL
    return level


def _budget_for_25(model_name: str, enabled: bool, budget: int) -> int | None:
    pro = _is_pro(model_name)
    if not enabled:
        return _PRO_MIN_BUDGET if pro else 0
    if budget < 0:
        return None  # động: model tự quyết
    if pro:
        return max(_PRO_MIN_BUDGET, min(budget, _PRO_MAX_BUDGET))
    return min(budget, _FLASH_MAX_BUDGET)


def resolve_thinking(
    model_name: str,
    *,
    enable_thinking: bool,
    thinking_budget: int,
    thinking_level: str,
    minimal_rejected: bool = False,
) -> ThinkingSettings | None:
    """Cấu hình suy nghĩ cho một lời gọi.

    Args:
        model_name: Tên model.
        enable_thinking: Người dùng bật suy nghĩ.
        thinking_budget: Ngân sách (Gemini 2.5): -1 động, 0 tắt, >0 số token.
        thinking_level: Mức chọn khi bật (Gemini 3+).
        minimal_rejected: Model đã từ chối mức ``minimal`` trong phiên này.

    Returns:
        :class:`ThinkingSettings`; ``None`` nếu không gửi ``thinking_config`` (model
        trước 2.5, hoặc tên không rõ thế hệ mà suy nghĩ đang tắt).
    """
    enabled = bool(enable_thinking) and thinking_budget != 0
    if uses_thinking_level(model_name):
        requested = thinking_level if enabled else MINIMAL_LEVEL
        return ThinkingSettings(level=_level_for(model_name, requested, minimal_rejected))
    version = gemini_model_version(model_name)
    if version is None:
        # Tên không rõ thế hệ: giữ cách cũ — chỉ gửi khi người dùng bật.
        if not enabled:
            return None
        return ThinkingSettings(budget=thinking_budget if thinking_budget > 0 else None)
    if version < _THINKING_FROM:
        return None
    return ThinkingSettings(budget=_budget_for_25(model_name, enabled, thinking_budget))
