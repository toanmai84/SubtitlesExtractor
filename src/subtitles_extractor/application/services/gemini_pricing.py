"""[v3.23.610] Bảng giá Gemini và ước tính chi phí của một lượt (hàm thuần).

Học từ ``pricing.json`` + ``_calculate_cost`` của gemini-srt-translator (MIT), bổ sung:

* token lấy từ cache ngầm tính theo tỉ lệ giá cache (``cached_input_ratio``) — ứng dụng
  đặt phần ổn định của prompt ở ``system_instruction`` nên cache chiếm phần đáng kể;
* model có hậu tố phiên bản (``gemini-3.1-flash-lite-preview-0612``) lấy giá của họ model
  gần nhất theo tiền tố dài nhất;
* model không có giá thì BÁO là chưa có giá, không đoán.

Bảng giá là dữ liệu (``data/gemini_pricing.json``), cập nhật được mà không sửa mã.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from subtitles_extractor.application.services.token_usage import (
    TokenUsage,
    total_usage,
)

__all__ = [
    "DEFAULT_TIER",
    "CostEstimate",
    "ModelRates",
    "PricingFormatError",
    "PricingTable",
    "describe_token_usage",
    "estimate_cost",
    "format_count",
    "parse_pricing",
]

DEFAULT_TIER = "standard"
_PER_MILLION = 1_000_000


class PricingFormatError(ValueError):
    """Bảng giá sai cấu trúc."""


@dataclass(frozen=True)
class ModelRates:
    """Giá một model ở một hạng dịch vụ (đơn vị tiền / 1 triệu token)."""

    input_per_million: float
    output_per_million: float


@dataclass(frozen=True)
class PricingTable:
    """Bảng giá.

    Attributes:
        currency: Đơn vị tiền (vd ``"USD"``).
        updated: Ngày cập nhật bảng giá (ISO).
        source: Trang nguồn.
        cached_input_ratio: Giá token đầu vào lấy từ cache so với giá đầu vào thường.
        rates: ``{model: {hạng: ModelRates}}``.
    """

    currency: str
    updated: str
    source: str
    cached_input_ratio: float
    rates: Mapping[str, Mapping[str, ModelRates]]

    def rates_for(self, model_name: str, tier: str = DEFAULT_TIER) -> ModelRates | None:
        """Giá của ``model_name`` (khớp nguyên tên, không thì tiền tố họ model dài nhất).

        Args:
            model_name: Tên model.
            tier: Hạng dịch vụ; không có thì dùng ``standard``.

        Returns:
            :class:`ModelRates`; ``None`` nếu bảng không có model này.
        """
        name = (model_name or "").lower()
        family = self.rates.get(name)
        if family is None:
            candidates = [key for key in self.rates if _is_version_of(name, key)]
            if not candidates:
                return None
            family = self.rates[max(candidates, key=len)]
        return family.get(tier) or family.get(DEFAULT_TIER)


#: Hậu tố chỉ là phiên bản của cùng model (không phải biến thể khác giá như "-lite").
_VERSION_SUFFIX_WORDS = frozenset({"preview", "exp", "experimental", "latest"})


def _is_version_of(name: str, family: str) -> bool:
    """``name`` là một phiên bản của ``family`` (vd ``family-preview-0612``)."""
    if not name.startswith(family + "-"):
        return False
    first = name[len(family) + 1:].split("-", 1)[0]
    return first in _VERSION_SUFFIX_WORDS or first.isdigit()


@dataclass(frozen=True)
class CostEstimate:
    """Chi phí ước tính.

    Attributes:
        amount: Số tiền (chỉ của các model có giá).
        currency: Đơn vị tiền.
        updated: Ngày của bảng giá.
        unpriced_models: Model không có trong bảng giá.
    """

    amount: float
    currency: str
    updated: str
    unpriced_models: tuple[str, ...] = ()


def _rate(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise PricingFormatError(f"Giá không hợp lệ ở {where}: {value!r}")
    return float(value)


def parse_pricing(raw: Mapping[str, Any]) -> PricingTable:
    """Đọc bảng giá từ dữ liệu JSON đã giải mã.

    Args:
        raw: Nội dung ``gemini_pricing.json``.

    Returns:
        :class:`PricingTable`.

    Raises:
        PricingFormatError: Thiếu ``models`` hoặc giá không phải số không âm.
    """
    models = raw.get("models") if isinstance(raw, Mapping) else None
    if not isinstance(models, Mapping) or not models:
        raise PricingFormatError("Bảng giá thiếu mục 'models'.")
    rates: dict[str, dict[str, ModelRates]] = {}
    for model_name, tiers in models.items():
        if not isinstance(tiers, Mapping):
            raise PricingFormatError(f"Giá của {model_name} không hợp lệ.")
        rates[str(model_name).lower()] = {
            str(tier): ModelRates(
                _rate(values.get("input"), f"{model_name}/{tier}/input"),
                _rate(values.get("output"), f"{model_name}/{tier}/output"),
            )
            for tier, values in tiers.items()
            if isinstance(values, Mapping)
        }
    return PricingTable(
        currency=str(raw.get("currency") or "USD"),
        updated=str(raw.get("updated") or ""),
        source=str(raw.get("source") or ""),
        cached_input_ratio=_rate(
            raw.get("cached_input_ratio", 1.0), "cached_input_ratio"
        ),
        rates=rates,
    )


def estimate_cost(
    by_model: Mapping[str, TokenUsage], table: PricingTable, tier: str = DEFAULT_TIER
) -> CostEstimate:
    """Chi phí ước tính nếu tính theo bảng giá trả phí.

    Token suy nghĩ tính giá đầu ra (như Gemini tính). Token đầu vào lấy từ cache tính
    theo ``cached_input_ratio``.

    Args:
        by_model: Token theo model.
        table: Bảng giá.
        tier: Hạng dịch vụ.

    Returns:
        :class:`CostEstimate`.
    """
    amount = 0.0
    unpriced: list[str] = []
    for model_name, usage in by_model.items():
        rates = table.rates_for(model_name, tier)
        if rates is None:
            unpriced.append(model_name)
            continue
        cached = min(usage.cached, usage.prompt)
        input_cost = (
            (usage.prompt - cached) + cached * table.cached_input_ratio
        ) * rates.input_per_million
        output_cost = (usage.thoughts + usage.output) * rates.output_per_million
        amount += (input_cost + output_cost) / _PER_MILLION
    return CostEstimate(amount, table.currency, table.updated, tuple(sorted(unpriced)))


def format_count(value: int) -> str:
    """Số có dấu chấm ngăn nghìn kiểu Việt Nam (``1.234.567``)."""
    return f"{value:,}".replace(",", ".")


def _format_money(amount: float, currency: str) -> str:
    text = f"{amount:.2f}" if amount >= 0.01 else f"{amount:.4f}"
    return f"{text.replace('.', ',')} {currency}"


def _format_month(updated: str) -> str:
    parts = updated.split("-")
    return f"{parts[1]}/{parts[0]}" if len(parts) >= 2 else updated


def describe_token_usage(
    by_model: Mapping[str, TokenUsage], estimate: CostEstimate | None
) -> str:
    """Một câu tóm tắt token đã dùng (và chi phí ước tính nếu có).

    Args:
        by_model: Token theo model.
        estimate: Chi phí ước tính; ``None`` nếu không có bảng giá.

    Returns:
        Câu tiếng Việt; rỗng nếu không có lời gọi nào.
    """
    total = total_usage(by_model)
    if total.is_empty:
        return ""
    prompt = format_count(total.prompt)
    if total.cached:
        prompt += f" ({format_count(total.cached)} lấy từ cache)"
    text = (
        f"Đã dùng {format_count(total.total)} token qua {total.requests} lượt gọi Gemini"
        f" — đầu vào {prompt}, suy nghĩ {format_count(total.thoughts)}, "
        f"đầu ra {format_count(total.output)}"
    )
    if len(by_model) > 1:
        parts = "; ".join(
            f"{model}: {format_count(usage.total)}"
            for model, usage in sorted(by_model.items(), key=lambda item: -item[1].total)
        )
        text += f" ({parts})"
    text += "."
    if estimate is not None and len(estimate.unpriced_models) < len(by_model):
        text += (
            f" Nếu tính phí: ≈ {_format_money(estimate.amount, estimate.currency)}"
            f" (bảng giá {_format_month(estimate.updated)}"
        )
        if estimate.unpriced_models:
            text += f", chưa có giá cho {', '.join(estimate.unpriced_models)}"
        text += "; dùng hạn mức miễn phí thì không mất tiền)."
    return text
