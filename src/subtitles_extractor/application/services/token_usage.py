"""[v3.23.610] Sổ token Gemini đã dùng — cộng theo model, an toàn khi dịch song song.

Học từ ``--token-stats`` của gemini-srt-translator: mỗi lời gọi được máy chủ trả lời đều
bị tính token (kể cả lượt trả JSON hỏng hay rỗng), nên mọi lượt đều được ghi. Bên dùng
đánh dấu (:meth:`TokenUsageLedger.snapshot`) lúc bắt đầu một việc rồi lấy phần dùng thêm
(:meth:`TokenUsageLedger.since`) lúc kết thúc.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

__all__ = ["TokenUsage", "TokenUsageLedger", "total_usage"]


@dataclass(frozen=True)
class TokenUsage:
    """Token của một hay nhiều lời gọi.

    Attributes:
        requests: Số lời gọi máy chủ đã trả lời.
        prompt: Token đầu vào (gồm cả phần lấy từ cache, video, âm thanh).
        cached: Phần đầu vào lấy từ cache ngầm của Gemini (tính giá rẻ hơn).
        thoughts: Token suy nghĩ (tính giá như đầu ra).
        output: Token đầu ra (câu trả lời).
    """

    requests: int = 0
    prompt: int = 0
    cached: int = 0
    thoughts: int = 0
    output: int = 0

    @property
    def total(self) -> int:
        """Tổng token tính phí (đầu vào + suy nghĩ + đầu ra)."""
        return self.prompt + self.thoughts + self.output

    @property
    def is_empty(self) -> bool:
        """Không có lời gọi nào."""
        return self.requests == 0 and self.total == 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.requests + other.requests,
            self.prompt + other.prompt,
            self.cached + other.cached,
            self.thoughts + other.thoughts,
            self.output + other.output,
        )

    def __sub__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            max(0, self.requests - other.requests),
            max(0, self.prompt - other.prompt),
            max(0, self.cached - other.cached),
            max(0, self.thoughts - other.thoughts),
            max(0, self.output - other.output),
        )


def total_usage(by_model: Mapping[str, TokenUsage] | Iterable[TokenUsage]) -> TokenUsage:
    """Cộng token của mọi model.

    Args:
        by_model: ``{model: TokenUsage}`` hoặc danh sách ``TokenUsage``.

    Returns:
        Tổng.
    """
    values = by_model.values() if isinstance(by_model, Mapping) else by_model
    result = TokenUsage()
    for usage in values:
        result = result + usage
    return result


class TokenUsageLedger:
    """Sổ cộng dồn token theo model (dùng chung cho cả phiên làm việc)."""

    def __init__(self) -> None:
        """Tạo sổ rỗng."""
        self._lock = threading.Lock()
        self._by_model: dict[str, TokenUsage] = {}

    def record(self, model_name: str, usage: TokenUsage) -> None:
        """Cộng token của một lời gọi.

        Args:
            model_name: Model đã gọi.
            usage: Token của lời gọi đó.
        """
        with self._lock:
            previous = self._by_model.get(model_name, TokenUsage())
            self._by_model[model_name] = previous + usage

    def snapshot(self) -> dict[str, TokenUsage]:
        """Bản chụp hiện tại (dùng làm mốc cho :meth:`since`)."""
        with self._lock:
            return dict(self._by_model)

    def since(self, mark: Mapping[str, TokenUsage]) -> dict[str, TokenUsage]:
        """Token dùng thêm từ mốc ``mark``.

        Args:
            mark: Kết quả :meth:`snapshot` lúc bắt đầu.

        Returns:
            ``{model: TokenUsage}`` của các model có dùng thêm.
        """
        current = self.snapshot()
        delta: dict[str, TokenUsage] = {}
        for model_name, usage in current.items():
            extra = usage - mark.get(model_name, TokenUsage())
            if not extra.is_empty:
                delta[model_name] = extra
        return delta
