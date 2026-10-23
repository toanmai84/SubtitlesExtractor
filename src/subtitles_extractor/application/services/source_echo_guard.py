"""[v3.23.579] Lô dịch thô mà model trả NGUYÊN VĂN câu gốc ở nhiều dòng.

Chuẩn đoán thật (phim "傻子落水…", 3902 dòng, v3.23.578): ở lô 3421-3480, model dịch 5
dòng đầu rồi trả nguyên văn chữ Hán cho **55 dòng liền** (3426-3480:
"既然事情都结束了", "你快开着你的大奔"…). Lô vẫn đủ số dòng, đúng ``line_no``, nên
mọi phép kiểm cấu trúc đều qua; phép kiểm "dịch không có tác dụng" chỉ bắt khi
**100%** số dòng giống bản gốc. Khâu tinh chỉnh phải dịch lại 55 dòng đó từ đầu mà
không có bản dịch thô nào để dựa vào.

Đo trên 37 lần chạy thật (2226 lô): ngoài ca này, bản dịch thô chỉ có tối đa 1 dòng giống
hệt câu gốc (dòng model trả rỗng, ứng dụng giữ câu gốc) — ngưỡng dưới đây chỉ bắt đúng
một lô hỏng kia.

Toàn bộ là hàm thuần: không I/O.
"""

from __future__ import annotations

import math
import re

#: Số dòng nguyên văn tối thiểu để coi là lô hỏng (một dòng lẻ là dòng model bỏ trống).
ECHO_MIN_LINES = 3
#: Tỉ lệ dòng nguyên văn tối thiểu trong lô.
ECHO_MIN_SHARE = 0.10

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def source_echo_lines(sources: dict[int, str], returned: dict[int, str]) -> list[int]:
    """Các dòng mà bản trả về GIỐNG HỆT câu gốc có chữ CJK (hàm thuần).

    Args:
        sources: ``số dòng → câu gốc``.
        returned: ``số dòng → bản model trả về``.

    Returns:
        Số dòng tăng dần. Câu gốc không có chữ CJK (tên Latin, "OK", số) không tính —
        giữ nguyên chúng là hợp lệ.
    """
    echoed: list[int] = []
    for line_no, source in sources.items():
        text = (returned.get(line_no) or "").strip()
        if text and text == source.strip() and _CJK_RE.search(text):
            echoed.append(line_no)
    return sorted(echoed)


def is_echo_batch(echo_count: int, batch_size: int) -> bool:
    """Lô có đủ nhiều dòng nguyên văn để phải dịch lại không (hàm thuần).

    Args:
        echo_count: Số dòng nguyên văn.
        batch_size: Số dòng của lô.

    Returns:
        True nếu ``echo_count`` ≥ max(:data:`ECHO_MIN_LINES`, 10% lô).
    """
    threshold = max(ECHO_MIN_LINES, math.ceil(ECHO_MIN_SHARE * batch_size))
    return echo_count >= threshold


__all__ = [
    "ECHO_MIN_LINES",
    "ECHO_MIN_SHARE",
    "is_echo_batch",
    "source_echo_lines",
]
