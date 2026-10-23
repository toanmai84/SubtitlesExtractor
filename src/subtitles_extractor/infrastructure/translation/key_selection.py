"""[v3.23.639] Chọn API key cho MỘT lời gọi văn bản (hàm thuần, không I/O).

Hiện trạng đo được (trạng thái quota thật của 3 key trong buổi làm việc): cách chọn cũ là "dính" một
key tới khi nó cạn rồi mới xoay — key #1 đã chạm 500/500 request/ngày cho CẢ HAI model flash-lite
trong khi key #2 chỉ dùng 81-134 và key #3 chỉ 45-49. Hệ quả:

* dồn tải: một key chạm trần RPM (15/phút) hay TPM (250k/phút) thì ứng dụng NGỒI CHỜ tới 60 giây
  dù hai key kia đang trống hoàn toàn;
* key cạn trước làm các lời gọi sau phải xoay + thử lại (mỗi lần tốn thêm thời gian), trong khi
  chia đều thì không key nào cạn sớm.

Chính sách (:func:`choose_key`):

1. Bỏ key hỏng và key đã hết hạn mức ngày.
2. Còn key gửi được NGAY (không phải chờ RPM/TPM/cooldown): giữ key hiện tại nếu nó không dùng nhiều
   hơn hẳn (hơn ``BALANCE_MARGIN`` hạn mức ngày) key nào khác gửi được ngay; ngược lại chọn key còn
   NHIỀU hạn mức ngày nhất. Ngưỡng này tránh nhảy qua lại từng lời gọi (mất bộ nhớ đệm prompt ngầm
   của Gemini) mà vẫn dàn đều tải theo thời gian.
3. Mọi key đều phải chờ: chọn key chờ NGẮN nhất (giữ key hiện tại nếu chênh dưới ``MIN_GAIN_S``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["BALANCE_MARGIN", "IMMEDIATE_S", "MIN_BALANCE_GAP", "MIN_GAIN_S", "KeyView", "choose_key"]

#: Chờ không quá chừng này giây thì coi là "gửi được ngay".
IMMEDIATE_S = 0.5
#: Key khác phải còn nhiều hơn key hiện tại chừng này (tỉ lệ hạn mức ngày) mới đáng đổi khi cả hai
#: đều gửi được ngay.
BALANCE_MARGIN = 0.15
#: ...VÀ phải hơn ít nhất chừng này request (model hạn mức ít như 20/ngày: 3 request đã là 15% nhưng
#: dàn đều chẳng có ý nghĩa — cứ dùng tới khi cạn rồi xoay như cũ).
MIN_BALANCE_GAP = 10
#: Đổi key chỉ để rút ngắn thời gian chờ khi tiết kiệm được ít nhất chừng này giây.
MIN_GAIN_S = 1.0


@dataclass(frozen=True, slots=True)
class KeyView:
    """Tình trạng một key đối với model + lời gọi sắp gửi.

    Attributes:
        index: Chỉ số key trong danh sách của adapter.
        wait_s: Số giây phải chờ nếu gửi ngay (``inf`` = hôm nay không gửi được).
        rpd_remaining: Số request/ngày còn lại (theo bộ đếm).
        rpd_limit: Hạn mức request/ngày của model.
        usable: ``False`` nếu key bị đánh dấu hỏng.
    """

    index: int
    wait_s: float
    rpd_remaining: int
    rpd_limit: int
    usable: bool = True

    @property
    def fraction(self) -> float:
        """Tỉ lệ hạn mức ngày còn lại (0-1)."""
        return self.rpd_remaining / self.rpd_limit if self.rpd_limit > 0 else 0.0


def choose_key(views: Sequence[KeyView], current: int) -> int | None:
    """Chỉ số key nên dùng cho lời gọi này, hoặc ``None`` nếu không key nào dùng được.

    Args:
        views: Tình trạng từng key.
        current: Key đang dùng (để giữ nếu không có lý do đổi).
    """
    candidates = [
        v for v in views if v.usable and v.rpd_remaining > 0 and not math.isinf(v.wait_s)
    ]
    if not candidates:
        return None
    by_index = {v.index: v for v in candidates}
    mine = by_index.get(current)
    ready = [v for v in candidates if v.wait_s <= IMMEDIATE_S]
    if ready:
        best = max(ready, key=lambda v: (v.fraction, v.rpd_remaining, -v.index))
        if mine is not None and mine.wait_s <= IMMEDIATE_S:
            worthwhile = (
                best.fraction - mine.fraction >= BALANCE_MARGIN
                and best.rpd_remaining - mine.rpd_remaining >= MIN_BALANCE_GAP
            )
            return best.index if worthwhile else current
        return best.index
    best = min(candidates, key=lambda v: (v.wait_s, -v.fraction, v.index))
    if mine is not None and mine.wait_s - best.wait_s < MIN_GAIN_S:
        return current
    return best.index
