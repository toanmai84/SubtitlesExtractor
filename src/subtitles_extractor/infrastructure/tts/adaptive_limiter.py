"""[v3.23.229] Bộ điều tiết song song THÍCH ỨNG cho dịch vụ có giới hạn không công bố.

Bằng chứng từ log chạy thật (Edge TTS, 95 câu, ``concurrency=64``)::

    06:50:43.165  audio rỗng cho 'toàn bộ tiền tiết kiệm củ…'
    06:50:43.176  audio rỗng cho 'Thiên phú thần thông:…'
    06:50:43.205  audio rỗng cho 'Đã can đảm bước bước đầu …'
    ... 20 lỗi DỒN trong ~2.4 giây, ngay sau khi phóng request
    06:51:10      "Edge TTS quá thời gian (30s) — bỏ qua dòng" x 9 dòng CÙNG LÚC
    06:51:54      'Họ tên:' phải thử tới lần 4/10 mới xong

Tổng thời gian TTS: 87 giây cho 95 câu, trong đó **~80 giây là retry và chờ timeout**.

Đặc điểm chẩn đoán: toàn bộ lỗi "audio rỗng" DỒN vào burst đầu tiên, không rải rác -> đây
không phải lỗi ngẫu nhiên của từng dòng, mà là **dịch vụ từ chối vì quá tải kết nối**.

Vấn đề thiết kế: khi lần thử đầu thất bại, mã cũ retry ở **cùng mức tải** -> lại bị chặn
(log: lần 2, lần 3, lần 4 vẫn rỗng). Càng retry càng đè nặng dịch vụ đang từ chối mình.

Edge TTS là dịch vụ miễn phí của Microsoft; giới hạn kết nối **không được công bố** và có
thể đổi bất cứ lúc nào. Vì vậy hardcode một con số "an toàn" là đoán mò. Cách đúng là **tự
dò và lùi** theo phản hồi thực tế — cùng nguyên lý AIMD (Additive Increase, Multiplicative
Decrease) mà TCP dùng để chia sẻ băng thông với mạng không biết trước dung lượng:

* Thất bại thành cụm -> **giảm một nửa** giới hạn (lùi nhanh khỏi vùng bị chặn).
* Chuỗi thành công dài -> **tăng dần từng nấc** (dò lại dung lượng, không nhảy vọt).
"""

from __future__ import annotations

import asyncio
import logging

__all__ = [
    "FAILURE_RATIO_TO_SHRINK",
    "INITIAL_CONCURRENCY",
    "MIN_CONCURRENCY",
    "SUCCESSES_TO_GROW",
    "AdaptiveConcurrencyLimiter",
    "grow_limit",
    "shrink_limit",
]

logger = logging.getLogger(__name__)

# Sàn: dưới mức này thì việc song song hoá không còn ý nghĩa, và nếu dịch vụ vẫn chặn ở
# mức 2 thì vấn đề nằm chỗ khác (mạng, IP bị khoá) — giảm tiếp cũng vô ích.
MIN_CONCURRENCY = 2

# Tỉ lệ thất bại trong một cửa sổ đủ để kết luận "đang bị chặn vì quá tải" chứ không phải
# lỗi lẻ. Đo thực: burst đầu ở concurrency=64 cho ~21/95 dòng rỗng (22%) — nhưng chúng dồn
# cục bộ, nên tính theo cửa sổ trượt sẽ vượt xa 0.25.
FAILURE_RATIO_TO_SHRINK = 0.25

# Số lần thành công LIÊN TIẾP trước khi nới thêm một nấc (tăng dần, không nhảy vọt).
SUCCESSES_TO_GROW = 8

# [KHỞI ĐỘNG CHẬM] Mức song song ban đầu, bất kể người dùng cấu hình bao nhiêu.
#
# Chỉ "giảm khi bị từ chối" là CHƯA ĐỦ: mô phỏng cho thấy nếu xuất phát thẳng ở 64 thì cả
# burst đầu tiên đã bay lên và bị chặn TRƯỚC KHI bộ điều tiết kịp phản ứng — đúng như log
# thật (20 lỗi dồn trong 2.4 giây đầu). Phải xuất phát THẤP rồi dò lên, giống TCP slow
# start: chi phí của việc dò lên là vài giây; chi phí của việc đâm vào tường là hàng chục
# giây timeout cộng nguy cơ mất thoại.
INITIAL_CONCURRENCY = 8


def shrink_limit(current: int, floor: int = MIN_CONCURRENCY) -> int:
    """Giảm một nửa giới hạn song song (hàm thuần).

    Lùi NHANH (chia đôi) chứ không giảm từng nấc: khi dịch vụ đã từ chối, mỗi lần thử ở
    mức quá cao lại tốn thêm một vòng timeout 30 giây.

    Args:
        current: Giới hạn hiện tại.
        floor: Sàn tối thiểu.

    Returns:
        Giới hạn mới (>= ``floor``).
    """
    return max(floor, current // 2)


def grow_limit(current: int, ceiling: int) -> int:
    """Nới giới hạn song song thêm MỘT nấc (hàm thuần).

    Tăng TỪ TỪ (cộng 1) chứ không nhân đôi: dung lượng thật không biết trước, nhảy vọt sẽ
    lại đâm vào tường và trả giá bằng một loạt timeout.

    Args:
        current: Giới hạn hiện tại.
        ceiling: Trần do người dùng cấu hình.

    Returns:
        Giới hạn mới (<= ``ceiling``).
    """
    return min(ceiling, current + 1)


class AdaptiveConcurrencyLimiter:
    """Cổng vào bất đồng bộ với giới hạn song song TỰ ĐIỀU CHỈNH.

    Thay cho ``asyncio.Semaphore`` cố định. Dùng đúng như semaphore::

        async with limiter:
            audio = await goi_dich_vu()
        limiter.report_success()   # hoặc report_failure() nếu audio rỗng

    Attributes:
        limit: Giới hạn hiện tại (chỉ đọc).
    """

    def __init__(
        self,
        ceiling: int,
        *,
        floor: int = MIN_CONCURRENCY,
        initial: int = INITIAL_CONCURRENCY,
    ) -> None:
        """Khởi tạo.

        Args:
            ceiling: Trần do người dùng cấu hình — mức song song TỐI ĐA được phép dò tới.
            floor: Sàn tối thiểu.
            initial: Mức xuất phát (khởi động chậm). Không bao giờ vượt ``ceiling``.

        Raises:
            ValueError: Nếu ``ceiling`` < 1.
        """
        if ceiling < 1:
            raise ValueError("ceiling phải >= 1")
        self._ceiling = ceiling
        self._floor = min(floor, ceiling)
        self._limit = max(self._floor, min(initial, ceiling))
        self._in_flight = 0
        self._streak_ok = 0
        self._recent_fail = 0
        self._recent_total = 0
        self._condition = asyncio.Condition()

    @property
    def limit(self) -> int:
        """Giới hạn song song hiện tại."""
        return self._limit

    async def __aenter__(self) -> AdaptiveConcurrencyLimiter:
        async with self._condition:
            await self._condition.wait_for(lambda: self._in_flight < self._limit)
            self._in_flight += 1
        return self

    async def __aexit__(self, *_exc: object) -> None:
        async with self._condition:
            self._in_flight -= 1
            self._condition.notify_all()

    def report_failure(self) -> None:
        """Báo một lượt gọi thất bại (audio rỗng / timeout).

        Khi tỉ lệ thất bại trong cửa sổ vượt :data:`FAILURE_RATIO_TO_SHRINK`, giới hạn bị
        chia đôi và cửa sổ được đặt lại — tránh giảm liên tiếp nhiều lần chỉ vì một burst.
        """
        self._streak_ok = 0
        self._recent_fail += 1
        self._recent_total += 1
        if self._recent_total < 4:  # mẫu quá nhỏ để kết luận
            return
        if self._recent_fail / self._recent_total < FAILURE_RATIO_TO_SHRINK:
            return
        new_limit = shrink_limit(self._limit, self._floor)
        if new_limit < self._limit:
            logger.warning(
                "Dịch vụ TTS đang từ chối (%d/%d lượt gần đây thất bại) — GIẢM song song "
                "%d -> %d để thoát vùng bị chặn.",
                self._recent_fail, self._recent_total, self._limit, new_limit,
            )
            self._limit = new_limit
        self._recent_fail = 0
        self._recent_total = 0

    def report_success(self) -> None:
        """Báo một lượt gọi thành công; nới dần giới hạn nếu chuỗi thành công đủ dài."""
        self._recent_total += 1
        self._streak_ok += 1
        if self._streak_ok < SUCCESSES_TO_GROW:
            return
        self._streak_ok = 0
        self._recent_fail = 0
        self._recent_total = 0
        new_limit = grow_limit(self._limit, self._ceiling)
        if new_limit > self._limit:
            logger.info(
                "Dịch vụ TTS ổn định — nới song song %d -> %d.", self._limit, new_limit
            )
            self._limit = new_limit
