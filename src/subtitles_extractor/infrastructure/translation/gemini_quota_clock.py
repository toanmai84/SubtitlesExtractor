"""[v3.23.607] Mốc đổi ngày hạn mức Gemini: 00:00 giờ Thái Bình Dương (hàm thuần).

Tài liệu Gemini API (trang "Rate limits"): "Requests per day (RPD) quotas reset at
midnight Pacific time" và "Rate limits are applied per project, not per API key".

Nhật ký thật khớp với tài liệu, KHÔNG khớp với ``retryDelay`` của lỗi 429 hết hạn
mức ngày:

* 12/10: gemini-3.8-flash hết hạn mức lúc 06:17 và 06:33 (giờ máy UTC+7); server
  gửi ``retryDelay`` ~26 phút (tức 07:00). Lúc 11:06 cả ba key vẫn bị từ chối —
  nếu hạn mức hồi lúc 07:00 thì phải dùng được.
* 08/10: cả ba key hết hạn mức gemini-3.8-flash từ 07:46 đến 08:05; lúc 14:19 gọi
  lại thành công — sau 14:00 giờ máy, tức 00:00 giờ Thái Bình Dương (PDT).

``retryDelay`` của lỗi hết hạn mức ngày luôn chỉ về 00:00 UTC (07:00 giờ máy) nên
không được dùng làm mốc hồi. Bản v3.23.606 đã tin con số đó — sai, bản này bỏ.

Giờ Thái Bình Dương lấy từ ``zoneinfo`` nếu máy có dữ liệu múi giờ; Windows thường
không có (cần gói ``tzdata``) nên có sẵn quy tắc giờ mùa hè của Mỹ (từ 2007): bắt
đầu 02:00 Chủ nhật thứ hai tháng 3, kết thúc 02:00 Chủ nhật đầu tháng 11.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, tzinfo

__all__ = [
    "describe_quota_reset",
    "next_quota_reset",
    "pacific_utc_offset",
    "quota_day",
]

_PACIFIC_STANDARD = timedelta(hours=-8)
_PACIFIC_DAYLIGHT = timedelta(hours=-7)
_SUNDAY = 6


def _nth_sunday(year: int, month: int, nth: int) -> int:
    first_weekday = date(year, month, 1).weekday()
    first_sunday = 1 + (_SUNDAY - first_weekday) % 7
    return first_sunday + 7 * (nth - 1)


def _us_rule_offset(now_utc: datetime) -> timedelta:
    """Độ lệch giờ Thái Bình Dương theo quy tắc giờ mùa hè của Mỹ (từ năm 2007)."""
    year = now_utc.year
    # 02:00 PST = 10:00 UTC; 02:00 PDT = 09:00 UTC.
    dst_start = datetime(year, 3, _nth_sunday(year, 3, 2), 10, tzinfo=UTC)
    dst_end = datetime(year, 11, _nth_sunday(year, 11, 1), 9, tzinfo=UTC)
    return _PACIFIC_DAYLIGHT if dst_start <= now_utc < dst_end else _PACIFIC_STANDARD


def pacific_utc_offset(now_utc: datetime) -> timedelta:
    """Độ lệch của giờ Thái Bình Dương so với UTC tại một thời điểm.

    Args:
        now_utc: Thời điểm (có múi giờ).

    Returns:
        ``-7 giờ`` (PDT) hoặc ``-8 giờ`` (PST).
    """
    moment = now_utc.astimezone(UTC)
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    except ImportError:
        return _us_rule_offset(moment)
    try:
        offset = moment.astimezone(ZoneInfo("America/Los_Angeles")).utcoffset()
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return _us_rule_offset(moment)
    return offset if offset is not None else _us_rule_offset(moment)


def quota_day(now_utc: datetime | None = None) -> str:
    """Ngày hạn mức (ngày theo giờ Thái Bình Dương) dạng ISO.

    Args:
        now_utc: Thời điểm; ``None`` = bây giờ.

    Returns:
        Vd ``"2026-10-11"`` cho 11:06 sáng 12/10 giờ Việt Nam.
    """
    moment = (now_utc or datetime.now(UTC)).astimezone(UTC)
    return (moment + pacific_utc_offset(moment)).date().isoformat()


def next_quota_reset(now_utc: datetime | None = None) -> datetime:
    """Thời điểm (UTC) hạn mức ngày hồi lần tới: 00:00 giờ Thái Bình Dương kế tiếp.

    Args:
        now_utc: Thời điểm; ``None`` = bây giờ.

    Returns:
        Thời điểm có múi giờ UTC.
    """
    moment = (now_utc or datetime.now(UTC)).astimezone(UTC)
    pacific_today = (moment + pacific_utc_offset(moment)).date()
    next_day = pacific_today + timedelta(days=1)
    midnight = datetime.combine(next_day, datetime.min.time(), UTC)
    # 07:00 UTC hôm đó luôn rơi vào 23:00-00:00 giờ Thái Bình Dương; giờ mùa hè chỉ đổi
    # lúc 02:00 nên độ lệch tại đó cũng là độ lệch lúc nửa đêm.
    return midnight - pacific_utc_offset(midnight + timedelta(hours=7))


def describe_quota_reset(
    now_utc: datetime | None = None, local_tz: tzinfo | None = None
) -> str:
    """Câu cho người dùng: lúc nào hạn mức ngày hồi, theo giờ máy, còn bao lâu.

    Args:
        now_utc: Thời điểm; ``None`` = bây giờ.
        local_tz: Múi giờ hiển thị; ``None`` = múi giờ của máy.

    Returns:
        Vd "Hạn mức ngày của Gemini hồi lúc 00:00 giờ Thái Bình Dương — 14:00 hôm nay theo
        giờ máy (còn khoảng 2 giờ 53 phút)."
    """
    moment = (now_utc or datetime.now(UTC)).astimezone(UTC)
    reset = next_quota_reset(moment)
    local_now = moment.astimezone(local_tz)
    local_reset = reset.astimezone(local_tz)
    if local_reset.date() == local_now.date():
        day = "hôm nay"
    elif local_reset.date() == local_now.date() + timedelta(days=1):
        day = "ngày mai"
    else:
        day = f"ngày {local_reset:%d/%m}"
    minutes = max(1, int((reset - moment).total_seconds() // 60))
    hours, minutes = divmod(minutes, 60)
    wait = (
        f"{hours} giờ {minutes} phút" if hours and minutes
        else f"{hours} giờ" if hours else f"{minutes} phút"
    )
    return (
        "Hạn mức ngày của Gemini hồi lúc 00:00 giờ Thái Bình Dương — "
        f"{local_reset:%H:%M} {day} theo giờ máy (còn khoảng {wait})."
    )
