"""[v3.23.123] Test: release() hoàn trả ĐẦY ĐỦ RPD/RPM/TPM.

Bug gốc: mỗi lần thử thất bại (503/429 — server KHÔNG trừ quota) vẫn cộng bộ đếm
request/ngày và không bao giờ hoàn → sau nhiều 503 thì báo "hết quota" oan.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    QuotaExhaustedError,
    RateLimit,
)


def _mgr(rpd: int = 3, rpm: int = 100, tpm: int = 10**9) -> GeminiQuotaManager:
    return GeminiQuotaManager(
        rate_limits={"m": RateLimit(rpm=rpm, tpm=tpm, rpd=rpd)}
    )


def test_release_rolls_back_daily_count() -> None:
    mgr = _mgr(rpd=3)
    res = mgr.acquire("m", 100, key_id="k")
    assert mgr.get_remaining("m", key_id="k")["rpd_used"] == 1
    mgr.release(res)
    assert mgr.get_remaining("m", key_id="k")["rpd_used"] == 0


def test_failed_attempts_do_not_exhaust_quota() -> None:
    # Mô phỏng 50 lần thử thất bại (acquire + release) trên RPD=3.
    mgr = _mgr(rpd=3)
    for _ in range(50):
        res = mgr.acquire("m", 100, key_id="k")
        mgr.release(res)  # lỗi server → hoàn trả
    # Quota vẫn còn nguyên — KHÔNG bị cạn oan.
    assert mgr.get_remaining("m", key_id="k")["rpd_remaining"] == 3
    # Vẫn còn đủ chỗ cho 3 request thành công.
    for _ in range(3):
        assert mgr.acquire("m", 100, key_id="k") is not None


def test_committed_requests_still_count() -> None:
    # Không release = coi như thành công (đã tiêu thụ) → đếm bình thường.
    mgr = _mgr(rpd=2)
    mgr.acquire("m", 100, key_id="k")
    mgr.acquire("m", 100, key_id="k")
    with pytest.raises(QuotaExhaustedError):
        mgr.acquire("m", 100, key_id="k")


def test_release_rolls_back_rpm_and_tpm() -> None:
    mgr = _mgr(rpd=100, rpm=5, tpm=1000)
    res = mgr.acquire("m", 500, key_id="k")
    rem = mgr.get_remaining("m", key_id="k")
    assert rem["rpm_used"] == 1 and rem["tpm_used"] == 500
    mgr.release(res)
    rem = mgr.get_remaining("m", key_id="k")
    assert rem["rpm_used"] == 0 and rem["tpm_used"] == 0


def test_release_none_is_safe() -> None:
    mgr = _mgr()
    mgr.release(None)  # không lỗi


def test_double_release_does_not_underflow() -> None:
    mgr = _mgr(rpd=3)
    res = mgr.acquire("m", 100, key_id="k")
    mgr.release(res)
    mgr.release(res)  # gọi lại không làm count âm
    assert mgr.get_remaining("m", key_id="k")["rpd_used"] == 0


def test_mixed_success_and_failure_counts_only_success() -> None:
    mgr = _mgr(rpd=5)
    # 2 thành công (giữ), 10 thất bại (hoàn).
    mgr.acquire("m", 100, key_id="k")
    mgr.acquire("m", 100, key_id="k")
    for _ in range(10):
        res = mgr.acquire("m", 100, key_id="k")
        mgr.release(res)
    assert mgr.get_remaining("m", key_id="k")["rpd_used"] == 2


# ── Persistence count ngày qua "restart" ─────────────────────────────────


def test_daily_count_persists_across_restart(tmp_path) -> None:
    path = tmp_path / "quota_state.json"
    mgr1 = GeminiQuotaManager(
        rate_limits={"m": RateLimit(rpm=100, tpm=10**9, rpd=5)}, state_path=path
    )
    mgr1.acquire("m", 100, key_id="k")
    mgr1.acquire("m", 100, key_id="k")
    assert path.exists()
    # "Khởi động lại": manager mới đọc lại count ngày từ đĩa.
    mgr2 = GeminiQuotaManager(
        rate_limits={"m": RateLimit(rpm=100, tpm=10**9, rpd=5)}, state_path=path
    )
    assert mgr2.get_remaining("m", key_id="k")["rpd_used"] == 2


def test_stale_day_not_restored(tmp_path) -> None:
    path = tmp_path / "quota_state.json"
    # Ghi tay một mục với period_key của NGÀY KHÁC -> không được nạp.
    path.write_text(
        '{"k|m": {"period_key": "2000-01-01", "count": 99}}', encoding="utf-8"
    )
    mgr = GeminiQuotaManager(
        rate_limits={"m": RateLimit(rpm=100, tpm=10**9, rpd=5)}, state_path=path
    )
    assert mgr.get_remaining("m", key_id="k")["rpd_used"] == 0


def test_no_state_path_is_memory_only(tmp_path) -> None:
    mgr = GeminiQuotaManager(rate_limits={"m": RateLimit(rpm=9, tpm=9, rpd=9)})
    mgr.acquire("m", 1, key_id="k")  # không lỗi dù không có state_path
    assert mgr.get_remaining("m", key_id="k")["rpd_used"] == 1


def test_period_key_is_pacific_date() -> None:
    from datetime import UTC, datetime
    # 08:00 UTC ngày 2 = 00:00/01:00 giờ Pacific -> vẫn còn ngày 1 hoặc đã sang ngày 2
    # tuỳ DST; kiểm chỉ cần trả ISO date hợp lệ và LÙI so với UTC.
    key = GeminiQuotaManager._period_key(datetime(2026, 6, 2, 3, 0, tzinfo=UTC))
    assert key == "2026-06-01"  # 03:00 UTC ngày 2 -> ~20:00 ngày 1 giờ Pacific
