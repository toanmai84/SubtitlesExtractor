"""Test [v3.23.17] GeminiQuotaManager: limit free tier đúng, reset Pacific, throttle."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    QuotaExhaustedError,
    RateLimit,
    _match_free_tier_limit,
)


class TestFreeTierLimits:
    def test_flash_lite_hi_limit_31(self) -> None:
        # [v3.23.36] gemini-3.1-flash-lite: RPM 15, RPD 500 (RPD cao nhất).
        limit = _match_free_tier_limit("gemini-3.1-flash-lite")
        assert (limit.rpm, limit.tpm, limit.rpd) == (15, 250_000, 500)

    def test_flash_lite_limit_generic(self) -> None:
        # flash-lite khác (vd 2.5): bảo thủ RPM 10, RPD 20.
        limit = _match_free_tier_limit("gemini-2.5-flash-lite")
        assert (limit.rpm, limit.tpm, limit.rpd) == (10, 250_000, 20)

    def test_flash_limit(self) -> None:
        # *-flash: RPM 5, RPD 20 (theo bảng quota thực tế).
        limit = _match_free_tier_limit("gemini-3.5-flash")
        assert (limit.rpm, limit.tpm, limit.rpd) == (5, 250_000, 20)

    def test_pro_limit(self) -> None:
        # pro không free → giữ thấp (RPM 5, RPD 20).
        limit = _match_free_tier_limit("gemini-2.5-pro")
        assert (limit.rpm, limit.tpm, limit.rpd) == (5, 250_000, 20)

    def test_flash_lite_before_flash(self) -> None:
        # 'flash-lite' chứa 'flash' → phải khớp flash-lite trước (3.1 = RPM 15).
        assert _match_free_tier_limit("gemini-3.1-flash-lite-Y").rpm == 15

    def test_unknown_defaults_conservative(self) -> None:
        limit = _match_free_tier_limit("gemini-something-new")
        assert limit.tpm == 250_000  # không lạc quan 1M


class TestPacificReset:
    def test_early_utc_is_previous_pacific_day(self) -> None:
        # 06:00 UTC = 22:00 PST hôm trước.
        dt = datetime(2026, 6, 16, 6, 0, 0, tzinfo=timezone.utc)
        assert GeminiQuotaManager._period_key(dt) == "2026-06-15"

    def test_later_utc_is_same_pacific_day(self) -> None:
        dt = datetime(2026, 6, 16, 9, 0, 0, tzinfo=timezone.utc)
        assert GeminiQuotaManager._period_key(dt) == "2026-06-16"


class TestThrottle:
    def test_rpm_fills_then_remaining_zero(self) -> None:
        qm = GeminiQuotaManager()
        for _ in range(10):  # flash rpm=10
            assert qm.acquire("gemini-3.5-flash", 100) is not None
        assert qm.get_remaining("gemini-3.5-flash")["rpm_remaining"] == 0

    def test_rpd_exhausted_raises(self) -> None:
        qm = GeminiQuotaManager(
            rate_limits={"m": RateLimit(rpm=1000, tpm=1_000_000, rpd=3)}
        )
        for _ in range(3):
            qm.acquire("m", 10)
        try:
            qm.acquire("m", 10)
            raise AssertionError("phải raise QuotaExhaustedError")
        except QuotaExhaustedError:
            pass

    def test_reconcile_updates_tokens(self) -> None:
        qm = GeminiQuotaManager()
        r = qm.acquire("gemini-3.5-flash", 100)
        before = qm.get_remaining("gemini-3.5-flash")["tpm_used"]
        qm.reconcile(r, 5000)
        after = qm.get_remaining("gemini-3.5-flash")["tpm_used"]
        assert after == before - 100 + 5000

    def test_release_frees_slot(self) -> None:
        qm = GeminiQuotaManager()
        r = qm.acquire("gemini-3.5-flash", 100)
        qm.release(r)
        assert qm.get_remaining("gemini-3.5-flash")["tpm_used"] == 0

    def test_thread_safe_counting(self) -> None:
        # 30 luồng acquire model rpd lớn → tổng đếm chính xác = 30 (không race).
        qm = GeminiQuotaManager(
            rate_limits={"m": RateLimit(rpm=10_000, tpm=10_000_000, rpd=10_000)}
        )
        ok = []
        lock = threading.Lock()

        def worker() -> None:
            r = qm.acquire("m", 10)
            with lock:
                ok.append(r is not None)

        threads = [threading.Thread(target=worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(ok) == 30
        assert qm.get_remaining("m")["rpm_used"] == 30
