"""[v3.23.586] Hết hạn mức NGÀY mà không còn key: dừng ngay, không ngủ theo lời server.

Nhật ký thật v3.23.585 (phim "灭门惨案II", phân tích bằng gemini-3.8-flash): đoạn 2 nhận
429 hết hạn mức ngày, cả ba API key đều đã cạn. Server gửi kèm ``retryDelay`` 36737 giây
và ứng dụng ghi "Chờ 36737.5s rồi thử lại" rồi đứng im từ 20:47 tới khi người dùng bấm
huỷ lúc 05:44 sáng hôm sau. [v3.23.607] Con số đó chỉ về 00:00 UTC, KHÔNG phải mốc hạn mức
hồi (00:00 giờ Thái Bình Dương) — xem ``test_quota_reset_pacific_v323607``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    MAX_SERVER_COOLDOWN_SEC,
    GeminiQuotaManager,
    QuotaExhaustedError,
    daily_quota_message,
    describe_wait,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from tests.unit.infrastructure.gemini_call_harness import (
    OK,
    REAL_503,
    REAL_DAILY_429,
    SCARCE_MODEL,
    FakeClock,
    ScriptedGemini,
    build_translator,
    call_gemini,
    confirmed_scarce_manager,
    daily_requests_used,
    install_fake_clock,
    real_daily_429,
)

#: Đúng con số trong nhật ký: "Chờ 36737.5s rồi thử lại (theo retryDelay server)".
OVERNIGHT_429 = real_daily_429(retry_seconds="36736.017170154", retry_delay="36736")

#: Hạn mức request/PHÚT: cùng metric ``free_tier_requests`` với hạn mức ngày, chỉ khác mã
#: quota. Nhật ký chưa có ca nào (bộ quản lý quota tự điều tiết RPM) nên dựng theo khuôn
#: của lỗi hạn mức ngày.
PER_MINUTE_429 = REAL_DAILY_429.replace(
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
    "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
).replace("limit: 20", "limit: 5")


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


class TestNoSleepOnDailyQuota:
    def test_real_case_stops_instead_of_sleeping_ten_hours(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Đoạn 2 của ca thật: một lần 503, rồi 429 hết hạn mức ngày; key kia đã cạn."""
        manager = confirmed_scarce_manager(tmp_path)
        manager.mark_daily_exhausted(
            SCARCE_MODEL, key_id=GeminiQuotaManager.key_fingerprint("KEY_B")
        )
        server = ScriptedGemini([REAL_503, OVERNIGHT_429])
        translator, slept = build_translator(
            monkeypatch, clock, manager, server, keys="KEY_A\nKEY_B"
        )
        # [v3.23.607] Mốc hồi theo 00:00 giờ Thái Bình Dương, không theo retryDelay.
        with pytest.raises(QuotaExhaustedError, match="00:00 giờ Thái Bình Dương"):
            call_gemini(translator, SCARCE_MODEL)
        assert slept == [30.0]  # chỉ lần chờ của 503; trước đây thêm 36737,5 giây
        assert server.calls == 2

    def test_single_key_stops_at_once(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        server = ScriptedGemini([OVERNIGHT_429])
        translator, slept = build_translator(monkeypatch, clock, manager, server)
        with pytest.raises(QuotaExhaustedError, match="hết hạn mức request/ngày"):
            call_gemini(translator, SCARCE_MODEL)
        assert slept == []
        assert server.calls == 1
        # Lượt gọi bị 429 không tính vào hạn mức, nhưng key được đánh dấu đã cạn.
        assert daily_requests_used(manager, SCARCE_MODEL) == 20

    def test_without_quota_manager_it_stops_too(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
    ) -> None:
        server = ScriptedGemini([OVERNIGHT_429])
        translator, slept = build_translator(monkeypatch, clock, None, server)
        with pytest.raises(QuotaExhaustedError):
            call_gemini(translator, SCARCE_MODEL)
        assert slept == []

    def test_spare_key_is_still_used(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        server = ScriptedGemini([OVERNIGHT_429, OK])
        translator, slept = build_translator(
            monkeypatch, clock, manager, server, keys="KEY_A\nKEY_B"
        )
        assert call_gemini(translator, SCARCE_MODEL) == {"subtitles": []}
        assert slept == []
        assert translator._api_key == "KEY_B"


class TestPerMinuteQuotaIsNotDaily:
    def test_classification(self) -> None:
        is_daily = GeminiSubtitleTranslator._is_daily_quota_error
        assert is_daily(RuntimeError(REAL_DAILY_429)) is True
        assert is_daily(RuntimeError(OVERNIGHT_429)) is True
        assert is_daily(RuntimeError(PER_MINUTE_429)) is False
        # Không có mã quota: giữ cách nhận diện cũ theo tên metric.
        bare = "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: free_tier_requests"
        assert is_daily(RuntimeError(bare)) is True

    def test_per_minute_limit_waits_and_retries(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Nhận nhầm hạn mức phút thành hạn mức ngày thì cả lượt chạy dừng oan."""
        manager = confirmed_scarce_manager(tmp_path)
        server = ScriptedGemini([PER_MINUTE_429, OK])
        translator, slept = build_translator(monkeypatch, clock, manager, server)
        assert call_gemini(translator, SCARCE_MODEL, video_refs=[object()]) == {
            "subtitles": []
        }
        assert slept == [24.5]  # retryDelay 23 giây + 1,5 giây đệm
        assert daily_requests_used(manager, SCARCE_MODEL) == 1


class TestServerWaitIsCapped:
    def test_absurd_wait_for_a_temporary_limit_is_capped(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        hours_long = PER_MINUTE_429.replace(
            "'retryDelay': '23s'", "'retryDelay': '7200s'"
        )
        server = ScriptedGemini([hours_long, OK])
        translator, slept = build_translator(monkeypatch, clock, manager, server)
        assert call_gemini(translator, SCARCE_MODEL, video_refs=[object()]) == {
            "subtitles": []
        }
        assert slept == [MAX_SERVER_COOLDOWN_SEC]


class TestMessages:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(36738.5, "10 giờ 12 phút"), (54644.5, "15 giờ 10 phút"), (7200.0, "2 giờ"),
         (2700.0, "45 phút"), (20.0, "1 phút")],
    )
    def test_wait_is_readable(self, seconds: float, expected: str) -> None:
        assert describe_wait(seconds) == expected

    def test_reset_time_follows_pacific_midnight(self) -> None:
        """[v3.23.607] Lỗi lúc 20:47 (giờ máy UTC+7, PDT): hồi lúc 14:00 hôm sau.

        Trước đây thông báo lấy ``retryDelay`` (36738 giây → "10 giờ 12 phút", tức 07:00)
        — mốc đó sai: nhật ký thật 12/10 lúc 11:06 vẫn bị từ chối.
        """
        from datetime import UTC, datetime

        now = datetime(2026, 10, 2, 13, 47, 44, tzinfo=UTC)
        message = daily_quota_message(SCARCE_MODEL, now)
        assert "00:00 giờ Thái Bình Dương" in message
        assert "còn khoảng 17 giờ 12 phút" in message
        assert "Server báo" not in message
