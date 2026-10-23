"""[v3.23.607] Hạn mức ngày Gemini hồi lúc 00:00 giờ Thái Bình Dương, không theo UTC.

Tài liệu Gemini API: "Requests per day (RPD) quotas reset at midnight Pacific time".
Bản v3.23.606 lại tin ``retryDelay`` của lỗi 429 (luôn chỉ về 00:00 UTC = 07:00 giờ
máy) và đổi mốc ngày sang UTC. Nhật ký thật (giờ máy UTC+7) bác bỏ điều đó:

* 12/10: gemini-3.8-flash hết hạn mức lúc 06:17 (key …53b9) và 06:33 (key …1f16);
  lúc 06:33 server gửi ``retryDelay`` ~26 phút. Lúc 11:06 đến 11:07, sau khi bản
  v3.23.606 tưởng hạn mức đã hồi từ 07:00, CẢ BA key vẫn bị từ chối; server lúc đó
  gửi "19 giờ 52 phút".
* 08/10: ba key hết hạn mức gemini-3.8-flash lúc 07:46, 08:00, 08:05; lúc 14:19 gọi
  lại thành công — sau 14:00 giờ máy = 00:00 PDT.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.translation import gemini_quota_clock
from subtitles_extractor.infrastructure.translation.gemini_quota_clock import (
    describe_quota_reset,
    next_quota_reset,
    pacific_utc_offset,
    quota_day,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    daily_quota_message,
)

VIETNAM = timezone(timedelta(hours=7))
MODEL = "gemini-3.8-flash"
KEY = GeminiQuotaManager.key_fingerprint("KEY_2")


def _vn(month: int, day: int, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, second, tzinfo=VIETNAM)


class _Clock:
    """Đồng hồ giả cho ``datetime.now`` trong mô-đun mốc hạn mức."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, start: datetime) -> None:
        self.now = start
        clock = self

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return clock.now.astimezone(tz) if tz is not None else clock.now

        monkeypatch.setattr(gemini_quota_clock, "datetime", _FrozenDatetime)


class TestRealTimelines:
    def test_twelfth_october_is_one_quota_day_until_two_pm(self) -> None:
        exhausted = [_vn(10, 12, 6, 17, 16), _vn(10, 12, 6, 33, 28)]
        still_refused = [_vn(10, 12, 11, 6, 29), _vn(10, 12, 11, 7, 43)]
        days = {quota_day(moment) for moment in exhausted + still_refused}
        assert days == {"2026-10-11"}
        assert quota_day(_vn(10, 12, 14, 0, 1)) == "2026-10-12"

    def test_eighth_october_succeeded_after_two_pm(self) -> None:
        refused = quota_day(_vn(10, 8, 8, 5, 36))
        succeeded = quota_day(_vn(10, 8, 14, 19, 54))
        assert refused != succeeded

    def test_message_at_eleven_names_two_pm_not_tomorrow_morning(self) -> None:
        message = daily_quota_message(MODEL, _vn(10, 12, 11, 7, 43))
        assert "00:00 giờ Thái Bình Dương" in message
        assert "còn khoảng 2 giờ 52 phút" in message
        assert describe_quota_reset(_vn(10, 12, 11, 7, 43), VIETNAM) == (
            "Hạn mức ngày của Gemini hồi lúc 00:00 giờ Thái Bình Dương — 14:00 hôm nay "
            "theo giờ máy (còn khoảng 2 giờ 52 phút)."
        )


class TestServerBlock:
    def test_block_from_dawn_still_holds_at_eleven_and_lifts_at_two(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        clock = _Clock(monkeypatch, _vn(10, 12, 6, 33, 28))
        manager = GeminiQuotaManager(state_path=tmp_path / "quota.json")
        manager.mark_daily_exhausted(MODEL, key_id=KEY)

        clock.now = _vn(10, 12, 11, 6, 29)
        assert manager.is_server_daily_blocked(MODEL, key_id=KEY)
        assert manager.get_remaining(MODEL, key_id=KEY)["rpd_remaining"] == 0

        clock.now = _vn(10, 12, 14, 0, 30)
        assert not manager.is_server_daily_blocked(MODEL, key_id=KEY)
        reopened = GeminiQuotaManager(state_path=tmp_path / "quota.json")
        assert not reopened.is_server_daily_blocked(MODEL, key_id=KEY)

    def test_state_written_by_v323606_does_not_block_the_next_pacific_day(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Tệp thật kiểu v3.23.606: dấu chặn ghi lúc 11:07 mang nhãn UTC "2026-10-12"."""
        state = tmp_path / "quota.json"
        model_key = f"{KEY}|{MODEL}"
        state.write_text(json.dumps({
            model_key: {"period_key": "2026-10-12", "count": 20},
            "__meta__": {
                "server_daily_block": {model_key: "2026-10-12"},
                "learned_rpd": {MODEL: 20},
                "daily_reset_offset_s": 0,
            },
        }), encoding="utf-8")
        _Clock(monkeypatch, _vn(10, 12, 14, 30))  # ngày hạn mức mới đã bắt đầu
        manager = GeminiQuotaManager(state_path=state)
        assert not manager.is_server_daily_blocked(MODEL, key_id=KEY)
        assert manager.get_remaining(MODEL, key_id=KEY)["rpd_remaining"] == 20


class TestPacificClock:
    def test_us_rule_matches_tz_database(self) -> None:
        zoneinfo = pytest.importorskip("zoneinfo")
        los_angeles = zoneinfo.ZoneInfo("America/Los_Angeles")
        moment = datetime(2024, 1, 1, tzinfo=UTC)
        while moment < datetime(2031, 1, 1, tzinfo=UTC):
            expected = moment.astimezone(los_angeles).utcoffset()
            assert gemini_quota_clock._us_rule_offset(moment) == expected, moment
            moment += timedelta(minutes=97)

    def test_windows_without_tzdata_uses_the_us_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import zoneinfo

        def _missing(_key: str) -> None:
            raise zoneinfo.ZoneInfoNotFoundError("No time zone found")

        monkeypatch.setattr(zoneinfo, "ZoneInfo", _missing)
        assert pacific_utc_offset(_vn(10, 12, 11, 7)) == timedelta(hours=-7)
        assert pacific_utc_offset(_vn(11, 10, 11, 7)) == timedelta(hours=-8)

    @pytest.mark.parametrize(
        ("now", "reset_local"),
        [
            (_vn(10, 12, 6, 33), _vn(10, 12, 14, 0)),
            (_vn(10, 12, 14, 0, 1), _vn(10, 13, 14, 0)),
            # Hết giờ mùa hè (01/11 02:00 PDT): mốc hồi lùi sang 15:00 giờ máy.
            (_vn(11, 1, 20, 0), _vn(11, 2, 15, 0)),
            # Vào giờ mùa hè (08/03 02:00 PST): nửa đêm 08/03 còn PST (15:00), nửa đêm
            # 09/03 đã PDT (14:00).
            (_vn(3, 8, 10, 0), _vn(3, 8, 15, 0)),
            (_vn(3, 8, 15, 30), _vn(3, 9, 14, 0)),
        ],
    )
    def test_next_reset_is_pacific_midnight(
        self, now: datetime, reset_local: datetime
    ) -> None:
        assert next_quota_reset(now) == reset_local.astimezone(UTC)


class TestStateWrittenByV323606BeforeTwoPm:
    def test_block_from_eleven_is_kept_until_two_pm(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Mở lại ứng dụng lúc 11:51 (ngày hạn mức vẫn là 11/10): dấu chặn lúc 11:07 còn
        đúng — giữ, để ứng dụng báo ngay thay vì tải 17 đoạn video rồi mới bị từ chối."""
        state = tmp_path / "quota.json"
        model_key = f"{KEY}|{MODEL}"
        state.write_text(json.dumps({
            model_key: {"period_key": "2026-10-12", "count": 20},
            "__meta__": {
                "server_daily_block": {model_key: "2026-10-12"},
                "daily_reset_offset_s": 0,
            },
        }), encoding="utf-8")
        clock = _Clock(monkeypatch, _vn(10, 12, 11, 51))
        manager = GeminiQuotaManager(state_path=state)
        assert manager.is_server_daily_blocked(MODEL, key_id=KEY)
        clock.now = _vn(10, 12, 14, 0, 30)
        assert not manager.is_server_daily_blocked(MODEL, key_id=KEY)
