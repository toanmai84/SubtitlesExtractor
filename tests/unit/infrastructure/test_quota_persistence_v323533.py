"""[v3.23.533] Dấu chặn từ server và hạn mức học được phải SỐNG SÓT qua lần khởi động lại.

Trước đây cả hai chỉ nằm trong bộ nhớ:
- Mất dấu "server đã từ chối key này hôm nay" → mở lại ứng dụng là thử lại key đã cạn, đốt request
  vô ích, mà hạn mức thật chỉ 20/ngày.
- Mất hạn mức học được → lần sau lại dùng con số chép tay lạc hậu, khiến v3.23.530 gần như vô nghĩa
  giữa các phiên.
"""
from __future__ import annotations

import json

import pytest

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)

MODEL = "m"


@pytest.fixture
def make_manager(tmp_path):
    path = tmp_path / "quota.json"

    def _make() -> GeminiQuotaManager:
        return GeminiQuotaManager(
            rate_limits={MODEL: RateLimit(rpm=100, tpm=10_000, rpd=70)},
            state_path=path,
        )

    _make.path = path
    return _make


class TestPersistence:
    def test_server_block_survives_restart(self, make_manager) -> None:
        make_manager().mark_daily_exhausted(MODEL, key_id="k1")
        assert make_manager().is_server_daily_blocked(MODEL, key_id="k1") is True

    def test_other_key_not_blocked_after_restart(self, make_manager) -> None:
        make_manager().mark_daily_exhausted(MODEL, key_id="k1")
        assert make_manager().is_server_daily_blocked(MODEL, key_id="k2") is False

    def test_learned_limit_survives_restart(self, make_manager) -> None:
        make_manager().learn_daily_limit_from_server(MODEL, 20)
        assert make_manager().get_remaining(MODEL, key_id="k9")["rpd_limit"] == 20

    def test_daily_counters_still_restored(self, make_manager) -> None:
        """Không được làm hỏng chức năng cũ khi thêm phần mới vào tệp."""
        first = make_manager()
        first.acquire(MODEL, estimated_tokens=1, key_id="k1")
        assert make_manager().get_remaining(MODEL, key_id="k1")["rpd_used"] >= 1


class TestFileFormat:
    def test_counters_stay_at_top_level(self, make_manager) -> None:
        """Giữ bộ đếm ở cấp gốc để tệp cũ và mới đọc lẫn nhau được."""
        manager = make_manager()
        manager.acquire(MODEL, estimated_tokens=1, key_id="k1")
        data = json.loads(make_manager.path.read_text(encoding="utf-8"))
        assert any(key != "__meta__" for key in data)
        assert "__meta__" in data

    def test_meta_key_not_treated_as_counter(self, make_manager) -> None:
        manager = make_manager()
        manager.mark_daily_exhausted(MODEL, key_id="k1")
        restored = make_manager()
        assert restored.get_remaining("__meta__", key_id="")["rpd_used"] == 0

    def test_corrupt_meta_is_ignored(self, make_manager) -> None:
        make_manager.path.write_text(
            json.dumps({"__meta__": "không phải dict"}), encoding="utf-8"
        )
        manager = make_manager()
        assert manager.is_server_daily_blocked(MODEL, key_id="k1") is False

    def test_invalid_learned_value_ignored(self, make_manager) -> None:
        make_manager.path.write_text(
            json.dumps({"__meta__": {"learned_rpd": {MODEL: "không phải số"}}}),
            encoding="utf-8",
        )
        assert make_manager().get_remaining(MODEL, key_id="k1")["rpd_limit"] == 70
