"""Đồ giả dùng chung để chạy THẬT vòng thử lại của ``_call_gemini`` trong kiểm thử.

Gom từ ``test_overload_quota_v323585`` khi ``test_daily_quota_wait_v323586`` cần cùng bộ
đồ giả: đồng hồ giả cho bộ quản lý quota, Gemini giả trả lỗi theo kịch bản, và nguyên văn
các thông báo lỗi chép từ nhật ký ứng dụng.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from subtitles_extractor.infrastructure.translation import gemini_quota_manager
from subtitles_extractor.infrastructure.translation.analysis_checkpoint import (
    AnalysisCheckpointStore,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    RateLimit,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

SCARCE_MODEL = "gemini-3.8-flash"

#: Bước "thành công" trong kịch bản của :class:`ScriptedGemini`.
OK = "ok"

REAL_503 = (
    "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently "
    "experiencing high demand. Spikes in demand are usually temporary. Please try again "
    "later.', 'status': 'UNAVAILABLE'}}"
)


def real_daily_429(retry_seconds: str = "23.017170154", retry_delay: str = "23") -> str:
    """Lỗi hết hạn mức request/NGÀY, chép nguyên từ nhật ký (chỉ thay thời gian chờ).

    Args:
        retry_seconds: Số giây trong câu "Please retry in …s".
        retry_delay: Số giây trong trường ``retryDelay``.

    Returns:
        Chuỗi lỗi như SDK Gemini trả về.
    """
    return (
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
        "current quota, please check your plan and billing details. \\n* Quota exceeded "
        "for metric: generativelanguage.googleapis.com/generate_content_free_tier_"
        f"requests, limit: 20, model: gemini-3.8-flash\\nPlease retry in {retry_seconds}s"
        ".', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/"
        "google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage."
        "googleapis.com/generate_content_free_tier_requests', 'quotaId': "
        "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': "
        "{'location': 'global', 'model': 'gemini-3.8-flash'}, 'quotaValue': '20'}]}, "
        "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': "
        f"'{retry_delay}s'}}]}}}}"
    )


REAL_DAILY_429 = real_daily_429()


class FakeClock:
    """Đồng hồ giả cho bộ quản lý quota: ``sleep`` chỉ đẩy thời gian, không chờ thật."""

    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def install_fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Thay ``time`` của module quản lý quota bằng đồng hồ giả và trả đồng hồ đó."""
    clock = FakeClock()
    monkeypatch.setattr(gemini_quota_manager, "time", clock)
    return clock


class ScriptedGemini:
    """Gemini giả: mỗi lượt gọi lấy một bước của kịch bản; hết thì lặp lại bước cuối.

    Bước là :data:`OK` (trả kết quả rỗng hợp lệ) hoặc nguyên văn một thông báo lỗi.
    """

    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.calls = 0

    def generate_content(self, model: str, contents: list, config: object) -> object:
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if step != OK:
            raise RuntimeError(step)
        return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None)


def quota_manager(
    tmp_path: Path, limits: dict[str, RateLimit] | None = None
) -> GeminiQuotaManager:
    """Bộ quản lý quota thật, lưu trạng thái vào thư mục tạm của test."""
    return GeminiQuotaManager(rate_limits=limits, state_path=tmp_path / "quota.json")


def confirmed_scarce_manager(tmp_path: Path) -> GeminiQuotaManager:
    """Hạn mức 20/ngày do chính SERVER nêu — đúng ca thật của gemini-3.8-flash."""
    manager = quota_manager(tmp_path)
    manager.learn_daily_limit_from_server(SCARCE_MODEL, 20)
    return manager


def build_translator(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    manager: GeminiQuotaManager | None,
    server: Any,
    *,
    keys: str = "KEY_A",
    retry_count: int = 3,
) -> tuple[GeminiSubtitleTranslator, list[float]]:
    """Adapter Gemini thật nối với server giả; trả kèm danh sách các lần "ngủ"."""
    translator = GeminiSubtitleTranslator(
        api_key=keys, retry_count=retry_count, quota_manager=manager
    )
    monkeypatch.setattr(
        translator, "_get_client", lambda: SimpleNamespace(models=server)
    )
    slept: list[float] = []

    def _sleep(seconds: float, _cancel_cb: object) -> bool:
        slept.append(seconds)
        clock.sleep(seconds)
        return False

    monkeypatch.setattr(translator, "_interruptible_sleep", _sleep)
    return translator, slept


def bare_analysis_adapter(
    manager: GeminiQuotaManager | None = None, checkpoint_dir: Path | None = None
) -> GeminiSubtitleTranslator:
    """Adapter KHÔNG chạy ``__init__`` (không cần SDK) cho các test thay ``_call_gemini``.

    Gán tay đúng các thuộc tính luồng phân tích đụng tới; thêm thuộc tính mới ở MỘT chỗ
    này thay vì ở từng file test.

    Args:
        manager: Bộ quản lý quota, ``None`` nếu test không quan tâm hạn mức.
        checkpoint_dir: Thư mục lưu tiến độ phân tích; ``None`` thì không lưu.
    """
    adapter = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    adapter._types_module = MagicMock()
    adapter._video_handle_cache = {}
    adapter._reading_cps_budget = 0.0
    adapter._emotion_cues = False
    adapter._api_key = "KEY_A"
    adapter._api_keys = ["KEY_A"]
    adapter._quota_manager = manager
    adapter._analysis_checkpoints = (
        AnalysisCheckpointStore(checkpoint_dir) if checkpoint_dir else None
    )
    return adapter


def call_gemini(translator: GeminiSubtitleTranslator, model: str, **kwargs: Any) -> dict:
    """Gọi ``_call_gemini`` với cấu hình tối thiểu."""
    config = SimpleNamespace(
        temperature=0.2, response_mime_type="application/json",
        response_schema={}, system_instruction="x",
    )
    return translator._call_gemini(
        model, "prompt", config, lambda _data: None, **kwargs
    )


def daily_requests_used(
    manager: GeminiQuotaManager, model: str, key: str = "KEY_A"
) -> int:
    """Số request/ngày bộ quản lý quota đã đếm cho ``key``."""
    key_id = GeminiQuotaManager.key_fingerprint(key)
    return manager.get_remaining(model, key_id=key_id)["rpd_used"]
