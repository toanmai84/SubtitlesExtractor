"""[v3.23.643] Lỗi do lượt rà soát độc lập tìm ra trong chính bản này."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from subtitles_extractor.infrastructure.translation.model_health import ModelHealth
from subtitles_extractor.infrastructure.tts import audio_mastering as am
from tests.unit.infrastructure.gemini_call_harness import (
    REAL_503,
    FakeClock,
    build_translator,
    call_gemini,
    install_fake_clock,
    quota_manager,
    real_daily_429,
)
from tests.unit.infrastructure.test_analysis_resume_v323564 import _adapter, _Gemini, _run

LITE_31 = "gemini-3.1-flash-lite"
FLASH_25 = "gemini-2.5-flash"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


def test_primary_thinking_config_is_restored_when_returning_to_it(
    monkeypatch, clock, tmp_path: Path
) -> None:
    seen: list[tuple[str, object]] = []
    overloaded = {LITE_31: 4}

    class _Server:
        def __init__(self) -> None:
            self.models = self

        def generate_content(self, model, contents, config):
            seen.append((model, getattr(config, "thinking_config", None)))
            if model == FLASH_25:
                raise RuntimeError(real_daily_429())
            if overloaded.get(model, 0) > 0:
                overloaded[model] -= 1
                raise RuntimeError(REAL_503)
            return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None)

    translator, _ = build_translator(monkeypatch, clock, quota_manager(tmp_path), _Server(), retry_count=6)
    translator._model_health = ModelHealth(lambda: clock.now, seed_models=[LITE_31, FLASH_25])
    marker = object()
    config = SimpleNamespace(
        temperature=0.2, response_mime_type="application/json", response_schema={},
        system_instruction="x", thinking_config=marker,
    )
    translator._call_gemini(LITE_31, "p", config, lambda d: None)
    primary_calls = [t for m, t in seen if m == LITE_31]
    assert FLASH_25 in [m for m, _ in seen]
    assert primary_calls and all(t is marker for t in primary_calls)      # kể cả lượt SAU khi quay về


def test_stereo_audio_still_works_in_the_mastering_helpers() -> None:
    stereo = (np.random.default_rng(0).standard_normal((50_000, 2)) * 0.1).astype(np.float32)
    assert am._peak_abs_chunked(stereo) == pytest.approx(float(np.max(np.abs(stereo))))
    assert am.dc_block(stereo, 16_000).shape == stereo.shape
    assert am.normalize_to_lufs(stereo, 16_000, -16.0).shape == stereo.shape


def test_analysis_hold_is_reentrant(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter.hold_analysis_progress()
    adapter.hold_analysis_progress()                           # lượt chồng nhau trên cùng bộ dịch
    _run(adapter, _Gemini())
    adapter.release_analysis_progress(clear=True)              # lượt trong thả: CHƯA xoá
    assert len(list(tmp_path.glob("analysis_*.json"))) == 1
    adapter.release_analysis_progress(clear=True)              # lượt ngoài cùng: xoá
    assert list(tmp_path.glob("analysis_*.json")) == []
