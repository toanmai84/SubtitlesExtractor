"""[v3.23.416] Test auto-tune batch size theo VRAM + tham số batch_size của adapter."""

from __future__ import annotations

import subprocess

import pytest

from subtitles_extractor.infrastructure.tts import vieneu_batch_tuning as tuning


def test_batch_size_for_vram_thresholds() -> None:
    assert tuning.batch_size_for_vram_gb(16.0) == 32
    assert tuning.batch_size_for_vram_gb(20.0) == 32
    assert tuning.batch_size_for_vram_gb(12.0) == 24
    assert tuning.batch_size_for_vram_gb(8.0) == 16
    assert tuning.batch_size_for_vram_gb(6.0) == 8
    assert tuning.batch_size_for_vram_gb(2.0) == 4


def test_detect_vram_parses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Result:
        returncode = 0
        stdout = "8192\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
    vram = tuning.detect_total_vram_gb()
    assert vram is not None
    assert abs(vram - 8.0) < 0.01  # 8192 MiB = 8 GiB


def test_detect_vram_none_when_no_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*a, **k):
        raise OSError("nvidia-smi không tồn tại")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert tuning.detect_total_vram_gb() is None


def test_suggest_auto_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuning, "detect_total_vram_gb", lambda: None)
    assert tuning.suggest_auto_batch_size() == tuning.DEFAULT_AUTO_BATCH_SIZE


def test_suggest_auto_uses_detected_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuning, "detect_total_vram_gb", lambda: 12.0)
    assert tuning.suggest_auto_batch_size() == 24


def test_adapter_batch_size_param_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tham số batch_size truyền vào phải ưu tiên hơn env."""
    from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
        VieNeuTtsAdapter,
    )

    monkeypatch.setenv("SUBEXT_VIENEU_BATCH", "99")
    adapter = VieNeuTtsAdapter(batch_size=16)
    assert adapter._batch_size == 16  # tham số thắng env
    adapter_off = VieNeuTtsAdapter(batch_size=0)
    assert adapter_off._batch_size == 0


def test_adapter_batch_size_none_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
        VieNeuTtsAdapter,
    )

    monkeypatch.setenv("SUBEXT_VIENEU_BATCH", "24")
    adapter = VieNeuTtsAdapter(batch_size=None)
    assert adapter._batch_size == 24  # None → lùi về env


def test_gpu_chunk_batch_thresholds() -> None:
    """[v3.23.420] Số chunk/forward GPU theo VRAM (đòn bẩy tốc độ thật)."""
    assert tuning.gpu_chunk_batch_for_vram_gb(8.0) == 64   # RTX 3070 Ti
    assert tuning.gpu_chunk_batch_for_vram_gb(12.0) == 96
    assert tuning.gpu_chunk_batch_for_vram_gb(16.0) == 128
    assert tuning.gpu_chunk_batch_for_vram_gb(24.0) == 192
    assert tuning.gpu_chunk_batch_for_vram_gb(6.0) == 32
    assert tuning.gpu_chunk_batch_for_vram_gb(4.0) == 16


def test_suggest_gpu_chunk_batch_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuning, "detect_total_vram_gb", lambda: None)
    assert tuning.suggest_gpu_chunk_batch() == tuning.DEFAULT_GPU_CHUNK_BATCH


def test_suggest_gpu_chunk_batch_uses_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tuning, "detect_total_vram_gb", lambda: 16.0)
    assert tuning.suggest_gpu_chunk_batch() == 128
