"""[v3.23.647] Danh mục giọng VieNeu đọc từ đĩa: không nạp engine 36 s chỉ để liệt kê giọng."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as module
from subtitles_extractor.infrastructure.tts import voice_catalog_cache as cache

PAIRS = [("Hải Đăng (Nam)", "Hải Đăng"), ("Thu Hà (Nữ)", "Thu Hà")]


def test_roundtrip_keeps_other_modes_and_checks_version(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    assert cache.save_catalog(path, "v3turbo", "3.8.0", PAIRS)
    assert cache.save_catalog(path, "standard", "3.8.0", PAIRS[:1])
    assert cache.load_catalog(path, "v3turbo", "3.8.0") == PAIRS
    assert cache.load_catalog(path, "standard", "3.8.0") == PAIRS[:1]
    assert cache.load_catalog(path, "v3turbo", "3.9.0") is None          # nâng bản vieneu ⇒ đọc lại
    assert cache.load_catalog(path, "turbo", "3.8.0") is None
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("payload", ["", "[]", "{", '{"format": 1}', '{"format": 1, "modes": {"v3turbo": 5}}',
                                      '{"format": 2, "modes": {}}'])
def test_corrupt_or_foreign_files_mean_no_cache(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "c.json"
    path.write_text(payload, encoding="utf-8")
    assert cache.load_catalog(path, "v3turbo", "3.8.0") is None
    assert cache.save_catalog(path, "v3turbo", "3.8.0", PAIRS)           # ghi đè được file hỏng
    assert cache.load_catalog(path, "v3turbo", "3.8.0") == PAIRS


def test_nothing_is_saved_without_a_version_or_voices(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    assert not cache.save_catalog(path, "v3turbo", None, PAIRS)
    assert not cache.save_catalog(path, "v3turbo", "3.8.0", [])
    assert not cache.save_catalog(None, "v3turbo", "3.8.0", PAIRS)
    assert cache.load_catalog(None, "v3turbo", "3.8.0") is None and not path.exists()


class _Engine:
    def list_preset_voices(self):
        return list(PAIRS)


def test_adapter_loads_the_engine_once_then_serves_the_catalog_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache, "default_cache_path", lambda: tmp_path / "c.json")
    monkeypatch.setattr(module, "installed_vieneu_version", lambda: "3.8.0")
    monkeypatch.setattr(module.VieNeuTtsAdapter, "is_available", lambda self: True)
    loads: list[int] = []

    def fake_load(self, device):          # noqa: ANN001
        loads.append(1)
        self._engine = _Engine()
        return self._engine

    monkeypatch.setattr(module.VieNeuTtsAdapter, "_get_or_load_engine", fake_load)
    first = module.VieNeuTtsAdapter()
    assert first.list_speakers("vi-VN") == ["Hải Đăng", "Thu Hà"] and loads == [1]
    second = module.VieNeuTtsAdapter()                                   # phiên/adapter mới: không nạp engine
    assert second.list_speakers("vi-VN") == ["Hải Đăng", "Thu Hà"]
    assert [p.voice_id for p in second.list_voice_catalog("vi-VN")] == ["Hải Đăng", "Thu Hà"]
    assert loads == [1] and second._engine is None
    monkeypatch.setattr(module, "installed_vieneu_version", lambda: "3.9.0")     # nâng bản ⇒ đọc lại từ engine
    third = module.VieNeuTtsAdapter()
    third.list_speakers("vi-VN")
    assert loads == [1, 1]
