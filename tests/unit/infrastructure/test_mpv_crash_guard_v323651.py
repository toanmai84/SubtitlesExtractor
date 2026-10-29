"""[v3.23.651] Từ 3 fault.log thật trên Windows: `access violation` trong mpv (lúc dò mpv khi khởi động và lúc nạp video).

Các dòng ``Windows fatal exception: code 0xe24c4a02`` là ngoại lệ LuaJIT đã được xử lý (faulthandler ghi nhầm là fatal).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.logging import crash_hooks

LUA_ONLY = "Windows fatal exception: code 0xe24c4a02\n\nThread 0x1 (most recent call first):\n  File \"mpv.py\", line 689 in _event_generator\n"
MPV_AV = LUA_ONLY + "Windows fatal exception: access violation\n\nThread 0x2 (most recent call first):\n  File \"mpv.py\", line 1058 in name\n"


@pytest.fixture
def hooks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(crash_hooks, "_INSTALLED", False)
    monkeypatch.setattr(crash_hooks, "_FAULT_FILE", None)
    monkeypatch.setattr(crash_hooks, "_PREVIOUS_CRASH", None)
    monkeypatch.setattr(crash_hooks, "_SESSION_MARKER", None)
    monkeypatch.setattr(crash_hooks, "_LOG_DIR", None)
    monkeypatch.setattr(sys, "excepthook", lambda *_a: None)
    monkeypatch.delenv("SUBTITLES_EXTRACTOR_FORCE_MPV", raising=False)
    return crash_hooks


def test_access_violation_in_mpv_disables_mpv_next_start_v323651(hooks, tmp_path: Path) -> None:
    (tmp_path / "fault.log").write_text(MPV_AV, encoding="utf-8")
    hooks.install_crash_hooks(tmp_path)
    assert (tmp_path / "mpv_disabled.flag").exists()
    assert hooks.mpv_disabled_by_crash() is True
    info = hooks.pop_previous_crash()
    assert info is not None and info["mpv_disabled"] is True


def test_benign_lua_exceptions_alone_do_not_disable_mpv_v323651(hooks, tmp_path: Path) -> None:
    (tmp_path / "fault.log").write_text(LUA_ONLY * 20, encoding="utf-8")
    hooks.install_crash_hooks(tmp_path)
    assert not (tmp_path / "mpv_disabled.flag").exists()
    assert hooks.mpv_disabled_by_crash() is False


def test_force_env_overrides_the_guard_v323651(hooks, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "fault.log").write_text(MPV_AV, encoding="utf-8")
    hooks.install_crash_hooks(tmp_path)
    monkeypatch.setenv("SUBTITLES_EXTRACTOR_FORCE_MPV", "1")
    assert hooks.mpv_disabled_by_crash() is False


def _fake_mpv(monkeypatch: pytest.MonkeyPatch, *, reject_lean: bool = False):
    created: list[dict] = []

    class MPVError(Exception):
        pass

    class FakeMPV:
        def __init__(self, **kwargs):
            if reject_lean and "load_scripts" in kwargs:
                raise MPVError("option not found")
            created.append(kwargs)
            self.calls: list[tuple] = []
            self.pause = False
            self.idle_active = False
            self.duration = 10.0

        def command(self, *args):
            self.calls.append(args)

        def wait_until_playing(self, *a, **k):  # pragma: no cover - không được gọi
            raise AssertionError("wait_until_playing (observer/Future) không được dùng nữa")

    module = types.ModuleType("mpv")
    module.MPV = FakeMPV
    module.MPVError = MPVError
    monkeypatch.setitem(sys.modules, "mpv", module)
    return created


def test_availability_probe_never_instantiates_mpv_v323651(monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.presentation import widgets

    created = _fake_mpv(monkeypatch)
    monkeypatch.delenv("SUBTITLES_EXTRACTOR_NO_MPV", raising=False)
    monkeypatch.setattr(crash_hooks, "_LOG_DIR", None)
    assert widgets._is_mpv_available() is True
    assert created == []                       # trước đây dựng mpv.MPV() rồi terminate() → AV ngay lúc khởi động


def test_availability_respects_guard_flag_and_env_v323651(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from subtitles_extractor.presentation import widgets

    _fake_mpv(monkeypatch)
    (tmp_path / "mpv_disabled.flag").write_text("x", encoding="utf-8")
    monkeypatch.setattr(crash_hooks, "_LOG_DIR", tmp_path)
    monkeypatch.delenv("SUBTITLES_EXTRACTOR_FORCE_MPV", raising=False)
    assert widgets._is_mpv_available() is False
    monkeypatch.setattr(crash_hooks, "_LOG_DIR", None)
    monkeypatch.setenv("SUBTITLES_EXTRACTOR_NO_MPV", "1")
    assert widgets._is_mpv_available() is False


def test_adapter_load_sets_pause_first_and_never_uses_future_wait_v323651(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.video.mpv_player_adapter import MpvPlayerAdapter

    created = _fake_mpv(monkeypatch)
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    adapter = MpvPlayerAdapter(wid=1234, mpv_options={})
    # [v3.23.665] created[0] có thể là lõi thăm dò tuỳ chọn (vo=null); lõi thật là bản cuối.
    assert created[-1]["load_scripts"] is False and created[-1]["ytdl"] is False
    adapter.load(video)                        # FakeMPV.wait_until_playing sẽ ném AssertionError nếu bị gọi
    assert adapter._mpv.calls[-1][:2] == ("loadfile", str(video))
    assert adapter._mpv.pause is True


def test_adapter_falls_back_when_libmpv_rejects_lean_options_v323651(monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.infrastructure.video.mpv_player_adapter import MpvPlayerAdapter

    created = _fake_mpv(monkeypatch, reject_lean=True)
    MpvPlayerAdapter(wid=1, mpv_options={})
    assert created and "load_scripts" not in created[0]
