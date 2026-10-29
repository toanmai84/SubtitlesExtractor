"""[v3.23.665] Crash khi kéo thanh seek video (fault.log người dùng 29/10, Windows).

Chuỗi gây lỗi:
1. libmpv đời mới vẫn nạp script Lua dựng sẵn (select/positioning/commands/auto_profiles/console) vì bộ tuỳ chọn tắt
   script cũ thiếu tên → mỗi lần seek LuaJIT ném ngoại lệ nội bộ 0xe24c4a02 (vô hại, LuaJIT tự bắt).
2. faulthandler (all_threads=True) trên Windows chạy cho từng ngoại lệ đó và đọc ngăn xếp mọi luồng không giữ GIL
   → access violation thật.
3. Kéo thanh bắn một ``seek exact`` đồng bộ cho từng pixel → hàng trăm lần/giây.
"""

from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import QCoreApplication, QElapsedTimer
from PySide6.QtWidgets import QApplication

from subtitles_extractor.infrastructure.video import mpv_player_adapter as adapter


# ── 1. Tắt đủ mọi script Lua dựng sẵn, chỉ với tuỳ chọn libmpv thật sự có ─────────────────────────────


class _FakeMpvModule:
    """Giả libmpv: chỉ biết ``known``; tuỳ chọn lạ → AttributeError như python-mpv (OPTION_NOT_FOUND)."""

    def __init__(self, known: set[str]) -> None:
        self.known = known
        self.instances: list[_FakeMpv] = []
        module = self

        class MPV:  # noqa: N801 — giống tên lớp python-mpv
            def __init__(self, **kwargs) -> None:
                for key in kwargs:
                    if key.replace("_", "-") not in module.known and key not in ("wid", "log_handler", "vo", "ao", "idle", "hwdec"):
                        raise AttributeError("mpv option does not exist", -5, key)
                self.kwargs = kwargs
                self.terminated = False
                module.instances.append(self)

            def __getitem__(self, name: str):
                if name not in module.known:
                    raise AttributeError("mpv property does not exist", -8, name)
                return "yes"

            def terminate(self) -> None:
                self.terminated = True

        self.MPV = MPV


_NEW_LIBMPV = {"load-scripts", "ytdl", "osc", "load-stats-overlay", "load-console", "load-auto-profiles",
               "load-select", "load-positioning", "load-commands"}  # không còn "load-osd-console"


@pytest.fixture(autouse=True)
def _reset_probe_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter, "_SUPPORTED_LEAN", None, raising=False)


def test_new_libmpv_gets_every_builtin_script_disabled() -> None:
    fake = _FakeMpvModule(_NEW_LIBMPV)
    player = adapter.create_lean_mpv(fake, {"wid": "1"})
    probe = fake.instances[0]
    assert probe.terminated and probe.kwargs.get("vo") == "null"  # lõi thăm dò không gắn cửa sổ
    # Trước đây: "load_osd_console" bị từ chối → cả bộ tắt-script bị bỏ, dựng lại với cấu hình gốc (Lua chạy hết).
    for name in ("load_scripts", "load_select", "load_positioning", "load_commands", "load_auto_profiles", "load_console"):
        assert player.kwargs.get(name) is False, name
    assert "load_osd_console" not in player.kwargs
    assert len(fake.instances) == 2  # thăm dò + lõi thật — không có lõi bỏ rơi gắn vào cửa sổ


def test_old_libmpv_keeps_old_names() -> None:
    fake = _FakeMpvModule({"load-scripts", "ytdl", "osc", "load-stats-overlay", "load-osd-console"})
    player = adapter.create_lean_mpv(fake, {"wid": "1"})
    assert player.kwargs["load_osd_console"] is False and "load_select" not in player.kwargs


def test_probe_runs_once_per_process() -> None:
    fake = _FakeMpvModule(_NEW_LIBMPV)
    adapter.create_lean_mpv(fake, {"wid": "1"})
    adapter.create_lean_mpv(fake, {"wid": "2"})
    assert len(fake.instances) == 3


def test_probe_that_cannot_answer_falls_back_to_base_set() -> None:
    class _Mute:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class _Module:
        MPV = _Mute

    player = adapter.create_lean_mpv(_Module, {"wid": "1"})
    assert player.kwargs["load_scripts"] is False and player.kwargs["ytdl"] is False


# ── 2. faulthandler không đọc ngăn xếp các luồng khác trên Windows ────────────────────────────────


def test_faulthandler_single_thread_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from subtitles_extractor.infrastructure.logging import crash_hooks

    assert crash_hooks._FAULT_ALL_THREADS is (sys.platform != "win32")
    calls: list[bool] = []
    monkeypatch.setattr(crash_hooks.faulthandler, "enable", lambda file=None, all_threads=True: calls.append(all_threads))
    monkeypatch.setattr(crash_hooks.faulthandler, "disable", lambda: None)
    with open(tmp_path / "fault.log", "w") as fh:
        monkeypatch.setattr(crash_hooks, "_FAULT_FILE", fh)
        monkeypatch.setattr(crash_hooks, "_FAULT_ALL_THREADS", False)  # như trên Windows
        crash_hooks.rearm()
    assert calls == [False]


# ── 3. Kéo thanh seek: gộp lệnh, vị trí cuối luôn được áp dụng ─────────────────────────────────


class _FakePlayer:
    def __init__(self) -> None:
        self.seeks: list[float] = []

    def seek(self, sec: float) -> None:
        self.seeks.append(sec)

    def pause(self) -> None:
        pass

    def release(self) -> None:
        pass


def _pump(ms: int) -> None:
    t = QElapsedTimer(); t.start()
    while t.elapsed() < ms:
        QCoreApplication.processEvents()


def test_drag_seek_is_coalesced() -> None:
    QApplication.instance() or QApplication([])
    from subtitles_extractor.presentation.widgets import mpv_video_widget as mvw

    w = mvw.MpvVideoWidget()
    player = _FakePlayer()
    w._player = player
    positions: list[float] = []
    w.position_changed.connect(positions.append)
    for i in range(200):  # kéo nhanh: 200 giá trị liên tiếp
        w.seek(i * 0.5)
    assert player.seeks == [0.0]  # lệnh đầu chạy ngay
    assert len(positions) == 200  # thanh trượt/nhãn vẫn cập nhật từng bước
    _pump(mvw._SEEK_COALESCE_MS * 4)
    assert player.seeks[-1] == 199 * 0.5  # vị trí cuối khi buông tay luôn được áp dụng
    assert len(player.seeks) <= 3  # trước đây: 200 lệnh seek exact đồng bộ
    w.seek(5.0)
    w.seek(6.0)
    w.release_player()
    _pump(mvw._SEEK_COALESCE_MS * 3)
    assert 6.0 not in player.seeks  # không seek vào lõi đã giải phóng
    w.deleteLater()
