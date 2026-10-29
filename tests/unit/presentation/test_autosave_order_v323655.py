"""[v3.23.655] Bug #15: hai lượt tự lưu trang Chỉnh sửa xong ngược thứ tự → bản cũ đè bản mới."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from subtitles_extractor.presentation.view_models import editor_page_view_model as module


class _Repo:
    def __init__(self) -> None:
        self.saved: list[list] = []
        self.lock = threading.Lock()

    def save_events(self, path, events) -> None:
        with self.lock:
            self.saved.append(list(events))


def test_older_snapshot_finishing_last_does_not_overwrite() -> None:
    repo = _Repo()
    gate = module._AutoSaveGate()
    old = module._AutoSaveRunnable(repo, "v.mp4", ["cũ"], gate=gate, seq=gate.next_seq("v.mp4"))
    new = module._AutoSaveRunnable(repo, "v.mp4", ["mới"], gate=gate, seq=gate.next_seq("v.mp4"))
    new.run()       # lượt mới xong trước
    old.run()       # lượt cũ xong sau
    assert repo.saved[-1] == ["mới"]
    assert ["cũ"] not in repo.saved


def test_in_order_both_written() -> None:
    repo = _Repo()
    gate = module._AutoSaveGate()
    first = module._AutoSaveRunnable(repo, "v.mp4", ["1"], gate=gate, seq=gate.next_seq("v.mp4"))
    first.run()
    second = module._AutoSaveRunnable(repo, "v.mp4", ["2"], gate=gate, seq=gate.next_seq("v.mp4"))
    second.run()
    assert repo.saved == [["1"], ["2"]]


def test_other_video_save_not_skipped() -> None:
    repo = _Repo()
    gate = module._AutoSaveGate()
    a = module._AutoSaveRunnable(repo, "a.mp4", ["A"], gate=gate, seq=gate.next_seq("a.mp4"))
    b = module._AutoSaveRunnable(repo, "b.mp4", ["B"], gate=gate, seq=gate.next_seq("b.mp4"))
    b.run()
    a.run()
    assert ["A"] in repo.saved and ["B"] in repo.saved


def test_next_seq_does_not_wait_for_running_write() -> None:
    """[v3.23.655] Rà soát: luồng giao diện xin số thứ tự không được chờ một lượt ghi CSDL đang chạy."""
    gate = module._AutoSaveGate()
    with gate.lock:                      # giả lập lượt ghi đang giữ khoá ghi
        done = threading.Event()
        threading.Thread(target=lambda: (gate.next_seq("v.mp4"), done.set()), daemon=True).start()
        assert done.wait(1.0)
