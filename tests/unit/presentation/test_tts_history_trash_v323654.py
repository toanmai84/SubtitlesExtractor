"""[v3.23.654] Sửa tay TTS lưu ở tệp có lịch sử (thay registry); audio dòng đã xoá vào thùng rác và được dùng lại."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.dubbing_line_edits import (
    LineEdits,
    decode_line_edits,
    encode_line_edits,
)
from subtitles_extractor.infrastructure.tts.line_edits_store import LineEditsFileStore, describe_edits
from subtitles_extractor.infrastructure.tts.segment_store import TRASH_DIR_NAME, segment_path
from tests.unit.presentation.test_dubbing_add_delete_lines_v323625 import _view_model
from tests.unit.presentation.test_dubbing_line_edits_v323598 import (  # noqa: F401
    HUSBAND,
    LINES,
    P495,
    _events,
    app,
    page,
)


def test_store_keeps_history_and_survives_reload(tmp_path: Path) -> None:
    now = [1000.0]
    store = LineEditsFileStore(tmp_path, clock=lambda: now[0])
    first = encode_line_edits(LineEdits(speakers={1: "A"}))
    second = encode_line_edits(LineEdits(speakers={1: "A", 2: "B"}))
    assert store.save("fp1", first)
    now[0] += 5
    assert store.save("fp1", second)                    # lần sửa đầu của phiên: cất bản trước
    now[0] += 5
    between = encode_line_edits(LineEdits(speakers={1: "A", 2: "C"}))
    assert store.save("fp1", between)                   # < 30 s, cùng cấu trúc: không cất thêm
    assert [entry.edits for entry in store.history("fp1")] == [first]
    now[0] += 5
    third = encode_line_edits(LineEdits(speakers={1: "A"}, lines=()))
    assert store.save("fp1", third)                     # danh sách dòng đổi: cất ngay dù < 30 s
    again = LineEditsFileStore(tmp_path)
    assert again.load("fp1") == third
    assert [entry.edits for entry in again.history("fp1")] == [between, first]
    assert "2 người nói" in describe_edits(second)
    second = third
    assert store.save("fp1", None) and again.load("fp1") is None
    assert again.history("fp1")[0].edits == third        # xoá hết sửa tay vẫn còn lịch sử


def test_store_rejects_bad_names_and_imports_legacy(tmp_path: Path) -> None:
    store = LineEditsFileStore(tmp_path)
    assert not store.save("../x", {"speakers": {"1": "A"}})
    assert store.import_legacy({"abc": {"speakers": {"1": "A"}}, "../bad": {"x": 1}}) == (1, 0)
    assert store.load("abc") == {"speakers": {"1": "A"}}


def test_page_migrates_registry_and_saves_to_file(page) -> None:  # noqa: F811
    controller = page._line_edits
    store = controller._edits_store
    assert store is not None
    page._view_model.set_source_events(_events())
    page._view_model.set_speaker_override([P495], HUSBAND)
    controller.flush_pending_save()                           # ghi gom sau ~0,4 s; ép ghi ngay để kiểm
    fingerprint = page._view_model.subtitle_fingerprint
    saved = decode_line_edits(store.load(fingerprint))
    assert saved.speakers == {P495: HUSBAND}
    assert page._settings.value("dubbing_line_edits", "", type=str) == ""   # không còn ghi registry


def test_history_dialog_reapplies_old_version(page, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    from subtitles_extractor.presentation.pages import tts_line_edits

    controller = page._line_edits
    page._view_model.set_source_events(_events())
    fingerprint = page._view_model.subtitle_fingerprint
    store = controller._edits_store
    store.save(fingerprint, encode_line_edits(LineEdits(speakers={P495: HUSBAND}, lines=None)))
    monkeypatch.setattr(store, "_clock", lambda: 10_000_000.0)
    page._view_model.delete_lines([0, 1, 2])                    # sửa mới → bản cũ vào lịch sử
    controller.flush_pending_save()
    assert store.history(fingerprint)
    monkeypatch.setattr(tts_line_edits, "ask_history_version", lambda *_a: 0)
    assert controller.show_history()
    assert len(page._view_model.source_events) == len(LINES)
    assert page._view_model.speaker_overrides == {P495: HUSBAND}
    assert page._view_model.undo()
    assert len(page._view_model.source_events) == len(LINES) - 3


def test_deleted_line_audio_goes_to_trash_and_comes_back(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    vm, _started = _view_model(tmp_path, monkeypatch)
    folder = Path(vm._last_results[0].segment_audio_path).parent
    audio_3 = Path(vm._last_results[3].segment_audio_path).read_bytes()
    vm.delete_lines([3])
    assert not segment_path(folder, len(LINES) - 1).exists()
    assert list((folder / TRASH_DIR_NAME).glob("*.wav"))
    assert vm.restore_original_lines([3]) == 1
    restored = {int(r.event_index): r for r in vm._last_results}[3]
    assert not restored.was_skipped                          # dùng lại audio, không phải TTS lại
    assert Path(restored.segment_audio_path).read_bytes() == audio_3
    assert 3 not in vm.edited_positions


def test_clearing_edits_always_keeps_previous_version(tmp_path: Path) -> None:
    """Rà soát độc lập: xoá hết sửa tay trong vòng 30 s từng làm mất bản trước khỏi lịch sử."""
    now = [0.0]
    store = LineEditsFileStore(tmp_path, clock=lambda: now[0])
    a = encode_line_edits(LineEdits(speakers={1: "A"}))
    b = encode_line_edits(LineEdits(speakers={1: "A", 2: "B", 3: "C"}))
    store.save("fp", a)
    store.save("fp", None)
    now[0] += 1
    store.save("fp", b)
    now[0] += 1
    store.save("fp", None)                             # "bỏ sửa tay" ngay sau đó
    assert b in [entry.edits for entry in store.history("fp")]


def test_failed_migration_keeps_registry_copy(page, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    """Rà soát độc lập: chép sang tệp lỗi (thư mục bị khoá/đầy đĩa) mà vẫn xoá khoá registry → mất mọi sửa tay."""
    import json

    controller = page._line_edits
    page._settings.setValue("dubbing_line_edits", json.dumps({"abcd": {"speakers": {"1": "A"}}}))
    monkeypatch.setattr(controller._edits_store, "import_legacy", lambda store: (0, 1))
    controller._import_legacy_store()
    assert page._settings.value("dubbing_line_edits", "", type=str)          # còn nguyên để thử lại lần sau


def test_trash_matches_original_line_not_just_text(tmp_path: Path) -> None:
    """Rà soát độc lập: hai dòng cùng lời "好" của hai vai — khôi phục phải lấy đúng audio của dòng gốc."""
    from subtitles_extractor.domain.ports.subtitle_tts_port import TTSSegmentResult
    from subtitles_extractor.infrastructure.tts.segment_store import clear_trash, move_to_trash, recover_from_trash

    for origin, payload in ((4, b"giong-A"), (9, b"giong-B")):
        path = segment_path(tmp_path, origin)
        path.write_bytes(payload)
        move_to_trash(TTSSegmentResult(event_index=origin, start_sec=0, end_sec=1, text="好",
                                       segment_audio_path=str(path)), origin)
    found = recover_from_trash(tmp_path, "好", 2, 4)
    assert found is not None and found[0].read_bytes() == b"giong-A"
    assert recover_from_trash(tmp_path, "好", 3, 7) is None            # dòng gốc khác: không lấy nhầm
    clear_trash(tmp_path)                                                # lượt TTS mới: bỏ audio cũ
    assert recover_from_trash(tmp_path, "好", 3, 9) is None
