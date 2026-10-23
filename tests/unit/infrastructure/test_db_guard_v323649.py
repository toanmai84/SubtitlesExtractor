"""[v3.23.649] Tệp SQLite hỏng không còn làm ứng dụng không khởi động được; có sao lưu xoay vòng và tự phục hồi.

Phần lớn ca biên ở đây đến từ lượt rà soát độc lập (mỗi ca tái hiện được lỗi MẤT DỮ LIỆU trên bản nháp đầu).
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.database import db_guard
from subtitles_extractor.infrastructure.database.maintenance import reset_database
from subtitles_extractor.infrastructure.database.sqlite_project_repository import SqliteProjectRepository
from subtitles_extractor.infrastructure.database.sqlite_subtitle_repository import SqliteSubtitleRepository
from subtitles_extractor.infrastructure.database.sqlite_translation_context_store import (
    SqliteTranslationContextStore,
)
from subtitles_extractor.infrastructure.database.sqlite_translation_memory_store import (
    SqliteTranslationMemoryStore,
)
from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
    SqliteTranslationSessionStore,
)
from subtitles_extractor.infrastructure.database.sqlite_video_state_repository import (
    SqliteVideoStateRepository,
)


@pytest.fixture(autouse=True)
def _fresh_guard() -> None:
    db_guard.reset_for_tests()
    yield
    db_guard.wait_for_background()


def _make_db(path: Path, rows: int = 300, value: str = "x", index: bool = False) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE projects (id INTEGER PRIMARY KEY, name TEXT, blob TEXT)")
    if index:
        conn.execute("CREATE INDEX idx_name ON projects(name)")
    conn.executemany("INSERT INTO projects VALUES (?,?,?)", [(i, f"dự án {i}", value * 200) for i in range(rows)])
    conn.commit()
    conn.close()


def _corrupt_middle(path: Path) -> None:
    data = bytearray(path.read_bytes())
    page = 4096
    for offset in range(page * 3, min(len(data), page * 6)):
        data[offset] = 0xFF if offset % 3 else 0x00
    path.write_bytes(bytes(data))


def _count(path: Path, table: str = "projects") -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


class TestHealthyFile:
    def test_missing_or_empty_file_is_left_alone(self, tmp_path: Path) -> None:
        assert db_guard.prepare_database(tmp_path / "none.db") == db_guard.GuardReport()
        (tmp_path / "empty.db").write_bytes(b"")
        assert db_guard.prepare_database(tmp_path / "empty.db").healthy

    def test_backup_is_made_once_per_interval_and_rotated(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path)
        t0 = time.time()
        assert db_guard.prepare_database(path, now=t0).backed_up
        assert (tmp_path / "app_state.db.bak1").exists()
        db_guard.reset_for_tests()
        assert not db_guard.prepare_database(path, now=t0 + 60).backed_up          # chưa tới 12 giờ
        for step in range(1, db_guard.BACKUP_SLOTS + 2):
            db_guard.reset_for_tests()
            old = t0 - step * 2 * 86400
            os.utime(tmp_path / "app_state.db.bak1", (old, old))                        # bản cũ hơn khoảng sao lưu
            assert db_guard.prepare_database(path, now=t0 + step).backed_up
        slots = sorted(p.name for p in tmp_path.glob("app_state.db.bak*"))
        assert slots == [f"app_state.db.bak{i}" for i in range(1, db_guard.BACKUP_SLOTS + 1)]
        assert not list(tmp_path.glob("*.tmp"))

    def test_tiny_database_is_not_backed_up(self, tmp_path: Path) -> None:
        path = tmp_path / "small.db"
        sqlite3.connect(path).execute("CREATE TABLE t (a)").connection.commit()
        assert not db_guard.prepare_database(path).backed_up

    def test_each_file_is_checked_once_per_process(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path)
        assert db_guard.prepare_database(path).backed_up
        assert db_guard.prepare_database(path) == db_guard.GuardReport()

    def test_a_bad_new_backup_never_pushes_out_a_good_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path)
        assert db_guard.prepare_database(path, now=time.time() - 3 * 86400).backed_up
        good = (tmp_path / "app_state.db.bak1").read_bytes()
        db_guard.reset_for_tests()
        real = db_guard._health
        monkeypatch.setattr(db_guard, "_health", lambda p, deep=True: "corrupt" if p.name.endswith(".tmp") else real(p, deep))
        assert not db_guard.prepare_database(path).backed_up
        assert (tmp_path / "app_state.db.bak1").read_bytes() == good and not (tmp_path / "app_state.db.bak2").exists()
        assert not list(tmp_path.glob("*.tmp"))

    def test_backup_counts_the_wal_file_in_the_size(self, tmp_path: Path) -> None:
        path = tmp_path / "small.db"
        sqlite3.connect(path).execute("CREATE TABLE t (a)").connection.commit()
        (tmp_path / "small.db-wal").write_bytes(b"\0" * 40_000)
        assert db_guard._footprint(path) >= 40_000


class TestUriSafePaths:
    """URI ``file:`` ghép thô làm SQLite cắt tên tệp tại ``#``/``?`` và bỏ ``mode=ro`` ⇒ tạo tệp rỗng lạ và kiểm nhầm."""

    @pytest.mark.parametrize("folder", ["Nam#1", "hoi?sao", "tram%41", "có dấu cách & chữ Hán 天骄", "a b"])
    def test_health_backup_and_corruption_are_detected_for_special_paths(self, tmp_path: Path, folder: str) -> None:
        root = tmp_path / folder
        root.mkdir()
        path = root / "app_state.db"
        _make_db(path)
        before = {p.name for p in tmp_path.iterdir()}
        assert db_guard._health(path) == "ok"
        assert db_guard.prepare_database(path).backed_up
        assert (root / "app_state.db.bak1").exists()
        assert {p.name for p in tmp_path.iterdir()} == before                  # không tạo tệp lạ ở thư mục cha
        _corrupt_middle(path)
        db_guard.reset_for_tests()
        assert db_guard._health(path) == "corrupt"
        assert db_guard.prepare_database(path).recovered_from in ("salvage", "merge")
        assert {p.name for p in tmp_path.iterdir()} == before


class TestCorruptFile:
    def test_unreadable_garbage_without_backup_starts_fresh_and_keeps_the_corrupt_copy(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        path.write_bytes(b"day khong phai co so du lieu" * 500)
        report = db_guard.prepare_database(path)
        assert not report.healthy and report.recovered_from == ""
        assert not path.exists()
        assert Path(report.corrupt_copy).read_bytes().startswith(b"day khong phai")
        assert any("bị hỏng" in n for n in db_guard.pop_notices()) and db_guard.pop_notices() == []

    def test_partially_corrupt_file_is_salvaged(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path, rows=2000)
        _corrupt_middle(path)
        report = db_guard.prepare_database(path)
        assert report.recovered_from == "salvage" and path.exists()
        saved = _count(path)
        assert 0 < saved < 2000
        assert Path(report.corrupt_copy).exists()
        assert sqlite3.connect(path).execute("PRAGMA quick_check").fetchone()[0] == "ok"

    def test_garbage_falls_back_to_the_backup(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path, rows=2000)
        assert db_guard.prepare_database(path).backed_up
        db_guard.reset_for_tests()
        path.write_bytes(b"\x00" * 100 + path.read_bytes()[100:200] + b"\xff" * 20000)
        report = db_guard.prepare_database(path)
        assert report.recovered_from == "backup" and _count(path) == 2000

    def test_complete_salvage_beats_an_older_backup_with_more_rows(self, tmp_path: Path) -> None:
        """Chỉ hỏng trang INDEX, dữ liệu bảng còn nguyên: bản cứu (mới) thắng bản sao lưu cũ (còn dự án đã xoá)."""
        path = tmp_path / "app_state.db"
        _make_db(path, rows=500, value="cu", index=True)
        assert db_guard.prepare_database(path).backed_up                   # .bak1: 500 dòng, giá trị CŨ
        db_guard.reset_for_tests()
        conn = sqlite3.connect(path)
        conn.execute("DELETE FROM projects WHERE id >= 400")               # người dùng đã xoá 100 dự án
        conn.execute("UPDATE projects SET blob = 'moi'")
        conn.commit()
        root = conn.execute("SELECT rootpage FROM sqlite_master WHERE name='idx_name'").fetchone()[0]
        conn.close()
        data = bytearray(path.read_bytes())
        page = int.from_bytes(data[16:18], "big") or 65536
        data[(root - 1) * page : root * page] = b"\xff" * page            # phá trang gốc của index
        path.write_bytes(bytes(data))
        assert db_guard._health(path) == "corrupt"
        report = db_guard.prepare_database(path)
        assert report.recovered_from == "salvage"
        assert _count(path) == 400                                         # KHÔNG sống lại 100 dự án đã xoá
        assert sqlite3.connect(path).execute("SELECT blob FROM projects LIMIT 1").fetchone()[0] == "moi"

    def test_incomplete_salvage_is_merged_into_the_backup_newest_rows_win(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path, rows=2000, value="cu")
        assert db_guard.prepare_database(path).backed_up
        db_guard.reset_for_tests()
        conn = sqlite3.connect(path)
        conn.execute("UPDATE projects SET blob = 'moi'")
        conn.commit()
        conn.close()
        _corrupt_middle(path)
        report = db_guard.prepare_database(path)
        assert report.recovered_from == "merge"
        assert _count(path) == 2000                                        # bản sao lưu bù phần không đọc được
        conn = sqlite3.connect(path)
        try:
            kinds = {row[0]: row[1] for row in conn.execute("SELECT substr(blob,1,2), count(*) FROM projects GROUP BY 1")}
        finally:
            conn.close()
        assert kinds.get("mo", 0) > 0 and kinds.get("cu", 0) > 0           # dòng cứu được là MỚI, phần còn lại lấy từ sao lưu

    def test_only_the_latest_corrupt_copies_are_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        total = db_guard.KEEP_CORRUPT_COPIES + 2
        for index in range(total):
            db_guard.reset_for_tests()
            path.write_bytes(b"hong" * 3000)
            db_guard.prepare_database(path, now=1_700_000_000 + index * 10)
        copies = sorted(p.name for p in tmp_path.glob("app_state.db.corrupt-*") if not p.name.endswith(("-wal", "-shm")))
        assert len(copies) == db_guard.KEEP_CORRUPT_COPIES
        assert copies[-1].endswith(time.strftime("%Y%m%d-%H%M%S", time.localtime(1_700_000_000 + (total - 1) * 10)))

    def test_the_just_moved_aside_copy_is_never_pruned_even_with_an_old_mtime(self, tmp_path: Path) -> None:
        """Cắt tỉa theo mtime từng xoá chính bản vừa cất (tệp phục hồi bằng copy2 giữ mtime cũ)."""
        for index in range(db_guard.KEEP_CORRUPT_COPIES):
            (tmp_path / f"app_state.db.corrupt-20300101-00000{index}").write_bytes(b"cu")
        path = tmp_path / "app_state.db"
        _make_db(path, rows=500)
        ancient = time.time() - 400 * 86400
        os.utime(path, (ancient, ancient))
        path.write_bytes(b"hong" * 3000)
        os.utime(path, (ancient, ancient))
        report = db_guard.prepare_database(path, now=time.time())
        assert Path(report.corrupt_copy).exists()

    def test_guard_never_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path)
        monkeypatch.setattr(db_guard, "_health", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bất ngờ")))
        assert db_guard.prepare_database(path) == db_guard.GuardReport()


class TestRecoveryFailsSafe:
    def test_write_failure_during_salvage_falls_back_to_the_backup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path, rows=1000)
        assert db_guard.prepare_database(path).backed_up
        db_guard.reset_for_tests()
        _corrupt_middle(path)

        def disk_full(*_a: object, **_k: object) -> None:
            raise sqlite3.OperationalError("database or disk is full")

        monkeypatch.setattr(db_guard, "_flush", disk_full)
        report = db_guard.prepare_database(path)
        assert report.recovered_from == "backup" and _count(path) == 1000
        assert Path(report.corrupt_copy).exists()
        assert not list(tmp_path.glob("*.salvage.tmp")) and any("bị hỏng" in n for n in db_guard.pop_notices())

    def test_every_recovery_step_failing_still_keeps_the_corrupt_file_and_reports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path, rows=1000)
        _corrupt_middle(path)
        monkeypatch.setattr(db_guard, "_flush", lambda *_a: (_ for _ in ()).throw(OSError("I/O")))
        report = db_guard.prepare_database(path)
        assert report.recovered_from == "" and not path.exists()
        assert Path(report.corrupt_copy).exists() and not list(tmp_path.glob("*.salvage.tmp"))
        assert db_guard.pop_notices()

    def test_a_wal_that_cannot_be_moved_aside_aborts_recovery_and_changes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "app_state.db"
        path.write_bytes(b"hong" * 3000)
        wal = tmp_path / "app_state.db-wal"
        wal.write_bytes(b"wal cu")
        real = os.replace

        def deny_wal(src: object, dst: object) -> None:
            if str(src).endswith("-wal"):
                raise PermissionError("antivirus đang giữ tệp")
            real(src, dst)

        monkeypatch.setattr(db_guard.os, "replace", deny_wal)
        assert db_guard.prepare_database(path) == db_guard.GuardReport()
        assert path.read_bytes().startswith(b"hong") and wal.exists()          # không đổi gì, không có bản hỏng nửa vời
        assert not list(tmp_path.glob("*.corrupt-*"))

    def test_foreign_wal_is_never_replayed_onto_the_restored_database(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path, rows=800)
        assert db_guard.prepare_database(path).backed_up
        db_guard.reset_for_tests()
        path.write_bytes(b"hong" * 3000)
        (tmp_path / "app_state.db-wal").write_bytes(b"\x37\x7f\x06\x82" + b"\0" * 4000)
        (tmp_path / "app_state.db-shm").write_bytes(b"\0" * 32768)
        report = db_guard.prepare_database(path)
        assert report.recovered_from == "backup"
        assert not (tmp_path / "app_state.db-wal").exists() and not (tmp_path / "app_state.db-shm").exists()
        assert _count(path) == 800


class TestNeverMistakeALockedFileForACorruptOne:
    """Lỗi tạm (khoá/không mở được) mà bị coi là hỏng sẽ đem dữ liệu người dùng đi cất và dựng DB rỗng."""

    def test_database_locked_by_another_connection_is_left_untouched(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path)
        holder = sqlite3.connect(path, isolation_level=None)
        holder.execute("BEGIN EXCLUSIVE")
        try:
            report = db_guard.prepare_database(path)
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        assert path.exists() and not list(tmp_path.glob("*.corrupt-*")) and report.recovered_from == ""
        assert _count(path) == 300

    def test_transient_error_is_unknown_not_corrupt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path)

        def locked(*_a: object, **_k: object) -> None:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(db_guard.sqlite3, "connect", locked)
        assert db_guard._health(path) == "unknown"
        monkeypatch.undo()
        assert not list(tmp_path.glob("*.corrupt-*"))

    def test_real_corruption_is_still_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        path.write_bytes(b"khong phai sqlite" * 400)
        assert db_guard._health(path) == "corrupt"
        _make_db(tmp_path / "b.db", rows=2000)
        _corrupt_middle(tmp_path / "b.db")
        assert db_guard._health(tmp_path / "b.db") == "corrupt"


class TestConcurrency:
    def test_concurrent_stores_in_one_process_wait_for_the_recovery(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        path.write_bytes(b"hong" * 3000)
        seen: list[bool] = []

        def open_store() -> None:
            db_guard.prepare_database(path)
            seen.append(not path.exists() or db_guard._health(path) in ("ok", "unknown"))

        threads = [threading.Thread(target=open_store) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert seen == [True] * 6
        assert len(list(tmp_path.glob("app_state.db.corrupt-*"))) == 1

    def test_another_process_holding_the_guard_lock_blocks_recovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "app_state.db"
        path.write_bytes(b"hong" * 3000)
        (tmp_path / "app_state.db.guardlock").write_text("99999")
        monkeypatch.setattr(db_guard, "_LOCK_WAIT_S", 0.3)
        assert db_guard.prepare_database(path) == db_guard.GuardReport()
        assert path.read_bytes().startswith(b"hong") and not list(tmp_path.glob("*.corrupt-*"))

    def test_a_stale_guard_lock_is_taken_over_and_released(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        path.write_bytes(b"hong" * 3000)
        lock = tmp_path / "app_state.db.guardlock"
        lock.write_text("99999")
        old = time.time() - db_guard._LOCK_STALE_S - 10
        os.utime(lock, (old, old))
        report = db_guard.prepare_database(path)
        assert not report.healthy and not lock.exists()

    def test_second_process_sees_the_recovered_file(self, tmp_path: Path) -> None:
        """Hai bản app: bản B khởi động sau khi A đã phục hồi thì thấy tệp lành, không tạo DB rỗng đè lên."""
        path = tmp_path / "app_state.db"
        _make_db(path, rows=900)
        assert db_guard.prepare_database(path).backed_up
        db_guard.reset_for_tests()
        path.write_bytes(b"hong" * 3000)
        assert db_guard.prepare_database(path).recovered_from == "backup"
        db_guard.reset_for_tests()                                 # "tiến trình B"
        assert db_guard.prepare_database(path).healthy and _count(path) == 900


class TestLargeFiles:
    def test_large_file_gets_a_light_check_at_startup_and_a_deep_one_in_the_background(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path, rows=2000)
        monkeypatch.setattr(db_guard, "DEEP_CHECK_MAX_BYTES", 1000)
        _corrupt_middle(path)
        assert db_guard._health(path, deep=False) == "ok"          # kiểm nhẹ chỉ đọc lược đồ: chưa thấy lỗi
        report = db_guard.prepare_database(path, background=True)
        assert report.recovered_from == ""                         # khởi động không bị chặn bởi kiểm sâu
        db_guard.wait_for_background()
        assert (tmp_path / "app_state.db.suspect").exists()       # kiểm sâu nền đặt cờ
        db_guard.reset_for_tests()                                 # lần khởi động sau
        report = db_guard.prepare_database(path, background=True)
        assert report.recovered_from in ("salvage", "merge")
        assert not (tmp_path / "app_state.db.suspect").exists()

    def test_background_backup_is_made_off_the_main_path(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path)
        db_guard.prepare_database(path, background=True)
        db_guard.wait_for_background()
        assert (tmp_path / "app_state.db.bak1").exists()


class TestResetDatabase:
    def test_successful_reset_removes_backups_and_corrupt_copies(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        _make_db(path)
        db_guard.prepare_database(path)
        (tmp_path / "app_state.db.corrupt-20260101-000000").write_bytes(b"x")
        (tmp_path / "khac.db.bak1").write_bytes("giữ nguyên".encode())
        assert (tmp_path / "app_state.db.bak1").exists()
        assert reset_database(path) == ["projects"]
        assert not list(tmp_path.glob("app_state.db.bak*")) and not list(tmp_path.glob("app_state.db.corrupt-*"))
        assert (tmp_path / "khac.db.bak1").exists()

    def test_failed_reset_keeps_every_backup(self, tmp_path: Path) -> None:
        """Reset thất bại (DB đang bị khoá/hỏng) mà xoá luôn sao lưu thì mất đường cứu duy nhất."""
        path = tmp_path / "app_state.db"
        _make_db(path)
        db_guard.prepare_database(path)
        (tmp_path / "app_state.db.corrupt-20260101-000000").write_bytes(b"x")
        holder = sqlite3.connect(path, isolation_level=None)
        holder.execute("BEGIN EXCLUSIVE")
        try:
            with pytest.raises(sqlite3.OperationalError):
                reset_database(path)
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        assert (tmp_path / "app_state.db.bak1").exists() and (tmp_path / "app_state.db.corrupt-20260101-000000").exists()
        assert _count(path) == 300


class TestEveryStoreSurvivesCorruption:
    @pytest.mark.parametrize("factory", [
        SqliteProjectRepository, SqliteSubtitleRepository, SqliteTranslationSessionStore,
        SqliteVideoStateRepository, SqliteTranslationMemoryStore, SqliteTranslationContextStore,
    ])
    def test_store_opens_after_garbage_file(self, tmp_path: Path, factory: type) -> None:
        path = tmp_path / f"{factory.__name__}.db"
        path.write_bytes(b"khong phai sqlite" * 400)
        store = factory(path)                       # trước bản này: DatabaseError → ứng dụng không khởi động được
        assert path.exists()
        with contextlib.suppress(Exception):
            store.close()
        assert list(tmp_path.glob(f"{path.name}.corrupt-*"))

    def test_project_repository_keeps_working_after_recovery(self, tmp_path: Path) -> None:
        path = tmp_path / "app_state.db"
        path.write_bytes(b"hong" * 3000)
        repo = SqliteProjectRepository(path)
        assert repo.list_summaries() == []
