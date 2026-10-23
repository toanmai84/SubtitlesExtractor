"""[v3.23.649] Bảo vệ tệp SQLite của ứng dụng: kiểm tra sức khoẻ, sao lưu xoay vòng, tự phục hồi khi hỏng.

Trước bản này, MỌI kho SQLite (dự án, phụ đề, phiên dịch, bộ nhớ dịch…) mở tệp ``app_state.db`` mà không kiểm tra gì: tệp
hỏng (mất điện giữa lúc ghi, ổ đĩa lỗi, bị chương trình khác ghi đè) làm ``sqlite3.DatabaseError: database disk image is
malformed`` ném thẳng ra ở mọi kho — người dùng có 79 dự án trong Thư viện không mở nổi ứng dụng, cũng không có bản sao lưu nào.

Nguyên tắc: **không bao giờ làm mất dữ liệu có thể cứu được** — tệp hỏng luôn được giữ nguyên (đổi tên, không xoá), mọi bước
phục hồi đều dựng ra tệp TẠM rồi mới thay, và mọi trạng thái "không chắc" (khoá, không mở được, lỗi I/O tạm) được coi là lành.

Quy trình (:func:`prepare_database`, gọi MỘT lần cho mỗi tệp trước khi kho đầu tiên kết nối, dưới khoá liên tiến trình):

1. Tệp chưa có / rỗng → không làm gì.
2. Kiểm tra: tệp nhỏ (≤ ``DEEP_CHECK_MAX_BYTES``) chạy ``PRAGMA quick_check``; tệp lớn chỉ kiểm nhẹ (đọc lược đồ) lúc khởi động
   — ``quick_check`` đầy đủ chạy NỀN và, nếu thấy hỏng, đặt cờ ``.suspect`` để lần khởi động sau kiểm kỹ rồi phục hồi.
3. Lành → sao lưu xoay vòng bằng API ``backup`` (an toàn với WAL), nền, tối đa mỗi ``BACKUP_INTERVAL_S``, giữ ``BACKUP_SLOTS``
   bản, bỏ qua nếu ổ đĩa không đủ chỗ.
4. Hỏng → đổi tên tệp hỏng (cùng ``-wal``/``-shm``) thành ``.corrupt-<thời điểm>``, rồi dựng lại: cứu các dòng đọc được; nếu đọc
   hết không lỗi thì dùng bản cứu; nếu còn thiếu thì GỘP với bản sao lưu lành mới nhất (dòng cứu được thắng); không có gì thì để
   ứng dụng tạo tệp mới. Thông báo được ghi lại để giao diện hiện cho người dùng.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

BACKUP_SLOTS = 2
BACKUP_INTERVAL_S = 12 * 3600.0
KEEP_CORRUPT_COPIES = 3
DEEP_CHECK_MAX_BYTES = 256 * 1024 * 1024
_MIN_BACKUP_BYTES = 16 * 1024
_LOCK_WAIT_S = 20.0
_LOCK_STALE_S = 180.0

_lock = threading.Lock()
#: Giữ suốt một lần kiểm tra/phục hồi: kho thứ hai (luồng khác) phải ĐỢI xong rồi mới kết nối, không mở tệp đang bị đổi tên.
_operation_lock = threading.Lock()
_checked: set[str] = set()
_notices: list[str] = []
_background: list[threading.Thread] = []

#: Thông điệp của SQLite chắc chắn là TỆP HỎNG. Mọi lỗi khác (đang bị khoá, không mở được, lỗi I/O tạm…) là "không rõ" và
#: KHÔNG được coi là hỏng — nhầm một tệp lành đang bị tiến trình khác khoá thành hỏng sẽ đem dữ liệu người dùng đi cất.
_CORRUPTION_MARKERS = ("malformed", "not a database", "corrupt", "disk image")


@dataclass(frozen=True, slots=True)
class GuardReport:
    """Kết quả :func:`prepare_database`."""

    healthy: bool = True
    backed_up: bool = False
    recovered_from: str = ""          # "salvage" | "merge" | "backup" | "" (không phục hồi được → tệp mới)
    corrupt_copy: str = ""


def pop_notices() -> list[str]:
    """Lấy (và xoá) các thông báo phục hồi chờ hiện cho người dùng."""
    with _lock:
        notices, _notices[:] = list(_notices), []
    return notices


def reset_for_tests() -> None:
    """Xoá bộ nhớ "đã kiểm" để test chạy lại được (đợi tác vụ nền đang chạy xong trước)."""
    wait_for_background()
    with _lock:
        _checked.clear()
        _notices.clear()


def wait_for_background(timeout: float = 60.0) -> None:
    """Đợi các tác vụ nền (sao lưu, kiểm sâu) xong — dùng cho test và lúc thoát."""
    deadline = time.monotonic() + timeout
    with _lock:
        threads = list(_background)
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    with _lock:
        _background[:] = [t for t in _background if t.is_alive()]


def _uri(path: Path, mode: str = "ro") -> str:
    """URI ``file:`` đúng chuẩn (mã hoá ``#``, ``?``, ``%``, dấu cách, Unicode) — ghép chuỗi thô làm SQLite cắt tên tệp tại ``#``/``?``."""
    return f"{Path(path).absolute().as_uri()}?mode={mode}"


def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(_uri(path), uri=True, timeout=2.0)


def _classify(exc: sqlite3.Error) -> str:
    text = str(exc).lower()
    return "corrupt" if any(marker in text for marker in _CORRUPTION_MARKERS) else "unknown"


def _health(path: Path, deep: bool = True) -> str:
    """``"ok"`` | ``"corrupt"`` | ``"unknown"`` (không kết luận được: khoá, không mở được, lỗi tạm…).

    ``deep=False`` chỉ đọc lược đồ (``sqlite_master``) — rẻ, bắt được tệp rác / lược đồ hỏng.
    """
    try:
        conn = _connect_ro(path)
    except sqlite3.Error:
        return "unknown"
    try:
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        if not deep:
            return "ok"
        rows = conn.execute("PRAGMA quick_check(1)").fetchall()
        if rows and rows[0][0] == "ok":
            return "ok"
        return "corrupt" if rows else "unknown"
    except sqlite3.Error as exc:
        return _classify(exc)
    finally:
        conn.close()


def _is_healthy(path: Path) -> bool:
    """Tệp lành rõ ràng (``quick_check`` = ok)."""
    return _health(path) == "ok"


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        name for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]


def _row_count(path: Path) -> int:
    """Tổng số dòng trong mọi bảng người dùng (0 nếu không đọc được)."""
    try:
        conn = _connect_ro(path)
    except sqlite3.Error:
        return 0
    try:
        total = 0
        for name in _user_tables(conn):
            with contextlib.suppress(sqlite3.Error):
                total += int(conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0])
        return total
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


# ── Khoá liên tiến trình ──────────────────────────────────────────────────────


@contextlib.contextmanager
def _cross_process_lock(path: Path) -> Iterator[bool]:
    """Khoá tệp ``<db>.guardlock`` (tạo độc quyền): hai bản ứng dụng khởi động cùng lúc không đổi tên/ghi đè tệp của nhau.

    Yield ``True`` nếu giữ được khoá; ``False`` nếu hết ``_LOCK_WAIT_S`` mà tiến trình kia chưa xong (khi đó KHÔNG phục hồi,
    chỉ để nguyên như trước). Khoá cũ hơn ``_LOCK_STALE_S`` (tiến trình sập) bị chiếm lại.
    """
    lock_path = path.with_name(path.name + ".guardlock")
    deadline = time.monotonic() + _LOCK_WAIT_S
    held = False
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
            held = True
            break
        except FileExistsError:
            with contextlib.suppress(OSError):
                if time.time() - lock_path.stat().st_mtime > _LOCK_STALE_S:
                    lock_path.unlink(missing_ok=True)
                    continue
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        except OSError:
            break                       # thư mục chỉ đọc… — không khoá được: để nguyên, không phục hồi
    try:
        yield held
    finally:
        if held:
            with contextlib.suppress(OSError):
                lock_path.unlink(missing_ok=True)


# ── Sao lưu ───────────────────────────────────────────────────────────────────


def _backup_path(path: Path, slot: int) -> Path:
    return path.with_name(f"{path.name}.bak{slot}")


def _footprint(path: Path) -> int:
    """Kích thước tệp tính cả ``-wal`` (dữ liệu sau sập có thể nằm toàn bộ ở đó)."""
    total = 0
    for suffix in ("", "-wal"):
        with contextlib.suppress(OSError):
            total += path.with_name(path.name + suffix).stat().st_size
    return total


def _rotate_backup(path: Path, now: float) -> bool:
    """Sao lưu nếu bản mới nhất đã cũ hơn ``BACKUP_INTERVAL_S``; xoay vòng ``.bak1`` → ``.bak2`` …"""
    newest = _backup_path(path, 1)
    try:
        size = _footprint(path)
        if size < _MIN_BACKUP_BYTES:
            return False
        if newest.exists() and now - newest.stat().st_mtime < BACKUP_INTERVAL_S:
            return False
        if shutil.disk_usage(path.parent).free < 2 * size:
            logger.info("db_guard: không đủ chỗ trống để sao lưu '{}' — bỏ qua lần này.", path.name)
            return False
    except OSError:
        return False
    temp = path.with_name(f"{path.name}.bak.{os.getpid()}.tmp")
    try:
        source = _connect_ro(path)
        try:
            target = sqlite3.connect(str(temp))
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        # Chỉ xoay vòng khi bản mới LÀNH — không bao giờ để một bản xấu đẩy bản tốt ra khỏi vòng.
        if _health(temp) != "ok":
            raise sqlite3.DatabaseError("bản sao lưu mới không qua kiểm tra")
        for slot in range(BACKUP_SLOTS, 1, -1):
            older = _backup_path(path, slot - 1)
            if older.exists():
                os.replace(older, _backup_path(path, slot))
        os.replace(temp, newest)
        return True
    except (sqlite3.Error, OSError) as exc:
        logger.warning("Không sao lưu được cơ sở dữ liệu {}: {}", path.name, exc)
        return False
    finally:
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            with contextlib.suppress(OSError):
                temp.with_name(temp.name + suffix).unlink(missing_ok=True)


# ── Phục hồi ──────────────────────────────────────────────────────────────────


def _columns(conn: sqlite3.Connection, table: str) -> tuple[list[str], bool]:
    """(cột thường — bỏ cột sinh/ẩn, có khoá chính hoặc WITHOUT ROWID không)."""
    info = conn.execute(f'PRAGMA table_xinfo("{table}")').fetchall()
    names = [row[1] for row in info if row[6] == 0]
    has_key = any(row[5] > 0 for row in info)
    return names, has_key


def _flush(target: sqlite3.Connection, insert: str, buffer: list[tuple]) -> None:
    target.executemany(insert, buffer)
    buffer.clear()


def _copy_rows(
    source: sqlite3.Connection, target: sqlite3.Connection, table: str, columns: list[str], order: str,
) -> bool:
    """Chép từng dòng của ``table`` (theo ``order``) tới khi gặp lỗi ĐỌC; giữ phần đã chép.

    Returns:
        ``True`` nếu đọc hết bảng không lỗi. Lỗi GHI (đầy đĩa…) KHÔNG được nuốt: ném ra để cả lần cứu bị bỏ.
    """
    names = ",".join(f'"{c}"' for c in columns)
    insert = f'INSERT OR IGNORE INTO "{table}" ({names}) VALUES ({",".join("?" * len(columns))})'
    buffer: list[tuple] = []
    complete = True
    try:
        rows = iter(source.execute(f'SELECT {names} FROM "{table}"{order}'))
    except sqlite3.DatabaseError as exc:
        logger.warning("Cứu dữ liệu bảng '{}': không đọc được ({}).", table, exc)
        return False
    while True:
        try:
            row = next(rows)
        except StopIteration:
            break
        except sqlite3.DatabaseError as exc:                    # chỉ lỗi ĐỌC nằm trong khối này
            complete = False
            logger.warning("Cứu dữ liệu bảng '{}' dừng giữa chừng: {}", table, exc)
            break
        buffer.append(row)
        if len(buffer) >= 200:
            _flush(target, insert, buffer)                      # lỗi GHI lan ra ngoài
    if buffer:
        _flush(target, insert, buffer)
    return complete


def _salvage(corrupt: Path, destination: Path) -> tuple[int, bool]:
    """Chép các dòng còn đọc được từ ``corrupt`` sang tệp mới ``destination``.

    Returns:
        ``(số dòng cứu được — đếm lại từ tệp đích, đọc hết mọi bảng không lỗi)``. Ném ``sqlite3.Error``/``OSError`` khi
        GHI thất bại (người gọi bỏ bản cứu và thử đường khác).
    """
    try:
        source = _connect_ro(corrupt)
    except sqlite3.Error:
        return 0, False
    try:
        try:
            schema = source.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        except sqlite3.Error:
            return 0, False
        target = sqlite3.connect(str(destination))
        complete = True
        try:
            for kind, _name, sql in schema:
                if kind == "table":
                    with contextlib.suppress(sqlite3.Error):
                        target.execute(sql)
            for kind, name, _sql in schema:
                if kind != "table":
                    continue
                try:
                    columns, has_key = _columns(source, name)
                except sqlite3.Error:
                    complete = False
                    continue
                if not columns:
                    continue
                if not _copy_rows(source, target, name, columns, ""):
                    complete = False
                    # Quét ngược lấy phần đuôi — chỉ an toàn khi bảng có khoá (INSERT OR IGNORE loại dòng trùng).
                    if has_key:
                        _copy_rows(source, target, name, columns, " ORDER BY rowid DESC")
            for kind, _name, sql in schema:
                if kind in ("index", "trigger", "view"):
                    with contextlib.suppress(sqlite3.Error):
                        target.execute(sql)
            target.commit()
            saved = sum(
                int(target.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]) for t in _user_tables(target)
            )
        finally:
            target.close()
    finally:
        source.close()
    return saved, complete


def _merge_into_backup(backup: Path, salvaged: Path, destination: Path) -> bool:
    """``destination`` = bản sao lưu + các dòng cứu được (dòng cứu được THẮNG vì mới hơn). ``False`` nếu không gộp được."""
    try:
        shutil.copy2(backup, destination)
        conn = sqlite3.connect(str(destination))
        try:
            conn.execute("ATTACH DATABASE ? AS salvaged", (str(salvaged),))
            for table in _user_tables(conn):
                mine, _ = _columns(conn, table)
                try:
                    theirs = {row[1] for row in conn.execute(f'PRAGMA salvaged.table_xinfo("{table}")') if row[6] == 0}
                except sqlite3.Error:
                    continue
                shared = [c for c in mine if c in theirs]
                if not shared:
                    continue
                names = ",".join(f'"{c}"' for c in shared)
                conn.execute(f'INSERT OR REPLACE INTO main."{table}" ({names}) SELECT {names} FROM salvaged."{table}"')
            conn.commit()
        finally:
            conn.close()
        return True
    except (sqlite3.Error, OSError) as exc:
        logger.warning("Gộp bản cứu với bản sao lưu thất bại: {}", exc)
        return False


def _move_aside(path: Path, stamp: str) -> Path:
    """Đổi tên tệp hỏng (kèm ``-wal``/``-shm``) thành ``.corrupt-<stamp>`` — hoàn tác nếu có bước thất bại."""
    corrupt = path.with_name(f"{path.name}.corrupt-{stamp}")
    done: list[tuple[Path, Path]] = []
    try:
        for suffix in ("-wal", "-shm", ""):
            source = path.with_name(path.name + suffix)
            if suffix and not source.exists():
                continue
            target = corrupt.with_name(corrupt.name + suffix)
            os.replace(source, target)
            done.append((source, target))
    except OSError:
        for source, target in reversed(done):
            with contextlib.suppress(OSError):
                os.replace(target, source)
        raise
    _prune_corrupt_copies(path, keep=corrupt)
    return corrupt


def _prune_corrupt_copies(path: Path, keep: Path) -> None:
    """Giữ ``KEEP_CORRUPT_COPIES`` bản hỏng MỚI NHẤT theo tên (tên chứa thời điểm); không bao giờ xoá ``keep``."""
    copies = sorted(
        (p for p in path.parent.glob(f"{path.name}.corrupt-*") if not p.name.endswith(("-wal", "-shm"))),
        key=lambda p: p.name, reverse=True,
    )
    for old in copies[KEEP_CORRUPT_COPIES:]:
        if old == keep:
            continue
        for victim in (old, old.with_name(old.name + "-wal"), old.with_name(old.name + "-shm")):
            with contextlib.suppress(OSError):
                victim.unlink(missing_ok=True)


def _clear_sidecars(path: Path) -> None:
    """Xoá ``-wal``/``-shm`` còn sót cạnh tệp vừa dựng lại (nếu không SQLite sẽ phát lại WAL lạ lên DB mới)."""
    for suffix in ("-wal", "-shm"):
        side = path.with_name(path.name + suffix)
        if side.exists():
            side.unlink()


def _recover(path: Path, now: float) -> GuardReport:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    corrupt = _move_aside(path, stamp)
    salvaged = path.with_name(path.name + ".salvage.tmp")
    merged = path.with_name(path.name + ".merge.tmp")
    for leftover in (salvaged, merged):
        with contextlib.suppress(OSError):
            leftover.unlink(missing_ok=True)
    source = ""
    try:
        try:
            saved, complete = _salvage(corrupt, salvaged)
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Cứu dữ liệu từ tệp hỏng thất bại khi ghi ({}) — thử bản sao lưu.", exc)
            saved, complete = 0, False
        _clear_sidecars(path)               # không để -wal lạ phát lại lên DB vừa dựng (lỗi ⇒ bỏ phục hồi, tệp hỏng vẫn được giữ)
        salvage_ok = saved > 0 and _is_healthy(salvaged)
        backups = [
            candidate for candidate in (_backup_path(path, slot) for slot in range(1, BACKUP_SLOTS + 1))
            if candidate.exists() and _is_healthy(candidate)
        ]
        newest_backup = backups[0] if backups else None
        if salvage_ok and complete:
            os.replace(salvaged, path)
            source = "salvage"
        elif salvage_ok and newest_backup is not None and _merge_into_backup(newest_backup, salvaged, merged):
            os.replace(merged, path)
            source = "merge"
        elif salvage_ok:
            os.replace(salvaged, path)
            source = "salvage"
        elif newest_backup is not None:
            shutil.copy2(newest_backup, path)
            source = "backup"
    finally:
        for leftover in (salvaged, merged):
            with contextlib.suppress(OSError):
                leftover.unlink(missing_ok=True)
    return GuardReport(healthy=False, recovered_from=source, corrupt_copy=str(corrupt))


# ── Điểm vào ──────────────────────────────────────────────────────────────────


def _spawn(target: object, *args: object) -> None:
    thread = threading.Thread(target=target, args=args, daemon=True, name="db_guard")  # type: ignore[arg-type]
    with _lock:
        _background.append(thread)
    thread.start()


def _suspect_flag(path: Path) -> Path:
    return path.with_name(path.name + ".suspect")


def _deep_check_in_background(path: Path) -> None:
    """Kiểm sâu tệp lớn sau khi ứng dụng đã khởi động; thấy hỏng thì đặt cờ để lần sau phục hồi (không đụng tệp đang dùng)."""
    try:
        if _health(path) == "corrupt":
            _suspect_flag(path).write_text("quick_check", encoding="utf-8")
            logger.warning("db_guard: '{}' có dấu hiệu hỏng — sẽ kiểm và phục hồi ở lần khởi động sau.", path.name)
    except Exception as exc:  # noqa: BLE001 - tác vụ nền: không bao giờ ném
        logger.debug("db_guard: kiểm sâu nền lỗi: {}", exc)


def _backup_in_background(path: Path, moment: float) -> None:
    try:
        _rotate_backup(path, moment)
    except Exception as exc:  # noqa: BLE001
        logger.debug("db_guard: sao lưu nền lỗi: {}", exc)


def prepare_database(db_path: str | Path, now: float | None = None, *, background: bool = False) -> GuardReport:
    """Kiểm tra / sao lưu / phục hồi tệp SQLite TRƯỚC khi mở. Mỗi tệp chỉ xử lý một lần trong tiến trình.

    Args:
        db_path: Đường dẫn tệp ``.db``.
        now: Thời điểm hiện tại (test).
        background: ``True`` = sao lưu và kiểm sâu tệp lớn chạy ở luồng nền (mặc định của ứng dụng thật); ``False`` = chạy ngay
            trong lời gọi (test, và để :class:`GuardReport` phản ánh kết quả sao lưu).

    Không bao giờ ném ngoại lệ: lỗi bất ngờ chỉ được ghi log để ứng dụng vẫn thử mở tệp như trước.
    """
    path = Path(db_path)
    with _operation_lock:
        key = str(path.absolute())
        with _lock:
            if key in _checked:
                return GuardReport()
            _checked.add(key)
        try:
            return _prepare_locked(path, now, background)
        except Exception as exc:  # noqa: BLE001 - lớp bảo vệ phụ: không bao giờ làm ứng dụng không khởi động được
            logger.warning("db_guard lỗi ngoài dự kiến với {}: {}", path.name, exc)
            return GuardReport()


def _prepare_locked(path: Path, now: float | None, background: bool) -> GuardReport:
    moment = time.time() if now is None else now
    if not path.exists() or _footprint(path) == 0:
        return GuardReport()
    large = path.stat().st_size > DEEP_CHECK_MAX_BYTES
    suspect = _suspect_flag(path).exists()
    state = _health(path, deep=suspect or not large)
    if state == "unknown":
        logger.info("db_guard: chưa kết luận được tình trạng '{}' (đang khoá/không mở được) — để nguyên.", path.name)
        return GuardReport()
    if state == "ok":
        if suspect:
            with contextlib.suppress(OSError):
                _suspect_flag(path).unlink(missing_ok=True)
        if background:
            if large and not suspect:
                _spawn(_deep_check_in_background, path)
            _spawn(_backup_in_background, path, moment)
            return GuardReport()
        return GuardReport(backed_up=_rotate_backup(path, moment))
    with _cross_process_lock(path) as held:
        if not held:
            logger.warning("db_guard: tiến trình khác đang xử lý '{}' — không phục hồi lần này.", path.name)
            return GuardReport()
        if _health(path) != "corrupt":                    # tiến trình kia có thể vừa phục hồi xong
            return GuardReport()
        report = _recover(path, moment)
    with contextlib.suppress(OSError):
        _suspect_flag(path).unlink(missing_ok=True)
    outcome = {
        "salvage": "đã cứu được các dòng còn đọc được sang tệp mới",
        "merge": "đã gộp các dòng cứu được với bản sao lưu gần nhất",
        "backup": "đã khôi phục từ bản sao lưu gần nhất",
    }.get(report.recovered_from, "không phục hồi được — ứng dụng tạo cơ sở dữ liệu mới")
    message = (
        f"Cơ sở dữ liệu '{path.name}' bị hỏng; {outcome}. "
        f"Tệp hỏng được giữ nguyên tại: {report.corrupt_copy}"
    )
    logger.warning("{}", message)
    with _lock:
        _notices.append(message)
    return report


__all__ = [
    "BACKUP_INTERVAL_S", "BACKUP_SLOTS", "GuardReport", "pop_notices", "prepare_database", "reset_for_tests",
    "wait_for_background",
]
