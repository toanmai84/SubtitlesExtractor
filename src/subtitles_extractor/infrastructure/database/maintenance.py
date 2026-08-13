"""[v3.23.88] Bảo trì cơ sở dữ liệu: đưa DB về trạng thái "như mới tạo".

Hàm :func:`reset_database` xoá sạch DỮ LIỆU trong mọi bảng người dùng (GIỮ schema) của một
tệp SQLite, dùng cho nút "Dọn dẹp Database" trong Cài đặt nâng cao. Giữ schema (không xoá
tệp) để tránh xung đột với các kết nối đang mở của những store khác trỏ cùng tệp.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from loguru import logger


def reset_database(db_path: str | Path) -> list[str]:
    """Xoá toàn bộ dữ liệu trong mọi bảng người dùng của tệp SQLite (giữ schema).

    Args:
        db_path: Đường dẫn tệp ``.db``. Nếu tệp không tồn tại, không làm gì.

    Returns:
        Danh sách tên bảng đã được xoá dữ liệu (rỗng nếu DB chưa tồn tại/không có bảng).
    """
    path = Path(db_path)
    if not path.exists():
        logger.info("reset_database: tệp DB chưa tồn tại, bỏ qua: {}", path)
        return []

    conn = sqlite3.connect(str(path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            # Tên bảng từ sqlite_master nên an toàn; vẫn bọc ngoặc kép cho chắc.
            cursor.execute(f'DELETE FROM "{table}"')
        conn.commit()
        # VACUUM thu hồi dung lượng; có thể thất bại nếu còn kết nối khác giữ khoá
        # -> chỉ best-effort, không coi là lỗi nghiêm trọng.
        try:
            cursor.execute("VACUUM")
        except sqlite3.OperationalError as exc:
            logger.warning("reset_database: VACUUM bỏ qua ({}).", exc)
        logger.info("reset_database: đã xoá dữ liệu {} bảng tại {}.", len(tables), path)
        return tables
    finally:
        conn.close()
