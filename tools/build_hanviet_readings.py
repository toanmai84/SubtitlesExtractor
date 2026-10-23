"""Dựng bảng âm Hán-Việt theo TỪNG CHỮ cho bộ soát phiên âm tên nhân vật (v3.23.567).

Kết quả: ``src/subtitles_extractor/data/hanviet_readings.json`` — mỗi chữ Hán (giản thể
lẫn phồn thể) ứng với MỌI âm Hán-Việt đã biết của nó. Bộ soát chỉ cần biết "âm này có
phải một âm hợp lệ của chữ này không", nên gộp càng đủ âm càng AN TOÀN (ít sửa nhầm).

Nguồn:
    1. ``hanviet-pinyin-words`` (npm, MIT, © 2024 Phong Phan) — ``src/hanvietData.js``:
       âm Hán-Việt theo chữ phồn thể + pinyin.
    2. OpenCC ``STCharacters.txt`` (Apache-2.0) — ánh xạ giản thể → phồn thể, để chữ
       giản thể nhận âm của (mọi) chữ phồn thể tương ứng.
    3. Unihan ``kVietnamese`` (Unicode License v3) — bổ sung âm còn thiếu (vd 任 "nhậm",
       兑 "đoài"). Trường này lẫn cả âm Nôm nên chỉ dùng để CHẤP NHẬN (thêm âm chỉ làm
       bộ soát dễ dãi hơn), KHÔNG BAO GIỜ dùng làm âm đề xuất sửa.

Định dạng mỗi chữ: ``"âm chính 1 âm chính 2|âm phụ 1 âm phụ 2"`` — âm chính (nguồn 1) được
phép đề xuất khi sửa; âm phụ (nguồn 3) chỉ để chấp nhận. Không có âm phụ thì bỏ ``|``.

Cách dùng:
    npm pack hanviet-pinyin-words cjk-unihan && tar xzf ...
    pip download opencc-python-reimplemented --no-deps && unzip ...
    python tools/build_hanviet_readings.py \\
        --hanviet package/src/hanvietData.js \\
        --opencc opencc/dictionary/STCharacters.txt \\
        --unihan unihan/package/data/unihan.db
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "subtitles_extractor"
    / "data"
    / "hanviet_readings.json"
)

_SCHEMA = "subtitles_extractor.hanviet_readings/1"

_SOURCES = [
    "hanviet-pinyin-words 2.1.7 (MIT, (c) 2024 Phong Phan)",
    "OpenCC STCharacters.txt (Apache-2.0)",
    "Unicode Unihan kVietnamese (Unicode License v3)",
    "Âm địa danh quy ước (bổ sung tay, xem tools/build_hanviet_readings.py)",
]

#: [v3.23.585] Âm chỉ gặp trong ĐỊA DANH quy ước mà ba nguồn trên không có. Chuẩn đoán
#: thật (phim "灭门惨案II", nhân vật là cựu binh chiến tranh biên giới): bảng thuật ngữ
#: ghi ``諒山 = Lạng Sơn`` — đúng tên tỉnh — nhưng bị báo "諒 đọc lượng". Chỉ để CHẤP NHẬN
#: (âm phụ), không bao giờ dùng làm âm đề xuất sửa.
_PLACE_NAME_READINGS: dict[str, tuple[str, ...]] = {
    "諒": ("lạng",),
}

_SYLLABLE_RE = re.compile(r"^[^\W\d_]+$")


def _clean_reading(raw: str) -> str:
    reading = unicodedata.normalize("NFC", raw.strip().lower())
    return reading if _SYLLABLE_RE.match(reading) else ""


Readings = dict[str, tuple[list[str], list[str]]]


def _add(table: Readings, char: str, readings: list[str], primary: bool) -> None:
    main, extra = table.setdefault(char, ([], []))
    for raw in readings:
        reading = _clean_reading(raw)
        if not reading or reading in main:
            continue
        if primary:
            main.append(reading)
            if reading in extra:
                extra.remove(reading)
        elif reading not in extra:
            extra.append(reading)


def load_hanviet_js(path: Path) -> Readings:
    """Đọc ``hanvietData.js`` (``export const hanvietData = {...}``) làm âm CHÍNH."""
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text[text.index("{") : text.rindex("}") + 1])
    table: Readings = {}
    for char, by_pinyin in payload.items():
        for readings in by_pinyin.values():
            _add(table, char, list(readings), primary=True)
    return table


def add_unihan(table: Readings, db_path: Path) -> int:
    """Bổ sung âm ``kVietnamese`` của Unihan làm âm PHỤ; trả số chữ được thêm âm."""
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT character, kVietnamese FROM unihan "
            "WHERE kVietnamese IS NOT NULL AND kVietnamese != ''"
        ).fetchall()
    finally:
        connection.close()
    touched = 0
    for char, readings in rows:
        before = sum(map(len, table.get(char, ([], []))))
        _add(table, char, str(readings).split(), primary=False)
        touched += int(sum(map(len, table[char])) > before)
    return touched


def add_simplified(table: Readings, st_path: Path) -> int:
    """Chữ giản thể nhận âm của mọi chữ phồn thể tương ứng; trả số chữ giản thể mới."""
    added = 0
    for line in st_path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        simplified, traditional = line.split("\t", 1)
        existed = any(table.get(simplified, ([], [])))
        for trad in traditional.split():
            main, extra = table.get(trad, ([], []))
            _add(table, simplified, list(main), primary=True)
            _add(table, simplified, list(extra), primary=False)
        if not existed and any(table.get(simplified, ([], []))):
            added += 1
    return added


def build(hanviet_js: Path, st_chars: Path, unihan_db: Path | None) -> dict[str, str]:
    """Dựng bảng ``chữ → "âm chính…|âm phụ…"``."""
    table = load_hanviet_js(hanviet_js)
    if unihan_db is not None:
        logger.info("Unihan bổ sung âm cho %d chữ.", add_unihan(table, unihan_db))
    # Trước bước giản thể để chữ giản thể (谅) nhận luôn âm bổ sung của chữ phồn thể (諒).
    for char, place_readings in _PLACE_NAME_READINGS.items():
        _add(table, char, list(place_readings), primary=False)
    logger.info("Thêm %d chữ giản thể.", add_simplified(table, st_chars))
    rendered: dict[str, str] = {}
    for char, (main, extra) in sorted(table.items()):
        if not main and not extra:
            continue
        rendered[char] = " ".join(main) + ("|" + " ".join(extra) if extra else "")
    return rendered


def main() -> None:
    """Điểm vào dòng lệnh."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hanviet", type=Path, required=True)
    parser.add_argument("--opencc", type=Path, required=True)
    parser.add_argument("--unihan", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    readings = build(args.hanviet, args.opencc, args.unihan)
    payload = {"schema": _SCHEMA, "sources": _SOURCES, "readings": readings}
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    logger.info("Đã ghi %d chữ vào %s", len(readings), args.output)


if __name__ == "__main__":
    main()
