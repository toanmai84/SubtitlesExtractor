"""[v3.23.675] Đóng gói Windows: icon thanh tác vụ, thông tin phiên bản, phụ thuộc trùng, màu Nhật ký.

Bằng chứng (build log 03/11 + ảnh chụp bản exe trên Windows 11):
* Nút thanh tác vụ hiện icon mặc định "ứng dụng không có icon" dù thanh tiêu đề đúng — app.ico chỉ có khung nén PNG.
* Exe không có VERSIONINFO ("Copying 0 resources to EXE").
* Cài CẢ opencv-contrib-python 4.10.0.84 (paddlex ghim) LẪN opencv-python 5.0.0.93 (requirements) — cùng thư mục cv2/.
* gradio/fastapi/uvicorn (vieneu khai báo, không dùng) bị gom vào exe.
* Trang Nhật ký: dòng INFO chữ #e0e0e0 trên nền sáng — gần như vô hình.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src" / "subtitles_extractor" / "data"


def _ico_entries(path: Path) -> list[tuple[int, bool]]:
    data = path.read_bytes()
    count = struct.unpack("<H", data[4:6])[0]
    out = []
    for i in range(count):
        w, _h, _c, _r, _p, _bpp, size, off = struct.unpack("<BBBBHHII", data[6 + 16 * i: 22 + 16 * i])
        out.append((w or 256, data[off:off + 4] == b"\x89PNG"))
    return out


def test_icon_du_co_va_khung_nho_la_bitmap() -> None:
    entries = dict(_ico_entries(DATA / "app.ico"))
    for size in (16, 20, 24, 32, 40, 48, 64, 256):
        assert size in entries, size
    assert not any(is_png for size, is_png in entries.items() if size < 256)
    assert entries[256] is True


def test_icon_dung_tu_anh_goc_bang_cong_cu() -> None:
    import importlib.util

    from PIL import Image

    spec = importlib.util.spec_from_file_location("make_app_icon", ROOT / "tools" / "make_app_icon.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build_ico(Image.open(DATA / "app.png")) == (DATA / "app.ico").read_bytes()


def test_qt_doc_duoc_moi_co_icon(qtbot) -> None:
    from PySide6.QtGui import QIcon

    sizes = {s.width() for s in QIcon(str(DATA / "app.ico")).availableSizes()}
    assert {16, 24, 32, 48, 256} <= sizes


def test_spec_co_thong_tin_phien_ban_va_loai_goi_web() -> None:
    spec = (ROOT / "SubtitlesExtractor.spec").read_text(encoding="utf-8")
    assert spec.count("version=_version_info") == 2
    assert 'StringStruct("FileDescription", "Subtitle Studio")' in spec
    excludes = spec[spec.index("excludes = ["):]
    excludes = excludes[:excludes.index("\n]")]
    for name in ("gradio", "gradio_client", "uvicorn", "fastapi", "starlette"):
        assert re.search(rf'^\s*"{name}",', excludes, re.M), name


@pytest.mark.parametrize("req", ["requirements.txt", "requirements-gpu.txt"])
def test_khong_cai_hai_ban_opencv(req: str) -> None:
    lines = [ln.split("#")[0].strip() for ln in (ROOT / req).read_text(encoding="utf-8").splitlines()]
    names = {re.split(r"[<>=!~ ]", ln)[0].lower() for ln in lines if ln}
    assert "opencv-python" not in names and "opencv-contrib-python" in names


def _contrast(a: str, b: str) -> float:
    def lum(c: str) -> float:
        rgb = [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_nhat_ky_doc_duoc_tren_nen_sang() -> None:
    from subtitles_extractor.presentation.pages.log_page import level_palette
    from subtitles_extractor.presentation.theme.colors import LIGHT_PALETTE

    levels, time_color, name_color = level_palette(False)
    bg = LIGHT_PALETTE["mono_bg"]
    for level, color in levels.items():
        assert _contrast(color, bg) >= 4.5, (level, color)
    assert _contrast(time_color, bg) >= 4.5 and _contrast(name_color, bg) >= 4.5
    dark_levels, _t, _n = level_palette(True)
    assert dark_levels["INFO"] == "#e0e0e0"           # nền tối giữ như cũ
