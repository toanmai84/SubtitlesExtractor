"""[v3.23.674] Sắp xếp giao diện: điều hướng chia nhóm, nút không cắt chữ, nhãn hiển thị đúng.

Khảo sát bằng ảnh chụp từng trang ở màn hình 1366×768 (harness ui_shots): tên tab "Engine _Nguồn" (dấu & bị Qt hiểu là
phím tắt), nút "\\u29d6 Lặp câu" (mã ký tự lộ ra), hàng nút trang Dịch/TTS bị cắt chữ, ô cấu hình Edge TTS có khoảng
trống lớn, 11 tab Cài đặt không phân nhóm, "Trích xuất" và "Xuất bản" cùng một biểu tượng.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data"
#: Tiền tố khoá chuỗi dùng làm chữ trên nút / tab / ô chọn / tiêu đề nhóm (nơi "&" thành phím tắt).
_WIDGET_KEY = re.compile(r"^(btn_|tab_|cb_|sec_|grp_|sc_grp_|dbg_sec_|dbg_btn_|ed_btn_|ex_btn_|ex_tab_|group_)")


def _walk(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, key)
    elif isinstance(node, str):
        yield prefix, node


@pytest.mark.parametrize("lang", ["vi", "en"])
def test_chu_tren_nut_tab_khong_co_dau_va_le(lang: str) -> None:
    data = json.loads((DATA / f"strings_{lang}.json").read_text(encoding="utf-8"))
    bad = [(k, v) for k, v in _walk(data) if _WIDGET_KEY.match(k) and re.search(r"(?<!&)&(?!&)", v)]
    assert not bad, bad


@pytest.mark.parametrize("lang", ["vi", "en"])
def test_khong_lo_ma_ky_tu(lang: str) -> None:
    text = (DATA / f"strings_{lang}.json").read_text(encoding="utf-8")
    assert "\\\\u" not in text


def test_dieu_huong_chia_nhom(qtbot) -> None:
    from PySide6.QtWidgets import QLabel, QWidget

    from subtitles_extractor.presentation.fluent_compat import FluentWindow, NavigationItemPosition

    win = FluentWindow()
    qtbot.addWidget(win)
    pages = [QWidget() for _ in range(4)]
    win.addSubInterface(pages[0], None, "A", section="QUY TRÌNH")
    win.addSubInterface(pages[1], None, "B", section="QUY TRÌNH")
    win.addSubInterface(pages[2], None, "C", section="CÔNG CỤ")
    win.addSubInterface(pages[3], None, "D", position=NavigationItemPosition.BOTTOM)
    headers = [lb.text() for lb in win.findChildren(QLabel) if lb.objectName() == "navSection"]
    assert headers == ["QUY TRÌNH", "CÔNG CỤ"]


def test_hang_nut_tu_xuong_dong(qtbot) -> None:
    from PySide6.QtWidgets import QPushButton

    from subtitles_extractor.presentation.widgets.flow_layout import flow_row

    holder, layout = flow_row()
    qtbot.addWidget(holder)
    buttons = [QPushButton(f"Nút số {i} khá dài") for i in range(6)]
    for b in buttons:
        layout.addWidget(b)
    one_line = layout.heightForWidth(5000)
    narrow = layout.heightForWidth(buttons[0].sizeHint().width() + 10)
    assert narrow > one_line * 3          # hẹp → mỗi nút một hàng, không ép chữ
    holder.resize(300, layout.heightForWidth(300))
    holder.show()
    for b in buttons:                      # mọi nút đủ bề ngang chữ
        assert b.width() >= b.sizeHint().width()


def test_khung_engine_cao_theo_trang_dang_hien(qtbot) -> None:
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    from subtitles_extractor.presentation.widgets.current_page_stack import CurrentPageStack

    def page(lines: int) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        for i in range(lines):
            lay.addWidget(QLabel(f"dòng {i}"))
        return w

    stack = CurrentPageStack()
    qtbot.addWidget(stack)
    short, tall = page(2), page(20)
    stack.addWidget(short)
    stack.addWidget(tall)
    assert stack.sizeHint().height() < tall.sizeHint().height()
    seen: list[int] = []
    stack.currentChanged.connect(seen.append)
    stack.setCurrentIndex(1)
    assert seen == [1] and stack.currentWidget() is tall
    assert stack.sizeHint().height() >= tall.sizeHint().height()


def test_cai_dat_danh_muc_chia_nhom(qtbot) -> None:
    from PySide6.QtWidgets import QWidget

    from subtitles_extractor.presentation.widgets.category_pages import CategoryPages

    pages = CategoryPages()
    qtbot.addWidget(pages)
    pages.addSection("CHUNG")
    a, b = QWidget(), QWidget()
    pages.addTab(a, "Giao diện")
    pages.addSection("NÂNG CAO")
    pages.addTab(b, "Khác")
    assert pages.count() == 2 and pages.tabText(1) == "Khác"
    pages.setCurrentIndex(1)
    assert pages.currentIndex() == 1
    pages._list.setCurrentRow(2)   # tiêu đề nhóm không chọn được → không đổi trang
    assert pages.currentIndex() == 1


def test_nut_chinh_co_kieu_noi_bat() -> None:
    from subtitles_extractor.presentation.theme.app_style import build_app_stylesheet

    qss = build_app_stylesheet()
    assert 'QPushButton[primary="true"]' in qss
    assert 'QPushButton[navButton="true"]' in qss


# ── [rà độc lập] tương phản chữ phụ, bề rộng tối thiểu hàng nút, gợi ý cài đặt ─────────────


def _lum(hex_color: str) -> float:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize(("dark", "window_bg"), [(False, "#efefef"), (True, "#202020")])
def test_chu_phu_doc_duoc_ca_nen_sang_lan_toi(dark: bool, window_bg: str) -> None:
    from subtitles_extractor.presentation.theme.app_style import _MUTED, build_app_stylesheet

    section_fg, disabled_bg, disabled_fg = _MUTED[dark]
    assert _contrast(section_fg, window_bg) >= 4.5          # tiêu đề nhóm thanh bên
    assert _contrast(disabled_fg, disabled_bg) >= 4.5       # nút chính đang khoá
    qss = build_app_stylesheet(dark)
    assert "palette(mid)" not in qss and section_fg in qss


def test_hang_nut_bo_rong_toi_thieu_du_nut_rong_nhat(qtbot) -> None:
    from PySide6.QtWidgets import QPushButton

    from subtitles_extractor.presentation.widgets.flow_layout import flow_row

    holder, layout = flow_row()
    qtbot.addWidget(holder)
    wide = QPushButton("Một nút có nhãn rất dài để thử bề rộng tối thiểu")
    layout.addWidget(QPushButton("Ngắn"))
    layout.addWidget(wide)
    assert layout.minimumSize().width() >= wide.sizeHint().width()


def test_goi_y_cai_dat_chi_cho_engine_thieu() -> None:
    from types import SimpleNamespace

    from subtitles_extractor.presentation.pages import tts_page as mod

    shown: dict = {}
    hint = SimpleNamespace(setText=lambda t: shown.__setitem__("text", t),
                           setVisible=lambda v: shown.__setitem__("visible", v))
    page = SimpleNamespace(_install_hint=hint, _INSTALL_COMMANDS=mod.TTSPage._INSTALL_COMMANDS,
                           _engine_avail={mod._ENGINE_EDGE: True, mod._ENGINE_GEMINI: True,
                                          mod._ENGINE_VIENEU: True})
    mod.TTSPage._refresh_install_hint(page)
    assert shown["visible"] is False and shown["text"] == ""
    page._engine_avail[mod._ENGINE_VIENEU] = False
    mod.TTSPage._refresh_install_hint(page)
    assert shown["visible"] is True
    assert "vieneu" in shown["text"] and "edge-tts" not in shown["text"]
