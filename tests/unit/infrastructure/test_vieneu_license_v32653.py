"""[v3.23.253] Cập nhật theo tài liệu VieNeu 2.7.0 (PyPI + README chính thức).

Bug licence (nghiêm trọng về pháp lý): App ghi VieNeu dùng licence phi-thương-mại cần xin
phép. README + PyPI chính thức (2.7.0, 7/5/2026) ghi rõ Apache 2.0 (Free to use). Ghi chú
sai có thể khiến người dùng tưởng KHÔNG được dùng thương mại -> mất cơ hội dùng hợp pháp.
Đã sửa cả docstring adapter lẫn nhãn cảnh báo trên UI.

Cảnh báo Turbo cho câu ngắn (theo tài liệu): README cảnh báo Turbo mode chất lượng thấp
hơn, có thể lỗi/nhiễu với câu rất ngắn (< 5 từ). Phụ đề có RẤT NHIỀU câu ngắn -> tooltip
mode nay nêu rõ để người dùng chọn Standard cho phụ đề.

Các tính năng VieNeu v2 khác app đã hỗ trợ sẵn: emotion natural/storytelling (UI +
save/restore), song ngữ Anh-Việt code-switching (SDK tự xử lý), Standard là mode mặc định.
"""

from __future__ import annotations

import pathlib

_ADAPTER_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/vieneu_tts_adapter.py"
).read_text(encoding="utf-8")
_PAGE_SRC = pathlib.Path(
    "src/subtitles_extractor/presentation/pages/tts_page.py"
).read_text(encoding="utf-8")

# Ghép chuỗi để không tự match chính docstring/assert của test này.
_BAD_LICENSE = "CC BY" + "-NC"


def test_không_còn_license_sai() -> None:
    assert _BAD_LICENSE not in _ADAPTER_SRC
    assert _BAD_LICENSE not in _PAGE_SRC


def test_ghi_đúng_apache_2() -> None:
    assert "Apache 2.0" in _ADAPTER_SRC
    assert "Apache 2.0" in _PAGE_SRC


def test_không_còn_cảnh_báo_phi_lợi_nhuận_trên_ui() -> None:
    assert "phi lợi nhuận" not in _PAGE_SRC


def test_tooltip_cảnh_báo_turbo_câu_ngắn() -> None:
    idx = _PAGE_SRC.find("self._vieneu_mode.setToolTip")
    assert idx != -1
    doan = _PAGE_SRC[idx : idx + 400]
    assert "5 từ" in doan
    assert "Standard" in doan


def test_standard_vẫn_là_mặc_định() -> None:
    idx_std = _PAGE_SRC.find('userData="standard"')
    idx_turbo = _PAGE_SRC.find('userData="turbo"')
    assert idx_std != -1 and idx_turbo != -1
    assert idx_std < idx_turbo
