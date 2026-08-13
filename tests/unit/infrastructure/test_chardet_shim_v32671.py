"""[v3.23.271] Shim thay chardet (LGPL) bằng charset-normalizer (MIT).

chardet ≤6.x là LGPL 2.1 (copyleft); bị paddlex kéo vào. Khi đóng gói PyInstaller,
thư viện Python thuần bị nhúng tĩnh → LGPL khó thoả mãn. Shim đăng ký module
``chardet`` giả dựa trên charset-normalizer (MIT) vào sys.modules trước khi paddlex
import. Xem docs/LICENSE_ANALYSIS.md.
"""

from __future__ import annotations

import sys

from subtitles_extractor.infrastructure.compat.chardet_shim import (
    _detect,
    _detect_all,
    install_chardet_shim,
)


def test_detect_utf8_tieng_viet() -> None:
    # Phát hiện UTF-8 cho text tiếng Việt có dấu.
    result = _detect("Xin chào tiếng Việt".encode())
    assert result["encoding"] == "utf-8"
    assert result["confidence"] > 0


def test_detect_empty_tra_none() -> None:
    # Byte rỗng -> encoding None (như chardet).
    result = _detect(b"")
    assert result["encoding"] is None
    assert result["confidence"] == 0.0


def test_detect_co_du_cac_khoa_chardet() -> None:
    # Kết quả phải có đủ 3 khoá như chardet.detect: encoding, confidence, language.
    result = _detect(b"hello")
    assert set(result.keys()) == {"encoding", "confidence", "language"}


def test_detect_all_tra_list() -> None:
    # detect_all trả list (API tương thích chardet.detect_all).
    results = _detect_all("Tiếng Việt có dấu".encode())
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["encoding"] == "utf-8"


def test_detect_all_empty() -> None:
    results = _detect_all(b"")
    assert results == []


def test_install_shim_dang_ky_sys_modules(monkeypatch) -> None:
    # install_chardet_shim đăng ký module 'chardet' giả vào sys.modules.
    monkeypatch.delitem(sys.modules, "chardet", raising=False)
    install_chardet_shim()
    assert "chardet" in sys.modules
    import chardet

    assert chardet.__version__ == "5.9.9"
    assert chardet.__subext_shim__ == "charset-normalizer"
    assert chardet.detect("Tiếng Việt".encode())["encoding"] == "utf-8"


def test_install_shim_khong_ghi_de_chardet_da_co(monkeypatch) -> None:
    # Nếu chardet thật đã nạp, shim KHÔNG ghi đè (tránh xung đột).
    sentinel = object()
    monkeypatch.setitem(sys.modules, "chardet", sentinel)  # type: ignore[arg-type]
    install_chardet_shim()
    assert sys.modules["chardet"] is sentinel


def test_encoding_dung_dinh_dang_gach_ngang() -> None:
    # charset-normalizer trả 'utf_8', shim chuẩn hoá thành 'utf-8' (như chardet).
    result = _detect(b"some ascii text")
    if result["encoding"]:
        assert "_" not in result["encoding"]
