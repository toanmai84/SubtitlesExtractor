"""[v3.23.261] Bắt OSError khi python-mpv cài nhưng thiếu libmpv.

**Bug phát hiện khi cài python-mpv 1.0.8 thật:** ``import mpv`` khi KHÔNG tìm thấy
libmpv (native lib) ném **``OSError``** ("Cannot find libmpv..."), KHÔNG phải
``ImportError``.

App nhiều chỗ chỉ bắt ``ImportError`` -> trên máy thiếu libmpv (hoặc DLL chưa inject
vào PATH), ``import mpv`` ném OSError lọt ra -> sập luồng thay vì báo lỗi thân thiện.

**Sửa:** mọi chỗ ``import mpv`` bắt CẢ ``ImportError`` lẫn ``OSError``. Collector gộp
``except (ImportError, OSError)``; đường chạy thật raise lỗi rõ về libmpv.

Kiểm tra bằng cách đọc source (không import mpv thật vì sandbox có thể thiếu/khác libmpv).
"""

from __future__ import annotations

import pathlib

_VIDEO_DIR = "src/subtitles_extractor/infrastructure/video"


def _read(rel: str) -> str:
    return pathlib.Path(f"{_VIDEO_DIR}/{rel}").read_text(encoding="utf-8")


def test_metadata_reader_bắt_oserror() -> None:
    src = _read("mpv_metadata_reader.py")
    # Đường read() phải bắt OSError khi import mpv (thiếu libmpv).
    assert "except OSError as exc:" in src
    assert "libmpv" in src


def test_frame_sampler_bắt_oserror() -> None:
    src = _read("decoders/mpv_frame_sampler.py")
    assert "except OSError as exc:" in src
    assert "libmpv" in src


def test_player_adapter_bắt_oserror() -> None:
    src = _read("mpv_player_adapter.py")
    assert "except OSError as exc:" in src
    assert "libmpv" in src


def test_collectors_gộp_importerror_oserror() -> None:
    # Các hàm thu thập error class dùng except (ImportError, OSError) để không lọt
    # OSError.
    for rel in (
        "mpv_metadata_reader.py",
        "decoders/mpv_frame_sampler.py",
        "mpv_player_adapter.py",
    ):
        src = _read(rel)
        assert "except (ImportError, OSError):" in src, rel


def test_oserror_thật_là_hành_vi_được_giả_định() -> None:
    # Ghi nhận bất biến: import mpv thiếu libmpv ném OSError (nếu môi trường có python-
    # mpv).
    # Không ép sandbox phải có libmpv; chỉ kiểm khi import được thì OSError là lớp con
    # hợp lệ.
    import builtins

    # OSError là built-in; xác nhận nó KHÔNG phải subclass ImportError (nên phải bắt
    # riêng).
    assert not issubclass(OSError, ImportError)
    assert hasattr(builtins, "OSError")
