"""[v3.23.646] Hồi quy: OCR không chạy được ("Thiếu lõi OCR … 'libpaddle' is not a package") sau v3.23.643.

Nguyên nhân gốc: bộ chặn torch (``sys.modules["torch"] = None``) làm ``scipy.stats``/``sklearn.cluster`` chết ở lần nạp
đầu; v3.23.643 chuyển nạp sklearn sang trễ nên lần nạp đầu rơi vào giữa ``import paddle``. Nhánh "dọn rồi thử lại" lại
xoá ``paddle.base.libpaddle.pir`` (do native đăng ký) và che mất lỗi gốc.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[3] / "src")


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin", "HOME": "/tmp"}, timeout=240, check=False,
    )


def test_scientific_imports_fail_under_the_torch_blocker_but_prewarm_makes_them_safe() -> None:
    result = _run("""
        import sys
        from subtitles_extractor.infrastructure.torch_import_blocker import install_torch_import_blocker
        install_torch_import_blocker()
        try:
            import scipy.stats
            print("UNEXPECTED_OK")
        except AttributeError as exc:
            print("BLOCKED", exc)
        for name in [n for n in sys.modules if n.startswith(("scipy", "sklearn"))]:
            del sys.modules[name]
        from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import _prewarm_scientific_imports
        _prewarm_scientific_imports()
        import scipy.stats, scipy.signal
        print("PREWARMED", sys.modules["torch"] is None)
    """)
    out = result.stdout
    assert "PREWARMED True" in out, result.stderr[-800:]       # chặn torch vẫn còn nguyên sau khi nạp trước
    assert "BLOCKED" in out or "UNEXPECTED_OK" in out            # (scipy cũ hơn 1.15 không dò torch)


def test_dbscan_loads_under_the_torch_blocker() -> None:
    result = _run("""
        import sys
        from subtitles_extractor.infrastructure.torch_import_blocker import install_torch_import_blocker
        install_torch_import_blocker()
        from subtitles_extractor.infrastructure.video.bbox_analyzer import load_dbscan_class
        cls = load_dbscan_class()
        print("DBSCAN", cls is not None, sys.modules["torch"] is None)
    """)
    assert "DBSCAN True True" in result.stdout, result.stderr[-800:]


def _fake_packages(root: Path, paddle_init: str) -> None:
    (root / "paddle").mkdir()
    (root / "paddle" / "__init__.py").write_text(textwrap.dedent(paddle_init), encoding="utf-8")
    (root / "paddleocr").mkdir()
    (root / "paddleocr" / "__init__.py").write_text("class PaddleOCR: ...\n", encoding="utf-8")


@pytest.fixture
def isolated_paddle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    saved = {n: m for n, m in sys.modules.items() if n.split(".")[0] in {"paddle", "paddleocr"}}
    for name in saved:
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [n for n in sys.modules if n.split(".")[0] in {"paddle", "paddleocr"}]:
        del sys.modules[name]
    sys.modules.update(saved)


def test_half_initialised_paddle_reports_the_first_error_not_the_pir_error(isolated_paddle: Path) -> None:
    """Lõi native đã nạp mà ``import paddle`` lỗi ⇒ không nạp lại (vô vọng) và báo đúng lỗi GỐC."""
    from subtitles_extractor.domain.exceptions import OcrModelLoadError
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    _fake_packages(isolated_paddle, """
        import sys, types
        sys.modules["paddle.base.libpaddle"] = types.ModuleType("paddle.base.libpaddle")
        raise AttributeError("'NoneType' object has no attribute 'Tensor'")
    """)
    with pytest.raises(OcrModelLoadError) as info:
        PaddleOcrAdapter._import_paddle()
    assert "AttributeError: 'NoneType' object has no attribute 'Tensor'" in str(info.value)
    assert "pir" not in str(info.value)


def test_paddle_never_reached_native_core_is_retried_once(isolated_paddle: Path) -> None:
    from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter

    marker = isolated_paddle / "attempts.txt"
    _fake_packages(isolated_paddle, f"""
        from pathlib import Path
        marker = Path({str(marker)!r})
        attempts = int(marker.read_text()) if marker.exists() else 0
        marker.write_text(str(attempts + 1))
        if attempts == 0:
            raise AttributeError("partially initialized module")
        __version__ = "fake"
    """)
    module, ocr_class = PaddleOcrAdapter._import_paddle()
    assert module.__version__ == "fake" and ocr_class.__name__ == "PaddleOCR"
    assert marker.read_text() == "2"


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("paddleocr") is None, reason="chưa cài paddleocr"
)
def test_real_paddle_imports_with_the_torch_blocker_installed() -> None:
    """Đúng kịch bản của exe: bộ chặn torch bật từ lúc khởi động, rồi nạp OCR (trước bản vá: lỗi 'libpaddle.pir')."""
    result = _run("""
        import sys
        from subtitles_extractor.infrastructure.torch_import_blocker import install_torch_import_blocker
        install_torch_import_blocker()
        from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter
        try:
            module, cls = PaddleOcrAdapter._import_paddle()
            print("PADDLE_OK", cls.__name__)
        except Exception as exc:
            print("PADDLE_FAIL", exc)
    """)
    assert "PADDLE_OK PaddleOCR" in result.stdout, result.stdout[-600:] + result.stderr[-600:]
