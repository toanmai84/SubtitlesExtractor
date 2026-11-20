"""[v3.23.734] Chốt chặn ``builtins.__import__`` giả rò sang test khác (lỗi chập chờn test pedalboard)."""

from __future__ import annotations

import builtins

import pytest

from tests import conftest


def test_restore_noop_when_real() -> None:
    assert builtins.__import__ is conftest._REAL_IMPORT
    assert conftest.restore_real_import("không rò") is False


def test_restore_fixes_leaked_fake_and_names_culprit() -> None:
    real = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "json":
            raise ImportError("No module named 'json'")
        return real(name, *args, **kwargs)  # type: ignore[arg-type]

    builtins.__import__ = _fake_import  # cố ý rò, KHÔNG qua monkeypatch
    try:
        with pytest.warns(pytest.PytestWarning, match="_fake_import"):
            assert conftest.restore_real_import("test giả lập") is True
        assert builtins.__import__ is real
        assert conftest.IMPORT_LEAKS[-1] == "test giả lập"
        assert __import__("json").dumps([]) == "[]"  # import chạy lại bình thường
    finally:
        builtins.__import__ = real


def test_sys_modules_fake_survives_background_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nguyên nhân gốc: ``__feature_import__`` (PySide6) tạm tráo ``builtins.__import__`` khi import, không an toàn đa
    luồng — đo thật: 2000 vòng thay/khôi phục ``__import__`` giả khi một luồng nền import → cuối cùng còn bản GIẢ.
    Giả lập thiếu module qua ``sys.modules`` không chạm ``__import__`` nên không bị ghi đè."""
    import sys
    import threading

    pytest.importorskip("PySide6.QtCore")
    stop = threading.Event()

    def _background() -> None:
        while not stop.is_set():
            import json  # noqa: F401 — đi qua __feature_import__

    before = builtins.__import__
    worker = threading.Thread(target=_background, daemon=True)
    worker.start()
    try:
        for _ in range(500):
            with monkeypatch.context() as patch:
                patch.setitem(sys.modules, "json_fake_missing_v323734", None)
                with pytest.raises(ImportError):
                    __import__("json_fake_missing_v323734")
            assert "json_fake_missing_v323734" not in sys.modules
    finally:
        stop.set()
        worker.join()
    # shiboken có thể để lại __lazy_import__ (chuyển tiếp đúng tới import gốc) — nhưng không bao giờ là bản giả.
    assert getattr(builtins.__import__, "__module__", None) in (None, "builtins") or builtins.__import__ is before


def test_shared_modules_not_faked_via_builtin_import() -> None:
    """Không test nào được thay ``builtins.__import__`` để giả lập thiếu pedalboard / google-genai (dùng sys.modules)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("test_*.py"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        if 'setattr(builtins, "__import__"' in text and ("pedalboard" in text or "google" in text):
            offenders.append(path.name)
    assert offenders == []
