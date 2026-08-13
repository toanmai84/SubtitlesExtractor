"""Fixtures và Cấu hình dùng chung cho toàn bộ test suite."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# -------------------------------------------------------------------------
# [CRITICAL FIX] Tự động chèn thư mục 'src' vào biến môi trường sys.path
# Giải quyết triệt để lỗi "ModuleNotFoundError: No module named 'subtitles_extractor'"
# khi chạy test trực tiếp bằng Visual Studio Test Explorer hoặc VS Code.
# -------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Chèn project root để import gói 'tools.calibration' (khung tự hiệu chuẩn).
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# -------------------------------------------------------------------------
# [v2.31] Stub qfluentwidgets — package custom không có trên PyPI nên không
# pip install được trong CI/minimal environment. Stub MagicMock đáp ứng mọi
# import pattern ``from subtitles_extractor.presentation.fluent_compat import X, Y, ...``.
# -------------------------------------------------------------------------
def _install_qfluentwidgets_stub_if_missing() -> None:
    if "qfluentwidgets" in sys.modules:
        return
    try:
        import qfluentwidgets  # noqa: F401 — kiểm tra có thực không.
        return
    except ImportError:
        pass

    # [v3.23.89] Stub bằng WIDGET PYQT6 THẬT (thay vì MagicMock) để các test giao diện
    # (SectionCard, theme colors, dựng TranslatePage) chạy ĐÚNG trong môi trường
    # headless không có qfluentwidgets — thay vì luôn fail. Symbol phi-widget
    # InfoBar, FluentIcon…) vẫn dùng MagicMock cho linh hoạt.
    from types import ModuleType

    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFrame,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
    )

    class _PermissiveWidget:
        """Mixin: nuốt mọi tham số khởi tạo qfluent-đặc thù + method lạ -> no-op.

        Nhờ vậy code dựng UI thật chạy được mà không cần qfluentwidgets thật.
        """

        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            text = next((a for a in args if isinstance(a, str)), None)
            if text is not None and hasattr(self, "setText"):
                self.setText(text)

        def __getattr__(self, name: str):  # method qfluent-đặc thù -> no-op
            return lambda *a, **k: None

    def _w(base):
        return type(f"Stub{base.__name__}", (_PermissiveWidget, base), {})

    class StubHeaderCardWidget(_PermissiveWidget, QFrame):
        """HeaderCardWidget thật-hoá: lưu title + có viewLayout là layout thật."""

        def __init__(self, *args, **kwargs) -> None:
            QFrame.__init__(self)
            self._stub_title = ""
            self.viewLayout = QVBoxLayout(self)
            text = next((a for a in args if isinstance(a, str)), None)
            if text is not None:
                self._stub_title = text

        def setTitle(self, title: str) -> None:  # noqa: N802
            self._stub_title = title

        def getTitle(self) -> str:  # noqa: N802
            return self._stub_title

        def __getattr__(self, name: str):
            return lambda *a, **k: None

    real_widgets = {
        "PushButton": _w(QPushButton),
        "PrimaryPushButton": _w(QPushButton),
        "ToolButton": _w(QToolButton),
        "ComboBox": _w(QComboBox),
        "LineEdit": _w(QLineEdit),
        "TextEdit": _w(QTextEdit),
        "Slider": _w(QSlider),
        "ProgressBar": _w(QProgressBar),
        "SpinBox": _w(QSpinBox),
        "DoubleSpinBox": _w(QDoubleSpinBox),
        "CheckBox": _w(QCheckBox),
        "ScrollArea": _w(QScrollArea),
        "CaptionLabel": _w(QLabel),
        "StrongBodyLabel": _w(QLabel),
        "HeaderCardWidget": StubHeaderCardWidget,
    }

    stub_module = ModuleType("qfluentwidgets")
    for name, cls in real_widgets.items():
        setattr(stub_module, name, cls)
    # Hàm theme: trả giá trị THẬT để test colors nhận chuỗi hex hợp lệ.
    stub_module.themeColor = lambda: QColor("#0078d4")  # type: ignore[attr-defined]

    class _StubTheme:
        LIGHT = "light"
        DARK = "dark"
        AUTO = "auto"
        _current = "light"

    def _set_theme(theme=None, *args, **kwargs) -> None:
        _StubTheme._current = theme

    stub_module.Theme = _StubTheme  # type: ignore[attr-defined]
    stub_module.setTheme = _set_theme  # type: ignore[attr-defined]
    stub_module.isDarkTheme = (  # type: ignore[attr-defined]
        lambda: _StubTheme._current == _StubTheme.DARK
    )
    # Symbol phi-widget (cửa sổ/điều hướng/icon/thông báo) -> MagicMock linh hoạt.
    for name in (
        "FluentWindow", "FluentIcon", "NavigationItemPosition",
        "InfoBar", "InfoBarPosition", "MessageBox",
        "qrouter", "FluentTranslator",
    ):
        setattr(stub_module, name, MagicMock(return_value=MagicMock()))
    # Bất kỳ symbol nào KHÁC chưa liệt kê -> MagicMock (giữ tính linh hoạt cũ).
    stub_module.__getattr__ = lambda name: MagicMock()  # type: ignore[attr-defined]
    sys.modules["qfluentwidgets"] = stub_module


_install_qfluentwidgets_stub_if_missing()


@pytest.fixture
def tmp_config_file(tmp_path: Path) -> Iterator[Path]:
    """Trả về đường dẫn tệp config tạm — tự dọn sau mỗi test."""
    config_path = tmp_path / "config.json"
    yield config_path
    if config_path.exists():
        config_path.unlink()
