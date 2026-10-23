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

# [v3.23.553] Script đo tham số PaddleOCR trên GPU chạy suy luận NGAY Ở CẤP MODULE (cần
# GPU + ảnh mẫu, không có hàm test) — không phải test tự động. Để pytest thu thập nó thì
# lệnh ``pytest`` ở thư mục gốc hỏng ngay bước collect. Chạy tay: ``python tests/<file>``.
collect_ignore = ["test_all_parameter_Paddle_with_gpu.py"]


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


@pytest.fixture(autouse=True)
def _isolate_torch_import_blocker() -> Iterator[None]:
    """[v3.23.602] Test nào cài bộ chặn import torch thì tự gỡ khi xong.

    ``bootstrap_for_gui`` cài bộ chặn cho CẢ tiến trình (đúng cho ứng dụng thật). Trong bộ
    test, nó rò sang các test chạy sau: scipy đọc ``sys.modules["torch"] = None`` rồi báo
    ``'NoneType' object has no attribute 'Tensor'`` — hai test âm thanh hỏng tuỳ thứ tự
    chạy (đo được khi một test ở ``tests/unit/application`` gọi ``bootstrap_for_gui``).
    """
    from subtitles_extractor.infrastructure.torch_import_blocker import (
        is_torch_import_blocked,
        uninstall_torch_import_blocker,
    )

    blocked_before = is_torch_import_blocked()
    yield
    if not blocked_before and is_torch_import_blocked():
        uninstall_torch_import_blocker()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """[v3.23.619] Huỷ tường minh các cửa sổ cấp cao còn sống trước khi tiến trình thoát.

    Nhiều widget được tạo không cha rồi gắn vào layout: Qt đặt cha nhưng PySide vẫn coi
    chúng thuộc Python. Khi thoát, ``PySide::destroyQCoreApplication`` huỷ mọi đối tượng
    thuộc Python theo thứ tự tuỳ ý → có lần huỷ trùng → SIGSEGV trong ``QObject::~QObject``
    (đo được: chạy riêng ``test_ui_bug_sweep_v323613.py`` crash ~50% số lần, mọi test vẫn
    qua). Huỷ cửa sổ cấp cao trước để Qt tự huỷ con theo đúng cây cha–con. Ứng dụng thật
    không bị: ``MainWindow`` thoát sạch 10/10 lần.
    """
    try:
        import shiboken6
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    app = QApplication.instance()
    if app is None:
        return
    for widget in list(QApplication.topLevelWidgets()):
        if shiboken6.isValid(widget):
            widget.close()
            shiboken6.delete(widget)
    app.processEvents()
