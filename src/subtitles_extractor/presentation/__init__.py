"""Tầng Presentation — Qt UI và view model.

Quy tắc:
    * View KHÔNG biết Paddle/OpenCV/QSettings — chỉ biết ViewModel.
    * ViewModel tiêu thụ use case + container — không tạo adapter trực tiếp.
"""

from __future__ import annotations

# Bảo vệ import PySide6: trong môi trường unit test không có PyQt6 (headless CI),
# import này sẽ thất bại. Dùng try/except để các sub-module thuần Python
# (is_mpv_available, factory functions...) vẫn import được bình thường.
try:
    from subtitles_extractor.presentation.main_window import MainWindow
    __all__ = ["MainWindow"]
except ModuleNotFoundError:
    # PyQt6 chưa được cài — môi trường headless / unit test
    __all__ = []
