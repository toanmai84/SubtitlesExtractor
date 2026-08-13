"""Gói tests của Subtitles Extractor.

Đảm bảo thư mục src được đưa vào sys.path để cho phép chạy độc lập từng test case
mà không bị lỗi ModuleNotFoundError trên các IDE (Visual Studio, VS Code).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))