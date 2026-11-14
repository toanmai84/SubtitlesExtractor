"""[v3.23.694] Mã nguồn các trang đã tách (U6) cho test đọc mã: trang gốc + các module tách, nối thành một văn bản.

Thay cho ``Path(...<trang>.py)`` trong test cũ: cùng giao diện ``read_text``. Chỉ giữ một dòng ``from __future__`` ở đầu
nên văn bản nối vẫn ``ast.parse`` được.
"""

from __future__ import annotations

from pathlib import Path

PAGES_DIR = Path(__file__).resolve().parents[2] / "src" / "subtitles_extractor" / "presentation" / "pages"
_FUTURE = "from __future__ import annotations\n"


class SplitPageSource:
    """Mã nguồn một trang đã tách: ``files`` theo thứ tự đọc; ``main`` là file trang gốc."""

    def __init__(self, main: str, files: tuple[str, ...]) -> None:
        self.name = main
        self.files = files

    def read_text(self, encoding: str = "utf-8") -> str:
        parts = [(PAGES_DIR / name).read_text(encoding=encoding) for name in self.files]
        return _FUTURE + "".join("\n" + part.replace(_FUTURE, "") for part in parts)

    def exists(self) -> bool:
        return all((PAGES_DIR / name).exists() for name in self.files)

    def __fspath__(self) -> str:
        return str(PAGES_DIR / self.name)


#: Theo thứ tự của file gốc trước khi tách: hằng số, hộp thoại, rồi lớp trang và các mixin của nó.
EDITOR_PAGE_SOURCE = SplitPageSource("editor_page.py", (
    "editor_constants.py", "editor_dialogs.py", "editor_page.py",
    "editor_reocr.py", "editor_find_replace.py", "editor_export.py",
))
TRANSLATE_PAGE_SOURCE = SplitPageSource("translate_page.py", (
    "translate_constants.py", "translate_widgets.py", "translate_page.py",
    "translate_ui.py", "translate_keys.py", "translate_models.py", "translate_analysis.py",
    "translate_table.py", "translate_tools.py", "translate_batch.py", "translate_export.py", "translate_cues.py",
))
#: [v3.23.695] Trang Lồng tiếng: hằng số, luồng nạp giọng, lớp trang, rồi các mixin (3.23.692 và 3.23.695).
TTS_PAGE_SOURCE = SplitPageSource("tts_page.py", (
    "tts_constants.py", "tts_voice_loader.py", "tts_page.py",
    "tts_ui.py", "tts_engines.py", "tts_results.py", "tts_review.py", "tts_settings.py",
    "tts_batch.py", "tts_gpu_install.py", "tts_export_details.py",
))
