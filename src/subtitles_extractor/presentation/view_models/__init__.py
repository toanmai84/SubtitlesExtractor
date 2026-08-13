"""ViewModel cho UI — pure Python logic, không phụ thuộc widget cụ thể."""

from __future__ import annotations

from subtitles_extractor.presentation.view_models.editor_page_view_model import (
    EditorPageViewModel,
)
from subtitles_extractor.presentation.view_models.extract_page_view_model import (
    ExtractPageViewModel,
)
from subtitles_extractor.presentation.view_models.settings_page_view_model import (
    SettingsPageViewModel,
)

__all__ = [
    "EditorPageViewModel",
    "ExtractPageViewModel",
    "SettingsPageViewModel",
]
