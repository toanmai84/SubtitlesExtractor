"""Tests bảo vệ v3.6 editor_page bugfixes session 4.

FO-1: _on_text_focus_out dùng current_events[idx] → list copy per keypress
SC-2: _on_split_clicked dùng current_events[idx]
WD-1: _on_waveform_drag dùng len(current_events) cho bounds check
WCS-1: _on_waveform_create_sub tạo 2 list copy liên tiếp
Batch: bool/len(current_events) → _table_model.events
"""
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[3] / "src/subtitles_extractor/presentation/pages/editor_page.py"


def _snip(source: str, method: str, class_hint: str = "") -> str:
    if class_hint:
        start_class = source.find(f"class {class_hint}")
        start = source.find(f"def {method}", start_class)
    else:
        start = source.find(f"def {method}")
    end = source.find("\n    def ", start + 1)
    return source[start:end if end != -1 else len(source)]


def _code_lines(snippet: str) -> list[str]:
    in_doc = False
    out = []
    for line in snippet.splitlines():
        s = line.strip()
        if s.startswith(('"""', "'''")):
            in_doc = not in_doc
            continue
        if in_doc or s.startswith("#") or not s:
            continue
        out.append(line)
    return out


# ── FO-1: _on_text_focus_out ─────────────────────────────────────────────────

class TestFocusOutNoCurrentEvents:
    def test_no_current_events(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_text_focus_out", "EditorPage")
        bad = [l for l in _code_lines(snippet) if "current_events" in l]
        assert not bad, f"FO-1: _on_text_focus_out không được dùng current_events:\n" + "\n".join(bad)

    def test_uses_table_model_for_text_compare(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_text_focus_out", "EditorPage")
        assert "_table_model.events" in snippet, "_on_text_focus_out phải dùng _table_model.events[idx]"


# ── SC-2: _on_split_clicked ───────────────────────────────────────────────────

class TestSplitClickedNoCurrentEvents:
    def test_no_current_events(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_split_clicked")
        bad = [l for l in _code_lines(snippet) if "current_events" in l]
        assert not bad, f"SC-2: _on_split_clicked không được dùng current_events:\n" + "\n".join(bad)

    def test_uses_table_model(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_split_clicked")
        assert "_table_model.events" in snippet


# ── WD-1: _on_waveform_drag ──────────────────────────────────────────────────

class TestWaveformDragNoCurrentEvents:
    def test_no_current_events_in_drag(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_waveform_drag")
        bad = [l for l in _code_lines(snippet) if "current_events" in l]
        assert not bad, f"WD-1: _on_waveform_drag không được dùng current_events:\n" + "\n".join(bad)

    def test_uses_table_model_for_bounds(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_waveform_drag")
        assert "_table_model.events" in snippet


# ── WCS-1: _on_waveform_create_sub ───────────────────────────────────────────

class TestWaveformCreateSubNoCurrentEvents:
    def test_no_current_events(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_waveform_create_sub")
        bad = [l for l in _code_lines(snippet) if "current_events" in l]
        assert not bad, f"WCS-1: _on_waveform_create_sub không được dùng current_events:\n" + "\n".join(bad)

    def test_single_table_model_reference(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        snippet = _snip(src, "_on_waveform_create_sub")
        assert "_table_model.events" in snippet


# ── Batch: bool/len/not current_events ───────────────────────────────────────

class TestNoBoolLenCurrentEvents:
    """Tất cả bool/len/not checks phải dùng _table_model.events."""

    def test_no_bool_current_events(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        # Chỉ tìm trong EditorPage (sau dòng "class EditorPage")
        editor_start = src.find("class EditorPage(")
        editor_src = src[editor_start:]
        bad = [
            l.strip() for l in editor_src.splitlines()
            if "bool(self._view_model.current_events)" in l
            and not l.strip().startswith("#")
        ]
        assert not bad, f"Phải dùng bool(_table_model.events): {bad}"

    def test_no_len_current_events(self) -> None:
        src = _PAGE.read_text(encoding="utf-8")
        editor_start = src.find("class EditorPage(")
        editor_src = src[editor_start:]
        bad = [
            l.strip() for l in editor_src.splitlines()
            if "len(self._view_model.current_events)" in l
            and not l.strip().startswith("#")
        ]
        assert not bad, f"Phải dùng len(_table_model.events): {bad}"
