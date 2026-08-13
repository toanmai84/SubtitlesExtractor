"""Tests bảo vệ v3.6 editor_page bugfixes.

Bug 1: _on_text_editing mutate SubtitleEvent in-place → undo không hoạt động
Bug 2: _replace_all dùng mixed source (ViewModel + table cache) → IndexError/stale
"""
from pathlib import Path

_PAGE = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/presentation/pages/editor_page.py"
)


def _get_method_snippet(source: str, method_name: str) -> str:
    start = source.find(f"def {method_name}(self")
    end = source.find("\n    def ", start + 1)
    return source[start:end if end != -1 else len(source)]


# ---------------------------------------------------------------------------
# Bug 1: _on_text_editing không được mutate SubtitleEvent
# ---------------------------------------------------------------------------


class TestTextEditingNoMutation:
    """Bug 1: _on_text_editing KHÔNG ĐƯỢC gán .text = ... trực tiếp vào event."""

    def test_no_direct_text_assignment_in_on_text_editing(self) -> None:
        """current_events[idx].text = ... phải bị loại bỏ hoàn toàn."""
        source = _PAGE.read_text(encoding="utf-8")
        snippet = _get_method_snippet(source, "_on_text_editing")

        # Không được có pattern: .text = text (gán trực tiếp vào SubtitleEvent)
        real_assignments = [
            line for line in snippet.splitlines()
            if ".text = text" in line and not line.strip().startswith("#")
        ]
        assert len(real_assignments) == 0, (
            f"Bug 1: _on_text_editing vẫn còn gán .text trực tiếp:\n"
            + "\n".join(real_assignments)
            + "\nViệc này bypass undo system và corrupt undo history."
        )

    def test_on_text_editing_still_calls_schedule_preview(self) -> None:
        """_on_text_editing vẫn phải gọi _schedule_preview() cho real-time preview."""
        source = _PAGE.read_text(encoding="utf-8")
        snippet = _get_method_snippet(source, "_on_text_editing")
        assert "_schedule_preview()" in snippet, (
            "_on_text_editing phải vẫn gọi _schedule_preview() dù không mutate event."
        )

    def test_on_text_editing_no_current_events_call(self) -> None:
        """_on_text_editing không cần gọi current_events (tốn kém per keystroke)."""
        source = _PAGE.read_text(encoding="utf-8")
        snippet = _get_method_snippet(source, "_on_text_editing")
        real_calls = [
            line for line in snippet.splitlines()
            if "current_events" in line and not line.strip().startswith("#")
        ]
        assert len(real_calls) == 0, (
            "_on_text_editing không cần gọi current_events — "
            "tốn kém mỗi keystroke và không cần thiết sau khi bỏ mutation."
        )

    def test_on_text_focus_out_calls_update_text(self) -> None:
        """_on_text_focus_out phải gọi update_text khi text thay đổi."""
        source = _PAGE.read_text(encoding="utf-8")
        snippet = _get_method_snippet(source, "_on_text_focus_out")
        assert "update_text(" in snippet, (
            "_on_text_focus_out phải gọi update_text() — đây là điểm DUY NHẤT "
            "lưu text thay đổi vào service với undo entry."
        )

    def test_undo_not_corrupted_by_text_editing(self) -> None:
        """Verify logic: không mutate → undo hoạt động đúng."""
        from subtitles_extractor.application.services.subtitle_editor_service import (
            SubtitleEditorService,
        )
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.confidence import Confidence
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        svc = SubtitleEditorService()
        ev = SubtitleEvent(1, "Original", TimeInterval(0.0, 2.0), Confidence(0.9), 5)
        svc.load([ev])

        # Simulate WITHOUT mutation (correct behavior):
        # - _on_text_editing fires (no mutation)
        # - _on_text_focus_out fires → update_text called
        svc.update_text(0, "Edited")

        assert svc._events[0].text == "Edited"
        assert svc._is_dirty is True

        # Undo phải khôi phục về "Original"
        state = svc.undo()
        assert state.events[0].text == "Original", (
            "Undo phải khôi phục về text trước khi edit"
        )


# ---------------------------------------------------------------------------
# Bug 2: _replace_all dùng consistent source
# ---------------------------------------------------------------------------


class TestReplaceAllConsistentSource:
    """Bug 2: _replace_all phải dùng nhất quán từ _table_model."""

    def test_replace_all_uses_table_model_events(self) -> None:
        """_replace_all phải dùng _table_model.events, không dùng current_events."""
        source = _PAGE.read_text(encoding="utf-8")
        snippet = _get_method_snippet(source, "_replace_all")

        real_current_events_calls = [
            line for line in snippet.splitlines()
            if "current_events" in line and not line.strip().startswith("#")
        ]
        assert len(real_current_events_calls) == 0, (
            f"Bug 2: _replace_all vẫn dùng current_events:\n"
            + "\n".join(real_current_events_calls)
            + "\nPhải dùng _table_model.events để nhất quán với cache."
        )
        assert "_table_model.events" in snippet, (
            "_replace_all phải dùng self._table_model.events làm nguồn dữ liệu chính."
        )

    def test_replace_all_has_length_guard(self) -> None:
        """_replace_all phải có guard kiểm tra cache và events cùng độ dài."""
        source = _PAGE.read_text(encoding="utf-8")
        snippet = _get_method_snippet(source, "_replace_all")

        # Phải có kiểm tra len() để tránh IndexError
        assert "len(cache)" in snippet or "len(" in snippet, (
            "_replace_all phải có length safety check để tránh IndexError "
            "khi cache và events mất sync."
        )

    def test_replace_all_rebuilds_stale_cache(self) -> None:
        """_replace_all phải rebuild cache khi phát hiện stale."""
        source = _PAGE.read_text(encoding="utf-8")
        snippet = _get_method_snippet(source, "_replace_all")

        # Phải có fallback rebuild cache
        assert ".lower()" in snippet and "len(cache) != len(" in snippet, (
            "_replace_all phải rebuild cache khi len(cache) != len(events)."
        )


# ---------------------------------------------------------------------------
# Integration: text edit → focus out flow đúng
# ---------------------------------------------------------------------------


class TestTextEditFocusOutFlow:
    """Đảm bảo flow text edit → focus out → undo hoạt động đúng."""

    def test_service_update_text_sets_dirty_and_pushes_undo(self) -> None:
        """update_text phải set _is_dirty=True và push undo entry."""
        from subtitles_extractor.application.services.subtitle_editor_service import (
            SubtitleEditorService,
        )
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.confidence import Confidence
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        svc = SubtitleEditorService()
        svc.load([SubtitleEvent(1, "Before", TimeInterval(0.0, 2.0), Confidence(0.9), 5)])

        undo_depth_before = len(svc._undo_stack)
        svc.update_text(0, "After")

        assert svc._is_dirty is True, "update_text phải set _is_dirty=True"
        assert len(svc._undo_stack) == undo_depth_before + 1, (
            "update_text phải push 1 undo entry"
        )

    def test_undo_after_update_text_restores_original(self) -> None:
        """Undo sau update_text phải khôi phục text gốc chính xác."""
        from subtitles_extractor.application.services.subtitle_editor_service import (
            SubtitleEditorService,
        )
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.confidence import Confidence
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        svc = SubtitleEditorService()
        svc.load([
            SubtitleEvent(1, "Row 1", TimeInterval(0.0, 1.0), Confidence(0.9), 5),
            SubtitleEvent(2, "Row 2", TimeInterval(2.0, 3.0), Confidence(0.9), 5),
        ])

        svc.update_text(0, "Modified Row 1")
        svc.update_text(1, "Modified Row 2")

        state = svc.undo()
        assert state.events[1].text == "Row 2", (
            "Undo lần 1 phải khôi phục Row 2 về 'Row 2'"
        )

        state = svc.undo()
        assert state.events[0].text == "Row 1", (
            "Undo lần 2 phải khôi phục Row 1 về 'Row 1'"
        )
