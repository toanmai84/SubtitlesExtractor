"""Unit tests bảo vệ v3.6 bugfix: xuất SRT không treo ứng dụng.

Dùng AST/text analysis — không import runtime để tránh dependency chain.

Bảo vệ 3 lỗi đã sửa:
1. _ExportRunnable.setAutoDelete(False) → ngăn PyQt6 GC huỷ signals.
2. ViewModel lưu strong reference _export_worker cho đến khi signal delivery.
3. Watchdog timer 60s là safety net khi signal bị drop bất ngờ.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_VM_PATH = (
    _PROJECT_ROOT / "src" / "subtitles_extractor"
    / "presentation" / "view_models" / "editor_page_view_model.py"
)
_PAGE_PATH = (
    _PROJECT_ROOT / "src" / "subtitles_extractor"
    / "presentation" / "pages" / "editor_page.py"
)


@pytest.fixture(scope="module")
def vm_source() -> str:
    return _VM_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page_source() -> str:
    return _PAGE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Bug 1: _ExportRunnable.setAutoDelete(False)
# ---------------------------------------------------------------------------


class TestExportRunnableAutoDelete:
    """_ExportRunnable phải có setAutoDelete(False) trong __init__ để ViewModel
    quản lý lifetime và ngăn PyQt6 GC huỷ signals trước khi queued event
    được main thread xử lý."""

    def test_set_auto_delete_false_present_in_runnable_init(
        self, vm_source: str
    ) -> None:
        """setAutoDelete(False) phải xuất hiện trong class _ExportRunnable."""
        # Tìm đoạn định nghĩa _ExportRunnable
        start = vm_source.find("class _ExportRunnable")
        assert start != -1, "_ExportRunnable class không tìm thấy"

        # Lấy thân class (đến class tiếp theo hoặc cuối file)
        next_class = vm_source.find("\nclass ", start + 1)
        snippet = vm_source[start:next_class if next_class != -1 else len(vm_source)]

        assert "setAutoDelete(False)" in snippet, (
            "Bug fix v3.6: _ExportRunnable PHẢI gọi self.setAutoDelete(False). "
            "Nếu không, Qt auto-delete runnable sau run() → PyQt6 GC huỷ "
            "signals → queued signal bị cancelled → _on_export_worker_success "
            "không bao giờ được gọi → _is_busy kẹt True → TREO UI."
        )

    def test_auto_delete_called_in_init_not_run(self, vm_source: str) -> None:
        """setAutoDelete phải được gọi trong __init__, không phải trong run()."""
        start = vm_source.find("class _ExportRunnable")
        next_class = vm_source.find("\nclass ", start + 1)
        snippet = vm_source[start:next_class if next_class != -1 else len(vm_source)]

        init_start = snippet.find("def __init__")
        run_start = snippet.find("def run")
        auto_delete_pos = snippet.find("setAutoDelete(False)")

        assert init_start < auto_delete_pos < run_start, (
            "setAutoDelete(False) phải được gọi bên trong __init__, "
            "trước khi run() được định nghĩa."
        )


# ---------------------------------------------------------------------------
# Bug 1 (tiếp): ViewModel lưu strong reference đến worker
# ---------------------------------------------------------------------------


class TestViewModelExportWorkerReference:
    """EditorPageViewModel phải lưu strong reference đến _ExportRunnable
    sau khi gọi thread_pool.start() để ngăn Python GC."""

    def test_export_worker_attribute_initialized_in_init(
        self, vm_source: str
    ) -> None:
        """_export_worker phải được khởi tạo trong __init__."""
        init_start = vm_source.find("class EditorPageViewModel")
        # Tìm __init__ của class
        init_def = vm_source.find("def __init__(self, container", init_start)
        # Tìm def tiếp theo sau __init__
        next_def = vm_source.find("\n    def ", init_def + 1)
        init_snippet = vm_source[init_def:next_def]

        assert "_export_worker" in init_snippet, (
            "_export_worker phải được khởi tạo trong __init__ "
            "(ví dụ: self._export_worker = None)"
        )

    def test_export_worker_assigned_in_export_to_file(
        self, vm_source: str
    ) -> None:
        """export_to_file phải gán worker vào self._export_worker."""
        method_start = vm_source.find("def export_to_file(self,")
        assert method_start != -1, "export_to_file không tìm thấy"

        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method]

        assert "self._export_worker = worker" in snippet or \
               "self._export_worker=worker" in snippet, (
            "Bug fix v3.6: export_to_file PHẢI gán worker vào self._export_worker "
            "để giữ strong Python reference. Không có reference → GC huỷ worker.signals "
            "→ queued signal bị drop → treo UI."
        )

    def test_export_worker_cleared_in_success_handler(
        self, vm_source: str
    ) -> None:
        """_on_export_worker_success phải xoá _export_worker reference."""
        method_start = vm_source.find("def _on_export_worker_success(self,")
        assert method_start != -1

        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method]

        assert "self._export_worker = None" in snippet, (
            "_on_export_worker_success phải set self._export_worker = None "
            "sau khi signal được deliver để tránh memory leak."
        )

    def test_export_worker_cleared_in_error_handler(
        self, vm_source: str
    ) -> None:
        """_on_export_worker_error phải xoá _export_worker reference."""
        method_start = vm_source.find("def _on_export_worker_error(self,")
        assert method_start != -1

        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method]

        assert "self._export_worker = None" in snippet, (
            "_on_export_worker_error phải set self._export_worker = None "
            "sau khi signal được deliver."
        )


# ---------------------------------------------------------------------------
# Bug 2: Watchdog timer
# ---------------------------------------------------------------------------


class TestExportWatchdogTimer:
    """Watchdog timer 60s là safety net: nếu signal bị drop vì bất kỳ lý do nào,
    watchdog force-reset busy state sau 60 giây để UI không bị treo vĩnh viễn."""

    def test_watchdog_initialized_in_init(self, vm_source: str) -> None:
        """_export_watchdog phải được khởi tạo trong __init__."""
        init_def = vm_source.find("def __init__(self, container")
        next_def = vm_source.find("\n    def ", init_def + 1)
        init_snippet = vm_source[init_def:next_def]

        assert "_export_watchdog" in init_snippet, (
            "_export_watchdog QTimer phải được khởi tạo trong __init__."
        )
        assert "60_000" in init_snippet or "60000" in init_snippet, (
            "Watchdog interval phải là 60 giây (60_000 ms)."
        )
        assert "setSingleShot(True)" in init_snippet, (
            "Watchdog phải là single-shot timer (chỉ fire 1 lần)."
        )

    def test_watchdog_started_in_export_to_file(self, vm_source: str) -> None:
        """export_to_file phải start watchdog."""
        method_start = vm_source.find("def export_to_file(self,")
        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method]

        assert "watchdog.start()" in snippet or "_export_watchdog.start()" in snippet, (
            "export_to_file phải gọi self._export_watchdog.start() "
            "để kích hoạt safety timer."
        )

    def test_watchdog_stopped_in_success_handler(self, vm_source: str) -> None:
        """_on_export_worker_success phải stop watchdog."""
        method_start = vm_source.find("def _on_export_worker_success(self,")
        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method]

        assert "watchdog.stop()" in snippet or "_export_watchdog.stop()" in snippet, (
            "_on_export_worker_success phải dừng watchdog khi export thành công."
        )

    def test_watchdog_stopped_in_error_handler(self, vm_source: str) -> None:
        """_on_export_worker_error phải stop watchdog."""
        method_start = vm_source.find("def _on_export_worker_error(self,")
        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method]

        assert "watchdog.stop()" in snippet or "_export_watchdog.stop()" in snippet, (
            "_on_export_worker_error phải dừng watchdog khi có lỗi."
        )

    def test_watchdog_timeout_handler_exists(self, vm_source: str) -> None:
        """_on_export_watchdog_timeout phải tồn tại."""
        assert "_on_export_watchdog_timeout" in vm_source, (
            "Phải có method _on_export_watchdog_timeout để xử lý khi watchdog timeout."
        )

    def test_watchdog_timeout_resets_busy_and_emits_error(
        self, vm_source: str
    ) -> None:
        """_on_export_watchdog_timeout phải reset _is_busy và emit error."""
        method_start = vm_source.find("def _on_export_watchdog_timeout(self)")
        assert method_start != -1, "_on_export_watchdog_timeout không tìm thấy"

        next_method = vm_source.find("\n    def ", method_start + 1)
        snippet = vm_source[method_start:next_method if next_method != -1 else len(vm_source)]

        assert "_set_busy(False)" in snippet or "_is_busy" in snippet, (
            "Watchdog timeout handler phải reset busy state."
        )
        assert "error_occurred.emit" in snippet, (
            "Watchdog timeout handler phải emit error_occurred "
            "để thông báo cho user."
        )


# ---------------------------------------------------------------------------
# Bug 3: _show_error trong editor_page phải re-enable nút
# ---------------------------------------------------------------------------


class TestShowErrorReenablesExportButton:
    """_show_error phải gọi setEnabled(...) để re-enable nút Export.

    Trước đây chỉ setText() → nút vẫn disabled sau khi lỗi.
    """

    def test_show_error_calls_set_enabled(self, page_source: str) -> None:
        """_show_error phải có setEnabled để re-enable export button."""
        method_start = page_source.find("def _show_error(self, message: str)")
        assert method_start != -1, "_show_error không tìm thấy trong editor_page"

        next_method = page_source.find("\n    def ", method_start + 1)
        snippet = page_source[method_start:next_method]

        assert "setEnabled(" in snippet, (
            "Bug fix v3.6: _show_error PHẢI gọi setEnabled() trên export button "
            "để re-enable nó sau lỗi. Trước đây chỉ có setText() → nút kẹt disabled."
        )
