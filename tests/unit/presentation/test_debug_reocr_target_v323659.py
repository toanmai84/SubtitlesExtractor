"""[v3.23.659] OCR lại 1 khung (trang Gỡ lỗi): kết quả phải ghi vào khung ĐÃ chọn, không phải khung đang xem."""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.presentation.view_models.debug_page_view_model import DebugPageViewModel


class _Sig:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


def _fake_vm(frames, idx):
    return SimpleNamespace(
        _frames=frames, _current_idx=idx,
        _container=SimpleNamespace(apply_settings_changes=lambda: None,
                                   translator=SimpleNamespace(translate=lambda k: k)),
        frame_changed=_Sig(), live_images_ready=_Sig(), action_success=_Sig(), is_busy=_Sig(),
    )


def test_result_goes_to_requested_frame_after_navigation():
    frames = ["f0", "f1", "f2", "f3"]
    vm = _fake_vm(frames, 1)
    vm._reocr_target = (frames, 1)
    vm._current_idx = 3  # người dùng bấm "Sau" trong lúc chờ
    DebugPageViewModel._on_worker_success(vm, "NEW", None, None, "", "{}", False)
    assert frames == ["f0", "NEW", "f2", "f3"]
    assert vm.frame_changed.calls == []
    assert vm.is_busy.calls[-1] == (False,)


def test_result_dropped_when_new_file_loaded():
    old = ["a", "b"]
    vm = _fake_vm(["x", "y", "z"], 1)
    vm._reocr_target = (old, 1)
    DebugPageViewModel._on_worker_success(vm, "NEW", None, None, "", "{}", False)
    assert vm._frames == ["x", "y", "z"] and old == ["a", "b"]
    assert vm.is_busy.calls[-1] == (False,)


def test_same_frame_updates_view():
    frames = ["f0", "f1"]
    vm = _fake_vm(frames, 1)
    vm._reocr_target = (frames, 1)
    DebugPageViewModel._on_worker_success(vm, "NEW", "qi", "qo", "", "{}", False)
    assert frames[1] == "NEW" and len(vm.frame_changed.calls) == 1
