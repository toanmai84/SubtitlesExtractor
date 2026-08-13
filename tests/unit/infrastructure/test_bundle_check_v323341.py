"""Test kiểm bản đóng gói — v3.23.341.

SỰ CỐ ĐƯỢC CHỐNG: v3.23.340 phát hiện ``whisperx_subprocess.py`` không vào bundle. Log
build **không hề nói gì** vì ``datas.append(...)`` trong tệp spec không in ra dòng nào,
nên chỉ biết khi tính năng hỏng lúc chạy — với thông điệp "thoát mã 2" rất khó chẩn đoán.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _load_tool():
    import importlib.util

    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    path = root / "tools" / "check_bundle.py"
    if not path.is_file():
        pytest.skip("Không tìm thấy tools/check_bundle.py")
    spec = importlib.util.spec_from_file_location("_bundle_tool", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bundle_tool"] = module
    spec.loader.exec_module(module)
    return module


def _make_complete_bundle(root: Path, tool) -> None:
    """Tạo bundle có đủ mọi thành phần bắt buộc."""
    for relative, _reason in tool.REQUIRED_ENTRIES:
        target = root / relative
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)


def test_detects_missing_whisperx_worker(tmp_path: Path) -> None:
    """Tái hiện đúng sự cố v3.23.340."""
    tool = _load_tool()
    _make_complete_bundle(tmp_path, tool)
    (tmp_path / "_internal/subtitles_extractor/infrastructure/stt"
     "/whisperx_subprocess.py").unlink()

    issues = tool.check_bundle(tmp_path)
    assert any("whisperx_subprocess" in issue.path for issue in issues)
    assert all(issue.missing for issue in issues)


def test_complete_bundle_has_no_issues(tmp_path: Path) -> None:
    """KHÔNG được báo động giả với bundle đủ."""
    tool = _load_tool()
    _make_complete_bundle(tmp_path, tool)
    assert tool.check_bundle(tmp_path) == []


def test_detects_torch_leaking_into_bundle(tmp_path: Path) -> None:
    """torch trong bundle = đóng gói sai; nó thuộc môi trường riêng."""
    tool = _load_tool()
    _make_complete_bundle(tmp_path, tool)
    (tmp_path / "_internal" / "torch").mkdir(parents=True)

    issues = tool.check_bundle(tmp_path)
    torch_issues = [i for i in issues if i.path.endswith("torch")]
    assert torch_issues
    assert not torch_issues[0].missing  # có mà không nên có


def test_empty_bundle_reports_everything(tmp_path: Path) -> None:
    tool = _load_tool()
    issues = tool.check_bundle(tmp_path)
    assert len(issues) == len(tool.REQUIRED_ENTRIES)


@pytest.mark.parametrize(
    "worker",
    ["whisperx_subprocess.py", "edge_tts_subprocess.py"],
)
def test_both_subprocess_workers_are_required(worker: str) -> None:
    """Cả hai worker chạy bằng ``python <script>`` đều phải có trong bundle."""
    tool = _load_tool()
    paths = [relative for relative, _ in tool.REQUIRED_ENTRIES]
    assert any(worker in path for path in paths)


def test_every_issue_explains_consequence() -> None:
    """Mỗi mục phải nêu HỆ QUẢ, để người đọc biết vì sao nó quan trọng."""
    tool = _load_tool()
    for _relative, reason in tool.REQUIRED_ENTRIES:
        assert reason
        assert len(reason) > 20


def test_build_script_runs_bundle_check() -> None:
    """Phải chạy TRONG build, nếu không sẽ chẳng ai nhớ chạy tay."""
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    script = root / "build_windows.bat"
    if not script.is_file():
        pytest.skip("Không tìm thấy build_windows.bat")
    assert "check_bundle.py" in script.read_text(encoding="utf-8", errors="replace")


def test_missing_bundle_directory_is_not_an_error(tmp_path: Path) -> None:
    """Chưa build thì không được coi là lỗi."""
    tool = _load_tool()
    assert tool.find_bundle_root(tmp_path) is None
