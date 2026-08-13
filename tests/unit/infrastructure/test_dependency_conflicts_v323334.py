"""Test công cụ kiểm xung đột phụ thuộc — v3.23.334.

BỐI CẢNH: v3.23.333 phát hiện ``whisperx`` ghim ``huggingface-hub<1.0.0`` trong khi ứng
dụng dùng ``1.24.0``. Nguy hiểm ở chỗ PaddleOCR/PaddleX/VieNeu KHÔNG ràng buộc phiên bản
gói đó, nên pip **hạ cấp âm thầm** — hỏng mà không báo lỗi.

Công cụ phải làm đúng hai việc, và bộ test này canh cả hai:
    * BẮT ĐÚNG khi ràng buộc thực sự vi phạm.
    * KHÔNG BÁO ĐỘNG GIẢ với ràng buộc có điều kiện không áp dụng — đây là chỗ dễ sai
      nhất, vì metadata thường có ``numpy<2.3.0 ; python_version == "3.10"`` trông rất
      giống xung đột nhưng thực ra không áp dụng.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("packaging", reason="Cần packaging để phân tích ràng buộc")


def _load_tool():
    """Nạp công cụ từ thư mục tools/ (không phải package cài đặt)."""
    import importlib.util

    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    path = root / "tools" / "check_dependency_conflicts.py"
    if not path.is_file():
        pytest.skip("Không tìm thấy tools/check_dependency_conflicts.py")

    spec = importlib.util.spec_from_file_location("_dep_tool", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dep_tool"] = module
    spec.loader.exec_module(module)
    return module


#: Môi trường THẬT của người dùng, lấy từ log build.
_USER_ENV = {
    "numpy": "2.3.5",
    "huggingface-hub": "1.24.0",
    "onnxruntime": "1.28.0",
    "scipy": "1.18.0",
    "pandas": "3.0.5",
    "pillow": "12.3.0",
    "tokenizers": "0.23.1",
    "aiohttp": "3.14.3",
}


# ── Chuẩn hoá tên gói ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("huggingface_hub", "huggingface-hub"),
        ("Huggingface-Hub", "huggingface-hub"),
        ("scikit_learn", "scikit-learn"),
        ("PySide6-Fluent-Widgets", "pyside6-fluent-widgets"),
        ("edge.tts", "edge-tts"),
    ],
)
def test_name_normalisation(raw: str, expected: str) -> None:
    """Tên gói phải chuẩn hoá — nếu không sẽ bỏ sót xung đột do khác cách viết."""
    assert _load_tool().normalise(raw) == expected


# ── Bắt đúng xung đột ────────────────────────────────────────────────────────
def test_detects_the_real_whisperx_conflict() -> None:
    """Tái hiện chính xung đột đã xảy ra — nếu test này hỏng là công cụ vô dụng."""
    tool = _load_tool()
    conflicts = tool.find_conflicts(
        "whisperx", ["huggingface-hub<1.0.0", "torch~=2.8.0"], _USER_ENV
    )
    assert any(c.dependency == "huggingface-hub" for c in conflicts)


def test_conflict_message_names_the_consequence() -> None:
    """Thông điệp phải nói HỆ QUẢ (pip sẽ đổi phiên bản), không chỉ báo lệch."""
    tool = _load_tool()
    conflicts = tool.find_conflicts("x", ["numpy<2.0"], _USER_ENV)
    assert conflicts
    described = conflicts[0].describe()
    assert "2.3.5" in described and "<2.0" in described
    assert "ĐỔI PHIÊN BẢN" in described


@pytest.mark.parametrize(
    "constraint",
    ["numpy<2.0", "numpy==1.26.0", "onnxruntime<1.20", "pillow>=99.0"],
)
def test_detects_various_violated_constraints(constraint: str) -> None:
    tool = _load_tool()
    assert tool.find_conflicts("x", [constraint], _USER_ENV)


# ── KHÔNG báo động giả ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "constraint",
    ["numpy>=1.26", "numpy>=2.1,<3.0", "onnxruntime!=1.24.0", "pillow>=10.3,<13.0"],
)
def test_satisfied_constraints_are_not_reported(constraint: str) -> None:
    tool = _load_tool()
    assert not tool.find_conflicts("x", [constraint], _USER_ENV)


def test_markers_for_other_python_versions_are_ignored() -> None:
    """CHỖ DỄ SAI NHẤT: ``numpy<2.3.0 ; python_version == "3.10"`` KHÔNG phải xung đột.

    Đọc bằng mắt rất dễ tưởng là xung đột. Công cụ phải đánh giá điều kiện trước.
    """
    tool = _load_tool()
    other_version = "3.10" if sys.version_info[:2] != (3, 10) else "3.9"
    conflicts = tool.find_conflicts(
        "fastembed",
        [f'numpy (>=1.21,<2.3.0) ; python_version == "{other_version}"'],
        _USER_ENV,
    )
    assert not conflicts


def test_extra_requirements_are_ignored() -> None:
    """Ràng buộc thuộc ``extra`` chỉ cài khi yêu cầu tường minh — không tính."""
    tool = _load_tool()
    assert not tool.find_conflicts(
        "vieneu", ['torch (<1.0) ; extra == "legacy"'], _USER_ENV
    )


def test_unknown_dependency_is_ignored() -> None:
    """Gói chưa cài thì không có gì để xung đột."""
    tool = _load_tool()
    assert not tool.find_conflicts("x", ["goi-khong-ton-tai<1.0"], _USER_ENV)


def test_dependency_without_specifier_is_ignored() -> None:
    tool = _load_tool()
    assert not tool.find_conflicts("x", ["numpy"], _USER_ENV)


def test_malformed_requirement_does_not_crash() -> None:
    tool = _load_tool()
    assert tool.find_conflicts("x", ["=== không hợp lệ ==="], _USER_ENV) == []


# ── Gói cố ý cài riêng ───────────────────────────────────────────────────────
def test_isolated_packages_are_declared() -> None:
    """whisperx + torch cố ý cài riêng — công cụ phải biết để không báo như lỗi."""
    tool = _load_tool()
    for package in ("whisperx", "torch", "torchaudio", "torchvision"):
        assert package in tool.ISOLATED_PACKAGES


# ── Hợp đồng với build script ────────────────────────────────────────────────
def test_build_script_runs_the_checker() -> None:
    """Kiểm xung đột phải chạy TRONG build, nếu không sẽ chẳng ai nhớ chạy tay."""
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    script = root / "build_windows.bat"
    if not script.is_file():
        pytest.skip("Không tìm thấy build_windows.bat")
    text = script.read_text(encoding="utf-8", errors="replace")
    assert "check_dependency_conflicts.py" in text
