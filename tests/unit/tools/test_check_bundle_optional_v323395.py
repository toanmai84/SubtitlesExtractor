"""[v3.23.395] Test: model prefetch là TÙY CHỌN — bundle nhỏ không bị báo lỗi giả."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[3] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import check_bundle as cb  # noqa: E402


def _make_bundle_with_required(root: Path) -> Path:
    """Dựng bundle giả có ĐỦ thành phần bắt buộc (không có model prefetch)."""
    bundle = root / "SubtitlesExtractor"
    for rel, _ in cb.REQUIRED_ENTRIES:
        target = bundle / rel
        if target.suffix:  # tệp
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("")
        else:  # thư mục
            target.mkdir(parents=True, exist_ok=True)
    return bundle


def test_missing_prefetch_models_is_not_a_failure(tmp_path: Path) -> None:
    bundle = _make_bundle_with_required(tmp_path)
    issues = cb.check_bundle(bundle)
    required_missing = [i for i in issues if i.missing and not i.optional]
    optional_missing = [i for i in issues if i.missing and i.optional]
    assert required_missing == []
    # Hai model prefetch (paddle official_models + hf hub) là tùy chọn.
    assert len(optional_missing) == 2
    assert all(i.optional for i in optional_missing)


def test_missing_required_still_fails(tmp_path: Path) -> None:
    bundle = _make_bundle_with_required(tmp_path)
    # Xóa một thành phần BẮT BUỘC → phải bị bắt.
    ffmpeg = bundle / "_internal/vendor/ffmpeg/ffmpeg.exe"
    ffmpeg.unlink()
    issues = cb.check_bundle(bundle)
    required_missing = [i for i in issues if i.missing and not i.optional]
    assert any("ffmpeg" in i.path for i in required_missing)


def test_unexpected_torch_flagged(tmp_path: Path) -> None:
    bundle = _make_bundle_with_required(tmp_path)
    (bundle / "_internal/torch").mkdir(parents=True)
    issues = cb.check_bundle(bundle)
    unexpected = [i for i in issues if not i.missing]
    assert any("torch" in i.path for i in unexpected)


def test_prefetch_models_no_longer_required() -> None:
    """Model prefetch KHÔNG còn trong REQUIRED_ENTRIES (đã chuyển sang OPTIONAL)."""
    required_paths = {rel for rel, _ in cb.REQUIRED_ENTRIES}
    assert "_internal/models/paddle/official_models" not in required_paths
    assert "_internal/models/huggingface/hub" not in required_paths
    optional_paths = {rel for rel, _ in cb.OPTIONAL_ENTRIES}
    assert "_internal/models/paddle/official_models" in optional_paths
