"""[v3.23.397] Test logic thuần của setup_embedded_python (không đụng mạng)."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[3] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import setup_embedded_python as sep  # noqa: E402


def test_url_format() -> None:
    url = sep.embedded_python_url("3.11.9")
    assert url == (
        "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
    )


def test_current_version_shape() -> None:
    v = sep.current_python_version()
    parts = v.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_enable_site_uncomments(tmp_path: Path) -> None:
    pth = tmp_path / "python311._pth"
    pth.write_text("python311.zip\n.\n#import site\n", encoding="utf-8")
    assert sep.enable_site_in_pth(pth) is True
    content = pth.read_text(encoding="utf-8")
    assert "import site" in content
    assert "#import site" not in content


def test_enable_site_appends_if_absent(tmp_path: Path) -> None:
    pth = tmp_path / "python311._pth"
    pth.write_text("python311.zip\n.\n", encoding="utf-8")
    sep.enable_site_in_pth(pth)
    assert "import site" in pth.read_text(encoding="utf-8")


def test_enable_site_already_enabled_is_noop(tmp_path: Path) -> None:
    pth = tmp_path / "python312._pth"
    pth.write_text("python312.zip\n.\nimport site\n", encoding="utf-8")
    sep.enable_site_in_pth(pth)
    # Không nhân đôi dòng.
    assert pth.read_text(encoding="utf-8").count("import site") == 1


def test_find_pth(tmp_path: Path) -> None:
    (tmp_path / "python311._pth").write_text("", encoding="utf-8")
    found = sep._find_pth(tmp_path)
    assert found is not None
    assert found.name == "python311._pth"


def test_find_available_uses_exact_when_present() -> None:
    # Bản chính xác tồn tại → dùng luôn.
    ver = sep.find_available_embed_version(
        "3.12.10", exists=lambda url: "3.12.10" in url
    )
    assert ver == "3.12.10"


def test_find_available_falls_back_to_lower_micro() -> None:
    # 3.12.13 không có (security-only) → dò xuống 3.12.10.
    available = {"3.12.10", "3.12.9", "3.12.0"}
    ver = sep.find_available_embed_version(
        "3.12.13",
        exists=lambda url: any(f"/{v}/" in url for v in available),
    )
    assert ver == "3.12.10"


def test_find_available_none_when_nothing() -> None:
    ver = sep.find_available_embed_version("3.12.13", exists=lambda url: False)
    assert ver is None
