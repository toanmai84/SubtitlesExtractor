"""[v3.23.717] Công cụ tải binary vendor: lấy libmpv từ kho gương SourceForge khi GitHub chặn trang phát hành.

Môi trường kiểm thử của dự án (proxy chỉ cho tải tệp) trả 403 cho cả API lẫn trang phát hành GitHub → hai test vendor
luôn bị bỏ qua. Nay công cụ đọc nguồn cấp RSS của kho gương chính thức cùng tác giả.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "fetch_vendor_binaries.py"

_RSS = """<rss><channel>
<item><link>https://sourceforge.net/projects/mpv-player-windows/files/libmpv/mpv-dev-x86_64-v3-20261108-git-55d3feb45c.7z/download</link></item>
<item><link>https://sourceforge.net/projects/mpv-player-windows/files/libmpv/mpv-dev-x86_64-20261101-git-48fca90e8c.7z/download</link></item>
<item><link>https://sourceforge.net/projects/mpv-player-windows/files/libmpv/mpv-dev-x86_64-20261108-git-55d3feb45c.7z/download</link></item>
<item><link>https://sourceforge.net/projects/mpv-player-windows/files/libmpv/mpv-dev-aarch64-20261108-git-55d3feb45c.7z/download</link></item>
</channel></rss>"""


@pytest.fixture
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_vendor_binaries_v323717", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _serve(tool: ModuleType, monkeypatch: pytest.MonkeyPatch, pages: dict[str, str]) -> list[str]:
    asked: list[str] = []

    def fake_get(url: str) -> bytes:
        asked.append(url)
        for prefix, body in pages.items():
            if url.startswith(prefix):
                return body.encode()
        raise tool.FetchError(f"Không tải được {url}: HTTP Error 403: Forbidden")

    monkeypatch.setattr(tool, "_http_get", fake_get)
    return asked


def test_falls_back_to_sourceforge_when_github_is_blocked(tool: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _serve(tool, monkeypatch, {tool.MPV_SOURCEFORGE_RSS: _RSS})
    name, url = tool.resolve_latest_mpv_asset()
    assert name == "mpv-dev-x86_64-20261108-git-55d3feb45c.7z"          # bản thường mới nhất, không lấy -v3 / aarch64
    assert url == f"{tool.MPV_SOURCEFORGE_FILES}/{name}/download"
    assert asked[0].startswith("https://api.github.com") and asked[-1] == tool.MPV_SOURCEFORGE_RSS


def test_accepts_v3_build_only_when_no_plain_build_is_listed(tool: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    only_v3 = "\n".join(line for line in _RSS.splitlines() if "-v3-" in line or "rss>" in line or "channel" in line)
    _serve(tool, monkeypatch, {tool.MPV_SOURCEFORGE_RSS: only_v3})
    assert tool.resolve_latest_mpv_asset()[0] == "mpv-dev-x86_64-v3-20261108-git-55d3feb45c.7z"


def test_reports_clearly_when_every_source_fails(tool: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(tool, monkeypatch, {tool.MPV_SOURCEFORGE_RSS: "<rss></rss>"})
    with pytest.raises(tool.FetchError, match="SourceForge"):
        tool.resolve_latest_mpv_asset()


def test_sourceforge_gets_a_client_name_it_accepts(tool: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"ok"

    def fake_urlopen(request, timeout):  # noqa: ANN001, ARG001
        seen.append(request.get_header("User-agent"))
        return _Response()

    monkeypatch.setattr(tool.urllib.request, "urlopen", fake_urlopen)
    tool._http_get("https://sourceforge.net/projects/mpv-player-windows/rss?path=/libmpv")
    tool._http_get("https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/x.zip")
    assert seen == [tool.SOURCEFORGE_USER_AGENT, tool.USER_AGENT]
