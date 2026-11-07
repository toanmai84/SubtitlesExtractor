"""[v3.23.673] Edge TTS đi qua proxy hệ thống + báo đúng lý do khi không kết nối được.

Chạy thật (tập 3 phim mẫu, máy sau proxy HTTP): lượt thử kết nối báo "audio rỗng" ×3 rồi
"Edge TTS không tạo được âm thanh…" — lý do thật (``ClientConnectorDNSError`` tới
speech.platform.bing.com, vì aiohttp KHÔNG tự đọc ``HTTPS_PROXY``) chỉ nằm ở log DEBUG. Sau khi
truyền ``proxy=``: 58/58 câu đọc được, xuất video lồng tiếng + phụ đề cứng thành công.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSGenerationError, TTSRequest
from subtitles_extractor.infrastructure.tts import edge_tts_subprocess as worker
from subtitles_extractor.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter
from subtitles_extractor.infrastructure.tts.edge_tts_subprocess import resolve_system_proxy
from tests.unit.infrastructure.test_dubbing_engines_v323593 import EVENTS

_PROXY_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY",
               "all_proxy", "NO_PROXY", "no_proxy")


@pytest.fixture
def clean_proxy_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize(("env", "want"), [
    ({}, None),
    ({"HTTPS_PROXY": "http://10.0.0.1:3128"}, "http://10.0.0.1:3128"),
    ({"HTTP_PROXY": "http://10.0.0.2:8080"}, "http://10.0.0.2:8080"),
    ({"HTTPS_PROXY": "10.0.0.3:3128"}, "http://10.0.0.3:3128"),          # thiếu scheme
    ({"HTTPS_PROXY": "socks5://10.0.0.4:1080"}, None),                    # aiohttp không hỗ trợ
    ({"HTTPS_PROXY": "http://10.0.0.1:3128", "NO_PROXY": ".bing.com"}, None),
    ({"HTTPS_PROXY": "http://10.0.0.1:3128", "NO_PROXY": "*"}, None),
])
def test_doc_proxy_he_thong(clean_proxy_env: pytest.MonkeyPatch, env, want) -> None:
    for name, value in env.items():
        clean_proxy_env.setenv(name, value)
    assert resolve_system_proxy() == want


class _FakeCommunicate:
    calls: list[dict] = []

    def __init__(self, text: str, voice: str, **kwargs) -> None:
        type(self).calls.append(kwargs)

    async def save(self, _path: str) -> None:
        raise OSError("Cannot connect to host speech.platform.bing.com:443 ssl:default")


@pytest.fixture
def fake_edge(monkeypatch: pytest.MonkeyPatch, clean_proxy_env) -> type[_FakeCommunicate]:
    module = types.ModuleType("edge_tts")
    module.Communicate = _FakeCommunicate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    _FakeCommunicate.calls = []
    clean_proxy_env.setenv("HTTPS_PROXY", "http://10.9.9.9:3128")
    return _FakeCommunicate


def test_worker_truyen_proxy(fake_edge, tmp_path: Path, capsys) -> None:
    code = asyncio.run(worker._synthesize("xin chào", "vi-VN-HoaiMyNeural", "+0%",
                                          str(tmp_path / "o.wav")))
    assert code == 5
    assert fake_edge.calls[-1]["proxy"] == "http://10.9.9.9:3128"
    assert "speech.platform.bing.com" in capsys.readouterr().err


def test_trong_tien_trinh_truyen_proxy_va_ghi_ly_do(fake_edge, tmp_path: Path) -> None:
    EdgeTTSAdapter._last_failure = ""
    ok = asyncio.run(EdgeTTSAdapter._run_edge_in_process(
        "xin chào", "vi-VN-HoaiMyNeural", "+0%", str(tmp_path / "o.wav")))
    assert ok is False
    assert fake_edge.calls[-1]["proxy"] == "http://10.9.9.9:3128"
    assert "speech.platform.bing.com" in EdgeTTSAdapter._last_failure


def test_thu_ket_noi_hong_bao_ly_do_that(
    fake_edge, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Đi đường trong tiến trình (không cần Python ngoài) với edge_tts giả lỗi kết nối.
    monkeypatch.setattr(EdgeTTSAdapter, "_prefer_in_process", True)
    monkeypatch.setattr(EdgeTTSAdapter, "is_available", lambda self: True)
    request = TTSRequest(events=EVENTS, speaker="vi-VN-HoaiMyNeural", normalize=False,
                         retry_count=2, retry_delay_s=0.0)
    with pytest.raises(TTSGenerationError) as info:
        EdgeTTSAdapter().generate(request, tmp_path / "out.wav")
    assert "Lỗi cuối cùng" in str(info.value)
    assert "speech.platform.bing.com" in str(info.value)
    assert any("lý do:" in r.getMessage() and "bing.com" in r.getMessage()
               for r in caplog.records)


def test_ly_do_theo_tung_cau_khong_lan_sang_cau_khac(tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.tts import edge_tts_adapter as mod

    async def one(reason: str) -> str:
        mod._LINE_FAILURE.set("")
        await asyncio.sleep(0)
        if reason:
            mod._note_edge_failure(reason)
        await asyncio.sleep(0.01)
        return mod._LINE_FAILURE.get()

    async def run() -> list[str]:
        return list(await asyncio.gather(one("lỗi A"), one("")))

    assert asyncio.run(run()) == ["lỗi A", ""]
