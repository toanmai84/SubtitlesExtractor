"""[v3.23.709] Cửa sổ cmd chớp lên sau khi mở ứng dụng: thư viện bên thứ ba chạy lệnh con không ẩn cửa sổ.

Ghi vết khởi động thật: sau khi cửa sổ chính hiện ~3 giây, việc nạp ``paddle`` chạy ``where nvcc`` và ``where ccache``
(``paddle/utils/cpp_extension/extension_utils.py``) bằng ``subprocess.check_output`` trần. Mã của ứng dụng thì mọi lời
gọi đều đã ẩn cửa sổ (quét AST: 0 lời gọi thiếu cờ).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from subtitles_extractor.infrastructure.process import hidden_process as hp

ROOT = Path(hp.__file__).resolve().parents[4]
FLAG = 0x08000000  # CREATE_NO_WINDOW


class TestKwargs:
    def test_adds_flag_only_when_caller_did_not_choose(self) -> None:
        assert hp._with_hidden_default({}, FLAG) == {"creationflags": FLAG}
        assert hp._with_hidden_default({"creationflags": 0, "shell": True}, FLAG) == {"creationflags": FLAG, "shell": True}
        assert hp._with_hidden_default({"creationflags": 0x10}, FLAG) == {"creationflags": 0x10}   # CREATE_NEW_CONSOLE
        marker = object()
        assert hp._with_hidden_default({"startupinfo": marker}, FLAG) == {"startupinfo": marker}
        assert hp._with_hidden_default({}, 0) == {}


class TestInstall:
    def test_noop_off_windows(self) -> None:
        if os.name == "nt":
            pytest.skip("chỉ kiểm nhánh không phải Windows")
        original = subprocess.Popen.__init__
        assert hp.install_hidden_console_default() is False
        assert subprocess.Popen.__init__ is original

    def test_third_party_style_call_gets_hidden_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[dict] = []

        def fake_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            seen.append(kwargs)

        monkeypatch.setattr(subprocess.Popen, "__init__", fake_init)
        monkeypatch.setattr(subprocess.Popen, "_se_hidden_console_default", False, raising=False)
        monkeypatch.setattr(hp.sys, "platform", "win32")
        monkeypatch.setattr(hp, "_CREATE_NO_WINDOW", FLAG)
        original_system = os.system
        try:
            assert hp.install_hidden_console_default() is True
            assert hp.install_hidden_console_default() is False            # cài một lần
            subprocess.Popen(["where", "nvcc"], stderr=subprocess.DEVNULL)  # đúng kiểu paddle gọi
            subprocess.Popen(["x"], creationflags=0x10)                     # nơi gọi tự chọn → giữ nguyên
            assert seen[0]["creationflags"] == FLAG and seen[1]["creationflags"] == 0x10
            assert os.system is not original_system
        finally:
            os.system = original_system


class TestWiring:
    def test_main_installs_before_heavy_imports(self) -> None:
        text = (ROOT / "main.py").read_text(encoding="utf-8")
        assert text.index("install_hidden_console_default()") < text.index("configure_all_model_stores()")
        assert text.index("install_hidden_console_default()") < text.index("bootstrap_for_gui()")

    @pytest.mark.parametrize("relative", [
        "src/subtitles_extractor/infrastructure/tts/edge_tts_subprocess.py",
        "src/subtitles_extractor/infrastructure/tts/vieneu_gpu_subprocess.py",
        "src/subtitles_extractor/infrastructure/stt/whisperx_subprocess.py",
    ])
    def test_workers_hide_their_children(self, relative: str) -> None:
        text = (ROOT / relative).read_text(encoding="utf-8")
        main_at = text.index("\ndef main()")
        assert "def _hide_child_consoles()" in text[:main_at]
        assert "_hide_child_consoles()" in text[main_at:main_at + 400]
