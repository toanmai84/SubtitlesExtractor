"""[v3.23.628] Lỗi tìm thấy khi CHẠY VieNeu 3.8.3 thật (onnxruntime 1.31, scipy 1.18).

1. onnxruntime từ chối graph có external data khi cache HuggingFace là symlink.
2. ``sys.modules["torch"] = None`` (blocker của PaddleOCR) làm scipy sập ở khâu master âm thanh.
3. ImportError không tên của chế độ standard/turbo bị báo là "VieNeu chưa cài".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from subtitles_extractor.infrastructure.tts.hf_cache_fix import (
    install_hf_materialize_patch,
    materialize_symlink,
)
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    describe_vieneu_import_error,
)
from subtitles_extractor.infrastructure.torch_import_blocker import (
    torch_none_bypassed,
    without_torch_block,
)


def _hf_layout(tmp_path: Path) -> tuple[Path, Path]:
    blob = tmp_path / "blobs" / "abc123"
    blob.parent.mkdir()
    blob.write_bytes(b"external-data")
    link = tmp_path / "snapshots" / "rev" / "x.data"
    link.parent.mkdir(parents=True)
    link.symlink_to(os.path.relpath(blob, link.parent))
    return blob, link


@pytest.mark.skipif(os.name == "nt", reason="symlink cần quyền đặc biệt trên Windows")
class TestHfSymlinks:
    def test_symlink_becomes_a_real_file_in_the_same_directory(self, tmp_path: Path) -> None:
        blob, link = _hf_layout(tmp_path)
        assert link.is_symlink() and link.resolve().parent != link.parent  # lỗi của ORT
        assert materialize_symlink(link) == str(link)
        assert not link.is_symlink()
        assert link.read_bytes() == b"external-data"
        assert link.resolve().parent == link.parent                       # ORT chấp nhận
        assert os.stat(link).st_ino == os.stat(blob).st_ino               # hard link: không tốn chỗ
        assert not list(link.parent.glob("*.materialize_tmp"))

    def test_regular_files_and_missing_paths_are_left_alone(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain.json"
        plain.write_text("{}")
        assert materialize_symlink(plain) == str(plain) and plain.read_text() == "{}"
        assert materialize_symlink(tmp_path / "none") == str(tmp_path / "none")

    def test_patch_wraps_hf_hub_download_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import huggingface_hub

        _blob, link = _hf_layout(tmp_path)
        calls: list[tuple] = []

        def fake_download(*args, **kwargs):
            calls.append(args)
            return str(link)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        assert install_hf_materialize_patch() and install_hf_materialize_patch()
        wrapped = huggingface_hub.hf_hub_download
        assert wrapped is not fake_download
        assert install_hf_materialize_patch() and huggingface_hub.hf_hub_download is wrapped
        assert wrapped("repo", "x.data") == str(link) and calls == [("repo", "x.data")]
        assert not link.is_symlink()


class TestTorchNoneBypass:
    def test_entries_are_hidden_inside_and_restored_after(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        with torch_none_bypassed():
            assert "torch" not in sys.modules
        assert "torch" in sys.modules and sys.modules["torch"] is None

    def test_a_real_module_is_never_touched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = SimpleNamespace(Tensor=object)
        monkeypatch.setitem(sys.modules, "torch", fake)
        with torch_none_bypassed():
            assert sys.modules["torch"] is fake
        assert sys.modules["torch"] is fake

    def test_exception_still_restores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)

        @without_torch_block
        def boom() -> None:
            assert "torch" not in sys.modules
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            boom()
        assert sys.modules["torch"] is None

    def test_final_master_survives_the_paddle_blocker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tái hiện: ``torch=None`` + scipy mới → ``AttributeError: 'NoneType' ... 'Tensor'``."""
        from subtitles_extractor.application.use_cases.resynthesize_lines import (
            ResynthesizeLinesUseCase,
        )
        from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest

        monkeypatch.setitem(sys.modules, "torch", None)
        t = np.linspace(0, 4, 48000 * 4, endpoint=False)
        master = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        out = tmp_path / "m.wav"
        ResynthesizeLinesUseCase._write_master(  # noqa: SLF001
            master, 48000, TTSRequest(events=[], normalize=True), out
        )
        audio, sr = sf.read(str(out))
        assert sr == 48000 and len(audio) == len(master) and float(np.abs(audio).max()) <= 1.0
        assert sys.modules["torch"] is None               # blocker của PaddleOCR còn nguyên


class TestImportErrorMessage:
    def test_installed_sdk_without_llama_is_not_reported_as_missing(self) -> None:
        error = ImportError("Failed to import `llama_cpp`. Please install llama-cpp-python >= 0.3.16.")
        message = describe_vieneu_import_error(error, "standard", vieneu_installed=True)
        assert "chưa cài" not in message and "llama_cpp" in message and "v3 Turbo" in message

    def test_missing_sdk_still_says_install(self) -> None:
        message = describe_vieneu_import_error(ImportError("x"), "v3turbo", vieneu_installed=False)
        assert "pip install vieneu" in message

    def test_other_import_errors_point_to_the_full_update(self) -> None:
        message = describe_vieneu_import_error(ImportError("no onnx"), "v3turbo", True)
        assert "pip install -U vieneu sea-g2p" in message and "no onnx" in message
