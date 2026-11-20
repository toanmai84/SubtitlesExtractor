"""[v3.23.745] GLM-OCR làm bộ đọc gói kiểm định — tải khi dùng lần đầu, chạy ``llama-server`` ở tiến trình riêng.

Chủ dự án (18/11/2026): exe không nặng thêm; chỉ tải mô hình khi dùng. Đo 457 câu: GLM-OCR 57/57 câu bất đồng
(PaddleOCR-VL 50/57); bản GGUF Q8_0 qua llama.cpp 57/57 + 158/159.

Test tải thật qua máy chủ HTTP cục bộ (mã băm, huỷ, gói thiếu tệp). Test tích hợp với ``llama-server`` + GGUF THẬT chạy
khi có biến môi trường ``SE_TEST_LLAMA_SERVER`` (đường dẫn llama-server) và ``SE_TEST_GLM_GGUF_DIR``.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import http.server
import io
import json
import os
import shutil
import sys
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.ocr_engine_port import OcrEngineConfig
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.quality.glm_ocr_runtime import (
    LLAMA_CPP_BUILD,
    MODEL_ASSETS,
    WINDOWS_SERVER_ASSETS,
    Asset,
    GlmOcrError,
    GlmOcrInstaller,
    clean_reading,
    glm_ocr_auditor,
    server_command,
)

SERVER_EXE = "llama-server.exe" if sys.platform == "win32" else "llama-server"


# ── tải / cài ────────────────────────────────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _serve(files: dict[str, bytes]) -> Iterator[str]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = files.get(self.path.lstrip("/"))
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a: object) -> None: ...

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def _assets(base: str, model: bytes, mmproj: bytes, *, bad_hash: bool = False) -> tuple[list[Asset], list[Asset]]:
    models = [Asset("m.gguf", f"{base}/m.gguf", len(model), "0" * 64 if bad_hash else hashlib.sha256(model).hexdigest()),
              Asset("p.gguf", f"{base}/p.gguf", len(mmproj), hashlib.sha256(mmproj).hexdigest())]
    servers = [Asset("llama.zip", f"{base}/llama.zip", 0, required_members=(SERVER_EXE, "ggml-cuda.dll")),
               Asset("cudart.zip", f"{base}/cudart.zip", 0, required_members=("cudart64_12.dll",))]
    return models, servers


FILES = {
    "m.gguf": b"M" * 3000, "p.gguf": b"P" * 2000,
    "llama.zip": _zip({f"llama-{LLAMA_CPP_BUILD}/{SERVER_EXE}": b"exe", "ggml-cuda.dll": b"dll"}),
    "cudart.zip": _zip({"cudart64_12.dll": b"rt"}),
}


def test_install_downloads_verifies_and_extracts(tmp_path: Path) -> None:
    with _serve(FILES) as base:
        models, servers = _assets(base, FILES["m.gguf"], FILES["p.gguf"])
        installer = GlmOcrInstaller(tmp_path / "glm", server_assets=servers, model_assets=models)
        assert not installer.is_installed()
        seen: list[tuple[str, float]] = []
        installer.install(lambda message, fraction: seen.append((message, fraction)) or True)
    assert installer.is_installed()
    assert installer.server_binary.read_bytes() == b"exe" and (installer.bin_dir / "cudart64_12.dll").is_file()
    assert installer.model_path.read_bytes() == FILES["m.gguf"]
    assert not list((tmp_path / "glm").glob("*.zip")) and not list((tmp_path / "glm").glob("*.part"))
    assert any(f == 1.0 for _m, f in seen) and "GLM-OCR" in seen[0][0]
    installer.install(lambda *_a: pytest.fail("đã cài thì không tải lại"))      # type: ignore[func-returns-value]


def test_hash_mismatch_rejected_and_partial_removed(tmp_path: Path) -> None:
    with _serve(FILES) as base:
        models, servers = _assets(base, FILES["m.gguf"], FILES["p.gguf"], bad_hash=True)
        installer = GlmOcrInstaller(tmp_path / "glm", server_assets=servers, model_assets=models)
        with pytest.raises(GlmOcrError, match="SHA-256"):
            installer.install(lambda *_a: True)
    assert not installer.is_installed() and not list((tmp_path / "glm").glob("m.gguf*"))


def test_cancel_and_missing_member(tmp_path: Path) -> None:
    with _serve({**FILES, "llama.zip": _zip({"README.md": b"x"})}) as base:
        models, servers = _assets(base, FILES["m.gguf"], FILES["p.gguf"])
        installer = GlmOcrInstaller(tmp_path / "a", server_assets=servers, model_assets=models)
        with pytest.raises(GlmOcrError, match="huỷ"):
            installer.install(lambda *_a: False)
        assert not list((tmp_path / "a").glob("*.part"))
        with pytest.raises(GlmOcrError, match=SERVER_EXE.replace(".", r"\.")):
            installer.install(lambda *_a: True)
    with pytest.raises(GlmOcrError):
        GlmOcrInstaller(tmp_path / "b", server_assets=[Asset("x.zip", "http://127.0.0.1:9/x.zip", 0)],
                        model_assets=[Asset("m", "http://127.0.0.1:9/m", 1, "0"), Asset("p", "http://127.0.0.1:9/p", 1)]
                        ).install(lambda *_a: True)


def test_pinned_release_assets() -> None:
    assert LLAMA_CPP_BUILD == "b12387"                         # đúng bản đã đo chất lượng GLM-OCR (commit a956a85)
    urls = [asset.url for asset in WINDOWS_SERVER_ASSETS]
    assert urls[0].endswith("/b12387/llama-b12387-bin-win-cuda-12.4-x64.zip")
    assert urls[1].endswith("/b12387/cudart-llama-bin-win-cuda-12.4-x64.zip")
    assert [len(asset.sha256) for asset in MODEL_ASSETS] == [64, 64]
    assert all("resolve/65a42de" in asset.url for asset in MODEL_ASSETS)


def test_server_command_local_only_and_gpu_layers(tmp_path: Path) -> None:
    command = server_command(tmp_path / "s", tmp_path / "m", tmp_path / "p", 1234, use_gpu=True, device="CUDA0")
    assert command[command.index("--host") + 1] == "127.0.0.1" and "--no-webui" in command
    assert command[command.index("-ngl") + 1] == "999" and command[command.index("--device") + 1] == "CUDA0"
    cpu = server_command(tmp_path / "s", tmp_path / "m", tmp_path / "p", 1, use_gpu=False)
    assert cpu[-1] == "0" and cpu[cpu.index("--device") + 1] == "none"


def test_clean_reading() -> None:
    assert clean_reading("\n这娘们又想耍什么花样") == "这娘们又想耍什么花样"
    assert clean_reading("```\n师兄\n等等我\n```") == "师兄等等我"


# ── lượt trích xuất: chọn GLM-OCR, lùi về PaddleOCR-VL khi lỗi ─────────────────────────────────────────────────────
class _Reporter:
    def __init__(self, cancelled: bool = False) -> None:
        self.cancelled = cancelled
        self.messages: list[str] = []

    def report(self, _a: int, _b: int, message: str = "") -> None:
        self.messages.append(message)

    def is_cancelled(self) -> bool:
        return self.cancelled


def _use_case(tmp_path: Path, factory: object) -> tuple[object, dict[str, object]]:
    from subtitles_extractor.application.use_cases.extract_subtitles import (
        ExtractSubtitlesUseCase,
    )

    (tmp_path / "phim.roi.mp4").write_bytes(b"x")
    calls: dict[str, object] = {}

    class _PaddleAuditor:
        def audit(self, events, clip, sidecar, out, *, config, progress=None, frames=None):  # noqa: ANN001, ANN201
            calls["paddle"] = config.recognition_model_name
            return out

    use_case = ExtractSubtitlesUseCase.__new__(ExtractSubtitlesUseCase)
    use_case._quality_auditor, use_case._audio_exporter = _PaddleAuditor(), None
    use_case._glm_auditor_factory = factory
    return use_case, calls


def _run(use_case: object, tmp_path: Path, reader: str, reporter: _Reporter | None = None) -> _Reporter:
    reporter = reporter or _Reporter()
    request = SimpleNamespace(video_path=tmp_path / "phim.mp4", ocr=OcrEngineConfig(device="gpu"), qa_reader=reader)
    use_case._export_qa_data(request, [SubtitleEvent(index=1, text="字", interval=TimeInterval(0, 1))],  # type: ignore[attr-defined]
                             tmp_path / "phim.seraw.json.gz", reporter, frames=[])
    return reporter


def test_glm_reader_used_with_gpu_when_selected(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class _GlmAuditor:
        def audit(self, events, clip, sidecar, out, *, config, progress=None, frames=None):  # noqa: ANN001, ANN201
            seen["model"] = config.recognition_model_name
            return out

    @contextlib.contextmanager
    def factory(*, use_gpu: bool, progress, cancelled):  # noqa: ANN001, ANN202
        seen["gpu"] = use_gpu
        progress("Đang tải GLM-OCR (1/4)", 0.5)
        yield _GlmAuditor()

    use_case, calls = _use_case(tmp_path, factory)
    reporter = _run(use_case, tmp_path, "glm_ocr")
    assert seen == {"gpu": True, "model": "GLM-OCR (GGUF Q8_0, llama.cpp b12387)"} and "paddle" not in calls
    assert any("Đang tải GLM-OCR" in m and "50%" in m for m in reporter.messages)


def test_glm_failure_falls_back_to_paddle_vl(tmp_path: Path) -> None:
    def factory(**_k: object) -> object:
        raise GlmOcrError("Tải GLM-OCR-Q8_0.gguf lỗi: mất mạng")

    use_case, calls = _use_case(tmp_path, factory)
    _run(use_case, tmp_path, "glm_ocr")
    assert str(calls["paddle"]).startswith("PaddleOCR-VL")


def test_cancel_during_glm_does_not_start_paddle(tmp_path: Path) -> None:
    def factory(**_k: object) -> object:
        raise GlmOcrError("Đã huỷ tải GLM-OCR.")

    use_case, calls = _use_case(tmp_path, factory)
    _run(use_case, tmp_path, "glm_ocr", _Reporter(cancelled=True))
    assert "paddle" not in calls


def test_paddle_reader_default_never_touches_glm(tmp_path: Path) -> None:
    use_case, calls = _use_case(tmp_path, lambda **_k: pytest.fail("không được mở GLM-OCR"))
    _run(use_case, tmp_path, "paddle_vl")
    assert str(calls["paddle"]).startswith("PaddleOCR-VL")


def test_setting_reaches_request_and_settings_page() -> None:
    from subtitles_extractor.infrastructure.settings.application_settings import (
        OcrSettings,
    )

    assert OcrSettings().qa_reader == "paddle_vl"
    with pytest.raises(ValueError):
        OcrSettings(qa_reader="tesseract")
    root = Path(__file__).resolve().parents[3] / "src/subtitles_extractor"
    assert 'qa_reader=str(getattr(snapshot.ocr, "qa_reader"' in (root / "presentation/view_models/"
                                                                  "extract_page_view_model.py").read_text(encoding="utf-8")
    page = (root / "presentation/pages/settings_page.py").read_text(encoding="utf-8")
    assert '"qa_reader": self._cb_qa_reader.currentData()' in page and "_set_combo(self._cb_qa_reader" in page
    for lang in ("vi", "en"):
        strings = json.loads((root / f"data/strings_{lang}.json").read_text(encoding="utf-8"))["settings"]
        assert {"rl_qa_reader", "qa_reader_paddle_vl", "qa_reader_glm_ocr", "tip_qa_reader"} <= strings.keys()


# ── tích hợp thật: llama-server + GGUF GLM-OCR + bộ kiểm định trên video ROI thật ─────────────────────────────────────
DATA = Path(__file__).resolve().parents[3] / "test data" / "hmzw_v3_23_742"
_SERVER = os.environ.get("SE_TEST_LLAMA_SERVER", "")
_GGUF = os.environ.get("SE_TEST_GLM_GGUF_DIR", "")


@pytest.mark.skipif(not (_SERVER and _GGUF and (DATA / "hmzw.seraw.json.gz").is_file()),
                    reason="cần SE_TEST_LLAMA_SERVER + SE_TEST_GLM_GGUF_DIR (llama-server và GGUF GLM-OCR thật)")
def test_real_llama_server_reads_cropped_subtitles(tmp_path: Path) -> None:
    from subtitles_extractor.domain.entities.ocr_frame_result import (
        OcrFrameResult,
        OcrTextBox,
    )
    from subtitles_extractor.domain.value_objects.confidence import Confidence

    root = tmp_path / "glm"
    root.mkdir()
    for asset in MODEL_ASSETS:
        (root / asset.name).symlink_to(Path(_GGUF) / asset.name)
    (root / "installed.json").write_text(json.dumps({"build": LLAMA_CPP_BUILD}), encoding="utf-8")
    installer = GlmOcrInstaller(root, server_binary=Path(_SERVER))
    assert installer.is_installed()
    seraw = json.load(gzip.open(DATA / "hmzw.seraw.json.gz"))
    frames = [OcrFrameResult(k, f["ts"], [OcrTextBox(text=b["t"], confidence=Confidence(min(1.0, b["c"])),
                                                     polygon=[(int(p[0]), int(p[1])) for p in b["p"]]) for b in f["boxes"]])
              for k, f in enumerate(seraw["frames"]) if 2644 <= f["ts"] <= 2654]
    events = [SubtitleEvent(index=1, text=e["t"], interval=TimeInterval(e["s"], e["e"]))
              for e in seraw["meta"]["result"]["events"] if 2645 <= e["s"] <= 2652]
    shutil.copy(DATA / "clips" / "mon_2644.roi.mp4", tmp_path / "phim.roi.mp4")
    shutil.copy(DATA / "clips" / "mon_2644.roi.json", tmp_path / "phim.roi.json")
    out = tmp_path / "phim.qa.json.gz"
    with glm_ocr_auditor(installer, use_gpu=False, progress=lambda *_a: True, cancelled=lambda: False) as auditor:
        auditor.audit(events, tmp_path / "phim.roi.mp4", tmp_path / "phim.roi.json", out,
                      config=OcrEngineConfig(recognition_model_name="GLM-OCR (GGUF Q8_0, llama.cpp b12387)"), frames=frames)
    payload = json.loads(gzip.open(out, "rt", encoding="utf-8").read())
    assert payload["model"].startswith("GLM-OCR") and payload["input"] == "subtitle_box_crop"
    for line in payload["lines"]:                                   # "门没关", "阿姨这是在"
        assert sum(reading == line["t"] for reading in line["vl"]) >= 2, line


# ── các lỗi lượt rà độc lập tìm ra (sửa trước khi giao bản) ──────────────────────────────────────────────────────────
def test_interrupted_install_resumes_missing_cuda_runtime(tmp_path: Path) -> None:
    """Gói DLL CUDA tải hỏng sau khi llama-server đã giải nén → lần sau PHẢI tải lại gói đó (trước: coi như đã cài)."""
    broken = {k: v for k, v in FILES.items() if k != "cudart.zip"}
    root = tmp_path / "glm"
    with _serve(broken) as base:
        models, servers = _assets(base, FILES["m.gguf"], FILES["p.gguf"])
        with pytest.raises(GlmOcrError):
            GlmOcrInstaller(root, server_assets=servers, model_assets=models).install(lambda *_a: True)
    assert (root / f"llama-{LLAMA_CPP_BUILD}" / SERVER_EXE).is_file()
    with _serve(FILES) as base:
        models, servers = _assets(base, FILES["m.gguf"], FILES["p.gguf"])
        installer = GlmOcrInstaller(root, server_assets=servers, model_assets=models)
        assert not installer.is_installed()
        fetched: list[str] = []
        installer.install(lambda message, _f: fetched.append(message) or True)
    assert installer.is_installed() and (installer.bin_dir / "cudart64_12.dll").is_file()
    assert {m.split(": ")[-1].rstrip(")") for m in fetched} == {"cudart.zip"}           # chỉ tải lại phần thiếu


def test_windows_only_bundle_not_downloaded_elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    installer = GlmOcrInstaller(tmp_path / "glm")
    with pytest.raises(GlmOcrError, match="Windows"):
        installer.install(lambda *_a: pytest.fail("không được tải gì"))                # type: ignore[func-returns-value]


def test_cuda_devices_parsed_from_list_devices() -> None:
    """[v3.23.748] Dò GPU bằng ``--list-devices`` (3.23.745 đọc dòng "offloaded … to GPU" mà server không in ở mức log
    mặc định → máy chủ dự án RTX 3070 Ti bị báo nhầm không lên được GPU, log 19/11 00:16)."""
    from subtitles_extractor.infrastructure.quality.glm_ocr_runtime import cuda_devices

    listing = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 3070 Ti (8191 MiB, 7120 MiB free)\n"
    assert cuda_devices(listing) == [("CUDA0", "NVIDIA GeForce RTX 3070 Ti (8191 MiB, 7120 MiB free)")]
    assert cuda_devices("Available devices:\n  (none)\n") == []
    assert cuda_devices("ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected") == []


def test_gpu_run_without_cuda_device_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as sp

    from subtitles_extractor.infrastructure.quality import glm_ocr_runtime as runtime

    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: sp.CompletedProcess(a, 0, "Available devices:\n  (none)\n", ""))
    server = runtime.LlamaServer(GlmOcrInstaller(tmp_path, server_binary=tmp_path / "llama-server"), use_gpu=True)
    with pytest.raises(GlmOcrError, match="không thấy GPU CUDA"):
        server._pick_device({})


def test_single_bad_image_left_empty_but_broken_server_aborts() -> None:
    import numpy as np

    from subtitles_extractor.infrastructure.quality.glm_ocr_runtime import (
        GlmOcrEngine,
    )

    class _Engine(GlmOcrEngine):
        def __init__(self, replies: list[object]) -> None:
            super().__init__(server=None)  # type: ignore[arg-type]
            self._replies = replies

        def read(self, image_rgb: np.ndarray, timeout: float = 0.0) -> str:  # [v3.23.759] thời hạn theo lượt
            reply = self._replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return str(reply)

    image = np.zeros((20, 40, 3), np.uint8)
    engine = _Engine(["门没关", OSError("HTTP 500"), "阿姨"])
    results = engine.infer_batch([image] * 3, [0, 1, 2], [0.0, 0.1, 0.2])
    assert [r.text_boxes[0].text if r.text_boxes else "" for r in results] == ["门没关", "", "阿姨"]
    dead = _Engine([ConnectionRefusedError("refused")] * 5)
    with pytest.raises(GlmOcrError, match="5 ảnh liên tiếp"):
        dead.infer_batch([image] * 5, list(range(5)), [0.0] * 5)


def test_server_stopped_when_start_fails_unexpectedly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.infrastructure.quality import glm_ocr_runtime as runtime

    stopped: list[bool] = []
    monkeypatch.setattr(runtime.LlamaServer, "_start", lambda self, cancelled: (_ for _ in ()).throw(KeyError("x")))
    monkeypatch.setattr(runtime.LlamaServer, "stop", lambda self: stopped.append(True))
    with pytest.raises(KeyError):
        runtime.LlamaServer(GlmOcrInstaller(tmp_path), use_gpu=True).start()
    assert stopped == [True]


def test_glm_path_frees_paddle_vl_vram_first(tmp_path: Path) -> None:
    order: list[str] = []

    @contextlib.contextmanager
    def factory(**_k: object):  # noqa: ANN202
        order.append("glm")
        yield SimpleNamespace(audit=lambda *a, **k: None)

    use_case, _calls = _use_case(tmp_path, factory)
    use_case._release_auxiliary_models = lambda: order.append("release")  # type: ignore[attr-defined]
    _run(use_case, tmp_path, "glm_ocr")
    assert order[:2] == ["release", "glm"]
