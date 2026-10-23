"""[v3.23.595] VieNeu 3.8: tuỳ chọn int8 cho v3 Turbo trên CPU + dò VNNI.

VieNeu 3.8.3 (bản mới nhất trên PyPI, khớp bản ứng dụng đang dùng) có
``V3TurboVieNeuTTS(precision="fp32" | "int8")``: int8 "nhanh ~3x/frame, nhỏ 4x; cần CPU hỗ
trợ VNNI để không bị méo" (chú thích trong mã nguồn SDK). ``vieneu.Vieneu`` là hàm factory
``Vieneu(mode="v3turbo", **kwargs)`` — chữ ký KHÔNG liệt kê ``precision`` nên cách dò tên
tham số cũ (v3.23.255) không bao giờ truyền được ``precision``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from subtitles_extractor.composition.container import ApplicationContainer
from subtitles_extractor.infrastructure.tts import cpu_features
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.cpu_features import (
    vnni_from_cpu_flags,
    vnni_from_numpy_features,
)
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter


class _RecordingFactory:
    """Bắt chước đúng chữ ký ``vieneu.factory.Vieneu`` 3.8.3: ``(mode, **kwargs)``."""

    def __init__(self, reject_precision: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reject_precision = reject_precision

    def make(self) -> Any:
        def Vieneu(mode: str = "v3turbo", **kwargs: Any) -> Any:  # noqa: N802 - tên SDK
            if self.reject_precision and "precision" in kwargs:
                raise TypeError("unexpected keyword argument 'precision'")
            self.calls.append({"mode": mode, **kwargs})
            return {"mode": mode, **kwargs}

        return Vieneu


@pytest.fixture(autouse=True)
def _clear_engine_cache():
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


class TestConstructEngine:
    def test_int8_reaches_v3turbo_through_factory_kwargs(self) -> None:
        factory = _RecordingFactory()
        adapter = VieNeuTtsAdapter(mode="v3turbo", precision="int8")
        engine = adapter._construct_engine(factory.make())
        assert engine == {"mode": "v3turbo", "precision": "int8"}

    def test_default_sends_no_precision(self) -> None:
        factory = _RecordingFactory()
        engine = VieNeuTtsAdapter(mode="v3turbo")._construct_engine(factory.make())
        assert engine == {"mode": "v3turbo"}

    def test_other_modes_never_get_precision(self) -> None:
        factory = _RecordingFactory()
        adapter = VieNeuTtsAdapter(mode="standard", precision="int8")
        engine = adapter._construct_engine(factory.make())
        assert "precision" not in engine

    def test_sdk_rejecting_precision_still_builds_engine(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        factory = _RecordingFactory(reject_precision=True)
        adapter = VieNeuTtsAdapter(mode="v3turbo", precision="int8")
        with caplog.at_level("WARNING"):
            engine = adapter._construct_engine(factory.make())
        assert engine == {"mode": "v3turbo"}
        assert "precision=int8" in caplog.text

    @pytest.mark.parametrize("value", ["fp16", "INT4", "", None])
    def test_unknown_precision_is_ignored(self, value: str | None) -> None:
        assert VieNeuTtsAdapter(mode="v3turbo", precision=value)._precision is None


class TestEngineCache:
    def test_int8_and_fp32_engines_are_cached_separately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _RecordingFactory()
        fake_module = ModuleType("vieneu")
        fake_module.Vieneu = factory.make()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "vieneu", fake_module)

        fast = VieNeuTtsAdapter(mode="v3turbo", force_cpu=False, precision="int8")
        default = VieNeuTtsAdapter(mode="v3turbo", force_cpu=False)
        assert fast._get_or_load_engine("cpu") == {"mode": "v3turbo", "precision": "int8"}
        assert default._get_or_load_engine("cpu") == {"mode": "v3turbo"}
        # Lần hai lấy từ cache, không nạp lại model.
        again = VieNeuTtsAdapter(mode="v3turbo", force_cpu=False, precision="int8")
        again._get_or_load_engine("cpu")
        assert len(factory.calls) == 2
        assert ("v3turbo", "natural") in vmod._ENGINE_CACHE  # khoá cũ giữ nguyên


def test_container_passes_precision(tmp_path: Path) -> None:
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )

    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "config.json"),
        i18n_data_dir=Path(vmod.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    assert container.make_vieneu_tts_adapter(precision="int8")._precision == "int8"
    assert container.make_vieneu_tts_adapter()._precision is None


# ── Dò VNNI ───────────────────────────────────────────────────────────────────


class TestVnni:
    @pytest.mark.parametrize(("flags", "expected"), [
        ({"sse4_2", "avx2", "avx512f", "avx512_vnni"}, True),  # Xeon/Zen 4
        ({"avx2", "avx_vnni"}, True),  # Core đời 12+
        ({"sse4_2", "avx2", "fma"}, False),  # Core đời 8
        (set(), False),
    ])
    def test_cpuinfo_flags(self, flags: set[str], expected: bool) -> None:
        assert vnni_from_cpu_flags(flags) is expected

    @pytest.mark.parametrize(("features", "expected"), [
        ({"AVX512VNNI": True, "AVXVNNI": False}, True),
        ({"AVX512VNNI": False, "AVXVNNI": True}, True),
        ({"AVX512VNNI": False, "AVXVNNI": False}, False),
        ({"AVX512VNNI": False}, None),  # NumPy không dò AVX-VNNI → chưa kết luận được
        ({}, None),
    ])
    def test_numpy_features(
        self, features: dict[str, bool], expected: bool | None
    ) -> None:
        assert vnni_from_numpy_features(features) is expected

    def test_reads_proc_cpuinfo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cpuinfo = tmp_path / "cpuinfo"
        cpuinfo.write_text(
            "processor\t: 0\nflags\t\t: fpu sse4_2 avx2 avx_vnni\n", encoding="utf-8"
        )
        monkeypatch.setattr(cpu_features, "Path", lambda _path: cpuinfo)
        monkeypatch.setattr(cpu_features.sys, "platform", "linux")
        assert cpu_features.cpu_vnni_support() is True

    def test_this_machine_gives_definite_answer_on_linux(self) -> None:
        if not Path("/proc/cpuinfo").is_file():
            pytest.skip("Chỉ kiểm trên Linux có /proc/cpuinfo.")
        assert cpu_features.cpu_vnni_support() in (True, False)
