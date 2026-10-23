"""[v3.23.593] Lồng tiếng nhiều giọng: cả ba engine đọc MỖI DÒNG bằng giọng của vai.

Trước bản này ``TTSRequest`` chỉ có một ``speaker`` và cả ba engine giải quyết giọng MỘT
lần cho cả phim. Test ở đây chạy đường thật của từng engine (chỉ thay lời gọi dịch
vụ/model bằng bản giả ghi lại giọng được dùng):

* Edge: ID giọng biến thể ``"…Neural@-12Hz"`` phải tách thành giọng gốc + ``pitch`` trước
  khi gọi dịch vụ (cả worker subprocess lẫn đường chạy trong tiến trình).
* Gemini: nhánh Standard và nhánh Native Audio.
* VieNeu: đường CPU, worker GPU (từng câu và theo lô — một lô chỉ một giọng), và giao thức
  của chính tiến trình worker.
* TTS lại vài dòng: giọng của vai đi theo dòng, không lệch theo vị trí tập con.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from subtitles_extractor.application.use_cases.resynthesize_lines import (
    ResynthesizeLinesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_tts_port import (
    TTSCancelledError,
    TTSRequest,
    TTSSegmentResult,
)
from subtitles_extractor.domain.value_objects.voice_profile import (
    TONE_HIGH,
    TONE_LOW,
    VOICE_FEMALE,
    VOICE_MALE,
    gender_from_voice_label,
)
from subtitles_extractor.infrastructure.tts import edge_tts_subprocess
from subtitles_extractor.infrastructure.tts import vieneu_gpu_subprocess as gpu_worker
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter
from subtitles_extractor.infrastructure.tts.edge_voice_variants import (
    edge_voice_catalog,
    split_voice_variant,
)
from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import GeminiTTSAdapter
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter

_SR = 24_000
#: Nhãn giọng thật của VieNeu 3.8.3 (voices_v3_turbo.json / voices.json).
THIEN_MINH = "⭐ Thiện Minh — Nam · Bắc · Phong cách kể chuyện"
THUY_DUNG = "⭐ Thùy Dung — Nữ · Nam · Phong cách tin tức"
TRUC_LY = "⭐ Trúc Ly — Nữ · Bắc · Phong cách tự nhiên"


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


def _sine(seconds: float = 0.5) -> np.ndarray:
    samples = np.arange(int(_SR * seconds)) / _SR
    return (0.2 * np.sin(2 * np.pi * 220 * samples)).astype(np.float32)


EVENTS = [
    _Event(1.0, 2.5, "Trưởng đồn."),
    _Event(3.0, 4.5, "Lại có nhà chết sạch nữa à?"),
    _Event(5.0, 6.5, "Mày đúng là khốn nạn!"),
    _Event(7.0, 8.5, "Trưởng đồn ạ."),
]


@pytest.fixture(autouse=True)
def _clear_vieneu_cache():
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


# ── Cổng (domain) ─────────────────────────────────────────────────────────────


class TestRequestVoices:
    def test_voice_at_falls_back_to_the_shared_voice(self) -> None:
        request = TTSRequest(events=EVENTS, line_voices=("a", "", " b "))
        assert [request.voice_at(i, "chung") for i in range(4)] == [
            "a", "chung", "b", "chung",
        ]

    def test_single_voice_request_is_unchanged(self) -> None:
        assert TTSRequest(events=EVENTS).voice_at(2, "chung") == "chung"

    def test_subset_keeps_each_line_voice(self) -> None:
        request = TTSRequest(events=EVENTS, line_voices=("a", "b", "c", "d"))
        assert request.line_voices_for([3, 1]) == ("d", "b")
        assert TTSRequest(events=EVENTS).line_voices_for([3, 1]) == ()


# ── Edge ──────────────────────────────────────────────────────────────────────


class TestEdgeVoiceVariants:
    def test_split(self) -> None:
        assert split_voice_variant("vi-VN-NamMinhNeural@-12Hz") == (
            "vi-VN-NamMinhNeural", "-12Hz",
        )
        assert split_voice_variant("vi-VN-NamMinhNeural") == (
            "vi-VN-NamMinhNeural", "+0Hz",
        )

    def test_bad_pitch_is_dropped_instead_of_failing_the_line(self) -> None:
        assert split_voice_variant("vi-VN-HoaiMyNeural@cao") == (
            "vi-VN-HoaiMyNeural", "+0Hz",
        )

    def test_vietnamese_catalog_has_six_voices(self) -> None:
        catalog = EdgeTTSAdapter().list_voice_catalog("vi-VN")
        assert [(p.voice_id, p.gender, p.tone) for p in catalog] == [
            ("vi-VN-HoaiMyNeural", VOICE_FEMALE, ""),
            ("vi-VN-HoaiMyNeural@-12Hz", VOICE_FEMALE, TONE_LOW),
            ("vi-VN-HoaiMyNeural@+12Hz", VOICE_FEMALE, TONE_HIGH),
            ("vi-VN-NamMinhNeural", VOICE_MALE, ""),
            ("vi-VN-NamMinhNeural@-12Hz", VOICE_MALE, TONE_LOW),
            ("vi-VN-NamMinhNeural@+12Hz", VOICE_MALE, TONE_HIGH),
        ]
        assert catalog[4].label == "NamMinh · trầm (nam)"

    def test_catalog_genders_for_other_languages(self) -> None:
        catalog = edge_voice_catalog(["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"])
        assert {p.voice_id: p.gender for p in catalog if "@" not in p.voice_id} == {
            "zh-CN-XiaoxiaoNeural": VOICE_FEMALE, "zh-CN-YunxiNeural": VOICE_MALE,
        }


class TestEdgeGenerate:
    def test_each_line_is_read_with_its_role_voice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        used: dict[str, str] = {}

        async def fake_generate(
            text: str, voice: str, rate: str
        ) -> tuple[np.ndarray, int]:
            used.setdefault(text, voice)
            return _sine(), _SR

        monkeypatch.setattr(
            EdgeTTSAdapter, "_async_generate", staticmethod(fake_generate)
        )
        monkeypatch.setattr(EdgeTTSAdapter, "is_available", lambda self: True)
        request = TTSRequest(
            events=EVENTS, speaker="vi-VN-HoaiMyNeural", normalize=False, retry_count=1,
            line_voices=("vi-VN-HoaiMyNeural@+12Hz", "vi-VN-NamMinhNeural@-12Hz", "", ""),
        )
        EdgeTTSAdapter().generate(request, tmp_path / "out.wav")
        assert [used[event.text] for event in EVENTS] == [
            "vi-VN-HoaiMyNeural@+12Hz", "vi-VN-NamMinhNeural@-12Hz",
            "vi-VN-HoaiMyNeural", "vi-VN-HoaiMyNeural",
        ]

    def test_variant_is_split_before_reaching_the_worker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: list[tuple[str, str]] = []

        async def fake_worker(
            text: str, voice: str, rate: str, output_path: str, pitch: str = "+0Hz"
        ) -> bool:
            sent.append((voice, pitch))
            sf.write(output_path, _sine(), _SR)
            return True

        monkeypatch.setattr(
            EdgeTTSAdapter, "_run_edge_subprocess", staticmethod(fake_worker)
        )
        audio, _sr = asyncio.run(
            EdgeTTSAdapter._async_generate("Tôi...", "vi-VN-NamMinhNeural@-12Hz", "+0%")
        )
        assert len(audio) > 0
        assert sent == [("vi-VN-NamMinhNeural", "-12Hz")]

    def test_pitch_reaches_edge_tts_in_both_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        created: list[dict[str, Any]] = []

        class _Communicate:
            def __init__(self, text: str, voice: str, **kwargs: Any) -> None:
                created.append({"voice": voice, **kwargs})

            async def save(self, path: str) -> None:
                sf.write(path, _sine(), _SR, format="WAV")

        fake_module = ModuleType("edge_tts")
        fake_module.Communicate = _Communicate  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "edge_tts", fake_module)

        code = asyncio.run(edge_tts_subprocess._synthesize(
            "Tôi...", "vi-VN-NamMinhNeural", "+0%", str(tmp_path / "a.wav"), "-12Hz",
        ))
        ok = asyncio.run(EdgeTTSAdapter._run_edge_in_process(
            "Tôi...", "vi-VN-NamMinhNeural", "+0%", str(tmp_path / "b.wav"), "+12Hz",
        ))
        assert code == 0 and ok
        assert [entry["pitch"] for entry in created] == ["-12Hz", "+12Hz"]

    def test_negative_rate_and_pitch_survive_the_command_line(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """argparse coi "-10%"/"-12Hz" đứng riêng là một TUỲ CHỌN → worker thoát mã 2.

        Lỗi có từ trước giọng trầm: mọi lần Edge đọc CHẬM lại (tốc độ < 1, Pass 2.5 vi
        chỉnh câu ngắn hơn khung) qua worker đều hỏng âm thầm. Dựng lệnh bằng adapter
        thật rồi đọc lại bằng bộ đọc tham số thật của worker.
        """
        commands: list[tuple[str, ...]] = []

        class _Process:
            returncode = 0

            async def communicate(self) -> tuple[bytes, bytes]:
                return b"", b""

        async def fake_exec(*args: str, **_kwargs: Any) -> _Process:
            commands.append(args)
            parsed = edge_tts_subprocess.build_parser().parse_args(list(args[2:]))
            Path(parsed.output).write_bytes(b"RIFF")
            return _Process()

        monkeypatch.setattr(EdgeTTSAdapter, "_prefer_in_process", False)
        monkeypatch.setattr(
            EdgeTTSAdapter, "_resolve_python_for_worker", staticmethod(lambda: "python")
        )
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        ok = asyncio.run(EdgeTTSAdapter._run_edge_subprocess(
            "-Đi!", "vi-VN-NamMinhNeural", "-10%", str(tmp_path / "o.wav"), pitch="-12Hz",
        ))
        parsed = edge_tts_subprocess.build_parser().parse_args(list(commands[0][2:]))
        assert ok
        assert (parsed.text, parsed.rate, parsed.pitch) == ("-Đi!", "-10%", "-12Hz")

    def test_worker_main_passes_pitch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, ...]] = []

        async def fake_synthesize(*args: str) -> int:
            seen.append(args)
            return 0

        monkeypatch.setattr(edge_tts_subprocess, "_synthesize", fake_synthesize)
        monkeypatch.setattr(sys, "argv", [
            "edge_tts_subprocess.py", "--text=x", "--voice=v", "--pitch=-12Hz",
            "--output=o.wav",
        ])
        assert edge_tts_subprocess.main() == 0
        assert seen == [("x", "v", "+0%", "o.wav", "-12Hz")]


# ── Gemini ────────────────────────────────────────────────────────────────────


class TestGemini:
    def test_catalog_genders(self) -> None:
        catalog = {p.voice_id: p for p in GeminiTTSAdapter("k").list_voice_catalog("")}
        assert len(catalog) == 30
        assert catalog["Charon"].gender == VOICE_MALE
        assert catalog["Kore"].gender == VOICE_FEMALE
        assert catalog["Charon"].label == "Charon (nam)"
        assert sum(p.gender == VOICE_FEMALE for p in catalog.values()) == 14

    def test_standard_reads_each_line_with_its_voice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        adapter = GeminiTTSAdapter("k")
        used: list[str] = []

        def fake_process(**kwargs: Any) -> TTSSegmentResult:
            used.append(kwargs["voice"])
            event = kwargs["event"]
            return TTSSegmentResult(
                event_index=kwargs["idx"], start_sec=event.start_sec,
                end_sec=event.end_sec, text=event.text, was_skipped=True,
            )

        monkeypatch.setattr(adapter, "is_available", lambda: True)
        monkeypatch.setattr(adapter, "_make_client", lambda **_kw: object())
        monkeypatch.setattr(adapter, "_process_standard", fake_process)
        request = TTSRequest(
            events=EVENTS, speaker="Aoede", normalize=False,
            line_voices=("Kore", "Charon", "", "Kore"),
        )
        adapter.generate(request, tmp_path / "out.wav")
        assert used == ["Kore", "Charon", "Aoede", "Kore"]

    def test_native_audio_reads_each_line_with_its_voice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import (
            _NATIVE_AUDIO_MODELS,
        )

        adapter = GeminiTTSAdapter("k")
        used: dict[str, str] = {}

        async def fake_native(**kwargs: Any) -> None:
            used[kwargs["text"]] = kwargs["voice"]
            return None

        monkeypatch.setattr(adapter, "is_available", lambda: True)
        monkeypatch.setattr(adapter, "_async_native_single_with_retry", fake_native)
        request = TTSRequest(
            events=EVENTS, language=_NATIVE_AUDIO_MODELS[0], speaker="Aoede",
            normalize=False, line_voices=("", "Charon", "Puck", ""),
        )
        adapter.generate(request, tmp_path / "out.wav")
        assert [used[event.text] for event in EVENTS] == [
            "Aoede", "Charon", "Puck", "Aoede",
        ]


# ── VieNeu ────────────────────────────────────────────────────────────────────


class TestVieNeuLabels:
    @pytest.mark.parametrize(("label", "gender"), [
        (THIEN_MINH, VOICE_MALE),
        (THUY_DUNG, VOICE_FEMALE),  # "Nam" thứ hai là miền Nam, không phải giới tính
        ("Bình (nam miền Bắc)", VOICE_MALE),
        ("Đoan (nữ miền Nam)", VOICE_FEMALE),
        ("Adam", ""),
    ])
    def test_gender_from_real_labels(self, label: str, gender: str) -> None:
        assert gender_from_voice_label(label) == gender


class _FakeVieneu:
    """Engine giả: ghi lại giọng của mỗi lần ``infer``."""

    def __init__(self) -> None:
        self.voices_used: dict[str, str] = {}

    def list_preset_voices(self) -> list[tuple[str, str]]:
        return [
            (THIEN_MINH, "Thiện Minh"), (TRUC_LY, "Trúc Ly"), (THUY_DUNG, "Thùy Dung"),
        ]

    def get_preset_voice(self, voice_id: str) -> dict[str, str]:
        return {"id": voice_id}

    def infer(self, text: str, voice: Any = None, **_kwargs: Any) -> np.ndarray:
        self.voices_used[text] = voice["id"]
        return _sine()


class _FakeGpuProcess:
    """Tiến trình worker GPU giả: nhận dòng JSON, ghi WAV, trả lời như worker thật."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.stdin = self
        self.stdout = self
        self._reply = ""

    def poll(self) -> None:
        return None

    def write(self, line: str) -> None:
        request = json.loads(line)
        self.requests.append(request)
        if "batch" in request:
            for entry in request["batch"]:
                sf.write(entry["output"], _sine(), _SR)
            outputs = [entry["output"] for entry in request["batch"]]
            self._reply = json.dumps({"ok": True, "batch": outputs})
        else:
            sf.write(request["output"], _sine(), _SR)
            self._reply = json.dumps({"ok": True, "output": request["output"]})

    def flush(self) -> None:
        return None

    def readline(self) -> str:
        return self._reply + "\n"


def _vieneu(engine: _FakeVieneu, monkeypatch: pytest.MonkeyPatch) -> VieNeuTtsAdapter:
    adapter = VieNeuTtsAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    adapter._engine = engine
    return adapter


_ROLE_VOICES = ("Trúc Ly", "", "Thùy Dung", "Trúc Ly")


class TestVieNeu:
    def test_catalog_reads_gender_from_labels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _vieneu(_FakeVieneu(), monkeypatch)
        catalog = adapter.list_voice_catalog("vi-VN")
        assert [(p.voice_id, p.gender) for p in catalog] == [
            ("Thiện Minh", VOICE_MALE), ("Trúc Ly", VOICE_FEMALE),
            ("Thùy Dung", VOICE_FEMALE),
        ]
        assert adapter.list_speakers("vi-VN") == ["Thiện Minh", "Trúc Ly", "Thùy Dung"]

    def test_cpu_reads_each_line_with_its_voice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _FakeVieneu()
        adapter = _vieneu(engine, monkeypatch)
        adapter._gpu_worker_disabled = True
        request = TTSRequest(
            events=EVENTS, speaker="Thiện Minh", normalize=False,
            line_voices=_ROLE_VOICES,
        )
        adapter.generate(request, tmp_path / "out.wav")
        assert [engine.voices_used[event.text] for event in EVENTS] == [
            "Trúc Ly", "Thiện Minh", "Thùy Dung", "Trúc Ly",
        ]

    def test_unknown_role_voice_uses_the_shared_voice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _FakeVieneu()
        adapter = _vieneu(engine, monkeypatch)
        adapter._gpu_worker_disabled = True
        request = TTSRequest(
            events=EVENTS[:2], speaker="Thùy Dung", normalize=False,
            line_voices=("Giọng không có", ""),
        )
        adapter.generate(request, tmp_path / "out.wav")
        assert set(engine.voices_used.values()) == {"Thùy Dung"}

    def test_gpu_worker_receives_the_role_preset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        adapter = _vieneu(_FakeVieneu(), monkeypatch)
        adapter._force_cpu = False
        process = _FakeGpuProcess()
        adapter._gpu_server = process
        request = TTSRequest(
            events=EVENTS, speaker="Thiện Minh", normalize=False,
            line_voices=_ROLE_VOICES,
        )
        adapter.generate(request, tmp_path / "out.wav")
        presets = {
            item["text"]: item.get("voice_preset", "") for item in process.requests
        }
        assert [presets[event.text] for event in EVENTS] == [
            "Trúc Ly", "", "Thùy Dung", "Trúc Ly",
        ]

    def test_gpu_batch_holds_one_voice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        adapter = _vieneu(_FakeVieneu(), monkeypatch)
        adapter._force_cpu = False
        adapter._batch_size = 4
        process = _FakeGpuProcess()
        adapter._gpu_server = process
        request = TTSRequest(
            events=EVENTS, speaker="Thiện Minh", normalize=False,
            line_voices=_ROLE_VOICES,
        )
        adapter.generate(request, tmp_path / "out.wav")
        batches = [
            (item.get("voice_preset", ""), [entry["text"] for entry in item["batch"]])
            for item in process.requests if "batch" in item
        ]
        assert batches == [
            ("Trúc Ly", ["Trưởng đồn.", "Trưởng đồn ạ."]),
            ("", ["Lại có nhà chết sạch nữa à?"]),
            ("Thùy Dung", ["Mày đúng là khốn nạn!"]),
        ]


class TestGpuWorkerProtocol:
    def test_serve_uses_the_preset_of_each_item(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _FakeVieneu()
        written: list[str] = []
        monkeypatch.setattr(gpu_worker, "_load_engine", lambda _mode: engine)
        monkeypatch.setattr(
            gpu_worker, "_write_wav", lambda path, _audio, _sr: written.append(path)
        )
        lines = [
            {"mode": "v3turbo", "voice_preset": "Thiện Minh", "sample_rate": _SR},
            {"text": "Trưởng đồn.", "output": "a.wav", "voice_preset": "Trúc Ly"},
            {"text": "Tôi...", "output": "b.wav"},
            {"batch": [{"text": "Trưởng đồn ạ.", "output": "c.wav"}],
             "voice_preset": "Thùy Dung"},
            {"quit": True},
        ]
        monkeypatch.setattr(sys, "stdin", io.StringIO(
            "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
        ))
        engine.infer_batch = lambda texts, voice=None, **_kw: [  # type: ignore[attr-defined]
            engine.infer(text, voice) for text in texts
        ]
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        assert gpu_worker.serve() == gpu_worker.EXIT_OK
        assert engine.voices_used == {
            "Trưởng đồn.": "Trúc Ly", "Tôi...": "Thiện Minh",
            "Trưởng đồn ạ.": "Thùy Dung",
        }
        assert written == ["a.wav", "b.wav", "c.wav"]

    def test_role_voice_is_resolved_once(self) -> None:
        calls: list[str] = []
        engine = SimpleNamespace(
            get_preset_voice=lambda name: calls.append(name) or {"id": name},
            list_preset_voices=lambda: [("x", "Trúc Ly")],
        )
        cache: dict[str, Any] = {}
        for _ in range(3):
            gpu_worker._voice_for_item(
                engine, {"voice_preset": "Trúc Ly"}, "chung", cache
            )
        assert calls == ["Trúc Ly"]
        assert gpu_worker._voice_for_item(engine, {}, "chung", cache) == "chung"


# ── TTS lại vài dòng ─────────────────────────────────────────────────────────


class TestResynthesizeKeepsRoleVoices:
    def test_subset_request_carries_the_voices_of_the_chosen_lines(
        self, tmp_path: Path
    ) -> None:
        captured: list[TTSRequest] = []

        class _Adapter:
            def generate(self, request: TTSRequest, **_kwargs: Any) -> list:
                captured.append(request)
                raise TTSCancelledError("dừng sau khi ghi nhận yêu cầu")

        request = TTSRequest(events=EVENTS, line_voices=("a", "b", "c", "d"))
        with pytest.raises(TTSCancelledError):
            ResynthesizeLinesUseCase(_Adapter()).execute(  # type: ignore[arg-type]
                full_request=request, selected_indices={3, 1}, prior_results=[],
                output_path=tmp_path / "out.wav", sample_rate=_SR,
                total_length_samples=_SR,
            )
        assert [event.text for event in captured[0].events] == [
            EVENTS[1].text, EVENTS[3].text,
        ]
        assert captured[0].line_voices == ("b", "d")
