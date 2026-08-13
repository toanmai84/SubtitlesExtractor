"""Test STT WhisperX — phần logic thuần + use case (mock engine).

WhisperX là phụ thuộc GPU nặng không cài trong sandbox, nên test:
  * ``_segments_to_events`` (hàm thuần dựng SubtitleEvent từ segment WhisperX).
  * ``is_available`` trả False khi thiếu whisperx (không crash).
  * Use case uỷ quyền đúng cho engine (mock).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.use_cases.transcribe_speech import (
    TranscribeSpeechRequest,
    TranscribeSpeechUseCase,
)
from subtitles_extractor.domain.exceptions import SpeechToTextError
from subtitles_extractor.domain.ports.speech_to_text_port import (
    TranscriptionConfig,
    TranscriptionResult,
)
from subtitles_extractor.infrastructure.stt.whisperx_adapter import WhisperXAdapter


class TestSegmentsToEvents:
    def test_basic_segments(self) -> None:
        segments = [
            {"text": " Xin chào ", "start": 0.0, "end": 1.5},
            {"text": "thế giới", "start": 1.5, "end": 3.0},
        ]
        events = WhisperXAdapter._segments_to_events(segments)
        assert [e.text for e in events] == ["Xin chào", "thế giới"]
        assert events[0].interval.start_sec == 0.0
        assert events[1].interval.end_sec == 3.0
        assert events[0].index == 1 and events[1].index == 2

    def test_empty_text_skipped(self) -> None:
        segments = [
            {"text": "  ", "start": 0.0, "end": 1.0},
            {"text": "có chữ", "start": 1.0, "end": 2.0},
        ]
        events = WhisperXAdapter._segments_to_events(segments)
        assert len(events) == 1 and events[0].text == "có chữ"

    def test_speaker_label_prefixed(self) -> None:
        segments = [{"text": "alo", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"}]
        events = WhisperXAdapter._segments_to_events(segments)
        assert events[0].text == "[SPEAKER_01] alo"

    def test_end_before_start_clamped(self) -> None:
        segments = [{"text": "x", "start": 5.0, "end": 3.0}]
        events = WhisperXAdapter._segments_to_events(segments)
        assert events[0].interval.end_sec == events[0].interval.start_sec == 5.0

    def test_malformed_timestamps_skipped(self) -> None:
        segments = [{"text": "x", "start": "bad", "end": "bad"}]
        events = WhisperXAdapter._segments_to_events(segments)
        assert events == []


class TestAvailability:
    def test_is_available_false_without_whisperx(self) -> None:
        # Sandbox không cài whisperx → False, không ném lỗi.
        assert WhisperXAdapter().is_available() is False

    def test_engine_name(self) -> None:
        assert "WhisperX" in WhisperXAdapter().get_engine_name()

    def test_transcribe_missing_engine_raises(self, tmp_path: Path) -> None:
        media = tmp_path / "a.wav"
        media.write_bytes(b"\x00")
        with pytest.raises(SpeechToTextError):
            WhisperXAdapter().transcribe(media, TranscriptionConfig())

    def test_transcribe_missing_file_raises(self) -> None:
        with pytest.raises(SpeechToTextError):
            WhisperXAdapter().transcribe(Path("/khong/ton/tai.wav"), TranscriptionConfig())


class TestLoadWavWithoutTorio:
    """[v3.22.2] Đọc WAV bằng module `wave` chuẩn — không phụ thuộc torio/FFmpeg DLL."""

    def test_reads_pcm16_mono_as_float32(self, tmp_path: Path) -> None:
        import shutil
        import subprocess

        if shutil.which("ffmpeg") is None:
            pytest.skip("Cần ffmpeg")
        wav = tmp_path / "a.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", str(wav)],
            check=True, capture_output=True,
        )
        samples = WhisperXAdapter._load_wav_as_float32(wav)
        assert samples.dtype.name == "float32"
        assert 15000 < len(samples) < 17000
        assert -1.05 <= float(samples.min()) and float(samples.max()) <= 1.05


class TestTorioDisabled:
    """[v3.22.3] Tắt torio FFmpeg để tránh FileNotFoundError DLL lúc import whisperx."""

    def test_env_vars_set_at_import(self) -> None:
        import os

        # Import module adapter đã set sẵn (idempotent).
        import subtitles_extractor.infrastructure.stt.whisperx_adapter  # noqa: F401

        assert os.environ.get("TORIO_USE_FFMPEG") == "0"
        assert os.environ.get("TORCHAUDIO_USE_FFMPEG") == "0"

    def test_is_available_swallows_dll_errors(self, monkeypatch) -> None:
        # Giả lập import whisperx ném FileNotFoundError (DLL) → is_available trả False.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "whisperx":
                raise FileNotFoundError("libtorio_ffmpeg6.pyd not found")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert WhisperXAdapter().is_available() is False


class TestDeviceResolution:
    """[v3.22.5] Tự dò CUDA, fallback CPU khi torch không có CUDA."""

    def test_cpu_request_keeps_cpu(self) -> None:
        device, compute = WhisperXAdapter._resolve_device_and_compute("cpu", "float32")
        assert device == "cpu"
        assert compute == "float32"

    def test_cpu_float16_downgraded_to_int8(self) -> None:
        device, compute = WhisperXAdapter._resolve_device_and_compute("cpu", "float16")
        assert device == "cpu"
        assert compute == "int8"

    def test_cuda_without_gpu_falls_back_to_cpu(self, monkeypatch) -> None:
        # Sandbox không có CUDA → cuda phải rơi về cpu + int8.
        device, compute = WhisperXAdapter._resolve_device_and_compute("cuda", "float16")
        assert device == "cpu"
        assert compute == "int8"


class _FakeStt:
    def __init__(self, available: bool) -> None:
        self._available = available
        self.called_with = None

    def is_available(self) -> bool:
        return self._available

    def get_engine_name(self) -> str:
        return "FakeSTT"

    def transcribe(self, media_path, config, progress_callback=None, cancellation_check=None):
        self.called_with = (media_path, config)
        from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
        from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

        return TranscriptionResult(
            events=[SubtitleEvent(index=1, text="ok", interval=TimeInterval(0.0, 1.0))],
            detected_language="vi",
        )


class TestUseCase:
    def test_delegates_to_engine(self, tmp_path: Path) -> None:
        fake = _FakeStt(available=True)
        use_case = TranscribeSpeechUseCase(fake)
        assert use_case.is_available() is True
        assert use_case.engine_name() == "FakeSTT"

        result = use_case.execute(
            TranscribeSpeechRequest(tmp_path / "v.mp4", TranscriptionConfig(language="vi"))
        )
        assert result.detected_language == "vi"
        assert len(result.events) == 1
        assert fake.called_with[1].language == "vi"


class TestSentenceSplitting:
    """[v3.22.6] Tách segment dài (CJK) thành câu ngắn dùng word-timestamps."""

    @staticmethod
    def _make_char_words(text: str, start: float, per_char: float):
        return [
            {"word": ch, "start": start + i * per_char, "end": start + (i + 1) * per_char}
            for i, ch in enumerate(text)
        ]

    def test_splits_on_silence_gap(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        words = [
            {"word": "你", "start": 0.0, "end": 0.3},
            {"word": "好", "start": 0.3, "end": 0.6},
            # khoảng lặng 1.0s → ngắt
            {"word": "世", "start": 1.6, "end": 1.9},
            {"word": "界", "start": 1.9, "end": 2.2},
        ]
        segment = {"text": "你好世界", "start": 0.0, "end": 2.2, "words": words}
        cfg = TranscriptionConfig(split_gap_sec=0.5, max_chars_per_cue=50)
        events = WhisperXAdapter._segments_to_events([segment], cfg)
        assert len(events) == 2
        assert events[0].text == "你好"
        assert events[1].text == "世界"

    def test_splits_on_sentence_end_punct(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        words = [
            {"word": "好", "start": 0.0, "end": 0.3},
            {"word": "。", "start": 0.3, "end": 0.4},
            {"word": "走", "start": 0.5, "end": 0.8},
        ]
        segment = {"text": "好。走", "start": 0.0, "end": 0.8, "words": words}
        cfg = TranscriptionConfig(split_gap_sec=5.0, max_chars_per_cue=50)
        events = WhisperXAdapter._segments_to_events([segment], cfg)
        assert events[0].text == "好。"
        assert events[1].text == "走"

    def test_long_segment_split_into_many(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        text = "一二三四五六七八九十" * 3  # 30 ký tự liền
        words = self._make_char_words(text, 0.0, 0.5)  # 15s
        segment = {"text": text, "start": 0.0, "end": 15.0, "words": words}
        cfg = TranscriptionConfig(split_gap_sec=5.0, max_chars_per_cue=10, max_cue_duration_sec=10.0)
        events = WhisperXAdapter._segments_to_events([segment], cfg)
        # 30 ký tự / 10 mỗi cue → khoảng 3 cue, không còn 1 khối dài.
        assert len(events) >= 3
        assert all(len(e.text) <= 12 for e in events)

    def test_no_words_keeps_whole_segment(self) -> None:
        # Align fail (không có words) → giữ nguyên segment.
        segment = {"text": "câu dài không tách", "start": 0.0, "end": 20.0}
        events = WhisperXAdapter._segments_to_events([segment])
        assert len(events) == 1
        assert events[0].interval.end_sec == 20.0

    def test_disable_split_keeps_whole(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        words = self._make_char_words("你好世界你好世界", 0.0, 0.5)
        segment = {"text": "你好世界你好世界", "start": 0.0, "end": 4.0, "words": words}
        cfg = TranscriptionConfig(enable_sentence_split=False)
        events = WhisperXAdapter._segments_to_events([segment], cfg)
        assert len(events) == 1

    def test_words_missing_timestamps_inherit(self) -> None:
        words = [
            {"word": "甲", "start": 0.0, "end": 0.5},
            {"word": "乙"},  # thiếu timestamp → kế thừa
            {"word": "丙", "start": 1.0, "end": 1.5},
        ]
        segment = {"text": "甲乙丙", "start": 0.0, "end": 1.5, "words": words}
        events = WhisperXAdapter._segments_to_events([segment])
        joined = "".join(e.text for e in events)
        assert joined == "甲乙丙"  # không mất chữ


class TestRawSegmentExport:
    """[v3.22.7] Xuất dữ liệu STT thô để hiệu chuẩn offline."""

    def test_sanitize_raw_segments(self) -> None:
        segments = [
            {
                "start": 0.0, "end": 2.0, "text": "你好", "speaker": "SPK1",
                "words": [
                    {"word": "你", "start": 0.0, "end": 0.5, "score": 0.9},
                    {"word": "好", "start": 0.5, "end": 2.0},  # thiếu score
                ],
            },
            "không phải dict — bỏ qua",
        ]
        clean = WhisperXAdapter._sanitize_raw_segments(segments)
        assert len(clean) == 1
        assert clean[0]["text"] == "你好"
        assert clean[0]["speaker"] == "SPK1"
        assert clean[0]["words"][1]["score"] == 0.0  # default an toàn

    def test_serializer_round_trip(self, tmp_path) -> None:
        from subtitles_extractor.infrastructure.serializers.raw_stt_serializer import (
            load_raw_stt,
            save_raw_stt,
        )

        segs = [{"start": 0.0, "end": 1.0, "text": "x", "words": [
            {"word": "x", "start": 0.0, "end": 1.0, "score": 0.5}]}]
        out = tmp_path / "t.sestt.json"
        save_raw_stt(out, segs, "v.mp4", "zh", "small", "3.22.7")
        meta, loaded = load_raw_stt(out)
        assert meta.detected_language == "zh"
        assert meta.word_count == 1
        assert loaded[0]["text"] == "x"


class TestRjiebaWordSegmentation:
    """[v3.23] Tách từ tiếng Trung bằng rjieba để không cắt giữa từ ghép."""

    def test_segment_boundaries_with_rjieba(self) -> None:
        # Nếu rjieba có, ranh giới phải khớp phân từ; nếu không, tập rỗng.
        boundaries = WhisperXAdapter._segment_word_boundaries("第一步超神")
        try:
            import rjieba  # noqa: F401

            assert len(boundaries) > 0
        except ImportError:
            assert boundaries == set()

    def test_rjieba_keeps_compound_words(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        # "第一步" nên giữ nguyên khi bật jieba (không cắt thành "第一" / "步").
        words = [
            {"word": ch, "start": i * 0.3, "end": (i + 1) * 0.3}
            for i, ch in enumerate("第一步超神系统")
        ]
        segment = {"text": "第一步超神系统", "start": 0.0, "end": 2.1, "words": words}
        cfg = TranscriptionConfig(
            use_word_segmentation=True, target_chars_per_cue=2, max_chars_per_cue=4
        )
        events = WhisperXAdapter._segments_to_events([segment], cfg)
        joined = "".join(e.text for e in events)
        assert joined == "第一步超神系统"  # không mất chữ

    def test_disable_rjieba_still_works(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        words = [
            {"word": ch, "start": i * 0.3, "end": (i + 1) * 0.3}
            for i, ch in enumerate("你好世界朋友")
        ]
        segment = {"text": "你好世界朋友", "start": 0.0, "end": 1.8, "words": words}
        cfg = TranscriptionConfig(use_word_segmentation=False, max_chars_per_cue=4)
        events = WhisperXAdapter._segments_to_events([segment], cfg)
        assert "".join(e.text for e in events) == "你好世界朋友"


class TestSubprocessIsolation:
    """[v3.23] WhisperX chạy tiến trình con để tránh xung đột DLL với paddle."""

    def test_is_available_uses_find_spec_no_import(self, monkeypatch) -> None:
        # is_available KHÔNG được import whisperx (chỉ find_spec).
        import builtins

        real_import = builtins.__import__

        def fail_on_whisperx(name, *a, **k):
            if name == "whisperx":
                raise AssertionError("Không được import whisperx vào process chính!")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fail_on_whisperx)
        # Không raise → find_spec không import.
        result = WhisperXAdapter().is_available()
        assert isinstance(result, bool)

    def test_use_subprocess_default_true(self) -> None:
        assert WhisperXAdapter()._use_subprocess is True

    def test_subprocess_can_be_disabled(self) -> None:
        assert WhisperXAdapter(use_subprocess=False)._use_subprocess is False


class TestHallucinationFilter:
    """[v3.23] Lọc câu ảo giác phổ biến của Whisper."""

    def test_filters_common_hallucinations(self) -> None:
        assert WhisperXAdapter._is_hallucination("请订阅")
        assert WhisperXAdapter._is_hallucination("谢谢观看")
        assert WhisperXAdapter._is_hallucination("Thanks for watching")
        assert WhisperXAdapter._is_hallucination("啊啊啊啊啊")  # lặp 1 ký tự
        assert WhisperXAdapter._is_hallucination("")

    def test_keeps_normal_text(self) -> None:
        assert not WhisperXAdapter._is_hallucination("你好世界")
        assert not WhisperXAdapter._is_hallucination("今天天气很好")

    def test_filter_removes_from_events(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        segs = [
            {"start": 0, "end": 1, "text": "正常字幕"},
            {"start": 2, "end": 3, "text": "请订阅"},
        ]
        events = WhisperXAdapter._segments_to_events(segs, TranscriptionConfig())
        assert len(events) == 1
        assert events[0].text == "正常字幕"
        assert events[0].index == 1  # đánh số lại liên tục


class TestSubprocessCrashRecovery:
    """[v3.23.5] Align crash (access violation) → vẫn dùng kết quả pre-align."""

    def test_build_subprocess_env_returns_path(self) -> None:
        env = WhisperXAdapter._build_subprocess_env()
        assert isinstance(env, dict)
        assert "PATH" in env

    def test_recovers_result_when_subprocess_crashes_after_dump(self, tmp_path, monkeypatch) -> None:
        import json as _json

        import subtitles_extractor.infrastructure.stt.whisperx_adapter as mod
        from subtitles_extractor.domain.ports.speech_to_text_port import (
            TranscriptionConfig,
        )

        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")

        class CrashProc:
            def __init__(self, out):
                _json.dump(
                    {"segments": [{"start": 0, "end": 2, "text": "你好世界",
                                   "words": [{"word": c, "start": i * 0.5, "end": (i + 1) * 0.5}
                                             for i, c in enumerate("你好世界")]}],
                     "language": "zh"},
                    open(out, "w", encoding="utf-8"), ensure_ascii=False,
                )
                self.stderr = iter(["PROGRESS 65 100 align"])
                self.stdout = None

            def wait(self, timeout=None):
                return 3221225477  # access violation

            def terminate(self): ...
            def kill(self): ...

        def fake_popen(cmd, **kw):
            out = cmd[cmd.index("--output") + 1]
            return CrashProc(out)

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
        adapter = WhisperXAdapter(use_subprocess=True)
        result = adapter._run_whisperx_subprocess(
            tmp_path, wav, TranscriptionConfig(), lambda c, t, m: None, lambda: False
        )
        # Không raise, vẫn có kết quả.
        assert [e.text for e in result.events] == ["你好世界"]


class TestAlignDeviceCpu:
    """[v3.23.6] Align mặc định chạy CPU để tránh xung đột cuDNN GPU (access violation)."""

    def test_align_device_default_cpu(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        assert TranscriptionConfig().align_device == "cpu"

    def test_align_device_passed_to_subprocess_config(self, tmp_path, monkeypatch) -> None:
        import json as _json

        import subtitles_extractor.infrastructure.stt.whisperx_adapter as mod
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")
        captured = {}

        class P:
            def __init__(self, cmd):
                idx = cmd.index("--config")
                captured["config"] = _json.loads(cmd[idx + 1])
                out = cmd[cmd.index("--output") + 1]
                _json.dump({"segments": [], "language": "zh"},
                           open(out, "w", encoding="utf-8"))
                self.stderr = iter([])
                self.stdout = None

            def wait(self, timeout=None):
                return 0

            def terminate(self): ...
            def kill(self): ...

        monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd, **kw: P(cmd))
        adapter = WhisperXAdapter(use_subprocess=True)
        adapter._run_whisperx_subprocess(
            tmp_path, wav, TranscriptionConfig(align_device="cpu"),
            lambda c, t, m: None, lambda: False,
        )
        assert captured["config"]["align_device"] == "cpu"


class TestAlignSeparateSubprocess:
    """[v3.23.7] Align chạy ở tiến trình con RIÊNG (tách cuDNN khỏi transcribe)."""

    def test_transcribe_then_align_two_subprocesses(self, tmp_path, monkeypatch) -> None:
        import json as _json

        import subtitles_extractor.infrastructure.stt.whisperx_adapter as mod
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")
        modes = []

        class P:
            def __init__(self, cmd):
                mode = cmd[cmd.index("--mode") + 1] if "--mode" in cmd else "transcribe"
                modes.append(mode)
                out = cmd[cmd.index("--output") + 1]
                if mode == "align":
                    _json.dump({"segments": [{"start": 0, "end": 2, "text": "你好",
                                              "words": [{"word": "你", "start": 0, "end": 1},
                                                        {"word": "好", "start": 1, "end": 2}]}],
                                "language": "zh"},
                               open(out, "w", encoding="utf-8"), ensure_ascii=False)
                else:
                    _json.dump({"segments": [{"start": 0, "end": 2, "text": "你好"}],
                                "language": "zh"},
                               open(out, "w", encoding="utf-8"), ensure_ascii=False)
                self.stderr = iter(["PROGRESS 100 100 ok"])
                self.stdout = None

            def wait(self, timeout=None):
                return 0

            def terminate(self): ...
            def kill(self): ...

        monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd, **kw: P(cmd))
        adapter = WhisperXAdapter(use_subprocess=True)
        result = adapter._run_whisperx_subprocess(
            tmp_path, wav, TranscriptionConfig(enable_align=True),
            lambda c, t, m: None, lambda: False,
        )
        assert modes == ["transcribe", "align"]
        # raw_segments lấy từ bản align (có words).
        assert "words" in result.raw_segments[0]

    def test_no_align_when_disabled(self, tmp_path, monkeypatch) -> None:
        import json as _json

        import subtitles_extractor.infrastructure.stt.whisperx_adapter as mod
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")
        modes = []

        class P:
            def __init__(self, cmd):
                modes.append(cmd[cmd.index("--mode") + 1] if "--mode" in cmd else "transcribe")
                out = cmd[cmd.index("--output") + 1]
                _json.dump({"segments": [{"start": 0, "end": 2, "text": "你好"}], "language": "zh"},
                           open(out, "w", encoding="utf-8"), ensure_ascii=False)
                self.stderr = iter([])
                self.stdout = None

            def wait(self, timeout=None):
                return 0

            def terminate(self): ...
            def kill(self): ...

        monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd, **kw: P(cmd))
        adapter = WhisperXAdapter(use_subprocess=True)
        adapter._run_whisperx_subprocess(
            tmp_path, wav, TranscriptionConfig(enable_align=False),
            lambda c, t, m: None, lambda: False,
        )
        assert modes == ["transcribe"]  # không có align


class TestCjkAwareJoining:
    """[v3.23.8] Nối từ CJK-aware: Latin có dấu cách, CJK nối liền."""

    def test_detect_cjk(self) -> None:
        assert WhisperXAdapter._is_cjk_text("你好世界这是中文")
        assert not WhisperXAdapter._is_cjk_text("Hello world this is English")

    def test_join_latin_with_spaces(self) -> None:
        out = WhisperXAdapter._join_words(["Help", "everyone", "explore"], is_cjk=False)
        assert out == "Help everyone explore"

    def test_join_latin_no_space_before_punct(self) -> None:
        out = WhisperXAdapter._join_words(["ideas", "."], is_cjk=False)
        assert out == "ideas."

    def test_join_cjk_no_spaces(self) -> None:
        out = WhisperXAdapter._join_words(["你", "好", "世", "界"], is_cjk=True)
        assert out == "你好世界"

    def test_english_words_get_spaces_in_cues(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        words = [
            {"word": "Help", "start": 0.0, "end": 0.4},
            {"word": "everyone", "start": 0.5, "end": 0.9},
            {"word": "explore.", "start": 1.0, "end": 1.5},
        ]
        segment = {"text": "Help everyone explore.", "start": 0.0, "end": 1.5, "words": words}
        events = WhisperXAdapter._segments_to_events([segment], TranscriptionConfig())
        joined = " ".join(e.text for e in events)
        assert "Help everyone" in joined  # CÓ dấu cách, không dính liền
        assert "Helpeveryone" not in joined

    def test_english_longer_cues_than_cjk(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        # Câu tiếng Anh nên dài hơn ~5 ký tự (không vụn như CJK).
        words = [
            {"word": w, "start": i * 0.3, "end": i * 0.3 + 0.25}
            for i, w in enumerate("These aerial robots are replacing manned planes".split())
        ]
        segment = {"text": "x", "start": 0.0, "end": 3.0, "words": words}
        events = WhisperXAdapter._segments_to_events([segment], TranscriptionConfig())
        assert all(len(e.text) > 5 for e in events)


class TestSpeakerLabelOnChange:
    """[v3.23.9] Chỉ gắn [speaker] khi NGƯỜI NÓI THAY ĐỔI."""

    def test_label_only_on_speaker_change(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        segs = [
            {"start": 0, "end": 1, "text": "A1", "speaker": "SPEAKER_00"},
            {"start": 1, "end": 2, "text": "A2", "speaker": "SPEAKER_00"},
            {"start": 2, "end": 3, "text": "B1", "speaker": "SPEAKER_01"},
            {"start": 3, "end": 4, "text": "A3", "speaker": "SPEAKER_00"},
        ]
        events = WhisperXAdapter._segments_to_events(
            segs, TranscriptionConfig(enable_sentence_split=False)
        )
        texts = [e.text for e in events]
        assert texts[0].startswith("[SPEAKER_00]")  # đầu tiên → gắn
        assert not texts[1].startswith("[")          # cùng người → không gắn
        assert texts[2].startswith("[SPEAKER_01]")   # đổi người → gắn
        assert texts[3].startswith("[SPEAKER_00]")   # đổi lại → gắn

    def test_no_speaker_no_label(self) -> None:
        from subtitles_extractor.domain.ports.speech_to_text_port import TranscriptionConfig

        segs = [{"start": 0, "end": 1, "text": "Hello"}]
        events = WhisperXAdapter._segments_to_events(
            segs, TranscriptionConfig(enable_sentence_split=False)
        )
        assert not events[0].text.startswith("[")
