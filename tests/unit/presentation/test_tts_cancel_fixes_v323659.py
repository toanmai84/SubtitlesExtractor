"""[v3.23.659] Rà BUG khâu TTS: huỷ lượt TTS cả phim, huỷ Edge/Gemini, opus 320k, m4a, đếm tập xong."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.use_cases.generate_tts import GenerateTTSUseCase
from subtitles_extractor.application.use_cases.resynthesize_lines import ResynthesizeLinesUseCase
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSCancelledError, TTSRequest, TTSSegmentResult
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.tts.segment_store import segment_path, segments_dir_for
from subtitles_extractor.presentation.view_models.tts_page_view_model import TTSPageViewModel
from subtitles_extractor.presentation.workers import resynthesize_worker, tts_worker

SR = 1000


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _events(n: int = 3) -> list[SubtitleEvent]:
    return [SubtitleEvent(index=i + 1, text=f"line {i}", interval=TimeInterval(float(i * 2), float(i * 2 + 1)))
            for i in range(n)]


class _FakeAdapter:
    def __init__(self, level: float, shift: float, cancel_after: int | None = None) -> None:
        self.level, self.shift, self.cancel_after = level, shift, cancel_after

    def get_engine_name(self) -> str:
        return "Fake"

    def is_available(self) -> bool:
        return True

    def generate(self, request, output_path, progress_cb=None, cancel_cb=None):
        segdir = segments_dir_for(output_path, request.segments_dir)
        segdir.mkdir(parents=True, exist_ok=True)
        results = []
        master = np.zeros(SR * 8, dtype=np.float32)
        for i, ev in enumerate(request.events):
            if self.cancel_after is not None and i >= self.cancel_after:
                raise TTSCancelledError("cancel")
            audio = np.full(SR // 2, self.level, dtype=np.float32)
            path = segment_path(segdir, i)
            sf.write(str(path), audio, SR)
            start = ev.start_sec + self.shift
            master[int(start * SR):int(start * SR) + len(audio)] += audio
            results.append(TTSSegmentResult(event_index=i, start_sec=ev.start_sec, end_sec=ev.end_sec, text=ev.text,
                                            audio_duration_s=0.5, adjusted_start_sec=start,
                                            segment_audio_path=str(path)))
        sf.write(str(output_path), master, SR)
        return results


def _sync(monkeypatch) -> None:
    monkeypatch.setattr(tts_worker.TTSWorker, "start", lambda self: (self.run(), self.finished.emit()))
    monkeypatch.setattr(resynthesize_worker.ResynthesizeWorker, "start",
                        lambda self: (self.run(), self.finished.emit()))


def test_cancelled_full_run_does_not_pair_old_results_with_new_audio(app, tmp_path: Path, monkeypatch) -> None:
    """Huỷ lượt TTS cả phim thứ hai: tệp seg_* đã là của lượt mới mà kết quả vẫn của lượt cũ → ghép lại sai mốc."""
    _sync(monkeypatch)
    adapters = [_FakeAdapter(0.1, 0.0), _FakeAdapter(0.9, 0.5, cancel_after=1)]
    container = SimpleNamespace(
        make_edge_tts_adapter=lambda: adapters.pop(0),
        make_generate_tts_use_case=lambda a: GenerateTTSUseCase(a),
        make_resynthesize_use_case=lambda a: ResynthesizeLinesUseCase(a),
    )
    vm = TTSPageViewModel(container)
    vm.set_source_events(_events())
    out = tmp_path / "ep.wav"
    assert vm.start_tts("edge", output_path=out, normalize=False)
    assert vm.can_resynthesize
    assert vm.start_tts("edge", output_path=out, normalize=False)        # bị huỷ sau dòng 0
    assert not vm.can_resynthesize                                       # không còn ghép audio mới vào mốc cũ
    assert len(vm.last_results) == 3                                     # bảng kết quả vẫn hiện


def test_cancel_before_any_line_written_keeps_links(app, tmp_path: Path, monkeypatch) -> None:
    """Rà độc lập: huỷ trước khi ghi dòng nào (vd Edge lượt 1) — audio từng dòng của lượt trước vẫn đúng, giữ liên kết."""
    _sync(monkeypatch)
    adapters = [_FakeAdapter(0.1, 0.0), _FakeAdapter(0.9, 0.5, cancel_after=0)]
    container = SimpleNamespace(
        make_edge_tts_adapter=lambda: adapters.pop(0),
        make_generate_tts_use_case=lambda a: GenerateTTSUseCase(a),
        make_resynthesize_use_case=lambda a: ResynthesizeLinesUseCase(a),
    )
    vm = TTSPageViewModel(container)
    vm.set_source_events(_events())
    out = tmp_path / "ep.wav"
    assert vm.start_tts("edge", output_path=out, normalize=False)
    import os
    import time as _t
    for r in vm.last_results:  # tệp của lượt trước: cũ hơn lượt sau
        os.utime(r.segment_audio_path, (_t.time() - 60, _t.time() - 60))
    assert vm.start_tts("edge", output_path=out, normalize=False)        # huỷ ngay, chưa ghi dòng nào
    assert vm.can_resynthesize


def test_edge_single_pass_cancel_keeps_master(tmp_path: Path) -> None:
    """Huỷ Edge chế độ một lượt: trước đây ghi đè audio chính bằng bản câm rồi mới báo huỷ."""
    from subtitles_extractor.infrastructure.tts import edge_tts_adapter as ea

    adapter = ea.EdgeTTSAdapter()
    adapter.is_available = lambda: True  # type: ignore[method-assign]

    async def generate(text, voice, speed, request):
        return np.full(12000, 0.3, np.float32), 24000

    adapter._async_gen_with_retry = generate  # type: ignore[method-assign]
    adapter._sync_generate_with_retry = lambda *a, **k: (np.zeros(100, np.float32), 24000)  # type: ignore[method-assign]
    out = tmp_path / "ep.wav"
    sf.write(out, np.full(24000 * 10, 0.5, np.float32), 24000)
    events = [SubtitleEvent(index=i + 1, text=f"xin chào {i}", interval=TimeInterval(i * 2.0, i * 2 + 1.0))
              for i in range(4)]
    request = TTSRequest(events=events, language="vi-VN", double_pass=False, normalize=False, keep_line_audio=True)
    with pytest.raises(TTSCancelledError):
        adapter.generate(request, out, cancel_cb=lambda: True)
    master, _ = sf.read(out)
    assert abs(master).max() == pytest.approx(0.5, abs=1e-3)              # audio chính cũ còn nguyên


def test_gemini_native_cancel_raises(tmp_path: Path, monkeypatch) -> None:
    from subtitles_extractor.infrastructure.tts import gemini_tts_adapter as gm

    adapter = gm.GeminiTTSAdapter(api_key="k")
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    out = tmp_path / "ep.wav"
    sf.write(out, np.full(24000 * 8, 0.5, np.float32), 24000)
    request = TTSRequest(events=_events(), language=gm._NATIVE_AUDIO_MODELS[0], keep_line_audio=True, normalize=False)
    with pytest.raises(TTSCancelledError):
        adapter.generate(request, out, cancel_cb=lambda: True)
    master, _ = sf.read(out)
    assert abs(master).max() == pytest.approx(0.5, abs=1e-3)


def test_opus_default_bitrate_writes_opus(tmp_path: Path) -> None:
    """Mặc định 320 kbps: libopus (mono) từ chối → trước đây âm thầm ghi WAV mà app vẫn nhớ đường dẫn .opus."""
    import shutil

    from subtitles_extractor.infrastructure.tts.audio_mastering import write_audio

    if shutil.which("ffmpeg") is None:
        pytest.skip("cần ffmpeg")
    out = write_audio(np.zeros(24000, np.float32), 24000, tmp_path / "ep.opus", fmt="opus", bitrate_kbps=320)
    assert out.suffix == ".opus" and out.is_file()


def test_resynth_works_when_master_is_m4a(app, tmp_path: Path, monkeypatch) -> None:
    """libsndfile không đọc được .m4a → "TTS lại dòng chọn" / ghép lại sau xoá dòng từng luôn báo lỗi."""
    from tests.unit.presentation.test_dubbing_add_delete_lines_v323625 import _view_model

    vm, started = _view_model(tmp_path, monkeypatch)
    master = tmp_path / "out.m4a"
    master.write_bytes(b"\x00\x00\x00\x18ftypM4A ")                    # libsndfile không mở được
    vm._last_output_path = master
    errors: list[str] = []
    vm.error_occurred.connect(errors.append)
    assert vm.remerge_master(), errors
    assert started and started[-1]["sample_rate"] > 0 and started[-1]["total_length_samples"] > 0


def test_edge_line_files_include_ducking(tmp_path: Path) -> None:
    """Tệp audio từng dòng phải khớp đúng đoạn trong master (kể cả hạ âm khi câu sau chồng lên)."""
    from subtitles_extractor.infrastructure.tts import edge_tts_adapter as ea

    adapter = ea.EdgeTTSAdapter()
    adapter.is_available = lambda: True  # type: ignore[method-assign]

    async def generate(text, voice, speed, request):
        return np.full(36000, 0.3, np.float32), 24000                     # 1,5 s mỗi câu

    adapter._async_gen_with_retry = generate  # type: ignore[method-assign]
    adapter._sync_generate_with_retry = lambda *a, **k: (np.zeros(100, np.float32), 24000)  # type: ignore[method-assign]
    events = [SubtitleEvent(index=1, text="câu một dài", interval=TimeInterval(0.0, 0.8)),
              SubtitleEvent(index=2, text="câu hai", interval=TimeInterval(0.9, 2.0))]
    request = TTSRequest(events=events, language="vi-VN", double_pass=False, normalize=False, keep_line_audio=True,
                         media_duration_s=4.0)
    out = tmp_path / "ep.wav"
    results = adapter.generate(request, out)
    master, sr = sf.read(out, dtype="float32")
    rebuilt = np.zeros_like(master)
    for result in results:
        if not result.segment_audio_path:
            continue
        audio, _ = sf.read(result.segment_audio_path, dtype="float32")
        start = int(round((result.adjusted_start_sec if result.adjusted_start_sec >= 0 else result.start_sec) * sr))
        rebuilt[start:start + len(audio)] += audio[:len(rebuilt) - start]
    assert np.max(np.abs(rebuilt - master)) < 2e-3                         # PCM_16 làm tròn ~1e-4


def test_batch_summary_counts_finished_episodes(app, tmp_path: Path, monkeypatch) -> None:
    """2 tập xong, tập 3 không khởi động được → trước đây báo "xong 1/3"."""
    from PySide6.QtWidgets import QMessageBox

    from subtitles_extractor.presentation.pages import tts_page
    from tests.unit.presentation.test_series_pages_v323658 import _container

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    page = tts_page.TTSPage(_container(tmp_path))
    page._batch_items = [SimpleNamespace(video_path=tmp_path / f"ep{i}.mp4") for i in range(1, 4)]
    page._batch_index, page._batch_done, page._batch_failures, page._batch_cancelled = 2, 2, ["ep3.mp4"], True
    page._finish_batch_tts()
    assert "2/3" in page._stage_lbl.text()
    page.close()
