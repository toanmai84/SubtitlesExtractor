"""[v3.23.647] Điểm kiểm tra giữa lượt TTS cả phim: sập/huỷ giữa chừng thì lần sau chạy tiếp từ dòng đã xong.

Log thật: lượt TTS cả phim 3649 dòng mất ~19 phút; cache từng dòng (v3.23.630) chỉ được ghi SAU khi cả lượt xong, và lượt
dở dang dưới 30% dòng dùng lại còn bị coi là "đổi quá nhiều" ⇒ huỷ/sập ở phút 15 là chạy lại từ đầu.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services import tts_incremental_run as run_mod
from subtitles_extractor.application.services.tts_line_cache import (
    LineCache,
    PlanMode,
    line_signatures,
    plan_incremental,
)
from subtitles_extractor.infrastructure.tts.line_cache_store import cache_path, load_cache, save_cache
from subtitles_extractor.infrastructure.tts.segment_store import segment_path, segments_dir_for
from tests.unit.application.test_tts_incremental_v323630 import (
    SR,
    _cache_for,
    _events,
    _FakeEngine,
    _request,
    _start,
    _view_model,
    _wait,
)

app = pytest.fixture(scope="module")(lambda: QApplication.instance() or QApplication([]))


class TestPartialCachePlan:
    def test_interrupted_run_resumes_even_when_few_lines_are_done(self) -> None:
        request = _request(_events(20))
        full = _cache_for(request)
        two_done = LineCache(full.global_key, full.sample_rate, {k: full.entries[k] for k in (0, 1)}, complete=False)
        plan = plan_incremental(two_done, request, "fake|v1", lambda _p: True)
        assert plan.mode is PlanMode.INCREMENTAL and plan.reason == "resumed"
        assert sorted(plan.reuse.values()) == [0, 1] and plan.stale == frozenset(range(2, 20))

    def test_same_cache_marked_complete_still_obeys_the_reuse_ratio(self) -> None:
        request = _request(_events(20))
        full = _cache_for(request)
        two_done = LineCache(full.global_key, full.sample_rate, {k: full.entries[k] for k in (0, 1)})
        assert plan_incremental(two_done, request, "fake|v1", lambda _p: True).mode is PlanMode.FULL

    def test_partial_cache_with_nothing_reusable_runs_everything(self) -> None:
        request = _request(_events(6))
        full = _cache_for(request)
        partial = LineCache(full.global_key, full.sample_rate, dict(full.entries), complete=False)
        plan = plan_incremental(partial, request, "fake|v1", lambda _p: False)       # mọi file audio đã mất
        assert plan.mode is PlanMode.FULL

    def test_a_changed_setting_still_discards_a_partial_cache(self) -> None:
        request = _request(_events(6))
        full = _cache_for(request)
        partial = LineCache("khoá-cũ", full.sample_rate, dict(full.entries), complete=False)
        assert plan_incremental(partial, request, "fake|v1", lambda _p: True).reason == "settings_changed"


class TestStore:
    def test_complete_flag_roundtrips_and_old_files_default_to_complete(self, tmp_path: Path) -> None:
        full = _cache_for(_request(_events(3)))
        path = tmp_path / "line_cache.json"
        save_cache(path, LineCache(full.global_key, full.sample_rate, dict(full.entries), complete=False))
        assert load_cache(path).complete is False                                      # type: ignore[union-attr]
        raw = json.loads(path.read_text(encoding="utf-8"))
        del raw["complete"]                                                              # file của bản cũ
        path.write_text(json.dumps(raw), encoding="utf-8")
        assert load_cache(path).complete is True                                       # type: ignore[union-attr]


class _CrashingEngine(_FakeEngine):
    """Engine giả có điểm kiểm tra như VieNeu: ghi sau MỖI dòng, rồi sập sau ``crash_after`` dòng."""

    def __init__(self, crash_after: int | None = None) -> None:
        super().__init__()
        self.crash_after = crash_after
        self._cb = None

    def set_checkpoint_callback(self, callback) -> None:      # noqa: ANN001
        self._cb = callback

    def generate(self, request, output_path, progress_cb=None, cancel_cb=None):      # noqa: ANN001
        self.calls += 1
        callback, self._cb = self._cb, None
        segdir = segments_dir_for(output_path, request.segments_dir)
        segdir.mkdir(parents=True, exist_ok=True)
        results = []
        from subtitles_extractor.domain.ports.subtitle_tts_port import TTSSegmentResult

        for position, event in enumerate(request.events):
            if self.crash_after is not None and position >= self.crash_after:
                raise RuntimeError("máy sập giữa chừng")
            self.spoken.append(event.text)
            path = segment_path(segdir, position)
            sf.write(path, np.full(60, self.amplitude(event.text), dtype=np.float32), SR)
            results.append(TTSSegmentResult(
                event_index=position, start_sec=event.start_sec, end_sec=event.end_sec,
                text=event.text, subtitle_text=event.text, audio_duration_s=0.06,
                adjusted_start_sec=event.start_sec, segment_audio_path=str(path),
            ))
            if callback is not None:
                callback(list(results), SR)
        from subtitles_extractor.infrastructure.tts.timing_math import master_length_samples

        last_end = max((e.end_sec for e in request.events), default=0.0)
        sf.write(output_path, np.zeros(master_length_samples(last_end, SR, request.media_duration_s), dtype=np.float32), SR)
        return results


class TestResumeEndToEnd:
    def test_crash_then_rerun_only_speaks_the_missing_lines(self, app: QApplication, tmp_path: Path) -> None:
        out = tmp_path / "ep.wav"
        crashing = _CrashingEngine(crash_after=3)                     # 3/20 = 15% < 30%
        vm = _view_model(crashing)
        vm.set_source_events(_events(20))
        errors: list[str] = []
        vm.error_occurred.connect(errors.append)
        assert _start(vm, out)
        _wait(vm, app)
        assert errors and "sập" in errors[0]
        partial = load_cache(cache_path(segments_dir_for(out, "")))
        assert partial is not None and partial.complete is False and sorted(partial.entries) == [0, 1, 2]

        healthy = _CrashingEngine()
        vm2 = _view_model(healthy)                                     # như mở lại ứng dụng
        vm2.set_source_events(_events(20))
        assert _start(vm2, out)
        _wait(vm2, app)
        assert healthy.spoken == [f"Câu số {i}" for i in range(3, 20)]      # CHỈ 17 dòng còn lại
        assert vm2.reuse_summary == (3, 20, 17)
        assert len(vm2.last_results) == 20 and out.exists()
        final = load_cache(cache_path(segments_dir_for(out, "")))
        assert final is not None and final.complete is True and len(final.entries) == 20

    def test_callback_is_single_use_so_later_partial_runs_cannot_corrupt_the_cache(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        out = tmp_path / "ep.wav"
        engine = _CrashingEngine()
        vm = _view_model(engine)
        vm.set_source_events(_events(10))
        assert _start(vm, out)
        _wait(vm, app)
        assert engine._cb is None
        before = load_cache(cache_path(segments_dir_for(out, "")))
        vm.update_event_text(4, "Câu số 4 đã sửa")
        assert _start(vm, out)                                         # chạy tăng dần 1 dòng
        _wait(vm, app)
        after = load_cache(cache_path(segments_dir_for(out, "")))
        assert before is not None and after is not None and after.complete and len(after.entries) == 10

    def test_record_checkpoint_ignores_empty_or_bad_sample_rate(self, tmp_path: Path) -> None:
        engine = _FakeEngine()
        request = _request(_events(4), keep_line_audio=True)
        run = run_mod.prepare_run(request, engine, tmp_path / "ep.wav", True)
        assert run is not None
        assert run_mod.record_checkpoint(run, [], SR) is False
        assert run_mod.record_checkpoint(run, [object()], 0) is False       # type: ignore[list-item]
        assert not cache_path(segments_dir_for(tmp_path / "ep.wav", "")).exists()
        assert len(line_signatures(request)) == 4


class TestVieNeuCheckpointThrottle:
    def _adapter(self):
        from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter

        return VieNeuTtsAdapter()

    def test_fires_every_n_lines_or_seconds_and_swallows_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as module

        adapter = self._adapter()
        adapter._checkpoint_every_lines, adapter._checkpoint_every_s = 5, 30.0
        clock = [100.0]
        monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
        calls: list[int] = []

        def callback(results, rate):          # noqa: ANN001
            calls.append(len(results))
            if len(calls) == 2:
                raise OSError("đĩa đầy")

        state = [0.0, 100.0]
        results: list[object] = []
        for _ in range(12):
            results.append(object())
            adapter._maybe_checkpoint(callback, results, SR, state)   # type: ignore[arg-type]
        assert calls == [5, 10]                                       # lỗi lần 2 bị nuốt, vẫn tiếp tục
        clock[0] += 31.0
        results.append(object())
        adapter._maybe_checkpoint(callback, results, SR, state)       # type: ignore[arg-type]
        assert calls[-1] == 13                                        # đủ thời gian dù chưa đủ số dòng

    def test_no_callback_is_a_noop(self) -> None:
        adapter = self._adapter()
        adapter._maybe_checkpoint(None, [object()] * 50, SR, [0.0, 0.0])    # type: ignore[arg-type]


class TestResumeTwice:
    """Rà soát độc lập: lượt TIẾP TỤC cũng phải ghi điểm kiểm tra, nếu không bị ngắt lần nữa là mất cả lượt."""

    def test_second_interruption_keeps_the_progress_of_the_resumed_run(self, app: QApplication, tmp_path: Path) -> None:
        out = tmp_path / "ep.wav"
        segdir = segments_dir_for(out, "")
        for crash_after, expected_cached in ((3, 3), (10, 13)):
            engine = _CrashingEngine(crash_after=crash_after)
            vm = _view_model(engine)
            vm.set_source_events(_events(40))
            errors: list[str] = []
            vm.error_occurred.connect(errors.append)
            assert _start(vm, out)
            _wait(vm, app)
            assert errors, crash_after
            cache = load_cache(cache_path(segdir))
            assert cache is not None and cache.complete is False
            assert sorted(cache.entries) == list(range(expected_cached))
            # Segment đã nằm ở thư mục chính, không còn thư mục tạm.
            assert all(Path(e.result.segment_audio_path).parent == segdir for e in cache.entries.values())
            assert not list(segdir.glob("resynth_seg_*"))
        healthy = _CrashingEngine()
        vm3 = _view_model(healthy)
        vm3.set_source_events(_events(40))
        assert _start(vm3, out)
        _wait(vm3, app)
        assert healthy.spoken == [f"Câu số {i}" for i in range(13, 40)]       # chỉ 27 dòng còn lại
        assert len(vm3.last_results) == 40
        assert [r.text for r in vm3.last_results] == [f"Câu số {i}" for i in range(40)]
        final = load_cache(cache_path(segdir))
        assert final is not None and final.complete and len(final.entries) == 40

    def test_leftover_temp_dirs_of_a_hard_crash_are_cleaned(self, app: QApplication, tmp_path: Path) -> None:
        out = tmp_path / "ep.wav"
        engine = _CrashingEngine(crash_after=2)
        vm = _view_model(engine)
        vm.set_source_events(_events(12))
        assert _start(vm, out)
        _wait(vm, app)
        segdir = segments_dir_for(out, "")
        leftover = segdir / "resynth_seg_dead"
        leftover.mkdir()
        (leftover / "seg_000009.wav").write_bytes(b"x" * 100)
        healthy = _CrashingEngine()
        vm2 = _view_model(healthy)
        vm2.set_source_events(_events(12))
        assert _start(vm2, out)
        _wait(vm2, app)
        assert not leftover.exists()
