"""[v3.23.630] TTS tăng dần: chữ ký từng dòng, kế hoạch, cache bền, chạy thật qua ViewModel.

Log thật của phim 86 tập: 21 lượt "Tổng hợp" cả phim (3649 dòng, ~18 phút/lượt = 6h14m) dù mỗi lần
chỉ sửa vài chục dòng. Các test này khoá hành vi: dòng không đổi DÙNG LẠI audio cũ, chỉ dòng đổi
được đưa vào engine, thêm/xoá dòng và mở lại ứng dụng vẫn nhận lại đúng audio.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from subtitles_extractor.application.services import tts_incremental_run as run_mod
from subtitles_extractor.application.services.tts_line_cache import (
    MIN_REUSE_RATIO,
    CacheEntry,
    LineCache,
    PlanMode,
    assemble_prior_results,
    build_cache,
    global_key,
    line_signatures,
    plan_incremental,
    remap_cache,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest, TTSSegmentResult
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.infrastructure.tts.line_cache_store import (
    audio_ok,
    cache_path,
    load_cache,
    save_cache,
)
from subtitles_extractor.infrastructure.tts.segment_store import segment_path, segments_dir_for
from subtitles_extractor.infrastructure.tts.timing_math import master_length_samples

SR = 1000
ENGINE = "fake|v1"


def _event(i: int, text: str, start: float | None = None) -> SubtitleEvent:
    start = float(i) if start is None else start
    return SubtitleEvent(
        index=i + 1, text=text, interval=TimeInterval(start_sec=start, end_sec=start + 0.8)
    )


def _events(n: int = 10) -> list[SubtitleEvent]:
    return [_event(i, f"Câu số {i}") for i in range(n)]


def _request(events: list[SubtitleEvent], **kw: object) -> TTSRequest:
    kw.setdefault("normalize", False)
    kw.setdefault("speaker", "Binh")
    return TTSRequest(events=events, **kw)  # type: ignore[arg-type]


def _ok(_path: str) -> bool:
    return True


def _cache_for(request: TTSRequest, tmp: Path | None = None) -> LineCache:
    """Cache như sau một lượt chạy cả phim (segment giả nếu có ``tmp``)."""
    results = []
    for position, event in enumerate(request.events):
        path = ""
        if tmp is not None:
            tmp.mkdir(parents=True, exist_ok=True)
            file = segment_path(tmp, position)
            sf.write(file, np.full(80, 0.1 * (position + 1), dtype=np.float32), SR)
            path = str(file)
        results.append(TTSSegmentResult(
            event_index=position, start_sec=event.start_sec, end_sec=event.end_sec,
            text=event.text, adjusted_start_sec=event.start_sec, segment_audio_path=path or f"x{position}",
        ))
    return build_cache(request, ENGINE, SR, results, line_signatures(request))


class TestSignatures:
    def test_text_change_only_touches_that_line(self) -> None:
        a = line_signatures(_request(_events()))
        events = _events()
        events[4] = _event(4, "Lời đã sửa")
        b = line_signatures(_request(events))
        assert [i for i in range(10) if a[i] != b[i]] == [4]

    def test_voice_change_only_touches_that_line(self) -> None:
        base = _request(_events(), line_voices=tuple("v1" for _ in range(10)))
        voices = ["v1"] * 10
        voices[7] = "v2"
        other = replace(base, line_voices=tuple(voices))
        a, b = line_signatures(base), line_signatures(other)
        assert [i for i in range(10) if a[i] != b[i]] == [7]

    def test_start_change_also_touches_the_previous_line(self) -> None:
        """Khung của dòng phụ thuộc mốc đầu câu kế → dời câu kế phải đọc lại dòng trước nó."""
        a = line_signatures(_request(_events()))
        events = _events()
        events[5] = _event(5, "Câu số 5", start=5.4)
        b = line_signatures(_request(events))
        assert [i for i in range(10) if a[i] != b[i]] == [4, 5]

    def test_last_line_depends_on_media_duration(self) -> None:
        a = line_signatures(_request(_events(), media_duration_s=20.0))
        b = line_signatures(_request(_events(), media_duration_s=25.0))
        assert [i for i in range(10) if a[i] != b[i]] == [9]


class TestGlobalKey:
    def test_audio_affecting_settings_change_the_key(self) -> None:
        base = global_key(_request(_events()), ENGINE)
        assert global_key(_request(_events(), base_speed=1.3), ENGINE) != base
        assert global_key(_request(_events(), speaker="Doan"), ENGINE) != base
        assert global_key(_request(_events(), max_drift_s=1.0), ENGINE) != base
        assert global_key(_request(_events()), "fake|v2") != base

    def test_master_only_settings_keep_the_cache(self) -> None:
        base = global_key(_request(_events()), ENGINE)
        for change in (
            {"normalize": True}, {"target_lufs": -23.0}, {"output_format": "flac"},
            {"output_bitrate_kbps": 128}, {"wav_subtype": "PCM_24"}, {"voice_clarity": False},
            {"media_duration_s": 99.0}, {"retry_count": 9}, {"segments_dir": "/x"},
        ):
            assert global_key(_request(_events(), **change), ENGINE) == base, change


class TestPlan:
    def test_no_cache_runs_everything(self) -> None:
        plan = plan_incremental(None, _request(_events()), ENGINE, _ok)
        assert plan.mode is PlanMode.FULL and plan.reason == "no_cache"
        assert plan.stale == frozenset(range(10))

    def test_nothing_changed(self) -> None:
        request = _request(_events())
        plan = plan_incremental(_cache_for(request), request, ENGINE, _ok)
        assert plan.mode is PlanMode.UNCHANGED and not plan.stale
        assert plan.reuse == {i: i for i in range(10)}

    def test_one_edited_line_is_the_only_stale_one(self) -> None:
        cache = _cache_for(_request(_events()))
        events = _events()
        events[3] = _event(3, "Sửa lại")
        plan = plan_incremental(cache, _request(events), ENGINE, _ok)
        assert plan.mode is PlanMode.INCREMENTAL
        assert plan.stale == frozenset({3}) and plan.reused_count == 9

    def test_settings_change_runs_everything(self) -> None:
        cache = _cache_for(_request(_events()))
        plan = plan_incremental(cache, _request(_events(), base_speed=1.4), ENGINE, _ok)
        assert plan.mode is PlanMode.FULL and plan.reason == "settings_changed"

    def test_engine_change_runs_everything(self) -> None:
        cache = _cache_for(_request(_events()))
        plan = plan_incremental(cache, _request(_events()), "fake|v2", _ok)
        assert plan.mode is PlanMode.FULL

    def test_mostly_changed_runs_everything(self) -> None:
        cache = _cache_for(_request(_events(10)))
        events = [_event(i, f"Mới {i}") for i in range(10)]
        events[0] = _event(0, "Câu số 0")
        plan = plan_incremental(cache, _request(events), ENGINE, _ok)
        assert plan.reused_count < MIN_REUSE_RATIO * 10
        assert plan.mode is PlanMode.FULL and plan.reason == "too_many_changed"

    def test_missing_audio_file_makes_the_line_stale(self) -> None:
        request = _request(_events())
        cache = _cache_for(request)
        gone = cache.entries[2].result.segment_audio_path
        plan = plan_incremental(cache, request, ENGINE, lambda p: p != gone)
        assert plan.stale == frozenset({2}) and plan.mode is PlanMode.INCREMENTAL

    def test_failed_line_is_retried(self) -> None:
        request = _request(_events())
        cache = _cache_for(request)
        broken = replace(cache.entries[6].result, error_msg="engine lỗi")
        entries = dict(cache.entries)
        entries[6] = CacheEntry(cache.entries[6].signature, broken)
        plan = plan_incremental(replace(cache, entries=entries), request, ENGINE, _ok)
        assert plan.stale == frozenset({6})

    def test_failed_lines_never_enter_the_cache(self) -> None:
        request = _request(_events(3))
        results = [
            TTSSegmentResult(event_index=0, start_sec=0, end_sec=1, text="a",
                             segment_audio_path="a.wav"),
            TTSSegmentResult(event_index=1, start_sec=1, end_sec=2, text="b",
                             error_msg="hỏng"),
            TTSSegmentResult(event_index=2, start_sec=2, end_sec=3, text="c"),  # không audio
        ]
        cache = build_cache(request, ENGINE, SR, results, line_signatures(request))
        assert set(cache.entries) == {0}

    def test_deleting_a_line_shifts_positions_but_keeps_audio(self) -> None:
        """Xoá dòng 4: dòng sau dịch lên 1 vị trí vẫn dùng lại audio; chỉ dòng 3 (câu kế đổi)
        phải đọc lại."""
        cache = _cache_for(_request(_events()))
        events = _events()
        del events[4]
        plan = plan_incremental(cache, _request(events), ENGINE, _ok)
        assert plan.mode is PlanMode.INCREMENTAL
        assert plan.stale == frozenset({3})
        assert plan.reuse[5] == 4 and plan.reuse[9] == 8 and 4 not in plan.reuse

    def test_inserting_a_line(self) -> None:
        cache = _cache_for(_request(_events()))
        events = _events()
        events.insert(5, _event(5, "Dòng mới", start=4.9))
        for position in range(6, len(events)):  # giữ đúng mốc của các dòng cũ
            events[position] = _event(position - 1, events[position].text)
        plan = plan_incremental(cache, _request(events), ENGINE, _ok)
        assert plan.stale == frozenset({4, 5})   # dòng trước (câu kế đổi) + dòng mới
        assert plan.reuse[5] == 6

    def test_assemble_prior_covers_every_position(self) -> None:
        request = _request(_events(5))
        cache = _cache_for(request)
        kept = [cache.entries[i].result for i in (0, 1, 3)]
        prior = assemble_prior_results(request, kept)
        assert [r.event_index for r in prior] == [0, 1, 2, 3, 4]
        assert prior[2].was_skipped and prior[4].was_skipped and not prior[3].was_skipped

    def test_remap_cache_renames_paths(self, tmp_path: Path) -> None:
        cache = _cache_for(_request(_events(4)), tmp_path)
        shifted = remap_cache(cache, {0: 0, 2: 1, 3: 2})
        assert set(shifted.entries) == {0, 1, 2}
        assert shifted.entries[1].result.event_index == 1
        assert shifted.entries[1].result.segment_audio_path == str(segment_path(tmp_path, 1))


class TestStore:
    def test_roundtrip(self, tmp_path: Path) -> None:
        cache = _cache_for(_request(_events(4)))
        path = cache_path(tmp_path)
        assert save_cache(path, cache)
        loaded = load_cache(path)
        assert loaded is not None
        assert loaded.global_key == cache.global_key and loaded.sample_rate == SR
        assert loaded.entries == cache.entries
        assert not list(tmp_path.glob("*.tmp"))

    def test_corrupt_or_foreign_files_mean_no_cache(self, tmp_path: Path) -> None:
        path = cache_path(tmp_path)
        assert load_cache(path) is None
        path.write_text("{không phải json", encoding="utf-8")
        assert load_cache(path) is None
        path.write_text(json.dumps({"version": 999, "entries": {}}), encoding="utf-8")
        assert load_cache(path) is None
        path.write_text(
            json.dumps({"version": 1, "global_key": "k", "entries": {"0": {"sig": "s"}}}),
            encoding="utf-8",
        )
        assert load_cache(path) is None

    def test_unknown_result_fields_are_ignored(self, tmp_path: Path) -> None:
        """Bản sau thêm trường vào TTSSegmentResult / bản trước ghi thừa trường: vẫn đọc được."""
        cache = _cache_for(_request(_events(2)))
        path = cache_path(tmp_path)
        save_cache(path, cache)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["entries"]["0"]["result"]["truong_tuong_lai"] = 1
        path.write_text(json.dumps(raw), encoding="utf-8")
        assert load_cache(path) is not None

    def test_audio_ok(self, tmp_path: Path) -> None:
        good = tmp_path / "a.wav"
        sf.write(good, np.zeros(100, dtype=np.float32), SR)
        empty = tmp_path / "b.wav"
        empty.write_bytes(b"")
        assert audio_ok(str(good)) and not audio_ok(str(empty))
        assert not audio_ok(str(tmp_path / "none.wav"))


class _FakeEngine:
    """Engine giả: ghi audio từng dòng đúng như adapter VieNeu (``event_index`` = vị trí trong
    request nhận vào, file ``seg_<vị trí>.wav``). Biên độ phụ thuộc lời nên nhận ra audio mới."""

    supports_line_cache = True
    sample_rate = SR

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.calls = 0

    def line_cache_key(self) -> str:
        return ENGINE

    def get_engine_name(self) -> str:
        return "Fake"

    def is_available(self) -> bool:
        return True

    @staticmethod
    def amplitude(text: str) -> float:
        return 0.05 + (sum(map(ord, text)) % 80) / 100.0

    def generate(self, request, output_path, progress_cb=None, cancel_cb=None):
        self.calls += 1
        segdir = segments_dir_for(output_path, request.segments_dir)
        segdir.mkdir(parents=True, exist_ok=True)
        results = []
        for position, event in enumerate(request.events):
            self.spoken.append(event.text)
            path = segment_path(segdir, position)
            sf.write(path, np.full(60, self.amplitude(event.text), dtype=np.float32), SR)
            results.append(TTSSegmentResult(
                event_index=position, start_sec=event.start_sec, end_sec=event.end_sec,
                text=event.text, subtitle_text=event.text, audio_duration_s=0.06,
                adjusted_start_sec=event.start_sec, segment_audio_path=str(path),
            ))
        last_end = max((e.end_sec for e in request.events), default=0.0)
        length = master_length_samples(last_end, SR, request.media_duration_s)
        sf.write(output_path, np.zeros(length, dtype=np.float32), SR)
        return results


class TestPrepareAndRecord:
    def _full_run(self, tmp_path: Path, events: list[SubtitleEvent]):
        engine = _FakeEngine()
        out = tmp_path / "ep.wav"
        request = _request(events, keep_line_audio=True)
        prepared = run_mod.prepare_run(request, engine, out, True)
        assert prepared is not None and prepared.plan.mode is PlanMode.FULL
        results = engine.generate(request, out)
        run_mod.record_run(prepared, results)
        return engine, out, results

    def test_engines_without_cache_support_are_ignored(self, tmp_path: Path) -> None:
        engine = SimpleNamespace()
        assert run_mod.prepare_run(_request(_events()), engine, tmp_path / "a.wav") is None

    def test_second_run_with_one_edit_only_reads_that_line(self, tmp_path: Path) -> None:
        engine, out, _ = self._full_run(tmp_path, _events())
        assert len(engine.spoken) == 10
        events = _events()
        events[6] = _event(6, "Đã sửa lời")
        request = _request(events, keep_line_audio=True)
        prepared = run_mod.prepare_run(request, engine, out, True)
        assert prepared is not None and prepared.plan.mode is PlanMode.INCREMENTAL
        assert prepared.plan.stale == frozenset({6})
        assert prepared.sample_rate == SR and prepared.total_length_samples > 0
        assert [r.event_index for r in prepared.prior_results] == list(range(10))

    def test_not_incremental_discards_the_cache(self, tmp_path: Path) -> None:
        engine, out, _ = self._full_run(tmp_path, _events())
        request = _request(_events(), keep_line_audio=True)
        prepared = run_mod.prepare_run(request, engine, out, incremental=False)
        assert prepared is not None and prepared.plan.mode is PlanMode.FULL
        assert prepared.plan.reason == "disabled"
        assert not cache_path(segments_dir_for(out)).exists()

    def test_reuse_after_deleting_a_line_renames_the_audio(self, tmp_path: Path) -> None:
        engine, out, _ = self._full_run(tmp_path, _events())
        segdir = segments_dir_for(out)
        before = {
            i: sf.read(segment_path(segdir, i), dtype="float32")[0][0] for i in range(10)
        }
        events = _events()
        del events[2]
        request = _request(events, keep_line_audio=True)
        prepared = run_mod.prepare_run(request, engine, out, True)
        assert prepared is not None and prepared.plan.mode is PlanMode.INCREMENTAL
        # Dòng cũ 5 giờ ở vị trí 4: file seg_000004.wav chứa đúng audio của dòng cũ 5.
        for old, new in ((0, 0), (5, 4), (9, 8)):
            now = sf.read(segment_path(segdir, new), dtype="float32")[0][0]
            assert now == pytest.approx(before[old], abs=1e-3), (old, new)
        reloaded = load_cache(cache_path(segdir))
        assert reloaded is not None and reloaded.entries[4].result.event_index == 4

    def test_record_selected_updates_only_those_lines(self, tmp_path: Path) -> None:
        engine, out, results = self._full_run(tmp_path, _events())
        events = _events()
        events[1] = _event(1, "Sửa tay một dòng")
        request = _request(events, keep_line_audio=True)
        run_mod.record_selected(request, engine, out, results, {1})
        prepared = run_mod.prepare_run(request, engine, out, True)
        assert prepared is not None and prepared.plan.mode is PlanMode.UNCHANGED


# ── Chạy thật qua ViewModel (worker thật, engine giả) ───────────────────────────────────

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from subtitles_extractor.application.use_cases.generate_tts import GenerateTTSUseCase  # noqa: E402
from subtitles_extractor.application.use_cases.resynthesize_lines import (  # noqa: E402
    ResynthesizeLinesUseCase,
)
from subtitles_extractor.presentation.view_models.tts_page_view_model import (  # noqa: E402
    TTSPageViewModel,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait(vm: TTSPageViewModel, app: QApplication, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if not vm.is_busy:
            app.processEvents()
            return
        time.sleep(0.01)
    raise AssertionError("TTS không xong kịp")


def _view_model(engine: _FakeEngine) -> TTSPageViewModel:
    container = SimpleNamespace(
        make_vieneu_tts_adapter=lambda **_kw: engine,
        make_generate_tts_use_case=lambda adapter=None: GenerateTTSUseCase(adapter=adapter),
        make_resynthesize_use_case=lambda adapter: ResynthesizeLinesUseCase(adapter=adapter),
    )
    return TTSPageViewModel(container)  # type: ignore[arg-type]


def _start(vm: TTSPageViewModel, out: Path, **kw: object) -> bool:
    return vm.start_tts(
        engine="vieneu", language="vi-VN", speaker="Binh", output_path=out, normalize=False,
        **kw,  # type: ignore[arg-type]
    )


class TestViewModelEndToEnd:
    def test_second_run_reuses_audio_and_only_reads_the_edited_line(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        engine = _FakeEngine()
        vm = _view_model(engine)
        out = tmp_path / "ep.wav"
        vm.set_source_events(_events(12))
        assert _start(vm, out)
        _wait(vm, app)
        assert len(engine.spoken) == 12 and vm.reuse_summary is None
        first_master = sf.read(out, dtype="float32")[0]

        vm.update_event_text(5, "Câu số 5 đã sửa")
        assert _start(vm, out)
        _wait(vm, app)
        assert engine.spoken[12:] == ["Câu số 5 đã sửa"]          # CHỈ dòng sửa vào engine
        assert vm.reuse_summary == (11, 12, 1)
        assert len(vm.last_results) == 12
        assert vm.last_results[5].text == "Câu số 5 đã sửa"
        assert vm.edited_positions == frozenset()
        # Dòng 5 có audio MỚI, dòng khác giữ nguyên trong master.
        master = sf.read(out, dtype="float32")[0]
        assert len(master) == len(first_master)
        at = lambda audio, pos: audio[int(pos * SR) + 5]  # noqa: E731
        assert at(master, 5) == pytest.approx(_FakeEngine.amplitude("Câu số 5 đã sửa"), abs=0.02)
        assert at(master, 3) == pytest.approx(_FakeEngine.amplitude("Câu số 3"), abs=0.02)

    def test_nothing_changed_only_remerges(self, app: QApplication, tmp_path: Path) -> None:
        engine = _FakeEngine()
        vm = _view_model(engine)
        out = tmp_path / "ep.wav"
        vm.set_source_events(_events(8))
        assert _start(vm, out)
        _wait(vm, app)
        assert _start(vm, out)
        _wait(vm, app)
        assert engine.calls == 1 and vm.reuse_summary == (8, 8, 0)
        assert len(vm.last_results) == 8 and out.exists()

    def test_cache_survives_a_new_session(self, app: QApplication, tmp_path: Path) -> None:
        """Mở lại ứng dụng (ViewModel mới, chưa có kết quả trong bộ nhớ) vẫn dùng lại audio."""
        out = tmp_path / "ep.wav"
        engine = _FakeEngine()
        first = _view_model(engine)
        first.set_source_events(_events(9))
        assert _start(first, out)
        _wait(first, app)

        engine2 = _FakeEngine()
        second = _view_model(engine2)
        events = _events(9)
        events[0] = _event(0, "Mở đầu mới")
        second.set_source_events(events)
        assert _start(second, out)
        _wait(second, app)
        assert engine2.spoken == ["Mở đầu mới"]
        assert second.reuse_summary == (8, 9, 1)

    def test_deleted_line_in_a_new_session_does_not_rerun_the_tail(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        out = tmp_path / "ep.wav"
        first = _view_model(_FakeEngine())
        first.set_source_events(_events(10))
        assert _start(first, out)
        _wait(first, app)

        engine2 = _FakeEngine()
        second = _view_model(engine2)
        events = _events(10)
        del events[1]
        second.set_source_events(events)
        assert _start(second, out)
        _wait(second, app)
        assert engine2.spoken == ["Câu số 0"]        # dòng trước dòng bị xoá (câu kế đổi)
        assert len(second.last_results) == 9

    def test_unticked_runs_the_whole_film(self, app: QApplication, tmp_path: Path) -> None:
        engine = _FakeEngine()
        vm = _view_model(engine)
        out = tmp_path / "ep.wav"
        vm.set_source_events(_events(6))
        assert _start(vm, out)
        _wait(vm, app)
        assert _start(vm, out, incremental=False)
        _wait(vm, app)
        assert len(engine.spoken) == 12 and vm.reuse_summary is None

    def test_manual_resynthesis_keeps_the_cache_valid(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        """"TTS lại dòng chọn" xong thì lần Tổng hợp sau không đọc lại đúng dòng đó."""
        engine = _FakeEngine()
        vm = _view_model(engine)
        out = tmp_path / "ep.wav"
        vm.set_source_events(_events(8))
        assert _start(vm, out)
        _wait(vm, app)
        vm.update_event_text(2, "Sửa rồi TTS lại tay")
        assert vm.resynthesize_lines({2})
        _wait(vm, app)
        spoken = len(engine.spoken)
        assert _start(vm, out)
        _wait(vm, app)
        assert len(engine.spoken) == spoken and vm.reuse_summary == (8, 8, 0)

    def test_structure_edit_keeps_the_disk_cache_in_step(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        engine = _FakeEngine()
        vm = _view_model(engine)
        out = tmp_path / "ep.wav"
        vm.set_source_events(_events(8))
        assert _start(vm, out)
        _wait(vm, app)
        vm.delete_lines([3])
        _wait(vm, app)             # ghép lại master (không TTS)
        before = len(engine.spoken)
        assert _start(vm, out)
        _wait(vm, app)
        assert engine.spoken[before:] == ["Câu số 2"]   # chỉ dòng trước dòng bị xoá
        assert len(vm.last_results) == 7

    def test_cancel_during_incremental_run(
        self, app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tempfile

        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(scratch))
        engine = _FakeEngine()
        vm = _view_model(engine)
        out = tmp_path / "ep.wav"
        vm.set_source_events(_events(8))
        assert _start(vm, out)
        _wait(vm, app)
        vm.update_event_text(1, "đổi")
        original = engine.generate

        def cancelling(request, output_path, progress_cb=None, cancel_cb=None):
            from subtitles_extractor.domain.ports.subtitle_tts_port import TTSCancelledError
            raise TTSCancelledError("huỷ")

        engine.generate = cancelling  # type: ignore[method-assign]
        cancelled: list[bool] = []
        vm.tts_cancelled.connect(lambda: cancelled.append(True))
        assert _start(vm, out)
        _wait(vm, app)
        assert cancelled == [True]
        assert not list(scratch.glob("resynth_seg_*"))   # huỷ xong không để WAV tạm
        # Sau huỷ vẫn chạy lại được và chỉ đọc dòng đã đổi.
        engine.generate = original  # type: ignore[method-assign]
        before = len(engine.spoken)
        assert _start(vm, out)
        _wait(vm, app)
        assert engine.spoken[before:] == ["đổi"]


def test_full_run_keeps_going_when_the_cache_cannot_be_written(tmp_path: Path) -> None:
    engine = _FakeEngine()
    out = tmp_path / "ep.wav"
    request = _request(_events(3), keep_line_audio=True)
    prepared = run_mod.prepare_run(request, engine, out, True)
    assert prepared is not None
    results = engine.generate(request, out)
    # Thư mục cache bị thay bằng một FILE → ghi lỗi nhưng không ném.
    segdir = segments_dir_for(out)
    for item in segdir.iterdir():
        item.unlink()
    segdir.rmdir()
    segdir.write_text("chặn", encoding="utf-8")
    run_mod.record_run(prepared, results)
