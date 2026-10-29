"""[v3.23.658] Dây chuyền phim bộ — trạng thái từng tập theo tệp có sẵn, kế hoạch chạy theo khâu, trạng thái bền."""

from __future__ import annotations

from pathlib import Path

from subtitles_extractor.application.services.series_pipeline import (
    STATE_FILENAME,
    SeriesRunState,
    SeriesStage,
    StageState,
    episodes_for_stage,
    first_stage_with_work,
    load_run_state,
    remaining_work,
    save_run_state,
    scan_series,
)

S = SeriesStage


def _touch(path: Path, data: bytes = b"x") -> None:
    path.write_bytes(data)


def _series(tmp_path: Path) -> Path:
    """4 tập: ep1 xong cả bộ, ep2 đã dịch, ep3 mới trích xuất, ep4 chưa gì. Kèm tệp xuất bản (không là tập nguồn)."""
    for n in range(1, 5):
        _touch(tmp_path / f"ep{n}.mp4")
    for n in (1, 2, 3):
        (tmp_path / f"ep{n}.original.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\n你好\n", encoding="utf-8")
    for n in (1, 2):
        (tmp_path / f"ep{n}.translate.vi.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nChào\n", encoding="utf-8")
    (tmp_path / "ep1.tts.vi.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nChào\n", encoding="utf-8")
    _touch(tmp_path / "ep1.wav")
    _touch(tmp_path / "ep1_phude_thuyetminh.mkv")
    return tmp_path


def test_scan_reads_states_from_files(tmp_path: Path) -> None:
    statuses = {s.name: s for s in scan_series(_series(tmp_path))}
    assert list(statuses) == ["ep1.mp4", "ep2.mp4", "ep3.mp4", "ep4.mp4"]   # tệp xuất bản không là tập
    assert all(state is StageState.DONE for state in statuses["ep1.mp4"].states.values())
    assert statuses["ep1.mp4"].next_stage() is None
    ep2 = statuses["ep2.mp4"].states
    assert ep2[S.TRANSLATE] is StageState.DONE and ep2[S.TTS] is StageState.READY
    assert ep2[S.PUBLISH] is StageState.BLOCKED
    assert statuses["ep3.mp4"].next_stage() is S.TRANSLATE
    ep4 = statuses["ep4.mp4"].states
    assert ep4[S.EXTRACT] is StageState.READY and ep4[S.TRANSLATE] is StageState.BLOCKED


def test_default_plan_stops_after_translation(tmp_path: Path) -> None:
    statuses = scan_series(_series(tmp_path))
    assert episodes_for_stage(statuses, S.EXTRACT) == [tmp_path / "ep4.mp4"]
    assert episodes_for_stage(statuses, S.TRANSLATE) == [tmp_path / "ep3.mp4"]   # ep4 chưa có phụ đề gốc
    assert first_stage_with_work(statuses, S.TRANSLATE) is S.EXTRACT
    assert first_stage_with_work(statuses, S.TRANSLATE, after=S.EXTRACT) is S.TRANSLATE
    assert first_stage_with_work(statuses, S.TRANSLATE, after=S.TRANSLATE) is None   # không tự sang TTS
    assert remaining_work(statuses, S.TRANSLATE) == {S.EXTRACT: 1, S.TRANSLATE: 2}
    # Chạy hết tới Xuất bản: sau Dịch còn TTS (ep2) — xuất bản chưa có tập nào sẵn sàng.
    assert first_stage_with_work(statuses, S.PUBLISH, after=S.TRANSLATE) is S.TTS


def test_state_survives_restart_and_bad_files(tmp_path: Path) -> None:
    folder = _series(tmp_path)
    assert load_run_state(folder) == SeriesRunState()          # chưa có tệp → mặc định (dừng sau Dịch)
    state = SeriesRunState(stop_after=S.PUBLISH, running=True, stage="translate")
    state.record_failures(S.TRANSLATE, ["ep3.mp4"], "hết hạn mức")
    assert save_run_state(folder, state, clock=lambda: 123.0)
    again = load_run_state(folder)
    assert again.running and again.stop_after is S.PUBLISH and again.stage == "translate"
    assert again.failures == {"ep3.mp4": {"translate": "hết hạn mức"}} and again.updated_at == 123.0
    statuses = {s.name: s for s in scan_series(folder, state=again)}
    assert statuses["ep3.mp4"].states[S.TRANSLATE] is StageState.FAILED
    assert episodes_for_stage(list(statuses.values()), S.TRANSLATE) == [folder / "ep3.mp4"]   # lỗi vẫn chạy lại được
    again.clear_failures(S.TRANSLATE, ["ep3.mp4"])
    assert again.failures == {}
    for bad in ("{", "[]", '{"version": 99}', '{"version": 1, "stop_after": "xyz", "failures": {"a": 5}}'):
        (folder / STATE_FILENAME).write_text(bad, encoding="utf-8")
        loaded = load_run_state(folder)
        assert loaded.stop_after is S.TRANSLATE and loaded.failures == {}


def test_unwritable_state_is_not_fatal(tmp_path: Path) -> None:
    missing = tmp_path / "khong_co" / "sau"
    assert not save_run_state(missing, SeriesRunState())
