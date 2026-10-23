"""[v3.23.617] Quyết định cách nối cả bộ: sao chép luồng hay chuẩn hoá từng tập.

Bộ phim mẫu 86 tập có 6 tập 30 fps (time base 1/15360) lẫn giữa các tập 25 fps (1/12800)
và 4 bộ SPS khác nhau. Sao chép luồng qua concat demuxer cho tệp ra mất khung hình ở
19 tập giữa phim mà ffmpeg không báo lỗi.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from subtitles_extractor.application.services.concat_plan import (
    ConcatInputInfo,
    build_normalize_filter,
    choose_concat_target,
    copy_mismatch_reasons,
    duration_mismatch,
    parse_frame_rate,
)


def _info(name: str, **overrides: object) -> ConcatInputInfo:
    base = ConcatInputInfo(
        path=Path(f"/phim/{name}.mp4"), duration_sec=100.0, video_codec="hevc",
        width=720, height=1280, pix_fmt="yuv420p", frame_rate="25/1",
        time_base="1/12800", extradata_hash="sps-a", audio_codec="aac",
        sample_rate=44100, channels=2,
    )
    return replace(base, **overrides)


def test_identical_episodes_allow_stream_copy() -> None:
    infos = [_info("01"), _info("02"), _info("03")]
    assert copy_mismatch_reasons(infos) == []


def test_frame_rate_mismatch_blocks_stream_copy() -> None:
    infos = [_info("55"), _info("56", frame_rate="30/1", time_base="1/15360")]
    reasons = copy_mismatch_reasons(infos)
    assert any("tốc độ khung" in r and "56.mp4" in r for r in reasons)
    assert any("time base" in r for r in reasons)


def test_parameter_set_mismatch_blocks_stream_copy() -> None:
    """Cùng codec/fps nhưng khác SPS: tệp ra chỉ mang SPS của tập đầu → không sao chép."""
    infos = [_info("23"), _info("24", extradata_hash="sps-b")]
    assert any("SPS" in r for r in copy_mismatch_reasons(infos))


def test_missing_audio_blocks_stream_copy() -> None:
    infos = [_info("01"), _info("02", audio_codec="", sample_rate=0, channels=0)]
    assert copy_mismatch_reasons(infos)


def test_target_follows_majority_frame_rate() -> None:
    infos = [_info(f"{i:02d}") for i in range(1, 6)]
    infos.append(_info("56", frame_rate="30/1", time_base="1/15360"))
    target = choose_concat_target(infos)
    assert target.frame_rate == "25/1"
    assert (target.width, target.height) == (720, 1280)
    assert (target.sample_rate, target.channel_layout) == (44100, "stereo")


def test_target_tie_prefers_higher_frame_rate() -> None:
    infos = [_info("01"), _info("02", frame_rate="30/1")]
    assert choose_concat_target(infos).frame_rate == "30/1"


def test_normalize_filter_resets_timestamps_and_concats_all_inputs() -> None:
    infos = [_info("55"), _info("56", frame_rate="30/1"), _info("57", width=1080, height=1920)]
    graph = build_normalize_filter(infos, choose_concat_target(infos))
    # Mỗi tập về mốc 0 rồi ép cùng tốc độ khung trước khi nối.
    assert graph.count("setpts=PTS-STARTPTS,fps=25/1") == 3
    assert graph.count("asetpts=PTS-STARTPTS") == 3
    # Chỉ tập lệch kích thước mới bị scale/pad.
    assert graph.count("scale=720:1280") == 1
    assert graph.endswith("[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[vout][aout]")


def test_episode_without_audio_gets_silence_of_its_duration() -> None:
    infos = [_info("01"), _info("02", duration_sec=63.5, audio_codec="", channels=0)]
    graph = build_normalize_filter(infos, choose_concat_target(infos))
    assert "anullsrc=r=44100:cl=stereo,atrim=duration=63.500" in graph
    assert "[1:a:0]" not in graph


def test_duration_mismatch_tolerance() -> None:
    assert not duration_mismatch(371.8, 372.4)  # làm tròn ở ranh giới tập
    assert duration_mismatch(371.8, 446.2)  # lỗi thật trên 5 tập mẫu (hình giãn)
    assert not duration_mismatch(0.0, 10.0)  # không biết tổng → không kết luận


def test_parse_frame_rate() -> None:
    assert parse_frame_rate("25/1") == 25.0
    assert round(parse_frame_rate("30000/1001"), 2) == 29.97
    assert parse_frame_rate("0/0") == 0.0
    assert parse_frame_rate("rác") == 0.0
