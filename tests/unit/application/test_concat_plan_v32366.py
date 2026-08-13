"""[v3.23.366] Kiểm thử service lập kế hoạch nối video cả bộ."""
from __future__ import annotations

from pathlib import Path

from subtitles_extractor.application.services.concat_plan import (
    ConcatPlan,
    build_concat_list_content,
    default_concat_output,
    find_concat_videos,
    natural_sort_key,
)


def test_natural_sort_orders_episodes_by_number() -> None:
    names = [Path(f"第{n}集.mkv") for n in [10, 2, 1, 84, 11, 3]]
    ordered = sorted(names, key=natural_sort_key)
    assert [p.stem for p in ordered] == [
        "第1集", "第2集", "第3集", "第10集", "第11集", "第84集"
    ]


def test_natural_sort_mixed_text_and_numbers() -> None:
    names = [Path("ep2.mp4"), Path("ep10.mp4"), Path("ep1.mp4")]
    ordered = sorted(names, key=natural_sort_key)
    assert [p.stem for p in ordered] == ["ep1", "ep2", "ep10"]


def test_build_concat_list_escapes_single_quote() -> None:
    content = build_concat_list_content([Path("/v/a.mkv"), Path("/v/b's.mkv")])
    lines = content.strip().splitlines()
    assert lines[0].startswith("file '") and lines[0].endswith("a.mkv'")
    # Dấu nháy đơn trong tên phải được escape theo quy tắc ffmpeg.
    assert "b'\\''s.mkv" in lines[1]


def test_find_concat_videos_filters_and_sorts(tmp_path: Path) -> None:
    # Tạo vài tệp: 2 bản đã xuất bản + 1 video gốc + 1 tệp không phải video.
    (tmp_path / "第2集_out.mkv").write_bytes(b"x")
    (tmp_path / "第10集_out.mkv").write_bytes(b"x")
    (tmp_path / "第1集_out.mkv").write_bytes(b"x")
    (tmp_path / "第1集.mp4").write_bytes(b"x")       # gốc — bị lọc khi có name_filter
    (tmp_path / "ghichu.txt").write_text("x")       # không phải video

    published = find_concat_videos(tmp_path, name_filter="_out")
    assert [p.name for p in published] == [
        "第1集_out.mkv", "第2集_out.mkv", "第10集_out.mkv"
    ]

    # Không lọc → lấy tất cả video (kể cả gốc), vẫn sắp thứ tự tự nhiên.
    all_videos = find_concat_videos(tmp_path)
    assert len(all_videos) == 4


def test_find_concat_videos_excludes_output(tmp_path: Path) -> None:
    (tmp_path / "a.mkv").write_bytes(b"x")
    (tmp_path / "b.mkv").write_bytes(b"x")
    (tmp_path / "tron_bo.mkv").write_bytes(b"x")
    result = find_concat_videos(tmp_path, exclude_names={"tron_bo.mkv"})
    assert [p.name for p in result] == ["a.mkv", "b.mkv"]


def test_concat_plan_validity() -> None:
    assert ConcatPlan([Path("a.mkv"), Path("b.mkv")], Path("o.mkv")).is_valid
    assert not ConcatPlan([Path("a.mkv")], Path("o.mkv")).is_valid


def test_default_output_uses_folder_name(tmp_path: Path) -> None:
    out = default_concat_output(tmp_path, [tmp_path / "a.mkv"])
    assert out.name == f"{tmp_path.name}_tron_bo.mkv"
    assert out.parent == tmp_path
