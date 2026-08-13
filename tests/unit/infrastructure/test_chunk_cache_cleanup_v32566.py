"""[v3.23.166] Test dọn cache file đoạn video theo ngân sách (LRU + tuổi + protected).

Từ v157 file đoạn nén được GIỮ để tái dùng khi xoay key -> phải dọn tự động kẻo phình
ổ cứng. Kiểm luật thuần: giữ file phiên hiện tại; xoá quá hạn tuổi trước; nếu vẫn vượt
dung lượng thì xoá LRU (cũ nhất) tới khi về dưới ngân sách.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.chunk_cache_cleanup import (
    ChunkFileInfo,
    plan_chunk_cache_cleanup,
)

_MB = 1024 * 1024
_HOUR = 3600.0
_NOW = 1_000_000.0


def _f(path: str, mb: float, age_hours: float) -> ChunkFileInfo:
    return ChunkFileInfo(path, int(mb * _MB), _NOW - age_hours * _HOUR)


def test_deletes_over_age_files_first() -> None:
    files = [
        _f("old.mp4", 10, age_hours=100),   # quá 72h
        _f("fresh.mp4", 10, age_hours=1),
    ]
    result = plan_chunk_cache_cleanup(
        files, max_total_bytes=0, max_age_seconds=72 * _HOUR, now_epoch=_NOW,
    )
    assert result == ["old.mp4"]


def test_lru_eviction_when_over_budget() -> None:
    # Ngân sách 25MB, có 4 file 10MB (tổng 40) -> xoá 2 file CŨ NHẤT (15MB dư).
    files = [
        _f("a.mp4", 10, age_hours=4),
        _f("b.mp4", 10, age_hours=3),
        _f("c.mp4", 10, age_hours=2),
        _f("d.mp4", 10, age_hours=1),
    ]
    result = plan_chunk_cache_cleanup(
        files, max_total_bytes=25 * _MB, max_age_seconds=0, now_epoch=_NOW,
    )
    assert result == ["a.mp4", "b.mp4"]  # cũ nhất trước, tới khi <= 25MB


def test_protected_never_deleted() -> None:
    # File phiên hiện tại (protected) KHÔNG bị xoá dù cũ nhất và vượt ngân sách.
    files = [
        _f("session.mp4", 30, age_hours=200),  # cũ + to nhưng đang dùng
        _f("other.mp4", 10, age_hours=1),
    ]
    result = plan_chunk_cache_cleanup(
        files, max_total_bytes=5 * _MB, max_age_seconds=72 * _HOUR, now_epoch=_NOW,
        protected_paths=frozenset({"session.mp4"}),
    )
    assert "session.mp4" not in result
    assert "other.mp4" in result


def test_both_criteria_disabled_deletes_nothing() -> None:
    files = [_f("a.mp4", 999, age_hours=999)]
    result = plan_chunk_cache_cleanup(
        files, max_total_bytes=0, max_age_seconds=0, now_epoch=_NOW,
    )
    assert result == []


def test_empty_input() -> None:
    assert plan_chunk_cache_cleanup([], 10 * _MB, _HOUR, _NOW) == []


def test_age_and_budget_combined() -> None:
    # 1 file quá hạn (xoá theo tuổi) + phần còn lại vẫn vượt ngân sách (xoá thêm LRU).
    files = [
        _f("aged.mp4", 5, age_hours=100),   # quá 72h -> xoá theo tuổi
        _f("old_big.mp4", 20, age_hours=5),
        _f("recent.mp4", 20, age_hours=1),
    ]
    result = plan_chunk_cache_cleanup(
        files, max_total_bytes=25 * _MB, max_age_seconds=72 * _HOUR, now_epoch=_NOW,
    )
    # aged bị xoá theo tuổi; còn old_big(20)+recent(20)=40 > 25 -> xoá old_big (LRU).
    assert "aged.mp4" in result
    assert "old_big.mp4" in result
    assert "recent.mp4" not in result


def test_return_order_old_to_new() -> None:
    files = [
        _f("new.mp4", 10, age_hours=1),
        _f("old.mp4", 10, age_hours=5),
        _f("mid.mp4", 10, age_hours=3),
    ]
    result = plan_chunk_cache_cleanup(
        files, max_total_bytes=5 * _MB, max_age_seconds=0, now_epoch=_NOW,
    )
    # Xoá cả 3 (ngân sách 5MB < 10MB mỗi file) theo thứ tự cũ -> mới.
    assert result == ["old.mp4", "mid.mp4", "new.mp4"]


# ── Smoke test tích hợp sweep_chunk_cache với file thật ──────────────────


def test_sweep_deletes_real_files_over_budget(tmp_path) -> None:
    """[v3.23.166] sweep_chunk_cache quét work_dir, giữ protected, dọn phần vượt budget."""
    import os

    from subtitles_extractor.infrastructure.translation.gemini_video_context import (
        GeminiVideoContextProvider,
    )

    provider = GeminiVideoContextProvider(
        cache_db_path=tmp_path / "cache.db",
        work_dir=tmp_path,
        chunk_cache_max_total_mb=0,  # tắt tiêu chí dung lượng
        chunk_cache_max_age_hours=1,  # chỉ dọn theo tuổi > 1h
    )
    old_file = tmp_path / "movie.ctxpart00.h360f1.mp4"
    fresh_file = tmp_path / "movie.ctxpart01.h360f1.mp4"
    old_file.write_bytes(b"0" * 1024)
    fresh_file.write_bytes(b"0" * 1024)
    old_epoch = old_file.stat().st_mtime - 5 * 3600  # 5 giờ trước
    os.utime(old_file, (old_epoch, old_epoch))

    deleted, _freed = provider.sweep_chunk_cache()
    assert deleted == 1
    assert not old_file.exists()
    assert fresh_file.exists()  # trong hạn tuổi -> giữ


def test_sweep_noop_without_work_dir(tmp_path) -> None:
    from subtitles_extractor.infrastructure.translation.gemini_video_context import (
        GeminiVideoContextProvider,
    )

    provider = GeminiVideoContextProvider(
        cache_db_path=tmp_path / "cache.db", work_dir=None,
    )
    assert provider.sweep_chunk_cache() == (0, 0)
