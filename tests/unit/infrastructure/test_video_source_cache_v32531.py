"""[v3.23.131] Test: (1) cache video theo NGUỒN (tra cloud trước khi nén);
(2) trần lô cho model lite; (3) hằng số liên quan.
"""

from __future__ import annotations

from pathlib import Path

from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _LITE_MODEL_BATCH_CAP,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
    VideoChunk,
)


def _provider(tmp_path: Path) -> GeminiVideoContextProvider:
    return GeminiVideoContextProvider(api_key="KEY1", cache_db_path=tmp_path / "c.db")


def _chunk(idx: int, start: float, end: float, tmp_path: Path) -> VideoChunk:
    return VideoChunk(idx, tmp_path / "video.mkv", start, end)


def test_source_key_is_deterministic(tmp_path: Path) -> None:
    prov = _provider(tmp_path)
    vp = tmp_path / "video.mkv"
    vp.write_bytes(b"x" * 100)
    ch = _chunk(0, 0.0, 100.0, tmp_path)
    k1 = prov._chunk_source_key(vp, ch)
    k2 = prov._chunk_source_key(vp, ch)
    assert k1 == k2 and len(k1) == 64


def test_source_key_changes_with_chunk_range(tmp_path: Path) -> None:
    prov = _provider(tmp_path)
    vp = tmp_path / "video.mkv"
    vp.write_bytes(b"x" * 100)
    k_a = prov._chunk_source_key(vp, _chunk(0, 0.0, 100.0, tmp_path))
    k_b = prov._chunk_source_key(vp, _chunk(1, 100.0, 200.0, tmp_path))
    assert k_a != k_b


def test_source_key_changes_with_api_key(tmp_path: Path) -> None:
    vp = tmp_path / "video.mkv"
    vp.write_bytes(b"x" * 100)
    ch = _chunk(0, 0.0, 100.0, tmp_path)
    p1 = GeminiVideoContextProvider(api_key="KEY1", cache_db_path=tmp_path / "a.db")
    p2 = GeminiVideoContextProvider(api_key="KEY2", cache_db_path=tmp_path / "b.db")
    assert p1._chunk_source_key(vp, ch) != p2._chunk_source_key(vp, ch)


def test_resolve_from_source_cache_returns_none_when_no_split(tmp_path: Path) -> None:
    prov = _provider(tmp_path)
    plan = prov.plan_chunks  # chỉ kiểm import; tạo plan giả đơn giản bên dưới
    from types import SimpleNamespace

    fake_plan = SimpleNamespace(needs_split=False, chunks=[])
    assert prov._resolve_chunks_from_source_cache(fake_plan, tmp_path / "v.mkv") is None
    assert plan is not None  # plan_chunks tồn tại


def test_lite_batch_cap_value() -> None:
    # Trần phải nằm trong khoảng khuyến nghị 40-60 cho flash-lite.
    assert 40 <= _LITE_MODEL_BATCH_CAP <= 60
