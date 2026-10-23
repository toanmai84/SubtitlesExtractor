"""[v3.23.596] Elastic timing cho VieNeu/Gemini + mốc dòng kề thật khi TTS lại tập con.

Dữ liệu thật: lượt lồng tiếng 10/10/2026, phim "灭门惨案II借种" (VieNeu v3 Turbo,
base 1,3x, trần 2,5x, "Dung sai cộng dồn" 0,25 s, "Nén êm" 1,5). Mỗi dòng ghi
``(STT, start, end, giọng ở 1,0x)`` lấy từ file ``tts_debug.csv`` (giọng ở 1,0x = "Thời
lượng audio" x "Tốc độ dùng"). Cụm 37-47 là đoạn cãi nhau dày đặc: lượt thật nén dòng 40,
43, 45 lên 2,42x / 2,30x / 2,50x.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from subtitles_extractor.application.use_cases.resynthesize_lines import (
    ResynthesizeLinesUseCase,
    resynthesis_neighbors,
)
from subtitles_extractor.domain.ports.subtitle_tts_port import (
    LineNeighbors,
    TTSRequest,
    TTSSegmentResult,
)
from subtitles_extractor.infrastructure.tts import edge_tts_adapter
from subtitles_extractor.infrastructure.tts import vieneu_tts_adapter as vmod
from subtitles_extractor.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter
from subtitles_extractor.infrastructure.tts.line_placement import (
    ELASTIC_LATE_CAP_S,
    comfort_speed,
    elastic_late_seconds,
    line_context,
    place_line,
)
from subtitles_extractor.infrastructure.tts.timing_math import (
    effective_available_seconds,
    lead_in_seconds,
    total_speed_ratio,
)
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter

#: (STT, start, end, giọng ở 1,0x, tốc độ lượt thật) — tts_debug.csv 10/10/2026.
CLUSTER = [
    (37, 246.467, 247.333, 0.55, 1.3),
    (38, 247.4, 248.257, 1.251, 1.3),
    (39, 250.667, 251.133, 0.61, 1.3),
    (40, 251.333, 252.067, 2.25, 2.417),
    (41, 252.133, 253.067, 1.81, 1.551),
    (42, 253.333, 254.067, 1.629, 1.746),
    (43, 254.267, 255.067, 2.451, 2.297),
    (44, 255.333, 256.6, 1.55, 1.3),
    (45, 256.8, 257.4, 2.15, 2.5),
    (46, 257.467, 258.4, 1.81, 1.769),
    (47, 258.533, 259.0, 0.43, 1.3),
]
BASE, MAX_SPEED, LEAD = 1.3, 2.5, 0.25


def _run_cluster(late: float, comfort: float) -> list[tuple[float, float, float]]:
    """(tốc độ, mốc phát, mốc dứt) từng dòng khi đặt tuần tự như vòng tổng hợp."""
    prev_end = 0.0
    placed = []
    for position, (_n, start, end, voice, _speed) in enumerate(CLUSTER):
        following = CLUSTER[position + 1][1] if position + 1 < len(CLUSTER) else None
        result = place_line(
            start_sec=start, end_sec=end, next_start_sec=following,
            prev_audio_end_sec=prev_end, voice_duration_s=voice, pause_s=0.0,
            base_speed=BASE, max_speed=MAX_SPEED, max_lead_s=LEAD,
            late_s=late, comfort_speed=comfort,
        )
        prev_end = result.play_start_sec + voice / result.speed
        placed.append((result.speed, result.play_start_sec, prev_end))
    return placed


class TestRealCluster:
    def test_without_elastic_reproduces_the_real_run(self) -> None:
        speeds = [speed for speed, _s, _e in _run_cluster(0.0, BASE * 1.5)]
        assert speeds == pytest.approx([row[4] for row in CLUSTER], abs=0.01)

    def test_user_settings_take_lines_off_the_ceiling(self) -> None:
        late = elastic_late_seconds(True, 0.25)
        placed = _run_cluster(late, BASE * 1.5)
        speeds = [speed for speed, _s, _e in placed]
        # Lượt thật: 3 dòng (40, 43, 45).
        assert sum(speed >= 2.0 for speed in speeds) == 1
        assert max(speeds) < 2.3  # lượt thật: dòng 45 kịch trần 2,5x
        # Không dòng nào đè tiếng dòng trước, và không dòng nào lùi quá dung sai.
        for (_s1, _p1, end_prev), (_s2, play, _e2) in pairwise(placed):
            assert play >= end_prev - 1e-6
        for (_speed, play, _end), row in zip(placed, CLUSTER, strict=True):
            assert play - row[1] <= late + 1e-6


class TestPlaceLine:
    def test_matches_the_formula_used_before_v596_when_elastic_off(self) -> None:
        rng = random.Random(596)
        for _ in range(400):
            start = rng.uniform(0, 100)
            end = start + rng.uniform(0.05, 4.0)
            following = rng.choice([None, end + rng.uniform(-0.3, 3.0)])
            prev_end = start - rng.uniform(-0.5, 2.0)
            voice = rng.uniform(0.1, 5.0)
            pause = rng.choice([0.0, 0.3])
            base, ceiling = rng.choice([1.0, 1.3]), rng.choice([2.0, 2.5, 3.0])
            available = effective_available_seconds(start, end, following)
            lead = lead_in_seconds(
                start, prev_end, max_lead_s=LEAD,
                needed_lead_s=max(0.0, voice / max(1.0, base) + pause - available),
            )
            legacy_speed = total_speed_ratio(
                voice, max(0.05, available + lead - pause), base, ceiling
            )
            result = place_line(
                start_sec=start, end_sec=end, next_start_sec=following,
                prev_audio_end_sec=prev_end, voice_duration_s=voice, pause_s=pause,
                base_speed=base, max_speed=ceiling, max_lead_s=LEAD,
            )
            assert result.play_start_sec == pytest.approx(start - lead)
            assert result.effective_available_s == pytest.approx(available + lead)
            assert result.speed == pytest.approx(legacy_speed)

    def test_compresses_to_comfort_before_borrowing(self) -> None:
        # Khung 1,0 s, giọng 2,4 s: không elastic phải 2,4x; có elastic thì dừng ở mức êm
        # 1,95x và chỉ mượn phần còn thiếu của 0,25 s.
        result = place_line(
            start_sec=10.0, end_sec=11.0, next_start_sec=11.1, prev_audio_end_sec=9.0,
            voice_duration_s=2.4, pause_s=0.0, base_speed=1.3, max_speed=2.5,
            max_lead_s=0.0, late_s=0.25, comfort_speed=1.95,
        )
        assert result.speed == pytest.approx(1.95)
        assert result.play_start_sec == pytest.approx(10.0)

    def test_comfortable_line_does_not_borrow(self) -> None:
        result = place_line(
            start_sec=10.0, end_sec=11.0, next_start_sec=11.1, prev_audio_end_sec=9.0,
            voice_duration_s=1.5, pause_s=0.0, base_speed=1.3, max_speed=2.5,
            max_lead_s=0.0, late_s=0.25, comfort_speed=1.95,
        )
        assert result.effective_available_s == pytest.approx(1.0)  # tới 11,1 - 0,1

    def test_following_line_waits_for_the_borrowed_audio(self) -> None:
        result = place_line(
            start_sec=11.1, end_sec=12.0, next_start_sec=None, prev_audio_end_sec=11.3,
            voice_duration_s=0.5, pause_s=0.0, base_speed=1.3, max_speed=2.5,
            max_lead_s=0.25, late_s=0.0, defer_s=0.25,
        )
        assert result.play_start_sec == pytest.approx(11.35)

    def test_deferral_is_capped(self) -> None:
        result = place_line(
            start_sec=11.1, end_sec=12.0, next_start_sec=None, prev_audio_end_sec=12.0,
            voice_duration_s=0.5, pause_s=0.0, base_speed=1.3, max_speed=2.5,
            max_lead_s=0.25, late_s=0.0, defer_s=0.25,
        )
        assert result.play_start_sec == pytest.approx(11.35)

    def test_late_allowance_is_capped_like_edge(self) -> None:
        assert elastic_late_seconds(True, 2.5) == ELASTIC_LATE_CAP_S == 0.5
        assert elastic_late_seconds(True, 0.25) == 0.25
        assert elastic_late_seconds(False, 2.5) == 0.0

    def test_comfort_speed_follows_request(self) -> None:
        request = TTSRequest(
            events=[], base_speed=1.3, max_speed=2.5, comfort_speed_ratio=1.5
        )
        assert comfort_speed(request) == pytest.approx(1.95)
        assert comfort_speed(
            TTSRequest(events=[], base_speed=2.0, max_speed=2.2, comfort_speed_ratio=1.5)
        ) == pytest.approx(2.2)


@dataclass
class _Event:
    start_sec: float
    end_sec: float
    text: str


class TestLineContext:
    def _request(self, **kwargs: Any) -> TTSRequest:
        events = [_Event(1.0, 2.0, "a"), _Event(2.0, 2.6, "b"), _Event(2.6, 3.6, "c")]
        return TTSRequest(events=events, max_drift_s=0.25, media_duration_s=9.0, **kwargs)

    def test_full_run(self) -> None:
        context = line_context(self._request(), 1, 1.9)
        assert (context.next_start_sec, context.prev_audio_end_sec) == (2.6, 1.9)
        assert context.late_s == context.defer_s == 0.25

    def test_last_line_never_spills_past_the_video(self) -> None:
        context = line_context(self._request(), 2, 3.0)
        assert context.next_start_sec == 9.0
        assert context.late_s == 0.0
        assert context.defer_s == 0.25

    def test_elastic_off(self) -> None:
        context = line_context(self._request(elastic_timing=False), 0, 0.0)
        assert context.late_s == context.defer_s == 0.0

    def test_subset_uses_real_neighbours(self) -> None:
        request = self._request(neighbors=(
            LineNeighbors(next_start_sec=3.05, next_fixed=True, prev_audio_end_sec=2.0),
            LineNeighbors(next_start_sec=5.0, next_fixed=False, prev_audio_end_sec=None),
        ))
        first = line_context(request, 0, 0.0)
        assert (first.next_start_sec, first.prev_audio_end_sec, first.late_s) == (
            3.05, 2.0, 0.0,
        )
        second = line_context(request, 1, 3.0)
        assert (second.next_start_sec, second.prev_audio_end_sec, second.late_s) == (
            5.0, 3.0, 0.25,
        )


# ── TTS lại tập con: không đè lên câu không chọn ─────────────────────────────

_SR = 48_000


class _WordEngine:
    """Engine giả: 0,25 s mỗi chữ, 48 kHz như v3 Turbo."""

    sample_rate = _SR

    def list_preset_voices(self) -> list[tuple[str, str]]:
        return [("A — Nam · Bắc", "A")]

    def get_preset_voice(self, voice_id: str | None = None) -> dict[str, str]:
        return {"id": voice_id or "A"}

    def infer(self, text: str, voice: Any = None, **_kwargs: Any) -> np.ndarray:
        samples = np.arange(int(_SR * 0.25 * len(text.split()))) / _SR
        return (0.2 * np.sin(2 * np.pi * 200 * samples)).astype(np.float32)


@pytest.fixture(autouse=True)
def _clear_engine_cache():
    vmod._ENGINE_CACHE.clear()
    yield
    vmod._ENGINE_CACHE.clear()


def _vieneu() -> VieNeuTtsAdapter:
    adapter = VieNeuTtsAdapter(mode="v3turbo", force_cpu=True)
    adapter.is_available = lambda: True  # type: ignore[method-assign]
    adapter._engine = _WordEngine()
    return adapter


def _events() -> list[_Event]:
    return [
        _Event(1.0, 2.0, "một hai ba bốn"),
        _Event(2.0, 2.6, "năm sáu bảy tám chín mười"),
        _Event(2.6, 3.6, "mười một mười hai"),
    ]


@pytest.mark.parametrize("elastic", [False, True])
def test_resynthesized_line_stops_before_the_next_line(
    tmp_path: Path, elastic: bool
) -> None:
    import soundfile as sf

    adapter = _vieneu()
    request = TTSRequest(
        events=_events(), speaker="A", normalize=False, base_speed=1.0, max_speed=3.0,
        keep_line_audio=True, lead_in_s=0.25, elastic_timing=elastic, max_drift_s=0.25,
    )
    output = tmp_path / "out.wav"
    full = adapter.generate(request, output)
    merged = ResynthesizeLinesUseCase(adapter).execute(
        full_request=request, selected_indices={1}, prior_results=full,
        output_path=output, sample_rate=_SR,
        total_length_samples=sf.info(str(output)).frames,
    )
    redone, following = merged[1], merged[2]
    # Trước đây: TTS lại đọc 1,0x tới 3,5 s, đè câu sau (2,6 s) 0,9 giây.
    assert redone.adjusted_start_sec + redone.audio_duration_s <= (
        following.adjusted_start_sec + 1e-6
    )
    assert redone.speed_used > 1.0


class TestResynthesisNeighbors:
    def _prior(self) -> list[TTSSegmentResult]:
        def voiced(index: int, start: float, end: float, audio: float, played: float):
            return TTSSegmentResult(
                index, start, end, "abcde"[index], audio_duration_s=audio,
                adjusted_start_sec=played,
            )

        return [
            voiced(0, 1.0, 2.0, 0.9, 1.0),
            voiced(1, 2.0, 2.6, 0.6, 2.0),
            voiced(2, 2.6, 3.6, 1.0, 2.7),
            TTSSegmentResult(3, 4.0, 5.0, "d", was_skipped=True),
            voiced(4, 5.0, 6.0, 0.5, 5.0),
        ]

    def test_fixed_neighbours_come_from_the_mix(self) -> None:
        events = [_Event(r.start_sec, r.end_sec, r.text) for r in self._prior()]
        neighbors = resynthesis_neighbors(events, [1], self._prior(), 9.0)
        assert neighbors == (
            LineNeighbors(next_start_sec=2.7, next_fixed=True, prev_audio_end_sec=1.9),
        )

    def test_selected_neighbours_and_skipped_lines(self) -> None:
        events = [_Event(r.start_sec, r.end_sec, r.text) for r in self._prior()]
        first, second, last = resynthesis_neighbors(
            events, [1, 2, 4], self._prior(), 9.0
        )
        assert first.next_start_sec == 2.6 and not first.next_fixed
        assert second.prev_audio_end_sec is None  # dòng 1 cũng đang được TTS lại
        # Dòng 3 bị bỏ (không tiếng) → câu trước có tiếng của dòng 4 là dòng 2 — đang
        # được TTS lại nên engine tự theo dõi.
        assert last.prev_audio_end_sec is None
        assert (last.next_start_sec, last.next_fixed) == (9.0, True)


def test_edge_subset_windows_end_at_real_neighbour() -> None:
    events = [_Event(2.0, 2.6, "b"), _Event(5.0, 6.0, "e")]
    windows = EdgeTTSAdapter._compute_extended_windows(
        events, neighbors=(
            LineNeighbors(next_start_sec=2.7, next_fixed=True, prev_audio_end_sec=1.9),
            LineNeighbors(next_start_sec=None, next_fixed=True, prev_audio_end_sec=None),
        ),
    )
    guard = edge_tts_adapter._GAP_GUARD_S
    # Trước đây khung kéo tới tận dòng chọn sau (5,0 s) → 2,9 s.
    assert windows[0] == pytest.approx(2.7 - 2.0 - guard)
    assert windows[1] == pytest.approx(1.0)


# ── Cả hai engine dùng chung cách đặt câu (kỷ luật đồng bộ engine) ───────────

_TEXTS = [
    "A Mặc,", "Mày làm gì thế hả?", "Chồng này...",
    "Các người không phải chết sạch rồi cơ mà?", "Cả nhà mày mới chết sạch ấy!",
    "Tôi thấy tên khốn Hồng Chấn...", "Lúc cả nhà tao chết sạch thì không thấy,",
    "mày đúng lúc vợ tao ngất,", "lại tranh thủ sàm sỡ chiếm tiện nghi.",
    "Mày đúng là đồ khốn nạn!", "Tôi...",
]
_VOICE_BY_TEXT = {text: row[3] for text, row in zip(_TEXTS, CLUSTER, strict=True)}


def _tone(seconds: float, rate: int) -> np.ndarray:
    samples = np.arange(round(seconds * rate)) / rate
    return (0.2 * np.sin(2 * np.pi * 220 * samples)).astype(np.float32)


class _ClusterEngine(_WordEngine):
    """Trả giọng đúng thời lượng thật của từng câu (ở tốc độ 1,0x)."""

    def infer(self, text: str, voice: Any = None, **_kwargs: Any) -> np.ndarray:
        return _tone(_VOICE_BY_TEXT[text.strip()], _SR)


def _cluster_request(elastic: bool) -> TTSRequest:
    events = [
        _Event(row[1] - 246.0, row[2] - 246.0, text)
        for row, text in zip(CLUSTER, _TEXTS, strict=True)
    ]
    return TTSRequest(
        events=events, speaker="A", normalize=False, base_speed=BASE,
        max_speed=MAX_SPEED, lead_in_s=LEAD, elastic_timing=elastic, max_drift_s=0.25,
        comfort_speed_ratio=1.5, retry_count=1, dialog_pause_ms=0,
        media_duration_s=20.0,
    )


def _assert_cluster(results: list[TTSSegmentResult], elastic: bool) -> None:
    speeds = [result.speed_used for result in results]
    if not elastic:
        assert speeds == pytest.approx([row[4] for row in CLUSTER], abs=0.02)
        return
    assert sum(speed >= 2.0 for speed in speeds) == 1
    for previous, current in pairwise(results):
        assert current.adjusted_start_sec >= (
            previous.adjusted_start_sec + previous.audio_duration_s - 1e-3
        )


@pytest.mark.parametrize("elastic", [False, True])
def test_vieneu_places_the_real_cluster(tmp_path: Path, elastic: bool) -> None:
    adapter = _vieneu()
    adapter._engine = _ClusterEngine()
    results = adapter.generate(_cluster_request(elastic), tmp_path / "o.wav")
    _assert_cluster(results, elastic)


@pytest.mark.parametrize("elastic", [False, True])
def test_gemini_places_the_real_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, elastic: bool
) -> None:
    from types import SimpleNamespace

    from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import GeminiTTSAdapter

    rate = 24_000

    class _Models:
        def generate_content(self, *, contents: str, **_kwargs: Any) -> Any:
            pcm = (_tone(_VOICE_BY_TEXT[contents.strip()], rate) * 32767).astype("<i2")
            blob = SimpleNamespace(data=pcm.tobytes(), mime_type="audio/pcm")
            part = SimpleNamespace(inline_data=blob)
            return SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]
            )

    adapter = GeminiTTSAdapter("k")
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    monkeypatch.setattr(
        adapter, "_make_client", lambda **_kw: SimpleNamespace(models=_Models())
    )
    request = _cluster_request(elastic)
    request = TTSRequest(**{
        **{field: getattr(request, field) for field in request.__dataclass_fields__},
        "language": "gemini-3.8-flash-lite-tts",
    })
    _assert_cluster(adapter.generate(request, tmp_path / "o.wav"), elastic)
