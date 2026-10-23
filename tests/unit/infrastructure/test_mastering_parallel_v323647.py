"""[v3.23.647] Master âm thanh: bỏ qua đoạn im lặng tuyệt đối + lọc song song theo khối.

Đo thật (tín hiệu mô phỏng 400 s, 4 lõi): ``master_finalize`` 26,3 s → 5,9 s; lệch so với bản cũ −135 dB (max 1e-5),
LUFS giống tới 1e-12.
"""

from __future__ import annotations

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts import audio_mastering as am

SR = 24000


def _speechlike(seconds: int, seed: int = 0, gap_every: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    audio = np.zeros(SR * seconds, dtype=np.float32)
    t = np.arange(SR * 2) / SR
    burst = (np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 450 * t)) * np.hanning(t.size) * 0.4
    for start in range(1, seconds - 3, gap_every):
        audio[start * SR : start * SR + burst.size] += (burst * rng.uniform(0.6, 1.2)).astype(np.float32)
    return audio


class TestActiveIslands:
    def test_islands_split_only_on_long_exact_silence(self) -> None:
        audio = np.zeros(SR * 30, dtype=np.float32)
        audio[SR * 2 : SR * 3] = 0.1          # đảo 1
        audio[SR * 4 : SR * 5] = 0.1          # cách 1 s (< 2,5 s) → cùng đảo
        audio[SR * 20 : SR * 21] = 0.1        # cách ~15 s → đảo riêng
        islands = am.find_active_islands(audio, SR)
        assert len(islands) == 2
        assert islands[0][0] <= SR * 2 and islands[0][1] >= SR * 5 and islands[0][1] < SR * 8
        assert islands[1][0] <= SR * 20 <= SR * 21 <= islands[1][1]

    def test_all_zero_and_tiny_values(self) -> None:
        assert am.find_active_islands(np.zeros(SR * 5, dtype=np.float32), SR) == []
        tiny = np.zeros(SR * 5, dtype=np.float32)
        tiny[SR] = 1e-30
        assert len(am.find_active_islands(tiny, SR)) == 1          # khác 0 dù rất nhỏ → vẫn được lọc

    def test_length_not_multiple_of_frame_and_tail_activity(self) -> None:
        audio = np.zeros(SR * 3 + 123, dtype=np.float32)
        audio[-1] = 0.5
        (island,) = am.find_active_islands(audio, SR)
        assert island[1] == audio.size


class TestProcessInBlocks:
    def test_skip_silence_and_workers_match_the_sequential_result(self) -> None:
        audio = _speechlike(120)
        reference = am.process_in_blocks(audio, SR, am._voice_clarity_core, block_s=10.0, pad_s=0.5)
        fast = am.process_in_blocks(
            audio, SR, am._voice_clarity_core, block_s=10.0, pad_s=0.5, skip_silence=True, workers=4,
        )
        assert fast.shape == reference.shape and fast.dtype == np.float32
        diff = fast.astype(np.float64) - reference
        assert np.sqrt(np.mean(diff**2)) < 1e-4 * np.sqrt(np.mean(reference.astype(np.float64) ** 2))
        assert np.abs(diff).max() < 1e-3

    def test_silent_gaps_stay_exactly_zero_and_all_zero_input_returns_zeros(self) -> None:
        audio = _speechlike(60, gap_every=20)
        out = am.process_in_blocks(audio, SR, am._voice_clarity_core, block_s=10.0, pad_s=0.5, skip_silence=True)
        assert not np.any(out[SR * 25 : SR * 35])
        zeros = am.process_in_blocks(
            np.zeros(SR * 60, dtype=np.float32), SR, am._voice_clarity_core, block_s=10.0, pad_s=0.5, skip_silence=True,
        )
        assert zeros.shape == (SR * 60,) and not np.any(zeros)

    def test_worker_exception_is_propagated(self) -> None:
        def boom(chunk: np.ndarray, sr: int) -> np.ndarray:
            raise RuntimeError("lọc hỏng")

        with pytest.raises(RuntimeError, match="lọc hỏng"):
            am.process_in_blocks(_speechlike(60), SR, boom, block_s=10.0, pad_s=0.5, skip_silence=True, workers=3)


class TestParallelStages:
    def test_kweighted_by_islands_gives_the_same_loudness_as_the_sequential_filter(self) -> None:
        audio = _speechlike(400)
        sequential = am.measure_lufs(audio[: am._LONG_AUDIO_S.__int__() * SR], SR)      # đường tuần tự (≤ ngưỡng dài)
        parallel = am._gated_loudness_from_kweighted(am._kweighted_by_islands(
            audio[: int(am._LONG_AUDIO_S) * SR], SR), SR)
        assert abs(sequential - parallel) < 0.01

    def test_measure_lufs_long_audio_uses_islands_and_matches_the_old_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        audio = _speechlike(400)
        parallel = am.measure_lufs(audio, SR)
        monkeypatch.setattr(am, "_LONG_AUDIO_S", 1e9)           # buộc đường tuần tự cũ
        sequential = am.measure_lufs(audio, SR)
        assert abs(sequential - parallel) < 0.01

    def test_true_peak_limit_parallel_is_identical_to_sequential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        audio = _speechlike(150) * 3.0                                  # nhiều đỉnh vượt trần
        monkeypatch.setattr(am, "_MAX_MASTER_WORKERS", 1)
        sequential = am.true_peak_limit(audio.copy(), SR)
        monkeypatch.setattr(am, "_MAX_MASTER_WORKERS", 4)
        monkeypatch.setattr(am.os, "cpu_count", lambda: 4)
        parallel = am.true_peak_limit(audio.copy(), SR)
        assert np.array_equal(sequential, parallel)
        assert np.abs(parallel).max() <= 0.97

    def test_master_finalize_long_audio_matches_the_sequential_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        audio = _speechlike(360)
        fast = am.master_finalize(audio.copy(), SR)
        monkeypatch.setattr(am, "_MAX_MASTER_WORKERS", 1)
        slow_clarity = am.process_in_blocks(audio, SR, am._voice_clarity_core)       # tuần tự, không bỏ im lặng
        assert fast.shape == audio.shape
        assert abs(am.measure_lufs(fast, SR) - (-16.0)) < 0.5
        assert not np.any(np.isnan(fast)) and slow_clarity.shape == audio.shape

    def test_worker_count_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(am.os, "cpu_count", lambda: 64)
        assert am.master_worker_count() == am._MAX_MASTER_WORKERS
        monkeypatch.setattr(am.os, "cpu_count", lambda: None)
        assert am.master_worker_count() == 1


class TestIslandEdges:
    """Rà soát độc lập: bộ lọc zero-phase có tiền-vọng/đuôi ~20–50 ms ngoài câu — đảo phải chừa lề."""

    def test_filter_tails_next_to_a_frame_aligned_phrase_are_kept(self) -> None:
        audio = np.zeros(SR * 40, dtype=np.float32)
        t = np.arange(int(SR * 1.0)) / SR
        phrase = (np.sin(2 * np.pi * 220 * t) * 0.5).astype(np.float32)        # cắt cứng, không fade
        start = SR * 10                                                           # đúng biên khung 0,25 s
        audio[start : start + phrase.size] = phrase
        reference = am.process_in_blocks(audio, SR, am._voice_clarity_core, block_s=5.0, pad_s=1.0)
        fast = am.process_in_blocks(
            audio, SR, am._voice_clarity_core, block_s=5.0, pad_s=1.0, skip_silence=True, workers=2,
        )
        assert np.abs(fast - reference).max() < 1e-3
        before, after = slice(start - SR // 20, start), slice(start + phrase.size, start + phrase.size + SR // 20)
        assert np.abs(fast[before] - reference[before]).max() < 1e-3
        assert np.abs(fast[after] - reference[after]).max() < 1e-3

    def test_islands_are_widened_by_one_frame_but_never_overlap_or_leave_the_array(self) -> None:
        audio = np.zeros(SR * 30, dtype=np.float32)
        audio[0:10] = 0.1                                   # sát đầu mảng
        audio[SR * 10 : SR * 10 + 5] = 0.1
        audio[-1] = 0.1                                     # sát cuối mảng
        islands = am.find_active_islands(audio, SR)
        assert islands[0][0] == 0 and islands[-1][1] == audio.size
        for (_, end), (begin, _) in zip(islands, islands[1:]):
            assert end < begin
        frame = int(SR * 0.25)
        assert islands[1][0] <= SR * 10 - frame and islands[1][1] >= SR * 10 + 5 + frame
