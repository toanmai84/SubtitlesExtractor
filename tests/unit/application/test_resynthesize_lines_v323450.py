"""[v3.23.450] Test nền tảng TTS lại dòng chọn: segment_store, remerge, use case.

QUAN TRỌNG: ``selected_indices`` là VỊ TRÍ enumerate (0-based) = ``event_index`` của kết quả,
KHÔNG phải ``event.index`` (1-based). Test hồi quy: chọn dòng n → TTS lại đúng dòng n (không n-1),
và KHÔNG ghi đè segment dòng khác.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from subtitles_extractor.infrastructure.tts.segment_store import segments_dir_for, segment_path
from subtitles_extractor.application.services.tts_remerge import SegmentPlacement, remerge_master
from subtitles_extractor.application.use_cases.resynthesize_lines import ResynthesizeLinesUseCase
from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest, TTSSegmentResult
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

SR = 100


def test_segment_paths():
    o = Path("/tmp/out/ep01.tts.wav")
    d = segments_dir_for(o)
    assert d.name == "ep01.tts_segments"
    assert segment_path(d, 5).name == "seg_000005.wav"
    assert str(segments_dir_for(o, "/custom")) == "/custom"


def test_remerge_places_and_clips():
    segs = [SegmentPlacement(0.0, np.ones(50, dtype=np.float32)),
            SegmentPlacement(1.0, np.ones(50, dtype=np.float32) * 0.5)]
    m = remerge_master(segs, SR, 300)
    assert len(m) == 300 and m[0] == 1.0 and m[50] == 0.0 and m[100] == 0.5
    assert len(remerge_master([SegmentPlacement(2.9, np.ones(50, dtype=np.float32))], SR, 300)) == 300


def test_remerge_invalid_args():
    with pytest.raises(ValueError):
        remerge_master([], 0, 300)
    with pytest.raises(ValueError):
        remerge_master([], SR, 0)


def _mk(i, s, e, t):
    return SubtitleEvent(index=i, text=t, interval=TimeInterval(start_sec=s, end_sec=e))


class _FakeAdapter:
    """Mô phỏng adapter thật: event_index = enumerate CỦA request truyền vào; ghi segment vào
    request.segments_dir theo enumerate đó; biên độ 0.7 để nhận biết 'audio mới'."""

    def __init__(self, sr=SR):
        self.sr = sr
        self.synthesized_texts = []

    def generate(self, request, output_path, progress_cb=None, cancel_cb=None):
        out = []
        seg_dir = Path(request.segments_dir)
        seg_dir.mkdir(parents=True, exist_ok=True)
        for idx, e in enumerate(request.events):   # enumerate 0-based như adapter thật
            self.synthesized_texts.append(e.text)
            p = segment_path(seg_dir, idx)
            sf.write(str(p), np.ones(50, dtype=np.float32) * 0.7, self.sr)
            out.append(TTSSegmentResult(
                event_index=idx, start_sec=e.start_sec, end_sec=e.end_sec, text=e.text,
                adjusted_start_sec=e.start_sec, segment_audio_path=str(p)))
        return out


def _prior_3lines(seg_root: Path):
    """3 dòng, mỗi dòng 1 segment biên độ riêng (0.5/0.2/0.9) tại 0s/1s/2s."""
    seg_root.mkdir(parents=True, exist_ok=True)
    amps = [0.5, 0.2, 0.9]
    prior = []
    for pos, amp in enumerate(amps):
        p = segment_path(seg_root, pos)
        sf.write(str(p), np.ones(50, dtype=np.float32) * amp, SR)
        prior.append(TTSSegmentResult(
            event_index=pos, start_sec=pos, end_sec=pos + 0.5, text=f"cũ {pos}",
            adjusted_start_sec=float(pos), segment_audio_path=str(p)))
    return prior


def test_resynthesize_selects_correct_line_not_previous(tmp_path):
    """Hồi quy CHÍNH: chọn VỊ TRÍ 1 (dòng giữa) → TTS lại đúng dòng 1, KHÔNG phải dòng 0."""
    out_wav = tmp_path / "master.wav"
    seg_root = segments_dir_for(out_wav)
    prior = _prior_3lines(seg_root)
    # events .index = 1-based (1,2,3) — nếu use case lọc nhầm theo .index sẽ chọn sai
    events = [_mk(1, 0, 0.5, "text A"), _mk(2, 1, 1.5, "text B SỬA"), _mk(3, 2, 2.5, "text C")]
    adapter = _FakeAdapter()
    uc = ResynthesizeLinesUseCase(adapter)
    merged = uc.execute(full_request=TTSRequest(events=events, normalize=False),
                        selected_indices={1},  # VỊ TRÍ 1 (dòng giữa)
                        prior_results=prior, output_path=out_wav,
                        sample_rate=SR, total_length_samples=300)
    # Chỉ dòng vị trí 1 được tổng hợp lại, đúng text của nó
    assert adapter.synthesized_texts == ["text B SỬA"], adapter.synthesized_texts
    # event_index vẫn là vị trí gốc 1
    r1 = [r for r in merged if r.event_index == 1][0]
    assert r1.text == "text B SỬA"
    # Audio: dòng 0 giữ (0.5), dòng 1 MỚI (0.7), dòng 2 giữ (0.9)
    m, _ = sf.read(str(out_wav), dtype="float32")
    assert 0.45 < m[0] < 0.55        # dòng 0 KHÔNG đổi
    assert 0.65 < m[100] < 0.75      # dòng 1 audio mới
    assert 0.85 < m[200] < 0.95      # dòng 2 KHÔNG đổi


def test_resynthesize_does_not_overwrite_other_segment_files(tmp_path):
    """Segment dòng 0 KHÔNG bị ghi đè khi TTS lại dòng 2."""
    out_wav = tmp_path / "master.wav"
    seg_root = segments_dir_for(out_wav)
    prior = _prior_3lines(seg_root)
    seg0_before, _ = sf.read(str(segment_path(seg_root, 0)), dtype="float32")
    events = [_mk(1, 0, 0.5, "A"), _mk(2, 1, 1.5, "B"), _mk(3, 2, 2.5, "C mới")]
    uc = ResynthesizeLinesUseCase(_FakeAdapter())
    uc.execute(full_request=TTSRequest(events=events, normalize=False),
               selected_indices={2}, prior_results=prior, output_path=out_wav,
               sample_rate=SR, total_length_samples=300)
    seg0_after, _ = sf.read(str(segment_path(seg_root, 0)), dtype="float32")
    assert np.allclose(seg0_before, seg0_after)  # dòng 0 nguyên vẹn (0.5)
    # dòng 2 đã thành audio mới (0.7)
    seg2, _ = sf.read(str(segment_path(seg_root, 2)), dtype="float32")
    assert 0.65 < float(seg2[0]) < 0.75


def test_resynthesize_multiple_lines(tmp_path):
    out_wav = tmp_path / "master.wav"
    seg_root = segments_dir_for(out_wav)
    prior = _prior_3lines(seg_root)
    events = [_mk(1, 0, 0.5, "A"), _mk(2, 1, 1.5, "B"), _mk(3, 2, 2.5, "C")]
    adapter = _FakeAdapter()
    uc = ResynthesizeLinesUseCase(adapter)
    merged = uc.execute(full_request=TTSRequest(events=events, normalize=False),
                        selected_indices={0, 2}, prior_results=prior, output_path=out_wav,
                        sample_rate=SR, total_length_samples=300)
    assert set(adapter.synthesized_texts) == {"A", "C"}
    m, _ = sf.read(str(out_wav), dtype="float32")
    assert 0.65 < m[0] < 0.75        # dòng 0 mới
    assert 0.15 < m[100] < 0.25      # dòng 1 giữ (0.2)
    assert 0.65 < m[200] < 0.75      # dòng 2 mới


def test_resynthesize_no_selection_raises():
    uc = ResynthesizeLinesUseCase(_FakeAdapter())
    with pytest.raises(ValueError):
        uc.execute(full_request=TTSRequest(events=[]), selected_indices=set(),
                   prior_results=[], output_path=Path("/tmp/x.wav"),
                   sample_rate=SR, total_length_samples=100)


def test_resynthesize_out_of_range_position_raises(tmp_path):
    uc = ResynthesizeLinesUseCase(_FakeAdapter())
    events = [_mk(1, 0, 0.5, "A")]
    with pytest.raises(ValueError):
        uc.execute(full_request=TTSRequest(events=events, normalize=False),
                   selected_indices={99}, prior_results=[], output_path=tmp_path / "m.wav",
                   sample_rate=SR, total_length_samples=100)


# [v3.23.453] Hồi quy: TTSSegmentResult FROZEN → phải dùng replace, KHÔNG gán .text (gán ném
# FrozenInstanceError, làm handler dừng giữa chừng → 'TTS lại với nội dung cũ').
from dataclasses import replace as _dc_replace, FrozenInstanceError
import pytest as _pytest


def test_ttssegmentresult_is_frozen_use_replace():
    r = TTSSegmentResult(event_index=0, start_sec=0, end_sec=1, text="cũ")
    with _pytest.raises(FrozenInstanceError):
        r.text = "MỚI"          # gán trực tiếp → LỖI (bug cũ)
    r2 = _dc_replace(r, text="MỚI")   # cách đúng
    assert r2.text == "MỚI" and r.text == "cũ"


def test_subtitle_event_replace_updates_text():
    ev = _mk(1, 0, 1, "cũ")
    ev2 = _dc_replace(ev, text="ĐÃ SỬA")
    assert ev2.text == "ĐÃ SỬA" and ev2.index == 1


# [v3.23.454] Sau khi TTS lại, phụ đề đính kèm phải khớp nội dung mới.
from subtitles_extractor.application.services.tts_subtitle_export import export_tts_subtitle


def test_export_tts_subtitle_has_text():
    results = [
        TTSSegmentResult(event_index=0, start_sec=0, end_sec=1, text="Xin chào",
                         adjusted_start_sec=0.0, adjusted_end_sec=1.0),
        TTSSegmentResult(event_index=1, start_sec=1, end_sec=2, text="ĐÃ SỬA",
                         adjusted_start_sec=1.0, adjusted_end_sec=2.0),
    ]
    import tempfile
    import numpy as np
    import soundfile as sf
    d = Path(tempfile.mkdtemp())
    wav = d / "ep.tts.wav"
    sf.write(str(wav), np.zeros(100, dtype=np.float32), 48000)
    srt = export_tts_subtitle(results, wav, TTSRequest(events=[], language="vi"))
    assert srt is not None
    content = Path(srt).read_text(encoding="utf-8")
    assert "ĐÃ SỬA" in content and "Xin chào" in content


def test_resynthesize_reexports_subtitle_with_new_text(tmp_path):
    """TTS lại dòng (đã sửa text) → phụ đề đính kèm chứa text MỚI."""
    import numpy as np
    import soundfile as sf
    out_wav = tmp_path / "master.wav"
    seg_root = segments_dir_for(out_wav)
    prior = _prior_3lines(seg_root)
    events = [_mk(1, 0, 0.5, "A"), _mk(2, 1, 1.5, "B ĐÃ SỬA"), _mk(3, 2, 2.5, "C")]
    uc = ResynthesizeLinesUseCase(_FakeAdapter())
    uc.execute(full_request=TTSRequest(events=events, normalize=False, language="vi"),
               selected_indices={1}, prior_results=prior, output_path=out_wav,
               sample_rate=SR, total_length_samples=300)
    # SRT đính kèm phải tồn tại + chứa text mới của dòng vị trí 1
    from subtitles_extractor.domain.value_objects.output_naming import tts_subtitle_path
    srt = tts_subtitle_path(out_wav, "vi")
    assert srt.exists(), "Không xuất lại phụ đề sau TTS lại"
    content = srt.read_text(encoding="utf-8")
    assert "B ĐÃ SỬA" in content, content
