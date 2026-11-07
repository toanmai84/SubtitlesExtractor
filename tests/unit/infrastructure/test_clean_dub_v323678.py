"""[v3.23.678] Lồng tiếng sạch: tách lời thoại gốc khỏi nền (MDX-Net ONNX), trộn giọng Việt lên phần nền.

Đo thật tập 3 phim mẫu (152 s, CPU, model Kim_Vocal_2):
* Gemini chép lời 90 s đầu: tiếng gốc 204 chữ Hán; phần NỀN đã tách "KHÔNG CÓ LỜI" (2/2 lần).
* Xuất cả tập 91 s (tách ~0,6× thời lượng); xuất lại dùng phần nền đã lưu 5,8 s; xuất thử 60 s chỉ tách đúng đoạn: 65 s.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from subtitles_extractor.infrastructure.audio import voice_separator as vs  # noqa: E402
from subtitles_extractor.infrastructure.video.video_render_command import (  # noqa: E402
    AudioMode,
    RenderMode,
    RenderRequest,
    SubtitleMode,
    VideoRenderError,
    build_render_command,
)


class _Identity:
    """"Model" trả nguyên phổ vào: stem chính = cả bản trộn."""

    def run(self, _outputs, feeds):
        return [feeds["input"]]


def _tone(path: Path, seconds: float) -> np.ndarray:
    t = np.arange(int(vs.SAMPLE_RATE * seconds)) / vs.SAMPLE_RATE
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 440 * t), 0.2 * np.sin(2 * np.pi * 660 * t)], axis=1).astype(np.float32)
    sf.write(str(path), stereo, vs.SAMPLE_RATE, subtype="FLOAT")
    return stereo


@pytest.mark.parametrize("primary_is_vocals", [False, True])
def test_tach_theo_khoi_giu_nguyen_do_dai_va_tai_tao_dung(tmp_path: Path, primary_is_vocals: bool) -> None:
    # 6,3 s: không chia hết cho khối (≈ 5,7 s hữu ích) — kiểm phần đuôi và mép khối.
    mix = _tone(tmp_path / "in.wav", 6.3)
    model = vs.SeparationModel("x.onnx", 7680, primary_is_vocals, "", "", 0)
    out = tmp_path / "nen.flac"
    seen: list[float] = []
    assert vs.separate_file(_Identity(), model, tmp_path / "in.wav", out, progress=seen.append)
    background, sr = sf.read(str(out), dtype="float32")
    assert sr == vs.SAMPLE_RATE and background.shape == mix.shape
    expected = np.zeros_like(mix) if primary_is_vocals else mix   # nền = gốc − lời (lời = cả bản trộn)
    assert np.max(np.abs(background - expected)) < 2e-3            # FLAC 16 bit
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)
    assert not list(tmp_path.glob("*.part*"))


def test_huy_giua_chung_khong_de_lai_tep(tmp_path: Path) -> None:
    _tone(tmp_path / "in.wav", 12.0)
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    out = tmp_path / "nen.flac"
    assert not vs.separate_file(_Identity(), vs.KIM_VOCAL_2, tmp_path / "in.wav", out, cancelled=cancelled)
    assert not out.exists() and not list(tmp_path.glob("*.part*"))


def test_tu_choi_am_thanh_sai_dinh_dang(tmp_path: Path) -> None:
    sf.write(str(tmp_path / "mono.wav"), np.zeros(1000, dtype=np.float32), 16000)
    with pytest.raises(ValueError):
        vs.separate_file(_Identity(), vs.KIM_VOCAL_2, tmp_path / "mono.wav", tmp_path / "o.flac")


# ── Tải model ──────────────────────────────────────────────────────────────────────────────


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def test_tai_model_kiem_sha256(tmp_path: Path) -> None:
    payload = b"model-bytes" * 1000
    good = vs.SeparationModel("m.onnx", 7680, True, "http://x", hashlib.sha256(payload).hexdigest(), len(payload))
    path = vs.ensure_model(tmp_path, good, opener=lambda *_a, **_k: _Resp(payload))
    assert path.read_bytes() == payload
    # Đã có đủ: không tải lại.
    assert vs.ensure_model(tmp_path, good, opener=lambda *_a, **_k: pytest.fail("không được tải lại")) == path
    bad = vs.SeparationModel("hong.onnx", 7680, True, "http://x", "0" * 64, len(payload))
    with pytest.raises(vs.SeparationError):
        vs.ensure_model(tmp_path, bad, opener=lambda *_a, **_k: _Resp(payload))
    assert not (tmp_path / "hong.onnx").exists() and not list(tmp_path.glob("*.part"))


def test_loi_mang_bao_ro(tmp_path: Path) -> None:
    def broken(*_a, **_k):
        raise OSError("mất mạng")

    with pytest.raises(vs.SeparationError, match="Không tải được"):
        vs.ensure_model(tmp_path, vs.KIM_VOCAL_2, opener=broken)


def test_khoa_bo_nho_dem_doi_khi_video_doi(tmp_path: Path) -> None:
    import os

    video = tmp_path / "tap.mp4"
    video.write_bytes(b"a")
    first = vs.background_cache_path(tmp_path, video)
    os.utime(video, (1_000_000, 1_000_000))
    assert vs.background_cache_path(tmp_path, video) != first


# ── Lệnh ffmpeg ────────────────────────────────────────────────────────────────────────────


def _request(tmp_path: Path, **kw) -> RenderRequest:
    for name in ("v.mp4", "tts.wav", "nen.flac"):
        (tmp_path / name).write_bytes(b"x")
    base = dict(
        video_path=tmp_path / "v.mp4", output_path=tmp_path / "out.mkv", mode=RenderMode.SOFT_SUB,
        subtitle_mode=SubtitleMode.NONE, audio_mode=AudioMode.CLEAN_DUB, audio_path=tmp_path / "tts.wav",
        background_path=tmp_path / "nen.flac",
    )
    base.update(kw)
    return RenderRequest(**base)


def test_lenh_tron_giong_viet_len_phan_nen(tmp_path: Path) -> None:
    cmd = build_render_command("ffmpeg", _request(tmp_path))
    inputs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-i"]
    assert inputs == [str(tmp_path / "v.mp4"), str(tmp_path / "tts.wav"), str(tmp_path / "nen.flac")]
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "[2:a]aformat" in graph and "[0:a]" not in graph       # tiếng gốc KHÔNG vào bản trộn
    assert "sidechaincompress" in graph and "[aout]" in graph
    assert cmd[cmd.index("-c:v") + 1] == "copy"


def test_xuat_thu_doi_moc_phan_nen(tmp_path: Path) -> None:
    # Phần nền chỉ tách cho đoạn thử (bắt đầu ở 30 s) → không tua thêm.
    req = _request(tmp_path, clip_start_sec=30.0, clip_duration_sec=60.0, background_offset_sec=30.0)
    cmd = build_render_command("ffmpeg", req)
    bg = cmd.index(str(tmp_path / "nen.flac"))
    assert cmd[bg - 3] != "-ss"
    # Phần nền cả phim → tua tới 30 s như video.
    cmd = build_render_command("ffmpeg", _request(tmp_path, clip_start_sec=30.0, clip_duration_sec=60.0))
    bg = cmd.index(str(tmp_path / "nen.flac"))
    assert cmd[bg - 3: bg - 1] == ["-ss", "30.000"]


def test_thieu_phan_nen_bao_loi(tmp_path: Path) -> None:
    with pytest.raises(VideoRenderError, match="phần nền"):
        build_render_command("ffmpeg", _request(tmp_path, background_path=None))


def test_render_video_tach_truoc_roi_xuat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from subtitles_extractor.infrastructure.video import ffmpeg_video_renderer as mod

    monkeypatch.setattr(mod, "find_ffmpeg", lambda: "ffmpeg")
    req = _request(tmp_path, background_path=None, separation_dir=tmp_path / "sep")
    captured: dict = {}

    def fake_prepare(_ffmpeg, video, work_dir, *, progress, cancelled, start_sec, duration_sec):
        progress(1.0)
        captured.update(work_dir=work_dir, window=(start_sec, duration_sec))
        return tmp_path / "nen.flac"

    def fake_render(_ffmpeg, request, _dur, on_progress, _cancel):
        captured["request"] = request
        on_progress(1.0)
        return request.output_path

    monkeypatch.setattr(vs, "prepare_background", fake_prepare)
    monkeypatch.setattr(mod, "_render_resolved", fake_render)
    marks: list[float] = []
    mod.render_video(req, total_duration_sec=10.0, on_progress=marks.append)
    assert captured["request"].background_path == tmp_path / "nen.flac"
    assert captured["work_dir"] == tmp_path / "sep" and captured["window"] == (0.0, 0.0)
    assert marks == [pytest.approx(0.8), pytest.approx(1.0)]          # 80 % tách, 20 % trộn/ghép

    monkeypatch.setattr(vs, "prepare_background", lambda *_a, **_k: None)   # người dùng huỷ
    with pytest.raises(VideoRenderError, match="huỷ"):
        mod.render_video(req)
    with pytest.raises(VideoRenderError, match="thư mục"):
        mod.render_video(_request(tmp_path, background_path=None))


# ── Trang Xuất bản ─────────────────────────────────────────────────────────────────────────


def test_trang_xuat_ban_co_che_do_long_tieng_sach(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import publish_page

    _app = QApplication.instance() or QApplication([])
    QSettings().remove("publish")
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(publish_page.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    page = publish_page.PublishPage(container)
    try:
        page._subtitle_combo.setCurrentIndex(0)                              # Phụ đề: Không
        modes = [mode for _key, mode in publish_page._AUDIO_CHOICES]
        assert modes[:3] == [AudioMode.ORIGINAL, AudioMode.VOICE_OVER, AudioMode.REPLACE_TRACK]   # chỉ số cũ giữ nguyên
        page._audio_combo.setCurrentIndex(modes.index(AudioMode.CLEAN_DUB))
        assert page._current_output_suffix().endswith("_longtieng")
        assert not page._duck_combo.isHidden()                               # vẫn chỉnh được mức hạ nhạc nền
        assert "Lồng tiếng sạch" in page._mode_hint.text()
        video = tmp_path / "ep1.mp4"
        video.write_bytes(b"v")
        (tmp_path / "ep1.wav").write_bytes(b"a")
        page._video_edit.setText(str(video))
        page._audio_edit.setText(str(tmp_path / "ep1.wav"))
        page._output_edit.setText(str(tmp_path / "ep1_longtieng.mkv"))
        request = page._build_request()
        assert request.audio_mode is AudioMode.CLEAN_DUB and request.separation_dir == tmp_path / "separation"
    finally:
        page.close()
        QSettings().remove("publish")
