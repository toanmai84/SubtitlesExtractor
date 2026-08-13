"""[v3.23.257] Sửa 2 bug tìm khi nội soi API THẬT vieneu 3.2.3 (cài trong sandbox).

Điều tra bằng ``inspect.signature`` trên ``V3TurboVieNeuTTS`` thật:

**Bug 1 — trần thời lượng không hoạt động với v3 Turbo.** ``infer`` v3 Turbo KHÔNG
có ``max_duration_s`` v.v. (app tìm các tên này). Trần THẬT là ``max_new_frames``. Đo
từ source: hop_length=480 @ sr 48000 -> 100 frame/giây. Không đặt đúng -> cơ chế chống
"ngân dài" (câu 3 ký tự sinh 32s audio) KHÔNG chạy với v3 Turbo. Sửa: cap giây -> frames.

**Bug 2 — voice cloning bị bỏ âm thầm.** ``encode_reference`` của v3 Turbo trả TUPLE
``(speaker_emb, ref_codes)``, nhưng ``infer(voice=)`` chỉ nhận ``str|dict``. Truyền
tuple ->
``_resolve_ref`` bỏ qua -> rơi về giọng mặc định (MẤT cloning, không báo lỗi). Sửa: bọc
tuple thành ``{"speaker_emb":..., "codes":...}`` (đúng khoá ``_resolve_ref`` đọc).
"""

from __future__ import annotations

import numpy as np

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    _V3TURBO_FRAMES_PER_SECOND,
    VieNeuTtsAdapter,
)


class _FakeV3Turbo:
    """Mô phỏng CHÍNH XÁC signature ``infer`` của V3TurboVieNeuTTS 3.2.3."""

    sample_rate = 48_000

    def infer(
        self,
        text: str,
        ref_audio=None,
        voice=None,
        style: str = "tu_nhien",
        max_new_frames: int = 300,
        **kw,
    ):
        return {"style": style, "max_new_frames": max_new_frames, "voice": voice}


def _adapter(emotion: str = "natural") -> VieNeuTtsAdapter:
    a = VieNeuTtsAdapter.__new__(VieNeuTtsAdapter)
    a._mode = "v3turbo"
    a._emotion = emotion
    a._precision = None
    return a


# ── Bug 1: max_new_frames ───────────────────────────────────────────────────
def test_frame_rate_constant() -> None:
    # Đo từ source: hop_length=480 @ 48kHz -> 100 frame/s.
    assert _V3TURBO_FRAMES_PER_SECOND == 100


def test_max_new_frames_được_đặt() -> None:
    # v3 Turbo: trần phải đặt qua max_new_frames (khác mặc định 300).
    r = _adapter()._infer_once(
        _FakeV3Turbo(), "Một câu dài hơn ba mươi ký tự đây", {"v": 1}
    )
    assert r["max_new_frames"] != 300  # đã đặt trần theo độ dài câu
    assert r["max_new_frames"] > 0


def test_trần_chặn_ngân_dài_câu_ngắn() -> None:
    # Câu ngắn -> trần nhỏ -> chặn model ngân dài thảm hoạ.
    r = _adapter()._infer_once(_FakeV3Turbo(), "Ừm.", {"v": 1})
    # 3 ký tự -> cap ~3s -> ~350 frames. Không được lên tới hàng nghìn.
    assert r["max_new_frames"] < 600


# ── Bug 2: encode_reference tuple -> dict ───────────────────────────────────
def test_encode_reference_tuple_bọc_thành_dict() -> None:
    emb = np.ones(4, dtype=np.float32)
    codes = np.ones(8, dtype=np.float32)

    class EngineCloning:
        sample_rate = 48_000

        def encode_reference(self, path):
            return (emb, codes)  # v3 Turbo trả TUPLE

        def infer(self, text, voice=None, **kw):
            return {"voice": voice}

    from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest

    a = _adapter()
    a._voice_data_cache = None
    a._force_cpu = True
    req = TTSRequest(events=[], ref_audio_path="/fake/ref.wav")
    voice_data = a._resolve_voice_data(EngineCloning(), req, use_cloning=True)
    # Phải là dict đúng khoá, KHÔNG phải tuple.
    assert isinstance(voice_data, dict)
    assert "speaker_emb" in voice_data
    assert "codes" in voice_data
    assert np.array_equal(voice_data["speaker_emb"], emb)


def test_style_vẫn_truyền_cùng_max_frames() -> None:
    # Cả style lẫn max_new_frames cùng hoạt động (không loại trừ nhau).
    r = _adapter(emotion="storytelling")._infer_once(
        _FakeV3Turbo(), "Một câu kể chuyện dài hơn", {"v": 1}
    )
    assert r["style"] == "doc_truyen"
    assert r["max_new_frames"] != 300
