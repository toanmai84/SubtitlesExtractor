"""[v3.23.246] Nhiệt độ tuỳ chỉnh cho Gemini TTS + preamble cho standard TTS.

**Preamble mở rộng sang standard TTS.** v245 chỉ bọc đường native audio. Cookbook chính
thức nêu rõ: *"the model can only do TTS, so you should always tell it to say/read
something, otherwise it won't do anything"* — áp cho CẢ standard TTS (generateContent).

**Temperature tuỳ chỉnh.** Tài liệu cho thấy Gemini TTS nhận tham số ``temperature``. Hạ
thấp (vd 0.7) làm giọng ỔN ĐỊNH hơn, giảm "ngân dài ngẫu nhiên" — nguồn gốc
hallucination đo được ở v244. Nhưng quá thấp có thể làm giọng bớt biểu cảm; đây là
ĐÁNH ĐỔI chủ quan nên Toan chọn: để NGƯỜI DÙNG tự chỉnh (``gemini_temperature``, mặc
định None = mặc định model).
"""

from __future__ import annotations

import pathlib

from subtitles_extractor.domain.ports.subtitle_tts_port import TTSRequest

_GEMINI_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/tts/gemini_tts_adapter.py"
).read_text(encoding="utf-8")
_VM_SRC = pathlib.Path(
    "src/subtitles_extractor/presentation/view_models/tts_page_view_model.py"
).read_text(encoding="utf-8")


# ── Field trên request ──────────────────────────────────────────────────────
def test_request_có_field_temperature() -> None:
    r = TTSRequest(events=[])
    assert r.gemini_temperature is None  # mặc định: dùng mặc định của model


def test_request_nhận_temperature_tuỳ_chỉnh() -> None:
    r = TTSRequest(events=[], gemini_temperature=0.7)
    assert r.gemini_temperature == 0.7


# ── Adapter truyền temperature vào cả hai đường ─────────────────────────────
def test_standard_tts_truyền_temperature() -> None:
    """[v3.23.547] v3.23.419 dựng config qua ``config_kwargs`` để tắt AFC có điều kiện.

    Nhiệt độ vì thế chuyển từ đối số từ khoá sang MỘT MỤC trong dict rồi mới bung vào
    ``GenerateContentConfig(**config_kwargs)``. Hành vi y hệt; bộ kiểm bám theo dạng mới.
    """
    # [v3.23.595] Chỉ gửi khi người dùng đặt VÀ model còn nhận tham số lấy mẫu (3.6+,
    # Flash-Lite 3.5+ và TTS 3.8 bỏ qua) — hành vi kiểm ở test_gemini_tts_38_v323595.
    assert 'temperature = getattr(request, "gemini_temperature", None)' in _GEMINI_SRC
    assert (
        "if temperature is not None and not sampling_params_deprecated(model):"
        in _GEMINI_SRC
    )
    assert "types.GenerateContentConfig(**config_kwargs)" in _GEMINI_SRC


def test_native_audio_chỉ_thêm_khi_có_giá_trị() -> None:
    # Native audio: chỉ thêm temperature khi != None (tránh gửi None cho SDK cũ).
    assert 'temperature = getattr(request, "gemini_temperature", None)' in _GEMINI_SRC
    assert "if temperature is not None:" in _GEMINI_SRC
    assert 'config_kwargs["temperature"] = temperature' in _GEMINI_SRC


# ── Chuỗi truyền từ view-model ──────────────────────────────────────────────
def test_view_model_truyền_temperature() -> None:
    assert "gemini_temperature=gemini_temperature" in _VM_SRC


# ── Preamble áp cho cả standard TTS ─────────────────────────────────────────
def test_standard_tts_bọc_transcript() -> None:
    # generateContent cũng phải bọc (cookbook: luôn bảo model "say/read").
    # [v3.23.595] Trừ model TTS 3.8+: đọc nguyên văn, lời dẫn sẽ bị đọc thành tiếng.
    assert "else wrap_transcript_for_tts(text)" in _GEMINI_SRC
