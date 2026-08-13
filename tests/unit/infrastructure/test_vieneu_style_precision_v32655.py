"""[v3.23.255] Tận dụng API VieNeu 3.2.3 đã XÁC MINH từ tài liệu chính thức.

Nguồn: pypi.org/project/vieneu/3.2.3/ (phát hành 12/7/2026). Các sự thật đã xác minh:
- Mặc định v3 Turbo (48kHz), torch-free trên CPU (ONNX Runtime).
- ``precision``: "int8" (mặc định, nhanh ~1.6x, nhỏ ~4x, "quality preserved") | "fp32".
- ``voice="Tên"`` gọi trực tiếp; ``style`` per-call trong ``infer``:
  "tu_nhien"(natural) | "tin_tuc"(news) | "doc_truyen"(storytelling).
- ``infer_batch`` (GPU), ``infer_stream`` (streaming ~300ms), emotion cues.

Ánh xạ ``_EMOTION_TO_STYLE`` (natural->tu_nhien, storytelling->doc_truyen) KHỚP tài liệu.

Cải tiến phiên này (dùng introspection -> an toàn với mọi bản SDK):
1. Truyền ``style`` vào ``infer`` nếu SDK nhận (sắc thái đọc per-call cho API 3.x).
2. Truyền ``precision`` vào constructor nếu app đặt + constructor nhận
   (None -> SDK tự int8).
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    _EMOTION_TO_STYLE,
    VieNeuTtsAdapter,
)


def _adapter(emotion: str = "natural", precision: str | None = None) -> VieNeuTtsAdapter:
    a = VieNeuTtsAdapter.__new__(VieNeuTtsAdapter)
    a._mode = "standard"
    a._emotion = emotion
    a._precision = precision
    return a


# ── style truyền vào infer ──────────────────────────────────────────────────
def test_infer_nhận_style_thì_truyền() -> None:
    class EngineWithStyle:
        def infer(self, text, voice, style="tu_nhien"):
            return {"style": style}

    r = _adapter(emotion="storytelling")._infer_once(EngineWithStyle(), "x", {"v": 1})
    assert r["style"] == "doc_truyen"


def test_infer_không_nhận_style_thì_bỏ_qua() -> None:
    # Bản cũ: infer không có 'style' -> không truyền (không lỗi).
    class EngineNoStyle:
        def infer(self, text, voice):
            return {"ok": True}

    r = _adapter(emotion="storytelling")._infer_once(EngineNoStyle(), "x", {"v": 1})
    assert r == {"ok": True}


def test_style_natural_ánh_xạ_đúng() -> None:
    class EngineWithStyle:
        def infer(self, text, voice, style="tu_nhien"):
            return {"style": style}

    r = _adapter(emotion="natural")._infer_once(EngineWithStyle(), "x", {"v": 1})
    assert r["style"] == "tu_nhien"


# ── precision truyền vào constructor ────────────────────────────────────────
def test_precision_đặt_thì_truyền() -> None:
    class Ctor:
        def __init__(self, precision="int8"):
            self.p = precision

    engine = _adapter(precision="fp32")._construct_engine(Ctor)
    assert engine.p == "fp32"


def test_precision_none_để_sdk_mặc_định() -> None:
    class Ctor:
        def __init__(self, precision="int8"):
            self.p = precision

    engine = _adapter(precision=None)._construct_engine(Ctor)
    assert engine.p == "int8"  # mặc định SDK, app không ép


def test_precision_constructor_không_nhận_thì_bỏ_qua() -> None:
    # Constructor bản cũ không có 'precision' -> không truyền, không lỗi.
    class CtorNoPrec:
        def __init__(self, mode="standard"):
            self.mode = mode

    engine = _adapter(precision="fp32")._construct_engine(CtorNoPrec)
    assert engine.mode == "standard"


# ── ánh xạ style khớp tài liệu 3.2.3 ────────────────────────────────────────
def test_style_values_khớp_tài_liệu() -> None:
    # Giá trị chính thức từ tài liệu VieNeu 3.2.3.
    assert _EMOTION_TO_STYLE["natural"] == "tu_nhien"
    assert _EMOTION_TO_STYLE["storytelling"] == "doc_truyen"
