"""[v3.23.254] Khởi tạo Vieneu an toàn qua introspection constructor.

**Bối cảnh:** VieNeu thay đổi API qua các phiên bản. App từng gọi CỨNG
``Vieneu(mode=..., emotion=...)``. Nếu bản SDK người dùng cài KHÔNG còn nhận
``emotion`` (vd đổi sang ``style``), gọi cứng ném ``TypeError`` -> sập cả engine.

**Giải pháp (không phụ thuộc đoán đúng bản API):** ``_construct_engine`` dò chữ ký
constructor bằng ``inspect.signature`` (pattern đã dùng an toàn cho ``infer``) và
chỉ truyền tham số constructor THẬT SỰ khai báo:
* ``mode`` nếu nhận.
* ``emotion`` nếu nhận (API cũ); nếu không mà có ``style``, ánh xạ emotion -> style.
* Không nội soi được / giá trị không hợp lệ -> lùi dần ``{mode}`` rồi ``{}`` (mặc định).

**Lưu ý trung thực:** giá trị ``style`` ("tu_nhien"/"doc_truyen") là ánh xạ hợp lý
theo tên tiếng Việt, CHƯA xác minh với tài liệu SDK 3.x. An toàn nhờ fallback: nếu SDK
từ chối giá trị, khối except lùi về khởi tạo mặc định thay vì sập.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts import (
    vieneu_tts_adapter as _mod,
)
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    _EMOTION_TO_STYLE,
    VieNeuTtsAdapter,
)


def _adapter(mode: str = "standard", emotion: str = "natural") -> VieNeuTtsAdapter:
    a = VieNeuTtsAdapter.__new__(VieNeuTtsAdapter)
    a._mode = mode
    a._emotion = emotion
    return a


def test_api_cũ_mode_emotion() -> None:
    """Constructor cũ (mode + emotion) -> truyền cả hai."""

    class OldAPI:
        def __init__(self, mode: str = "standard", emotion: str = "natural") -> None:
            self.kw = (mode, emotion)

    engine = _adapter(emotion="storytelling")._construct_engine(OldAPI)
    assert engine.kw == ("standard", "storytelling")


def test_api_mới_mode_style() -> None:
    """Constructor mới (mode + style, KHÔNG emotion) -> ánh xạ emotion sang style."""

    class NewAPI:
        def __init__(self, mode: str = "turbo", style: str = "tu_nhien") -> None:
            self.kw = (mode, style)

    engine = _adapter(emotion="storytelling")._construct_engine(NewAPI)
    assert engine.kw == ("standard", "doc_truyen")


def test_api_tối_giản_không_tham_số() -> None:
    """Constructor không nhận tham số nào -> khởi tạo trần, không lỗi."""

    class MinimalAPI:
        def __init__(self) -> None:
            self.kw = ()

    engine = _adapter()._construct_engine(MinimalAPI)
    assert engine.kw == ()


def test_chỉ_mode_không_emotion_không_style() -> None:
    """Constructor chỉ có mode -> truyền mode, bỏ qua emotion/style."""

    class ModeOnly:
        def __init__(self, mode: str = "standard") -> None:
            self.kw = mode

    engine = _adapter(mode="turbo")._construct_engine(ModeOnly)
    assert engine.kw == "turbo"


def test_style_không_hợp_lệ_thì_fallback() -> None:
    """Nếu constructor nhận style nhưng TỪ CHỐI giá trị -> lùi về mặc định, không sập."""

    class PickyStyle:
        def __init__(self, mode: str = "standard", style: str = "tu_nhien") -> None:
            if style not in ("tu_nhien", "tin_tuc"):  # từ chối 'doc_truyen'
                raise TypeError("style không hợp lệ")
            self.kw = (mode, style)

    # emotion=storytelling -> style=doc_truyen bị từ chối -> fallback {mode} rồi {}.
    engine = _adapter(emotion="storytelling")._construct_engine(PickyStyle)
    # Fallback {mode} thành công (style về mặc định tu_nhien).
    assert engine.kw == ("standard", "tu_nhien")


def test_emotion_to_style_mapping() -> None:
    assert _EMOTION_TO_STYLE["natural"] == "tu_nhien"
    assert _EMOTION_TO_STYLE["storytelling"] == "doc_truyen"


def test_construct_engine_tồn_tại_và_dùng_inspect() -> None:
    # Call site khởi tạo phải dùng _construct_engine, không gọi cứng constructor.
    import pathlib

    src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/tts/vieneu_tts_adapter.py"
    ).read_text(encoding="utf-8")
    assert "self._construct_engine(Vieneu)" in src
    assert "Vieneu(mode=self._mode, emotion=self._emotion)" not in src
    assert hasattr(_mod, "inspect")
