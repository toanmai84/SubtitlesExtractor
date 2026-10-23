"""[v3.23.407] Test chuẩn hoá tên giọng cho worker GPU (xử lý tuple (nhãn, giá_trị))."""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    match_voice_name,
    normalize_voice_candidates,
)

_AVAILABLE = ["Minh Đức", "Phạm Tuyên", "Thái Sơn", "Xuân Vĩnh"]


def _resolve(voice_name, speaker="") -> str | None:
    for candidate in normalize_voice_candidates(voice_name, speaker):
        matched = match_voice_name(candidate, _AVAILABLE)
        if matched is not None:
            return matched
    return None


def test_tuple_label_value_resolves_to_value() -> None:
    """Bug thực tế: voice_name là tuple (nhãn dài, ID) → phải ra ID hợp lệ."""
    voice = ("Minh Đức — Nam · Bắc · Phong cách tin tức", "Minh Đức")
    assert _resolve(voice) == "Minh Đức"


def test_plain_string_unchanged() -> None:
    assert _resolve("Thái Sơn") == "Thái Sơn"


def test_list_form_resolves() -> None:
    assert _resolve(["Phạm Tuyên — Nam", "Phạm Tuyên"]) == "Phạm Tuyên"


def test_speaker_fallback_used() -> None:
    # voice_name rỗng → dùng speaker.
    assert _resolve("", "Xuân Vĩnh") == "Xuân Vĩnh"


def test_candidates_dedup_and_order() -> None:
    cands = normalize_voice_candidates(("Minh Đức", "Minh Đức"), "Minh Đức")
    assert cands == ["Minh Đức"]  # bỏ trùng


def test_empty_yields_no_candidates() -> None:
    assert normalize_voice_candidates("", "") == []


def test_label_alone_does_not_match_but_value_does() -> None:
    """Nhãn dài đơn lẻ không khớp (đúng), nhưng ứng viên GIÁ TRỊ trong tuple thì khớp."""
    # Nhãn dài đơn lẻ: không khớp (available ngắn hơn, không phải tiền tố của nhãn).
    assert match_voice_name("Minh Đức — Nam · Bắc", _AVAILABLE) is None
    # Nhưng khi đi qua tuple (nhãn, giá_trị), _resolve vẫn ra đúng nhờ giá trị.
    assert _resolve(("Minh Đức — Nam · Bắc", "Minh Đức")) == "Minh Đức"
