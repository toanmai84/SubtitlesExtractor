"""[v3.23.626] Cập nhật theo VieNeu-TTS 3.8.3: giọng đổi tên (bí danh), phong cách giọng,
phân vai ưu tiên giọng tự nhiên, cảnh báo bản SDK cũ."""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.application.services.dubbing_cast import (
    CastMember,
    auto_cast,
    canonical_voice_id,
    default_gender_voices,
)
from subtitles_extractor.domain.value_objects.voice_profile import (
    STYLE_NATURAL,
    STYLE_NEWS,
    STYLE_STORY,
    VOICE_FEMALE,
    VOICE_MALE,
    VoiceProfile,
    is_featured_label,
    style_from_voice_label,
)
from subtitles_extractor.infrastructure.tts.vieneu_gpu_plan import VIENEU_INSTALL_SPECS
from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    VIENEU_VOICE_ALIASES,
    VieNeuTtsAdapter,
    aliases_of,
    parse_version,
    resolve_voice_alias,
    vieneu_version_advice,
)

#: Nhãn thật của VieNeu 3.8.3 (`voice_label` + voices_v3_turbo.json).
LABELS = {
    "Adam bựa": "⭐ Adam bựa — Nam · Bắc · Phong cách tự nhiên",
    "Thiện Minh": "⭐ Thiện Minh — Nam · Bắc · Phong cách kể chuyện",
    "Mai Anh": "⭐ Mai Anh — Nữ · Bắc · Phong cách tin tức",
    "Trúc Ly": "⭐ Trúc Ly — Nữ · Bắc · Phong cách tự nhiên",
    "Minh Đức": "Minh Đức — Nam · Bắc · Phong cách tin tức",
    "Ngọc Huyền": "⭐ Ngọc Huyền — Nữ · Bắc · Giọng đọc tự nhiên",
    "Mỹ Duyên": "Mỹ Duyên — Nữ · Nam · Phong cách đọc truyện",
}


def test_style_and_featured_are_read_from_real_labels() -> None:
    assert style_from_voice_label(LABELS["Mai Anh"]) == STYLE_NEWS
    assert style_from_voice_label(LABELS["Thiện Minh"]) == STYLE_STORY
    assert style_from_voice_label(LABELS["Mỹ Duyên"]) == STYLE_STORY
    assert style_from_voice_label(LABELS["Ngọc Huyền"]) == STYLE_NATURAL
    assert style_from_voice_label("Bình (nam miền Bắc)") == ""
    assert is_featured_label(LABELS["Trúc Ly"]) and not is_featured_label(LABELS["Minh Đức"])


def test_old_names_resolve_to_renamed_presets() -> None:
    assert resolve_voice_alias("Minh Quân") == "Hải Đăng"
    assert resolve_voice_alias("Minh Quân Pro") == "Hải Đăng"
    assert resolve_voice_alias("Anh Khôi") == "Thiện Minh"
    assert resolve_voice_alias("Mạnh Dũng") == "Quốc Tuấn"
    assert resolve_voice_alias("Trúc Ly") == "Trúc Ly"
    assert set(aliases_of("Hải Đăng")) == {"Minh Quân", "Minh Quân Pro"}
    # Engine có bảng alias riêng (luôn đúng với bản SDK đang cài) thì ưu tiên nó.
    engine = SimpleNamespace(resolve_voice_name=lambda name: "Mới" if name == "Cũ" else None)
    assert resolve_voice_alias("Cũ", engine) == "Mới"
    assert resolve_voice_alias("Minh Quân", engine) == "Hải Đăng"
    assert VIENEU_VOICE_ALIASES


def test_saved_casting_under_the_old_name_still_applies() -> None:
    catalog = [
        VoiceProfile("Hải Đăng", "Hải Đăng", VOICE_MALE, aliases=("Minh Quân", "Minh Quân Pro")),
        VoiceProfile("Trúc Ly", "Trúc Ly", VOICE_FEMALE),
    ]
    assert canonical_voice_id("Minh Quân", catalog) == "Hải Đăng"
    assert canonical_voice_id("Hải Đăng", catalog) == "Hải Đăng"
    assert canonical_voice_id("Không có", catalog) == "Không có"


def test_line_voice_with_old_name_reaches_the_renamed_preset() -> None:
    """Trước: "Minh Quân" không khớp preset nào → dòng của vai đọc bằng giọng chung."""
    asked: list[str] = []

    class _Engine:
        def list_preset_voices(self):
            return [("Hải Đăng — Nam", "Hải Đăng"), ("Trúc Ly — Nữ", "Trúc Ly")]

        def get_preset_voice(self, name):
            asked.append(name)
            return {"voice": name}

    adapter = VieNeuTtsAdapter.__new__(VieNeuTtsAdapter)
    adapter._mode = "v3turbo"
    adapter._line_voice_presets = {}
    data, preset = adapter._resolve_line_voice(_Engine(), "Minh Quân")
    assert preset == "Hải Đăng" and data == {"voice": "Hải Đăng"} and asked == ["Hải Đăng"]


def _catalog() -> list[VoiceProfile]:
    return [
        VoiceProfile(
            name, LABELS[name],
            VOICE_FEMALE if "Nữ" in LABELS[name] else VOICE_MALE,
            style=style_from_voice_label(LABELS[name]), featured=is_featured_label(LABELS[name]),
        )
        for name in LABELS
    ]


def test_auto_cast_prefers_natural_voices_over_news_readers() -> None:
    cast = [
        CastMember("甲", "A", "male", "", 40, True), CastMember("乙", "B", "female", "", 30, True),
    ]
    casting = auto_cast(cast, _catalog(), {})
    assert casting["甲"] == "Adam bựa"          # tự nhiên, không phải Minh Đức (tin tức)
    assert casting["乙"] == "Trúc Ly"           # tự nhiên, không phải Mai Anh (tin tức)


def test_news_voice_is_last_resort_but_still_used() -> None:
    catalog = _catalog()
    cast = [CastMember(f"M{i}", f"M{i}", "male", "", 10 - i, True) for i in range(3)]
    casting = auto_cast(cast, catalog, {})
    assert set(casting.values()) == {"Adam bựa", "Thiện Minh", "Minh Đức"}
    assert casting["M0"] == "Adam bựa"
    assert casting["M2"] == "Minh Đức"          # tin tức đứng cuối


def test_default_two_voices_skip_the_news_reader() -> None:
    male, female = default_gender_voices(
        [p for p in _catalog() if p.voice_id in {"Minh Đức", "Thiện Minh", "Mai Anh", "Trúc Ly"}]
    )
    assert male == "Thiện Minh"                 # kể chuyện hơn tin tức (Minh Đức đứng trước)
    assert female == "Trúc Ly"                  # tự nhiên hơn tin tức (Mai Anh đứng trước)


def test_version_advice() -> None:
    assert parse_version("3.8.3.post1") == (3, 8, 3)
    assert parse_version("") == () and parse_version(None) == ()
    assert "pip install -U vieneu" in vieneu_version_advice("3.7.1")
    assert "frame đệm" in vieneu_version_advice("3.7.1")
    assert "khuyên dùng" in vieneu_version_advice("3.8.1")
    assert vieneu_version_advice("3.8.3") == ""
    assert vieneu_version_advice("3.9.0") == ""
    assert vieneu_version_advice(None) == ""


def test_install_specs_require_the_fixed_sdk() -> None:
    assert "vieneu[legacy]>=3.8.3,<4" in VIENEU_INSTALL_SPECS
    assert "sea-g2p>=0.9.1" in VIENEU_INSTALL_SPECS
