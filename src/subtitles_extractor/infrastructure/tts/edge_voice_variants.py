"""[v3.23.593] Biến thể cao độ cho giọng Edge TTS — đủ giọng để LỒNG TIẾNG nhiều vai.

Edge TTS chỉ có HAI giọng tiếng Việt (HoaiMy nữ, NamMinh nam). Phim thật có hàng chục
vai: lượt dịch thật v3.23.592 (phim "灭门惨案II") có 13 nhân vật nam nói ít nhất một câu
— với hai giọng, mọi người đàn ông trong phim cùng một giọng. Dịch vụ Edge nhận tham số
``pitch`` (Hz, tương đối) nên mỗi giọng gốc tách được thành ba giọng nghe khác nhau:
thường, trầm, cao. Giọng trầm hợp vai lớn tuổi, giọng cao hợp vai trẻ.

ID giọng biến thể ghi ``"<giọng gốc>@<pitch>"`` (vd ``"vi-VN-NamMinhNeural@-12Hz"``):
một chuỗi duy nhất đi qua mọi nơi đang truyền tên giọng (QSettings,
``TTSRequest.speaker``, ``line_voices``) mà không phải đổi kiểu dữ liệu. Chỉ adapter
Edge tách nó ra khi gọi dịch vụ.
"""

from __future__ import annotations

import re

from subtitles_extractor.domain.value_objects.voice_profile import (
    TONE_HIGH,
    TONE_LOW,
    VOICE_FEMALE,
    VOICE_MALE,
    VoiceProfile,
)

#: Ký tự ngăn giọng gốc và cao độ trong ID giọng biến thể.
PITCH_SEPARATOR = "@"
#: Cao độ mặc định của Edge (không đổi).
NEUTRAL_PITCH = "+0Hz"
#: (cao độ, hậu tố nhãn, sắc giọng). ±12Hz: đủ để tai phân biệt hai vai cùng giọng gốc mà
#: giọng vẫn tự nhiên (giọng nam nói ~100-150Hz → lệch ~10%).
EDGE_PITCH_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("", "", ""),
    ("-12Hz", "trầm", TONE_LOW),
    ("+12Hz", "cao", TONE_HIGH),
)
_PITCH_RE = re.compile(r"^[+-]\d{1,3}Hz$")
#: Giọng NỮ trong bảng giọng Edge của ứng dụng; các giọng còn lại là giọng nam.
_FEMALE_VOICES = frozenset({
    "vi-VN-HoaiMyNeural", "zh-CN-XiaoxiaoNeural", "zh-TW-HsiaoChenNeural",
    "en-US-JennyNeural", "en-US-AriaNeural", "en-GB-SoniaNeural", "ja-JP-NanamiNeural",
    "ko-KR-SunHiNeural", "fr-FR-DeniseNeural", "es-ES-ElviraNeural",
    "de-DE-KatjaNeural", "th-TH-PremwadeeNeural",
})


def split_voice_variant(voice: str) -> tuple[str, str]:
    """Tách ID giọng thành (giọng gốc gửi Edge, cao độ).

    Args:
        voice: ``"vi-VN-NamMinhNeural"`` hoặc ``"vi-VN-NamMinhNeural@-12Hz"``.

    Returns:
        ``(giọng gốc, pitch)``; pitch sai định dạng được bỏ (``"+0Hz"``) để dịch vụ không
        từ chối cả câu.
    """
    base, separator, pitch = (voice or "").partition(PITCH_SEPARATOR)
    if not separator or not _PITCH_RE.match(pitch.strip()):
        return base.strip(), NEUTRAL_PITCH
    return base.strip(), pitch.strip()


def _short_name(voice: str) -> str:
    """``"vi-VN-NamMinhNeural"`` → ``"NamMinh"`` (nhãn gọn cho bảng phân vai)."""
    name = voice.rsplit("-", 1)[-1]
    return name[: -len("Neural")] if name.endswith("Neural") else name


def edge_voice_catalog(voices: list[str]) -> list[VoiceProfile]:
    """Danh sách giọng Edge kèm biến thể cao độ, giới tính và sắc giọng.

    Args:
        voices: Giọng gốc của một ngôn ngữ (thứ tự gợi ý của ứng dụng).

    Returns:
        Mỗi giọng gốc ba mục: thường, trầm, cao.
    """
    catalog: list[VoiceProfile] = []
    for voice in voices:
        is_female = voice in _FEMALE_VOICES
        gender = VOICE_FEMALE if is_female else VOICE_MALE
        gender_label = "nữ" if is_female else "nam"
        for pitch, suffix, tone in EDGE_PITCH_VARIANTS:
            voice_id = f"{voice}{PITCH_SEPARATOR}{pitch}" if pitch else voice
            name = _short_name(voice) + (f" · {suffix}" if suffix else "")
            catalog.append(VoiceProfile(
                voice_id=voice_id, label=f"{name} ({gender_label})",
                gender=gender, tone=tone,
            ))
    return catalog


__all__ = [
    "EDGE_PITCH_VARIANTS",
    "NEUTRAL_PITCH",
    "PITCH_SEPARATOR",
    "edge_voice_catalog",
    "split_voice_variant",
]
