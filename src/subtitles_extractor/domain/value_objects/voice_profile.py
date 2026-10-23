"""[v3.23.593] Mô tả một giọng đọc để PHÂN VAI lồng tiếng (giọng nào hợp nhân vật nào).

Mỗi engine TTS chỉ biết ID giọng (``"vi-VN-NamMinhNeural"``, ``"Charon"``, ``"Thiện
Minh"``). Phân vai tự động cần thêm hai điều: giọng nam hay nữ, và giọng trầm/cao (để cụ
già không mang giọng thanh niên, trẻ con không mang giọng trầm). :class:`VoiceProfile`
gói các thông tin đó; engine điền chúng ở ``SubtitleTTSPort.list_voice_catalog``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Giới tính giọng — chuỗi cố định để so sánh/tuần tự hoá dễ dàng.
VOICE_MALE = "male"
VOICE_FEMALE = "female"
#: Sắc giọng tương đối trong cùng một giới.
TONE_LOW = "low"
TONE_HIGH = "high"

#: [v3.23.626] Phong cách đọc của giọng (VieNeu 3.8 ghi trong nhãn: "Phong cách tự nhiên" …).
STYLE_NATURAL = "natural"
STYLE_STORY = "story"
STYLE_NEWS = "news"

#: Gạch dài, gạch ngắn hoặc gạch nối, hay dấu mở ngoặc, rồi tới chữ giới tính.
_LABEL_GENDER_RE = re.compile(
    r"(?:[\u2014\u2013-]\s*|\(\s*)(nam|nữ)\b", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    """Một giọng có thể giao cho nhân vật.

    Attributes:
        voice_id: Giá trị engine nhận ở ``TTSRequest.speaker``/``line_voices``.
        label: Tên hiển thị cho người dùng.
        gender: :data:`VOICE_MALE`, :data:`VOICE_FEMALE` hoặc rỗng (không rõ).
        tone: :data:`TONE_LOW`, :data:`TONE_HIGH` hoặc rỗng (giọng thường).
        style: [v3.23.626] ``STYLE_*`` hoặc rỗng (engine không ghi). Thoại nhân vật hợp giọng
            "tự nhiên" nhất; giọng đọc tin tức / kể chuyện hợp lời dẫn hơn.
        featured: [v3.23.626] Giọng do tác giả engine tuyển chọn (VieNeu: dấu ⭐).
        aliases: [v3.23.626] Tên CŨ vẫn trỏ về giọng này (VieNeu đổi tên giọng: "Minh Quân"
            → "Hải Đăng") — lựa chọn giọng đã lưu dưới tên cũ vẫn dùng được.
    """

    voice_id: str
    label: str = ""
    gender: str = ""
    tone: str = ""
    style: str = ""
    featured: bool = False
    aliases: tuple[str, ...] = ()

    @property
    def display(self) -> str:
        """Nhãn hiển thị, lùi về ID khi engine không có nhãn."""
        return self.label or self.voice_id


def gender_from_voice_label(label: str) -> str:
    """Đọc giới tính từ nhãn giọng của VieNeu.

    Hai dạng nhãn thật của VieNeu 3.8: ``"⭐ Thiện Minh — Nam · Bắc · Phong cách kể
    chuyện"`` (v3 Turbo/v3 Nano: chữ đầu sau gạch dài là GIỚI, chữ thứ hai là vùng miền)
    và ``"Bình (nam miền Bắc)"`` (bản standard). Chỉ đọc chữ ĐẦU sau dấu ngăn — chữ
    "Nam" thứ hai trong ``"Nữ · Nam · …"`` là miền Nam, không phải giới tính.

    Args:
        label: Nhãn giọng.

    Returns:
        :data:`VOICE_MALE`, :data:`VOICE_FEMALE` hoặc rỗng nếu nhãn không ghi.
    """
    match = _LABEL_GENDER_RE.search(label or "")
    if match is None:
        return ""
    return VOICE_FEMALE if match.group(1).casefold() == "nữ" else VOICE_MALE


def _fold(text: str) -> str:
    """Bỏ dấu + thường hoá để so khớp từ khoá ("Kể chuyện" → "ke chuyen")."""
    decomposed = unicodedata.normalize("NFD", text or "")
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "d").casefold()


def style_from_voice_label(label: str) -> str:
    """Phong cách đọc ghi trong nhãn giọng VieNeu 3.8.

    Nhãn thật: ``"⭐ Mai Anh — Nữ · Bắc · Phong cách tin tức"``, ``"… Phong cách kể chuyện"``,
    ``"… Phong cách đọc truyện"``, ``"… Phong cách tự nhiên"``, ``"… Giọng đọc tự nhiên"``.

    Returns:
        ``STYLE_NATURAL`` / ``STYLE_STORY`` / ``STYLE_NEWS`` hoặc rỗng nếu nhãn không ghi.
    """
    folded = _fold(label)
    if "tin tuc" in folded:
        return STYLE_NEWS
    if "ke chuyen" in folded or "doc truyen" in folded:
        return STYLE_STORY
    if "tu nhien" in folded:
        return STYLE_NATURAL
    return ""


def is_featured_label(label: str) -> bool:
    """Nhãn giọng có dấu ⭐ (giọng tuyển chọn của VieNeu)."""
    return (label or "").lstrip().startswith("\u2b50")


def style_rank(profile: VoiceProfile) -> int:
    """Độ hợp thoại nhân vật (nhỏ hơn = hợp hơn): tự nhiên < chưa rõ/kể chuyện < tin tức."""
    if profile.style == STYLE_NATURAL:
        return 0
    return 2 if profile.style == STYLE_NEWS else 1


def gender_from_roster(gender: str) -> str:
    """Giới tính ghi trong bảng nhân vật: ``"nam"``/``"male"`` → nam, ``"nữ"`` → nữ.

    [v3.23.602] Chuyển từ ``dubbing_cast`` (hàm riêng) về đây để lưới soát nhãn người nói
    vừa hỏi lại dùng chung, không phải phụ thuộc khâu lồng tiếng.

    Args:
        gender: Trường giới tính của một nhân vật.

    Returns:
        :data:`VOICE_MALE`, :data:`VOICE_FEMALE` hoặc rỗng nếu không rõ.
    """
    # [v3.23.615] Dạng Unicode tổ hợp ("nữ" = n + u + dấu móc + dấu ngã) cũng là nữ.
    value = unicodedata.normalize("NFC", (gender or "").strip()).casefold()
    if value in {"nữ", "nu", "female", "f", "女"}:
        return VOICE_FEMALE
    if value in {"nam", "male", "m", "男"}:
        return VOICE_MALE
    # [v3.23.614] "Nam giới", "nữ giới", "nữ (cải trang nam)": chữ ĐẦU là giới tính.
    head = value.split(maxsplit=1)[0] if value else ""
    if head in {"nữ", "nu", "female"}:
        return VOICE_FEMALE
    if head in {"nam", "male"}:
        return VOICE_MALE
    return ""


__all__ = [
    "STYLE_NATURAL",
    "STYLE_NEWS",
    "STYLE_STORY",
    "TONE_HIGH",
    "TONE_LOW",
    "VOICE_FEMALE",
    "VOICE_MALE",
    "VoiceProfile",
    "gender_from_roster",
    "gender_from_voice_label",
    "is_featured_label",
    "style_from_voice_label",
    "style_rank",
]
