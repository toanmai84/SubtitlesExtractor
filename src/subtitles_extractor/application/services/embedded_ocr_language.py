"""[v3.23.102] Ánh xạ ngôn ngữ track phụ đề nhúng (ISO 639) sang mã ngôn ngữ PaddleOCR.

VÌ SAO: OCR phụ đề nhúng phải dùng model nhận dạng ĐÚNG ngôn ngữ; nếu dùng model CJK cho
phụ đề Latin/Hàn/Ả-Rập… sẽ ra ký tự nhiễu (□■…). Track nhúng thường khai báo ngôn ngữ (vd
``eng``), nên ta tự suy ra mã PaddleOCR phù hợp; người dùng vẫn có thể chọn tay khi cần.
"""

from __future__ import annotations

# ISO 639-2/B (và vài mã 639-1) -> mã ngôn ngữ PaddleOCR.
# PaddleOCR: 'ch' (Trung+Anh), 'chinese_cht' (phồn thể), 'en', 'japan', 'korean',
# 'latin' (chữ Latin chung), 'cyrillic', 'arabic', 'devanagari', 'vi', 'th'.
_ISO639_TO_PADDLE: dict[str, str] = {
    "eng": "en", "en": "en",
    "chi": "ch", "zho": "ch", "zh": "ch", "cmn": "ch",
    "cht": "chinese_cht", "yue": "chinese_cht",
    "jpn": "japan", "ja": "japan",
    "kor": "korean", "ko": "korean",
    "vie": "vi", "vi": "vi",
    "tha": "th", "th": "th",
    "ara": "arabic", "ar": "arabic", "fas": "arabic", "per": "arabic", "urd": "arabic",
    "rus": "cyrillic", "ru": "cyrillic", "ukr": "cyrillic", "bul": "cyrillic",
    "srp": "cyrillic", "bel": "cyrillic", "mkd": "cyrillic",
    "hin": "devanagari", "mar": "devanagari", "nep": "devanagari", "san": "devanagari",
    # Nhóm chữ Latin (Pháp, Đức, Tây Ban Nha, Ý, Bồ, Hà Lan, Ba Lan…)
    "fra": "latin", "fre": "latin", "deu": "latin", "ger": "latin", "spa": "latin",
    "ita": "latin", "por": "latin", "nld": "latin", "dut": "latin", "pol": "latin",
    "ron": "latin", "rum": "latin", "ces": "latin", "cze": "latin", "swe": "latin",
    "dan": "latin", "nor": "latin", "fin": "latin", "hun": "latin", "tur": "latin",
    "ind": "latin", "msa": "latin", "may": "latin",
}

# Lựa chọn hiển thị trên UI: (nhãn, mã PaddleOCR). Mục đầu = tự động.
AUTO_LANGUAGE = ""  # rỗng = tự suy từ track

# Ngôn ngữ mà model hợp nhất hiện đại (PP-OCRv6 medium/small, PP-OCRv5 server) xử lý chung
# trong MỘT model -> không cần engine OCR riêng, dùng luôn engine chính cho chất lượng
# nhất. Ngôn ngữ ngoài tập này (Hàn, Kirin, Ả-Rập, Thái, Devanagari) cần model riêng.
_UNIFIED_PADDLE_LANGS: frozenset[str] = frozenset(
    {"ch", "chinese_cht", "en", "japan", "latin", "vi"}
)

UI_LANGUAGE_CHOICES: list[tuple[str, str]] = [
    ("Tự động (theo track)", AUTO_LANGUAGE),
    ("Tiếng Anh / Latin (en)", "en"),
    ("Trung giản thể (ch)", "ch"),
    ("Trung phồn thể (chinese_cht)", "chinese_cht"),
    ("Nhật (japan)", "japan"),
    ("Hàn (korean)", "korean"),
    ("Việt (vi)", "vi"),
    ("Thái (th)", "th"),
    # [v3.23.324] SỬA LỖI: bốn mục trước đây dùng "latin"/"cyrillic"/"arabic"/
    # "devanagari" — đó là TÊN NHÓM HỆ CHỮ VIẾT, KHÔNG phải mã ngôn ngữ hợp lệ.
    # Đã đối chiếu mã nguồn paddleocr 3.7 (_utils/langs.py + _pipelines/ocr.py):
    # PaddleOCR nhận mã THÀNH VIÊN (vd "fr") rồi tự suy ra model nhóm
    # ("latin_PP-OCRv5_mobile_rec"). Bản thân "latin" KHÔNG nằm trong LATIN_LANGS
    # nên truyền vào luôn thất bại: "No models are available for lang='latin'".
    ("Pháp (fr) — chữ Latin", "fr"),
    ("Đức (de) — chữ Latin", "de"),
    ("Tây Ban Nha (es) — chữ Latin", "es"),
    ("Bồ Đào Nha (pt) — chữ Latin", "pt"),
    ("Indonesia (id) — chữ Latin", "id"),
    ("Nga (ru) — chữ Kirin", "ru"),
    ("Ả-Rập (ar)", "ar"),
    ("Hindi (hi) — Devanagari", "hi"),
]


def resolve_paddle_lang(iso_639: str) -> str | None:
    """Suy mã ngôn ngữ PaddleOCR từ mã ngôn ngữ track (ISO 639).

    Args:
        iso_639: Mã ngôn ngữ của track (vd ``"eng"``, ``"jpn"``); có thể rỗng.

    Returns:
        Mã PaddleOCR (vd ``"en"``, ``"japan"``) hoặc ``None`` nếu không nhận diện được
        (khi đó nên giữ engine OCR mặc định của người dùng).
    """
    if not iso_639:
        return None
    return _ISO639_TO_PADDLE.get(iso_639.strip().lower())


def is_covered_by_unified_model(paddle_lang: str | None) -> bool:
    """True nếu ngôn ngữ đã được model hợp nhất (PP-OCRv6/v5 server) xử lý chung.

    Khi True, OCR phụ đề nhúng nên dùng engine chính (PP-OCRv6) thay vì dựng engine riêng,
    để có chất lượng tốt nhất và không tải dư model.
    """
    return bool(paddle_lang) and paddle_lang in _UNIFIED_PADDLE_LANGS


def describe_paddle_lang(paddle_lang: str | None) -> str:
    """Nhãn thân thiện cho mã PaddleOCR (để log/hiển thị)."""
    if not paddle_lang:
        return "mặc định"
    for label, code in UI_LANGUAGE_CHOICES:
        if code == paddle_lang:
            return label
    return paddle_lang


__all__ = [
    "AUTO_LANGUAGE",
    "UI_LANGUAGE_CHOICES",
    "describe_paddle_lang",
    "is_covered_by_unified_model",
    "resolve_paddle_lang",
]
