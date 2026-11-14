"""[v3.23.724] Danh mục model Gemini sinh văn bản ("Text-out models") — miễn phí và trả phí.

Nguồn: bảng Rate Limit trong Google AI Studio của chủ dự án (14/11/2026) + danh sách ``models.list`` của Gemini API cùng
ngày (tên gọi API chính xác). Trước bản này ứng dụng chỉ liệt kê ``*flash*`` miễn phí: bỏ hẳn các model trả phí (2.5 Pro,
3.1 Pro, Gemini 4 Argon / Carbon) và cả model miễn phí mới không mang chữ "flash" (Gemini 4 Hydrogen).

``free`` = hạn mức bản MIỄN PHÍ (RPM, TPM, RPD); ``None`` = bản miễn phí 0/0 → chỉ dùng được với key trả phí.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import RateLimit

#: Hạn mức dùng để ĐIỀU TIẾT model chỉ trả phí (mức Tier 1 thấp nhất của Google); vượt thì máy chủ trả 429 và bộ quản
#: lý hạn mức học lại từ lỗi đó. Trước đây model Pro bị áp 5 request/phút, 20 request/ngày của bản miễn phí.
PAID_TIER_LIMIT = RateLimit(rpm=150, tpm=2_000_000, rpd=1_000)


@dataclass(frozen=True, slots=True)
class GeminiTextModel:
    """Một model Gemini sinh văn bản."""

    model_id: str
    label: str
    free: RateLimit | None
    note: str = ""

    @property
    def paid_only(self) -> bool:
        return self.free is None


_FREE_FLASH = RateLimit(rpm=5, tpm=250_000, rpd=20)
_FREE_LITE_HI = RateLimit(rpm=15, tpm=250_000, rpd=500)

#: Thứ tự hiển thị: miễn phí (RPD cao trước) rồi trả phí.
TEXT_MODELS: tuple[GeminiTextModel, ...] = (
    GeminiTextModel(
        "gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite", _FREE_LITE_HI, "mặc định dịch — đo tốt và nhanh nhất",
    ),
    GeminiTextModel("gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite", _FREE_LITE_HI),
    GeminiTextModel("gemini-3.8-flash", "Gemini 3.8 Flash", _FREE_FLASH, "mặc định phân tích"),
    GeminiTextModel("gemini-4-hydrogen", "Gemini 4 Hydrogen", _FREE_FLASH, "thế hệ 4"),
    GeminiTextModel("gemini-3.6-flash", "Gemini 3.6 Flash", _FREE_FLASH),
    GeminiTextModel("gemini-3-flash-preview", "Gemini 3 Flash", _FREE_FLASH),
    GeminiTextModel("gemini-2.5-flash", "Gemini 2.5 Flash", _FREE_FLASH),
    GeminiTextModel("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", RateLimit(rpm=10, tpm=250_000, rpd=20)),
    GeminiTextModel("gemini-4-argon", "Gemini 4 Argon", None, "thế hệ 4"),
    GeminiTextModel("gemini-4-carbon", "Gemini 4 Carbon", None, "thế hệ 4"),
    GeminiTextModel("gemini-3.1-pro-preview", "Gemini 3.1 Pro", None),
    GeminiTextModel("gemini-2.5-pro", "Gemini 2.5 Pro", None),
)
_BY_ID = {model.model_id: model for model in TEXT_MODELS}

#: Model trả về từ API nhưng KHÔNG sinh văn bản hội thoại (ảnh, giọng nói, thời gian thực, nhúng, robot…).
_NON_TEXT_MARKERS = (
    "embedding", "aqa", "imagen", "veo", "tts", "image", "banana", "live", "native-audio", "transcribe", "robotics",
    "computer-use", "customtools", "decision", "omni", "learnlm", "lyria", "gemma", "-exp", "antigravity",
    "deep-research",
)
#: Đã ngừng và tự định tuyến sang model khác (deprecations 10/2026) — không gợi ý.
_DEPRECATED = frozenset({"gemini-3.5-flash", "gemini-3.7-flash"})


def lookup(model_id: str) -> GeminiTextModel | None:
    return _BY_ID.get(_bare(model_id))


def _bare(model_id: str) -> str:
    name = (model_id or "").strip().lower()
    return name[len("models/"):] if name.startswith("models/") else name


def is_text_out_model(model_id: str) -> bool:
    """Model Gemini sinh văn bản dùng được cho dịch / phân tích — CẢ miễn phí lẫn trả phí."""
    name = _bare(model_id)
    if name in _BY_ID:
        return True
    if not name.startswith("gemini-") or name in _DEPRECATED:
        return False
    if any(marker in name for marker in _NON_TEXT_MARKERS):
        return False
    return not re.match(r"gemini-(1\.|2\.0|2-)", name)   # đời cũ đã ngừng


def default_limit(model_id: str) -> RateLimit | None:
    """Hạn mức điều tiết mặc định theo danh mục: miễn phí → hạn mức miễn phí; chỉ trả phí → :data:`PAID_TIER_LIMIT`;
    ``None`` nếu model không có trong danh mục (bộ quản lý hạn mức tự suy theo tên)."""
    model = lookup(model_id)
    if model is None:
        return None
    return model.free if model.free is not None else PAID_TIER_LIMIT


def sort_key(model_id: str) -> tuple[int, int, str]:
    """Sắp xếp danh sách: theo thứ tự danh mục, model lạ xếp sau (miễn phí đoán theo tên trước)."""
    name = _bare(model_id)
    if name in _BY_ID:
        return (0, TEXT_MODELS.index(_BY_ID[name]), name)
    return (1, 0 if "flash" in name else 1, name)


def describe(model_id: str) -> str:
    """Chú thích hạn mức cho ô chọn model (tiếng Việt)."""
    model = lookup(model_id)
    if model is None:
        return "Model khác — hạn mức xem trong Google AI Studio (Rate Limit)."
    if model.free is None:
        text = f"{model.label}: CHỈ dùng với key trả phí (bản miễn phí 0/0)."
    else:
        free = model.free
        text = (f"{model.label}: miễn phí {free.rpm} request/phút · {free.tpm // 1000}K token/phút · "
                f"{free.rpd} request/ngày; key trả phí được hạn mức cao hơn.")
    return f"{text} {model.note}".strip() if model.note else text


def default_model_ids() -> list[str]:
    """Danh sách model hiển thị khi chưa tải từ API."""
    return [model.model_id for model in TEXT_MODELS]


__all__ = [
    "PAID_TIER_LIMIT", "TEXT_MODELS", "GeminiTextModel", "default_limit", "default_model_ids", "describe",
    "is_text_out_model", "lookup", "sort_key",
]
