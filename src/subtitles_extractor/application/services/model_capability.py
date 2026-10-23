"""[v3.23.606] So sức mạnh hai model dịch qua tên — để dùng lại kết quả đã lưu.

Chuẩn đoán thật 12/10, 05:48 đến 08:04 (灭门惨案II, v3.23.605). Người dùng làm đúng
lời khuyên "hết hạn mức ngày → chọn model có hạn mức cao hơn":

* Lần 1: dịch thô xong bằng gemini-3.8-flash (22 request, 43 phút), giai đoạn tinh chỉnh
  hết hạn mức ngay request đầu.
* Lần 2: đổi sang gemini-3.7-flash → khoá checkpoint chứa tên model nên mất bản dịch thô
  của 3.8-flash, dịch thô LẠI (21 request), tinh chỉnh được 11 lô thì hết hạn mức.
* Lần 3: đổi sang gemini-3.5-flash-lite → lại dịch thô và tinh chỉnh từ đầu. Bản cuối
  hoàn toàn là của model yếu nhất: 240 dòng vượt ngân sách đọc (bản dịch thô 3.7-flash
  hôm trước: 122), 80 dòng lồng tiếng phải đọc nhanh hơn 1,6x.

Quy tắc: kết quả đã lưu của một giai đoạn được dùng lại khi model đã dịch nó MẠNH HƠN hoặc
NGANG model đang chọn (đổi xuống vì hết hạn mức). Đổi LÊN model mạnh hơn thì dịch lại —
người dùng muốn chất lượng cao hơn. Tên không đọc được hạng thì chỉ dùng lại khi trùng
tên.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from subtitles_extractor.domain.value_objects.model_alias import canonical_model_id

__all__ = [
    "ModelCapability",
    "model_capability",
    "result_reusable_for",
]

#: Hạng theo từ khoá trong tên model; dòng "lite"/"nano" xếp dưới bản thường cùng thế hệ.
_TIER_BY_WORD = {"nano": 0, "lite": 1, "flash": 2, "pro": 3, "ultra": 4}
_STANDARD_TIER = 2
#: Các đời flash-lite đã đo là ngang hạng (xem :func:`model_capability`).
_MEASURED_EQUAL_LITE = ((3, 1), (3, 5))
_VERSION_RE = re.compile(r"^\d+(?:\.\d+)*$")


@dataclass(frozen=True, order=True)
class ModelCapability:
    """Hạng so sánh được của một model.

    Attributes:
        tier: Dòng model (lite < flash < pro…), so trước.
        version: Thế hệ, vd ``(3, 8)``; so sau ``tier``.
    """

    tier: int
    version: tuple[int, ...]


def _family(model_name: str) -> str:
    return model_name.strip().lower().split("-", 1)[0]


def model_capability(model_name: str) -> ModelCapability | None:
    """Đọc hạng của model từ tên (hàm thuần).

    Args:
        model_name: Vd ``"gemini-3.8-flash"``, ``"gemini-3.5-flash-lite"``.

    Returns:
        Hạng so sánh được; ``None`` khi tên không có số thế hệ.
    """
    # [v3.23.622] Bí danh đã ngừng chạy trên model đích (3.5→3.6, 3.7→3.8).
    words = canonical_model_id(model_name).split("-")
    version = next(
        (tuple(int(part) for part in word.split(".")) for word in words
         if _VERSION_RE.match(word)),
        None,
    )
    if version is None:
        return None
    tiers = [_TIER_BY_WORD[word] for word in words if word in _TIER_BY_WORD]
    # "flash-lite" chứa cả "flash" lẫn "lite": từ hạ hạng thắng.
    tier = min(tiers) if tiers else _STANDARD_TIER
    # [v3.23.643] Đã đo (chấm mù, v3.23.635): 3.1-flash-lite dịch TỐT HƠN 3.5-flash-lite (điểm 30 so với
    # 42-52) nên "đời mới hơn" không có nghĩa mạnh hơn trong dòng lite. Nhật ký thật 22/10: người dùng chọn
    # lại 3.5-flash-lite → ứng dụng vứt bản dịch thô 3.1 đã hoàn tất (67 phút) vì cho là "yếu hơn".
    if tier == _TIER_BY_WORD["lite"] and version in _MEASURED_EQUAL_LITE:
        version = _MEASURED_EQUAL_LITE[0]
    return ModelCapability(tier=tier, version=version)


def result_reusable_for(stored_model: str, requested_model: str) -> bool:
    """Kết quả do ``stored_model`` dịch có được dùng thay cho ``requested_model`` không.

    Args:
        stored_model: Model đã dịch kết quả đang lưu.
        requested_model: Model người dùng đang chọn cho giai đoạn đó.

    Returns:
        True khi trùng tên, hoặc cùng họ model và ``stored_model`` mạnh hơn/ngang hạng.
    """
    if stored_model.strip().lower() == requested_model.strip().lower():
        return True
    if _family(stored_model) != _family(requested_model):
        return False
    stored = model_capability(stored_model)
    requested = model_capability(requested_model)
    if stored is None or requested is None:
        return False
    return stored >= requested
