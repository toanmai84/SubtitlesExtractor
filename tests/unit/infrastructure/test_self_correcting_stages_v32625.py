"""[v3.23.225] VÒNG TỰ SỬA giữa các giai đoạn dịch — chi phí thêm bằng 0.

Bằng chứng từ log chạy thật (v224, cùng video):

* Lưới ``under_translation_guard`` chạy 3 lần — sau Giai đoạn 2 (dịch thô), 3 (tinh
  chỉnh), 4 (bản địa hoá) — và báo **CÙNG 5 dòng, CÙNG lỗi** ở cả ba lần::

      22:11:02  GĐ2 -> 82, 52, 71, 73, 80
      22:11:28  GĐ3 -> 82, 52, 71, 73, 80   (y hệt)
      22:11:50  GĐ4 -> 82, 52, 71, 73, 80   (y hệt)

* Nghĩa là lỗi sinh ra ở GĐ2 và **không giai đoạn nào sửa**, dù GĐ3/GĐ4 ĐÃ gọi API và ĐÃ
  có bản gốc trong payload kép. Ta trả tiền cho hai lượt gọi rồi bỏ phí.

Đối chiếu gốc CJK (lần đầu có trong log) xác nhận lưới bắt ĐÚNG:

* ``自然会受到一些`` -> "sẽ chịu," (rớt "đương nhiên" + "một số")
* ``你这么多年在外面`` -> "Bao năm qua," (rớt "ở bên ngoài")
* ``作为王建强的侄女`` -> "Là cháu chú," (rớt tên riêng Vương Kiến Cường)
* ``内心是怎么想我的`` -> "Chú nghĩ gì?" (ĐẢO chủ thể — câu gốc lược chủ ngữ)

Giải pháp: đánh cờ ``needs_expansion`` cho các dòng đó trong payload của giai đoạn KẾ
TIẾP, kèm chỉ thị bổ sung. Tính từ chính ``input_lines`` của giai đoạn (= output giai đoạn
trước) nên KHÔNG cần lưu state, KHÔNG thêm lượt gọi API.
"""

from __future__ import annotations

import pathlib
import re

from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

_SRC = pathlib.Path(
    "src/subtitles_extractor/infrastructure/translation/gemini_translation_adapter.py"
).read_text(encoding="utf-8")


def _bare_translator() -> "GeminiSubtitleTranslator":
    """[v3.23.536] Thực thể TRỐNG để gọi các hàm thuần cần ``self``.

    ``_length_hint``, ``_line_to_payload`` và ``_line_to_dual_payload`` là phương thức thực thể
    (không phải staticmethod); phần thân chỉ dùng vài thuộc tính tuỳ chọn nên dựng bằng ``__new__``
    là đủ, không cần API key.
    """
    instance = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    instance._reading_cps_budget = 0.0
    instance._emotion_cues = False
    return instance


def _line(idx: int, text: str, original: str = "") -> TranslationLine:
    return TranslationLine(
        index=idx,
        start_ms=idx * 2000,
        end_ms=idx * 2000 + 1500,
        text=text,
        original_text=original,
    )


# ── Cờ needs_expansion trong payload ────────────────────────────────────────
def test_payload_kep_mang_co_can_bo_sung() -> None:
    line = _line(80, "Là cháu chú,", "作为王建强的侄女")
    payload = _bare_translator()._line_to_dual_payload(
        line, next_start_ms=162_000, needs_expansion=True
    )
    assert payload["needs_expansion"] is True
    assert payload["original"] == "作为王建强的侄女"  # có bản gốc để đối chiếu


def test_dong_binh_thuong_khong_mang_co() -> None:
    line = _line(5, "Hôm nay cháu tới đây làm gì", "你今天来这里做什么")
    payload = _bare_translator()._line_to_dual_payload(line, 12_000)
    assert "needs_expansion" not in payload


def test_payload_don_khong_co_co() -> None:
    # Giai đoạn LITERAL không có bản gốc trong payload -> cờ vô nghĩa, không được gửi.
    payload = _bare_translator()._line_to_payload(_line(1, "x", "原文"), 5000)
    assert "needs_expansion" not in payload


# ── Prompt dạy model xử lý cờ ───────────────────────────────────────────────
def test_prompt_huong_dan_bo_sung_noi_dung() -> None:
    assert "needs_expansion" in _SRC
    assert "BỔ SUNG DÒNG BỊ DỊCH THIẾU" in _SRC
    # Phải nói rõ: thà vượt max_chars còn hơn mất nghĩa.
    assert "kể cả khi phải dài hơn" in _SRC


def test_prompt_dung_ca_that_lam_vi_du() -> None:
    assert "作为王建强的侄女" in _SRC  # ca rớt tên riêng
    assert "内心是怎么想我的" in _SRC  # ca đảo chủ thể


def test_prompt_canh_bao_cau_luoc_chu_ngu() -> None:
    # Tiếng Trung hay lược chủ ngữ -> model đoán bừa rồi đảo vai. Đây là lỗi TÁI PHẠM
    # (v224 đã cấm chung chung nhưng model vẫn sai ở dòng 73).
    assert "LỖI TÁI PHẠM NHIỀU LẦN" in _SRC
    assert "câu gốc lược chủ ngữ" in _SRC


def test_chi_thi_bo_sung_chi_gan_vao_giai_doan_co_ban_goc() -> None:
    """Chỉ thị "diễn giải" chỉ gắn vào giai đoạn CÓ bản gốc để đối chiếu.

    Giai đoạn LITERAL (dịch thô) chưa có ``original`` nên gắn chỉ thị vào đó vừa vô
    nghĩa vừa gây nhiễu prompt.

    [v3.23.547] Kiểm bằng BẤT BIẾN thay vì so nguyên văn cả dòng template: các đợt sau
    đã chèn thêm ``{no_source_script}``, ``{emotion_cues}`` vào cùng chuỗi, làm phép so
    nguyên văn đỏ dù bất biến vẫn đúng.
    """
    # [v3.23.612] Template có thể xuống dòng thành nhiều f-string liền nhau (giữ dòng
    # ≤ 90 ký tự) — nối phần tiếp theo vào trước khi so.
    templates = [
        head + "".join(re.findall(r'f"([^"]*)"', tail))
        for head, tail in re.findall(
            r'f"## QUY TẮC\\n(\{[^"]*)"((?:\s*\n\s*f"[^"]*")*)', _SRC
        )
    ]
    assert len(templates) == 3, f"Kỳ vọng 3 template giai đoạn, thấy {len(templates)}"

    literal, style, loc = templates
    # Hai giai đoạn có bản gốc: PHẢI mang chỉ thị bổ sung, ngay sau mệnh đề mô tả.
    # [v3.23.633] Quy tắc giữ vai/ngôi của giai đoạn tinh chỉnh nằm giữa mệnh đề mô tả và chỉ thị.
    assert "{desc_clause_style}{PERSON_ANCHOR_RULE_REFINE}{expansion}" in style
    assert "{desc_clause_loc}{PERSON_ANCHOR_RULE_REFINE}{expansion}" in loc
    assert "PERSON_ANCHOR_RULE_REFINE" not in literal
    # Giai đoạn LITERAL: KHÔNG được mang chỉ thị bổ sung lẫn mệnh đề mô tả.
    assert "{expansion}" not in literal
    assert "desc_clause" not in literal
    # Các quy tắc dùng chung thì cả ba giai đoạn đều phải có.
    for shared in ("{concise}", "{pronoun}", "{examples}"):
        assert all(shared in tpl for tpl in templates), shared
