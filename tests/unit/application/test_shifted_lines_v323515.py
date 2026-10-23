"""[v3.23.515] Phát hiện dòng có bản dịch THUỘC VỀ dòng khác (lệch dòng / lệch khối).

Các cờ sẵn có chỉ bắt dấu hiệu GIÁN TIẾP (dòng quá ngắn, trùng dòng liền kề). Lệch KHỐI thì mọi
dòng vẫn dài bình thường và không trùng nhau nên lọt hết — dữ liệu thật có vùng dòng 257-262 lệch
3 dòng mà không cờ nào báo.

Cách đo tất định: dùng bảng thuật ngữ + bảng nhân vật đã duyệt làm MỐC NEO.
"""
from __future__ import annotations

from dataclasses import dataclass

from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
)

GLOSSARY = "支付宝 = Alipay\n心动值 = chỉ số rung động"
CHARACTERS = "林逸 (Lâm Dật) [nam] — nhân vật chính\n王梅 (Vương Mai) [nữ] — hàng xóm"


@dataclass
class _Event:
    index: int
    text: str


def _flags(source: list[_Event], translated: list[_Event]) -> dict:
    return detect_quality_flags(
        source, translated, glossary=GLOSSARY, characters=CHARACTERS
    )


class TestShiftDetection:
    """[v3.23.549] Chỉ báo lệch khi có BẰNG CHỨNG BỔ TRỢ.

    Trước đây một dòng lẻ bị cờ là đủ báo. Soi phim thật 3902 dòng: cả 5 dòng bị cờ đều là dương
    tính giả, mỗi dòng một độ lệch khác nhau (-1, +2, -3, +3, +2), không dòng nào kề dòng nào.
    Nay chỉ báo khi (a) hai dòng HOÁN ĐỔI cho nhau, hoặc (b) nhiều dòng gần nhau cùng độ lệch —
    đúng hai dạng mà lệch thật tạo ra.
    """

    def test_detects_mutual_swap(self) -> None:
        """Dòng 1 mang nội dung dòng 3 VÀ dòng 3 mang nội dung dòng 1 → lệch thật."""
        source = [_Event(1, "王梅人好"), _Event(2, "今天天气"), _Event(3, "支付宝到账")]
        translated = [
            _Event(1, "Alipay tiền vào tài khoản."),
            _Event(2, "Hôm nay trời đẹp."),
            _Event(3, "Vương Mai tốt bụng."),
        ]
        findings = _flags(source, translated)["shifted_lines"]
        assert {(item["index"], item["shift"]) for item in findings} == {(1, 2), (3, -2)}

    def test_detects_adjacent_swap(self) -> None:
        # [v3.23.522] Dòng trước KẾT THÚC CÂU → hai dòng độc lập, lệch là lệch thật.
        source = [_Event(1, "林逸来了。"), _Event(2, "支付宝到账。")]
        translated = [_Event(1, "Alipay tiền vào tài khoản."), _Event(2, "Lâm Dật đến rồi.")]
        findings = _flags(source, translated)["shifted_lines"]
        assert any(item["index"] == 2 and item["shift"] == -1 for item in findings)

    def test_report_includes_target_line(self) -> None:
        source = [_Event(1, "林逸来了。"), _Event(2, "支付宝到账。")]
        translated = [_Event(1, "Alipay tiền vào tài khoản."), _Event(2, "Lâm Dật đến rồi.")]
        findings = {item["index"]: item for item in _flags(source, translated)["shifted_lines"]}
        assert findings[2]["belongs_to_index"] == 1
        assert findings[1]["belongs_to_index"] == 2

    def test_isolated_single_flag_not_reported(self) -> None:
        """Một dòng lẻ mang mốc neo của dòng bên, không có gì bổ trợ → KHÔNG báo.

        Tái hiện dạng dương tính giả thật: model thêm hô ngữ/từ đồng nghĩa khiến một dòng tình cờ
        chứa chuỗi của dòng lân cận.
        """
        source = [_Event(1, "林逸来了。"), _Event(2, "他说话。")]
        translated = [_Event(1, "Anh ấy nói."), _Event(2, "Lâm Dật đến rồi.")]
        assert _flags(source, translated)["counts"]["shifted_lines"] == 0

    def test_correct_translation_not_flagged(self) -> None:
        source = [_Event(1, "林逸来了"), _Event(2, "支付宝到账")]
        translated = [
            _Event(1, "Lâm Dật đến rồi."),
            _Event(2, "Alipay tiền vào tài khoản."),
        ]
        assert _flags(source, translated)["counts"]["shifted_lines"] == 0

    def test_lines_without_anchors_ignored(self) -> None:
        """Dòng không chứa thuật ngữ/tên nào thì không đủ căn cứ — không báo bừa."""
        source = [_Event(1, "今天天气不错"), _Event(2, "是啊")]
        translated = [_Event(1, "Ừ đúng vậy."), _Event(2, "Hôm nay trời đẹp.")]
        assert _flags(source, translated)["counts"]["shifted_lines"] == 0

    def test_no_glossary_no_findings(self) -> None:
        source = [_Event(1, "林逸来了。"), _Event(2, "他说话。")]
        translated = [_Event(1, "Anh ấy nói."), _Event(2, "Lâm Dật đến rồi.")]
        flags = detect_quality_flags(source, translated)
        assert flags["shifted_lines"] == []


class TestSplitSentenceNotFlagged:
    """[v3.23.522] Câu tách đôi lệch ±1 là đảo trật tự hợp lệ — không được báo."""

    def test_split_pair_exempt(self) -> None:
        source = [_Event(1, "林逸来了"), _Event(2, "他很高兴")]
        translated = [_Event(1, "Anh ấy rất vui,"), _Event(2, "Lâm Dật đến rồi.")]
        assert _flags(source, translated)["counts"]["shifted_lines"] == 0
