"""[v3.23.169] Test quyết định chèn nghỉ hội thoại — bỏ nghỉ khi khung quá chật.

Dữ liệu thật (Along with the Gods, 1497 câu): STT 567 "- Tôi xin lỗi. / - Xin lỗi chỉ
huy." nhồi khung 0.81s, is_dialog -> chèn 100ms nghỉ -> nghỉ ăn vào khung -> đẩy nén từ
~1.4x lên 1.57x (giảm chất lượng giọng). Khoảng nghỉ hội thoại là trang trí; khi khung
chật thì ưu tiên giữ giọng rõ. Hàm thuần quyết định dựa trên tỉ lệ nén DỰ KIẾN sau khi
trừ nghỉ.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.edge_tts_adapter import (
    _should_insert_dialog_pause,
)


def test_inserts_when_room_is_ample() -> None:
    # Khung 3.0s, giọng 1.5s, nghỉ 0.1s -> nén dự kiến 1.5/2.9 ~ 0.52 -> chèn thoải mái.
    assert _should_insert_dialog_pause(True, 100, 1.5, 3.0, 0.1) is True


def test_skips_when_crowded() -> None:
    # Ca STT 567: giọng ~1.06s, khung an toàn ~1.06s, nghỉ 0.1s -> sau trừ nghỉ còn
    # 0.96s -> nén 1.06/0.96 ~ 1.10... vẫn dưới 1.30? Dùng số chật hơn cho rõ:
    # giọng 1.06, khung 1.06, nghỉ 0.1 -> room 0.96 -> ratio 1.104 (chèn được).
    # Ca thật chật: giọng 1.30, khung 1.06, nghỉ 0.1 -> room 0.96 -> 1.354 > 1.30 -> bỏ.
    assert _should_insert_dialog_pause(True, 100, 1.30, 1.06, 0.1) is False


def test_skips_when_no_room_after_pause() -> None:
    # Nghỉ gần bằng cả khung -> không còn chỗ cho giọng -> bỏ.
    assert _should_insert_dialog_pause(True, 100, 0.5, 0.12, 0.1) is False


def test_not_dialog_never_inserts() -> None:
    assert _should_insert_dialog_pause(False, 100, 1.0, 5.0, 0.1) is False


def test_zero_pause_config_never_inserts() -> None:
    assert _should_insert_dialog_pause(True, 0, 1.0, 5.0, 0.0) is False


def test_boundary_at_crowded_ratio() -> None:
    # room = 1.0, giọng = 1.30 -> ratio đúng 1.30 (<=) -> vẫn chèn (biên chấp nhận).
    assert _should_insert_dialog_pause(True, 100, 1.30, 1.10, 0.10) is True
    # giọng 1.31 -> ratio 1.31 > 1.30 -> bỏ.
    assert _should_insert_dialog_pause(True, 100, 1.31, 1.10, 0.10) is False


def test_custom_crowded_ratio() -> None:
    # Nới ngưỡng chật lên 1.6 -> ca 1.354 nay được chèn.
    assert _should_insert_dialog_pause(
        True, 100, 1.30, 1.06, 0.1, crowded_ratio=1.6
    ) is True
