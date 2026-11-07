"""[v3.23.687] Giọng lồng tiếng đã chọn theo vai được giữ qua các tập dù bảng nhân vật ghi tên khác nhau.

Tên vai thật từ phân tích phim mẫu: tập 1 "楚瑶", tập 5 "楚瑶 / 瑶瑶" (cùng một người).
"""

from __future__ import annotations

from subtitles_extractor.application.services.dubbing_cast import match_saved_casting


def test_alias_match_keeps_voice_of_returning_role() -> None:
    saved = {"叶然": "nam_1", "楚瑶": "nu_2"}
    assert match_saved_casting(["叶然", "楚瑶 / 瑶瑶", "秦执"], saved) == {"叶然": "nam_1", "楚瑶 / 瑶瑶": "nu_2"}
    # Ngược lại: đã lưu tên có bí danh, tập sau chỉ ghi một tên.
    assert match_saved_casting(["瑶瑶"], {"楚瑶 / 瑶瑶": "nu_2"}) == {"瑶瑶": "nu_2"}


def test_exact_name_wins_and_latest_saved_is_used() -> None:
    # Hai mục đã lưu cùng khớp bí danh → lấy mục lưu gần nhất (cuối dict).
    saved = {"楚瑶": "nu_old", "瑶瑶": "nu_new"}
    assert match_saved_casting(["楚瑶 / 瑶瑶"], saved) == {"楚瑶 / 瑶瑶": "nu_new"}
    # Vai trùng nguyên tên giữ giọng của chính nó; một mục đã lưu không cấp cho hai vai.
    saved = {"楚瑶 / 瑶瑶": "nu_2"}
    out = match_saved_casting(["楚瑶 / 瑶瑶", "瑶瑶"], saved)
    assert out == {"楚瑶 / 瑶瑶": "nu_2"}


def test_unrelated_names_and_annotations() -> None:
    assert match_saved_casting(["南宫仙子"], {"南宫梦": "nu_3"}) == {}
    # Chú thích trong ngoặc bị bỏ khi so khớp (như khi gán người nói).
    assert match_saved_casting(["叶然（师兄）"], {"叶然": "nam_1"}) == {"叶然（师兄）": "nam_1"}
    assert match_saved_casting([], {"叶然": "nam_1"}) == {}
    assert match_saved_casting(["叶然"], {}) == {}
