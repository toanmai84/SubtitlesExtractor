"""Test cho :mod:`infrastructure.translation.key_rotation_policy` (v3.23.294).

Kiểm chứng chính sách xoay API key, đặc biệt sửa lỗi upload video 2 lần: khi video
đã upload dưới key hiện tại (``avoid_reupload=True``) thì KHÔNG xoay key vì lý do
cơ hội ``much_better`` — chỉ xoay khi bắt buộc (cạn quota hoặc không đủ trọn phiên).

Kịch bản gốc từ log thực tế (tập 4, 5): analyze upload dưới key #1 (còn 5 req),
translate thấy key #2/#3 còn 20 → cũ: xoay → upload lại. Mới: giữ key #1 (5 >= 3
stage) → tái dùng video.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.key_rotation_policy import (
    is_insufficient,
    is_much_better,
    should_switch_for_viability,
)


class TestIsMuchBetter:
    def test_double_or_more(self) -> None:
        assert is_much_better(current_remaining=5, best_remaining=20) is True

    def test_plus_five_boundary(self) -> None:
        # 3*2=6, 3+5=8 -> ngưỡng = 8; best=8 -> đủ.
        assert is_much_better(current_remaining=3, best_remaining=8) is True
        assert is_much_better(current_remaining=3, best_remaining=7) is False

    def test_not_much_better(self) -> None:
        assert is_much_better(current_remaining=10, best_remaining=12) is False


class TestIsInsufficient:
    def test_cannot_finish_and_better_exists(self) -> None:
        # Cần 8 request, key hiện tại còn 3, key khác còn 20 -> thiếu.
        assert is_insufficient(3, 20, needed_requests=8) is True

    def test_enough_for_session(self) -> None:
        # Cần 3, còn 5 -> đủ, không thiếu.
        assert is_insufficient(5, 20, needed_requests=3) is False

    def test_no_better_key(self) -> None:
        # Thiếu nhưng key khác cũng không hơn -> không coi là insufficient (vô ích).
        assert is_insufficient(2, 2, needed_requests=8) is False


class TestShouldSwitchForViability:
    def test_exhausted_always_switches(self) -> None:
        """Key hiện tại cạn (0) -> luôn xoay, kể cả đang giữ video."""
        assert (
            should_switch_for_viability(
                current_remaining=0,
                best_remaining=5,
                needed_requests=3,
                avoid_reupload=True,
            )
            is True
        )

    def test_insufficient_switches_even_with_video(self) -> None:
        """Không đủ đi trọn phiên -> xoay dù phải upload lại (không còn lựa chọn)."""
        assert (
            should_switch_for_viability(
                current_remaining=2,
                best_remaining=20,
                needed_requests=8,
                avoid_reupload=True,
            )
            is True
        )

    def test_much_better_switches_without_video(self) -> None:
        """Không giữ video -> giữ hành vi cũ: xoay cơ hội khi key khác hơn hẳn."""
        assert (
            should_switch_for_viability(
                current_remaining=5,
                best_remaining=20,
                needed_requests=1,
                avoid_reupload=False,
            )
            is True
        )

    def test_much_better_suppressed_with_video(self) -> None:
        """SỬA LỖI CHÍNH: giữ video + key hiện tại đủ -> KHÔNG xoay cơ hội."""
        assert (
            should_switch_for_viability(
                current_remaining=5,
                best_remaining=20,
                needed_requests=1,
                avoid_reupload=True,
            )
            is False
        )

    def test_episode4_scenario_fixed(self) -> None:
        """Tái hiện tập 4/5: key #1 còn 5, key khác 20, phiên cần 3 stage.

        avoid_reupload=True -> giữ key #1 (5 >= 3) -> tái dùng video, không upload lại.
        """
        assert (
            should_switch_for_viability(
                current_remaining=5,
                best_remaining=20,
                needed_requests=3,
                avoid_reupload=True,
            )
            is False
        )

    def test_episode4_scenario_old_behavior(self) -> None:
        """Cùng số liệu tập 4 nhưng avoid_reupload=False -> xoay (hành vi cũ gây upload lại)."""
        assert (
            should_switch_for_viability(
                current_remaining=5,
                best_remaining=20,
                needed_requests=3,
                avoid_reupload=False,
            )
            is True
        )

    def test_current_best_no_switch(self) -> None:
        """Key hiện tại đủ và không có key nào hơn hẳn -> giữ nguyên."""
        assert (
            should_switch_for_viability(
                current_remaining=10,
                best_remaining=12,
                needed_requests=3,
                avoid_reupload=False,
            )
            is False
        )
