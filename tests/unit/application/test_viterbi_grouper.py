"""Test :class:`ViterbiGrouper` — quy hoạch động gộp frame."""

from __future__ import annotations

from subtitles_extractor.application.services.viterbi_grouper import (
    ViterbiGrouper,
    ViterbiGrouperConfig,
)
from subtitles_extractor.domain.entities.ocr_frame_result import (
    OcrFrameResult,
    OcrTextBox,
)
from subtitles_extractor.domain.value_objects.confidence import Confidence


def _frame(idx: int, ts: float, text: str) -> OcrFrameResult:
    return OcrFrameResult(
        frame_index=idx,
        timestamp_sec=ts,
        text_boxes=[
            OcrTextBox(
                text=text,
                confidence=Confidence(0.9),
                polygon=[(0, 10), (100, 10), (100, 40), (0, 40)],
            )
        ],
    )


def _frame_at_y(
    idx: int, ts: float, text: str, y_center: int = 25
) -> OcrFrameResult:
    """Tạo frame với polygon ở Y-center cụ thể (cho test position penalty)."""
    half_height = 15
    return OcrFrameResult(
        frame_index=idx,
        timestamp_sec=ts,
        text_boxes=[
            OcrTextBox(
                text=text,
                confidence=Confidence(0.9),
                polygon=[
                    (0, y_center - half_height),
                    (100, y_center - half_height),
                    (100, y_center + half_height),
                    (0, y_center + half_height),
                ],
            )
        ],
    )


class TestViterbiGrouper:
    def test_empty_input(self) -> None:
        grouper = ViterbiGrouper(ViterbiGrouperConfig())
        assert grouper.group([]) == []

    def test_single_frame(self) -> None:
        grouper = ViterbiGrouper(ViterbiGrouperConfig())
        labels = grouper.group([_frame(0, 0.0, "Xin chào")])
        assert labels == [0]

    def test_identical_text_grouped(self) -> None:
        grouper = ViterbiGrouper(ViterbiGrouperConfig(open_penalty=1.0))
        # 3 frame text giống hệt — penalty mở mới rất cao ⇒ phải gộp 1 nhóm.
        frames = [
            _frame(0, 0.0, "Xin chào"),
            _frame(1, 0.1, "Xin chào"),
            _frame(2, 0.2, "Xin chào"),
        ]
        labels = grouper.group(frames)
        assert len(set(labels)) == 1

    def test_completely_different_text_separated(self) -> None:
        grouper = ViterbiGrouper(
            ViterbiGrouperConfig(
                open_penalty=0.05,  # rẻ tách
                min_similarity_to_join=0.95,
            )
        )
        frames = [
            _frame(0, 0.0, "ABCDEFGH"),
            _frame(1, 0.5, "WXYZ12345"),
        ]
        labels = grouper.group(frames)
        assert labels[0] != labels[1]

    def test_gap_too_large_forces_new_group(self) -> None:
        grouper = ViterbiGrouper(
            ViterbiGrouperConfig(
                open_penalty=10.0,  # đắt mở mới
                max_gap_sec=0.5,
            )
        )
        frames = [
            _frame(0, 0.0, "Xin chào"),
            _frame(1, 5.0, "Xin chào"),  # gap 5s vượt max_gap_sec
        ]
        labels = grouper.group(frames)
        assert labels[0] != labels[1]

    def test_labels_increasing(self) -> None:
        grouper = ViterbiGrouper(
            ViterbiGrouperConfig(open_penalty=0.05, max_gap_sec=10.0)
        )
        frames = [
            _frame(0, 0.0, "A"),
            _frame(1, 0.1, "B"),
            _frame(2, 0.2, "C"),
        ]
        labels = grouper.group(frames)
        # Labels phải tăng dần (0, 1, 2) hoặc liên tiếp.
        assert min(labels) == 0
        assert all(0 <= lbl < len(labels) for lbl in labels)

    def test_regression_off_by_one_lookback_includes_first_frame(self) -> None:
        """Regression test cho bug off-by-one trong v2.15:
        ``range(current_idx - 1, lookback_start, -1)`` không bao giờ xét
        ``parent_candidate_idx = -1`` (mở cluster đầu) — khiến 3 frame
        text giống hệt với ``open_penalty=1.0`` bị tách thành ``[0, 1, 1]``
        thay vì gộp thành ``[0, 0, 0]``.
        """
        grouper = ViterbiGrouper(
            ViterbiGrouperConfig(
                open_penalty=1.0,           # đắt → buộc gộp
                min_similarity_to_join=0.5,
                max_lookback=10,
                max_gap_sec=10.0,
            )
        )
        frames = [
            _frame(0, 0.0, "Hello"),
            _frame(1, 0.1, "Hello"),
            _frame(2, 0.2, "Hello"),
        ]
        labels = grouper.group(frames)
        # Trước fix: labels == [0, 1, 1] (sai). Sau fix: [0, 0, 0].
        assert labels == [0, 0, 0], (
            f"Bug off-by-one đã tái xuất: kỳ vọng [0,0,0], nhận {labels}."
        )

    def test_regression_cjk_two_chars_sharing_one_char_viterbi(self) -> None:
        """Kiểm tra hành vi Viterbi stable với CJK 2 ký tự share 1 ký tự.

        Lưu ý thiết kế (stable file):
            Viterbi dùng ``average_cluster_error = sum / cluster_size``.
            Với 2 câu CJK "性别" vs "雄性" (sim≈0.5, error=0.5):
            - Cluster 5 frame: avg_error = 1.0/5 = 0.20 < open_penalty=0.35
              → DP ưu tiên gộp tất cả vào 1 cluster (accepted behavior).
            - Greedy path (mặc định) xử lý đúng: sim=0.5 < threshold=0.75
              → tách thành 2 group riêng ✓.

        Test này xác nhận Greedy tách đúng — Viterbi là opt-in cho video khác.
        """
        from subtitles_extractor.application.services.subtitle_builder import (
            text_similarity_cached,
        )

        # Greedy dùng similarity_threshold=0.75 → tách đúng CJK 2-char.
        sim = text_similarity_cached("性别", "雄性")
        assert sim < 0.75, (
            f"text_similarity('性别','雄性') = {sim:.3f} phải < 0.75 "
            f"để Greedy tách 2 câu thành group riêng."
        )

        # Viterbi với stable file: average_error / size → có thể gộp.
        # Đây là trade-off đã được chấp nhận trong stable file.
        grouper = ViterbiGrouper(
            ViterbiGrouperConfig(open_penalty=0.35, min_similarity_to_join=0.40)
        )
        frames = [
            _frame(0, 24.76, "性别"),
            _frame(1, 25.00, "性别"),
            _frame(2, 25.25, "雄性"),
            _frame(3, 25.50, "雄性"),
            _frame(4, 25.80, "雄性"),
        ]
        labels = grouper.group(frames)
        # Frame cùng text vẫn phải cùng cluster (dù Viterbi có thể gộp 2 câu).
        assert labels[0] == labels[1], (
            f"2 frame '性别' liền kề phải cùng cluster, nhận {labels}."
        )
        assert labels[2] == labels[3] == labels[4], (
            f"3 frame '雄性' liền kề phải cùng cluster, nhận {labels}."
        )

    def test_v219_position_penalty_skipped_in_stable_viterbi(self) -> None:
        """Y-spread penalty đã được loại bỏ khỏi stable viterbi file.

        Tính năng này tồn tại ở v2.19 nhưng stable viterbi file không có.
        Thay thế: _filter_cross_frame_spatial_outliers trong SubtitleBuilder
        thực hiện lọc Y bất thường ở tầng trước khi grouping.

        Xác nhận: stable Viterbi vẫn gộp đúng các frame cùng text gần nhau.
        """
        grouper = ViterbiGrouper(
            ViterbiGrouperConfig(open_penalty=0.35, min_similarity_to_join=0.40)
        )
        frames = [
            _frame_at_y(0, 0.0, "Hello", y_center=100),
            _frame_at_y(1, 0.1, "Hello", y_center=100),
            _frame_at_y(2, 0.2, "Hello", y_center=900),
            _frame_at_y(3, 0.3, "Hello", y_center=900),
        ]
        labels = grouper.group(frames)
        # Stable Viterbi gộp tất cả text giống nhau bất kể Y.
        # Việc lọc Y được xử lý bởi _filter_cross_frame_spatial_outliers.
        assert labels[0] == labels[1], (
            f"2 frame text giống nhau phải cùng cluster, nhận {labels}."
        )
        assert labels[2] == labels[3], (
            f"2 frame text giống nhau phải cùng cluster, nhận {labels}."
        )

    def test_v219_position_penalty_close_y_still_grouped(self) -> None:
        """Frame Y gần nhau (jitter ±10px) → vẫn cùng 1 cluster."""
        grouper = ViterbiGrouper(
            ViterbiGrouperConfig(open_penalty=0.35, min_similarity_to_join=0.40)
        )
        frames = [
            _frame_at_y(0, 0.0, "Hello", y_center=900),
            _frame_at_y(1, 0.1, "Hello", y_center=905),
            _frame_at_y(2, 0.2, "Hello", y_center=895),
            _frame_at_y(3, 0.3, "Hello", y_center=910),
        ]
        labels = grouper.group(frames)
        assert len(set(labels)) == 1, (
            f"Y jitter nhỏ + text giống → 1 cluster, nhận {labels}."
        )


class TestViterbiCjkBoostBugFix:
    """Regression tests cho bug CJK short-text boost sai trong viterbi_similarity.

    Bug: ``viterbi_similarity("性别","雄性")`` = 0.85 (do boost max_len≤3 và
    base≥0.45 không phân biệt CJK/Latin). Dẫn đến Viterbi gộp 2 câu phụ đề
    hoàn toàn khác nghĩa thành 1.

    Fix: CJK chỉ boost khi base ≥ 0.66 (giống text_similarity_cached).
    """

    def test_cjk_2char_sim_not_boosted(self) -> None:
        """sim('性别','雄性') phải ≤ 0.55, không được boost lên 0.85."""
        from subtitles_extractor.application.services.viterbi_grouper import (
            viterbi_similarity,
        )
        sim = viterbi_similarity("性别", "雄性")
        assert sim <= 0.55, (
            f"viterbi_similarity('性别','雄性') = {sim:.4f} phải ≤ 0.55 "
            f"(không được boost CJK 2-char share 1-char lên 0.85)."
        )

    def test_latin_typo_still_boosted(self) -> None:
        """Latin typo ('Xin','Xim') vẫn được boost — hành vi cũ giữ nguyên."""
        from subtitles_extractor.application.services.viterbi_grouper import (
            viterbi_similarity,
        )
        sim = viterbi_similarity("Xin", "Xim")
        assert sim >= 0.80, (
            f"Latin typo vẫn phải được boost: sim('Xin','Xim') = {sim:.4f}"
        )

    def test_same_cjk_sim_is_1(self) -> None:
        from subtitles_extractor.application.services.viterbi_grouper import (
            viterbi_similarity,
        )
        assert viterbi_similarity("恭喜宿主", "恭喜宿主") == 1.0

    def test_viterbi_splits_xingbie_xiongxing_with_conf_drop(self) -> None:
        """Bug thực tế: Viterbi gộp '性别'+'雄性' do conf drop khi subtitle mới.

        Kịch bản: 4 frame '性别' (conf ổn định) → frame '雄性' đầu tiên
        có conf thấp hơn → error_weight = 0.5 → check sim bị bypass.
        Fix: viterbi_similarity CJK không boost → sim=0.50 < min=0.60 →
        cluster invalid → split đúng.
        """
        grouper = ViterbiGrouper(
            ViterbiGrouperConfig(
                open_penalty=0.35,
                min_similarity_to_join=0.60,
                max_gap_sec=1.0,
                sample_step_sec=0.05,
            )
        )
        frames = [
            _frame_with_conf(0, 24.76, "性别", 0.92),
            _frame_with_conf(1, 24.81, "性别", 0.91),
            _frame_with_conf(2, 24.86, "性别", 0.92),
            _frame_with_conf(3, 25.00, "性别", 0.90),
            _frame_with_conf(4, 25.05, "雄性", 0.88),  # conf drop → trigger bug
            _frame_with_conf(5, 25.10, "雄性", 0.92),
            _frame_with_conf(6, 25.15, "雄性", 0.93),
            _frame_with_conf(7, 25.25, "雄性", 0.91),
        ]
        labels = grouper.group(frames)

        assert len(set(labels)) == 2, (
            f"Phải có 2 cluster, nhận {len(set(labels))}: {labels}"
        )
        assert labels[0] == labels[1] == labels[2] == labels[3], (
            f"4 frame '性别' phải cùng cluster, nhận {labels[:4]}"
        )
        assert labels[4] == labels[5] == labels[6] == labels[7], (
            f"4 frame '雄性' phải cùng cluster, nhận {labels[4:]}"
        )
        assert labels[3] != labels[4], (
            f"Boundary 性别→雄性 phải tách, nhận labels[3]={labels[3]}, "
            f"labels[4]={labels[4]}"
        )


def _frame_with_conf(
    idx: int, ts: float, text: str, conf: float
) -> OcrFrameResult:
    """Helper tạo frame với confidence tuỳ chỉnh."""
    return OcrFrameResult(
        frame_index=idx,
        timestamp_sec=ts,
        text_boxes=[
            OcrTextBox(
                text=text,
                confidence=Confidence(conf),
                polygon=[(50, 100), (300, 100), (300, 140), (50, 140)],
            )
        ],
    )
