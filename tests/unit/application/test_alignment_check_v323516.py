"""[v3.23.516] Kiểm CĂN DÒNG ngay tại lô để thử lại khi model trả kết quả lệch.

Các lớp kiểm trước chỉ xét CẤU TRÚC (đủ dòng, line_no hợp lệ) nên lệch KHỐI lọt hết: mọi dòng vẫn
đủ, vẫn dài bình thường, chỉ có điều nội dung nằm sai chỗ. Dữ liệu thật có vùng dòng 257-262 lệch
nguyên khối 3 dòng mà không cờ nào báo và cũng không có cơ hội thử lại.
"""
from __future__ import annotations

from subtitles_extractor.application.services.alignment_check import (
    build_anchor_pairs,
    find_misaligned_lines,
    is_batch_misaligned,
)

CHARACTERS = "林逸 (Lâm Dật) [nam] — chính\n王梅 (Vương Mai) [nữ] — hàng xóm"
GLOSSARY = "支付宝 = Alipay\n心动值 = chỉ số rung động"


def _anchors():
    return build_anchor_pairs(CHARACTERS, GLOSSARY)


class TestAnchorBuilding:
    def test_builds_from_both_tables(self) -> None:
        anchors = dict(_anchors())
        assert anchors["林逸"] == "lâm dật"
        assert anchors["支付宝"] == "alipay"

    def test_ignores_short_targets(self) -> None:
        # Bản dịch quá ngắn (<3 ký tự) dễ khớp bừa → loại.
        assert build_anchor_pairs("", "啊 = à") == []

    def test_empty_tables(self) -> None:
        assert build_anchor_pairs("", "") == []


class TestDetection:
    def test_block_shift_detected(self) -> None:
        sources = {1: "王梅人好", 2: "天气不错", 3: "支付宝到账"}
        translations = {
            1: "Alipay tiền vào tài khoản.",
            2: "Trời đẹp thật.",
            3: "Vương Mai tốt bụng.",
        }
        findings = find_misaligned_lines(sources, translations, _anchors())
        flagged = {item["index"] for item in findings}
        assert flagged == {1, 3}

    def test_correct_batch_not_flagged(self) -> None:
        sources = {1: "林逸来了", 2: "支付宝到账"}
        translations = {1: "Lâm Dật đến rồi.", 2: "Alipay đã nhận tiền."}
        assert find_misaligned_lines(sources, translations, _anchors()) == []

    def test_lines_without_anchor_ignored(self) -> None:
        sources = {1: "天气不错", 2: "是啊"}
        translations = {1: "Ừ đúng thế.", 2: "Trời đẹp thật."}
        assert find_misaligned_lines(sources, translations, _anchors()) == []


class TestRetryThreshold:
    def test_single_flagged_line_does_not_trigger_retry(self) -> None:
        """Một dòng đơn lẻ thường là ĐẢO TRẬT TỰ hợp lệ khi câu bị tách đôi → không bắt dịch lại."""
        # [v3.23.522] Dùng cặp câu ĐỘC LẬP (có dấu kết câu) để đúng một dòng bị cờ — cặp câu
        # tách đôi nay được miễn trừ hẳn nên không dùng làm ví dụ cho ngưỡng được nữa.
        sources = {1: "林逸来了。", 2: "天气不错。"}
        translations = {1: "Trời đẹp thật.", 2: "Lâm Dật đến rồi."}
        findings = find_misaligned_lines(sources, translations, _anchors())
        assert len(findings) == 1
        assert is_batch_misaligned(sources, translations, _anchors()) == []

    def test_two_flagged_lines_trigger_retry(self) -> None:
        sources = {1: "王梅人好", 2: "天气不错", 3: "支付宝到账"}
        translations = {
            1: "Alipay tiền vào tài khoản.",
            2: "Trời đẹp thật.",
            3: "Vương Mai tốt bụng.",
        }
        assert len(is_batch_misaligned(sources, translations, _anchors())) == 2

    def test_no_anchors_never_triggers_retry(self) -> None:
        sources = {1: "林逸来了", 2: "他说话"}
        translations = {1: "Anh ấy nói.", 2: "Lâm Dật đến rồi."}
        assert is_batch_misaligned(sources, translations, []) == []


class TestAdapterHook:
    def test_adapter_raises_for_misaligned_batch(self) -> None:
        from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )

        translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        translator._active_characters = CHARACTERS
        translator._active_glossary = GLOSSARY

        def line(index: int, text: str) -> TranslationLine:
            return TranslationLine(index=index, start_ms=0, end_ms=100, text=text)

        batch = [line(1, "王梅人好"), line(2, "天气不错"), line(3, "支付宝到账")]
        result = [
            line(1, "Alipay tiền vào tài khoản."),
            line(2, "Trời đẹp thật."),
            line(3, "Vương Mai tốt bụng."),
        ]
        assert len(translator._detect_batch_misalignment(batch, result)) == 2

    def test_adapter_safe_without_tables(self) -> None:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )

        translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        assert translator._detect_batch_misalignment([], []) == []


class TestLengthCorrelationSignal:
    """[v3.23.517] Tín hiệu thứ hai: TƯƠNG QUAN ĐỘ DÀI — phủ MỌI dòng, kể cả lô không có tên riêng.

    Mốc neo tên/thuật ngữ chỉ phủ ~13% số dòng nên vùng lệch không chứa tên riêng sẽ lọt. Độ dài
    câu thì dòng nào cũng có: căn ĐÚNG dòng thì hai dãy độ dài tương quan mạnh, còn cả khối bị đẩy
    đi k dòng thì tương quan tại 0 tụt và đỉnh chuyển sang k.
    """

    @staticmethod
    def _shifted_case(shift: int) -> tuple[dict[int, str], dict[int, str]]:
        """Dựng lô có độ dài biến thiên rõ rệt rồi đẩy bản dịch đi ``shift`` dòng."""
        lengths = [4, 30, 7, 25, 12, 40, 5, 33, 9, 28, 15, 45]
        sources = {index: "x" * length for index, length in enumerate(lengths, start=1)}
        translations = {
            index - shift: "y" * length for index, length in enumerate(lengths, start=1)
        }
        return sources, translations

    def test_detects_shift_of_one(self) -> None:
        from subtitles_extractor.application.services.alignment_check import (
            detect_length_shift,
        )

        sources, translations = self._shifted_case(1)
        # Bản dịch được đặt LÙI 1 dòng → bộ dò báo khoảng lệch -1.
        assert detect_length_shift(sources, translations) == -1

    def test_detects_shift_of_three(self) -> None:
        from subtitles_extractor.application.services.alignment_check import (
            detect_length_shift,
        )

        sources, translations = self._shifted_case(3)
        assert detect_length_shift(sources, translations) == -3

    def test_aligned_batch_not_flagged(self) -> None:
        from subtitles_extractor.application.services.alignment_check import (
            detect_length_shift,
        )

        lengths = [4, 30, 7, 25, 12, 40, 5, 33, 9, 28, 15, 45]
        sources = {i: "x" * n for i, n in enumerate(lengths, start=1)}
        translations = {i: "y" * n for i, n in enumerate(lengths, start=1)}
        assert detect_length_shift(sources, translations) is None

    def test_too_few_lines_never_flagged(self) -> None:
        """Lô quá ngắn thì hệ số tương quan quá nhiễu — không kết luận."""
        from subtitles_extractor.application.services.alignment_check import (
            detect_length_shift,
        )

        sources = {1: "x" * 5, 2: "x" * 30, 3: "x" * 8}
        translations = {0: "y" * 5, 1: "y" * 30, 2: "y" * 8}
        assert detect_length_shift(sources, translations) is None

    def test_uniform_lengths_never_flagged(self) -> None:
        """Mọi dòng dài như nhau → không có biến thiên để so, phải im lặng."""
        from subtitles_extractor.application.services.alignment_check import (
            detect_length_shift,
        )

        sources = {i: "x" * 20 for i in range(1, 15)}
        translations = {i: "y" * 20 for i in range(1, 15)}
        assert detect_length_shift(sources, translations) is None

    def test_adapter_uses_length_signal_without_anchors(self) -> None:
        """Lô KHÔNG có tên riêng/thuật ngữ nào vẫn phải bắt được lệch."""
        from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )

        translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        translator._active_characters = ""
        translator._active_glossary = ""
        lengths = [4, 30, 7, 25, 12, 40, 5, 33, 9, 28, 15, 45]
        batch = [
            TranslationLine(index=i, start_ms=0, end_ms=100, text="x" * n)
            for i, n in enumerate(lengths, start=1)
        ]
        result = [
            TranslationLine(index=i - 1, start_ms=0, end_ms=100, text="y" * n)
            for i, n in enumerate(lengths, start=1)
        ]
        assert translator._detect_batch_misalignment(batch, result)


class TestErrorStaysInsideRetryLoop:
    """[v3.23.518] Phép kiểm căn dòng phải nằm TRONG bộ kiểm truyền cho ``_call_gemini``.

    v3.23.516 đặt phép kiểm ở NGOÀI, sau khi ``_call_gemini`` đã trả về — nơi không có gì bắt
    ``_BatchValidationError``. Kết quả: lỗi thoát thẳng ra và làm SẬP cả tác vụ dịch
    ("Lỗi không lường trước khi dịch phụ đề"). Test này khoá lại vị trí đúng.
    """

    @staticmethod
    def _source() -> str:
        import inspect

        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )

        return inspect.getsource(GeminiSubtitleTranslator._translate_single_batch)

    def test_check_runs_inside_validator(self) -> None:
        source = self._source()
        assert source.index("def _validator") < source.index(
            "_detect_batch_misalignment_payload"
        ) < source.index("data = self._call_gemini")

    def test_no_check_after_model_call(self) -> None:
        assert "self._detect_batch_misalignment(batch, batch_result)" not in self._source()


class TestPayloadDetector:
    """Bộ dò chạy trên payload THÔ (trước khi ghép) — nơi vòng thử-lại có thể bắt lỗi."""

    @staticmethod
    def _translator():
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            GeminiSubtitleTranslator,
        )

        translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        translator._active_characters = CHARACTERS
        translator._active_glossary = GLOSSARY
        return translator

    @staticmethod
    def _batch():
        from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine

        return [
            TranslationLine(index=1, start_ms=0, end_ms=100, text="王梅人好"),
            TranslationLine(index=2, start_ms=0, end_ms=100, text="天气不错"),
            TranslationLine(index=3, start_ms=0, end_ms=100, text="支付宝到账"),
        ]

    def test_detects_shift_in_payload(self) -> None:
        payload = {"subtitles": [
            {"line_no": 1, "text": "Alipay tiền vào tài khoản."},
            {"line_no": 2, "text": "Trời đẹp thật."},
            {"line_no": 3, "text": "Vương Mai tốt bụng."},
        ]}
        assert len(self._translator()._detect_batch_misalignment_payload(self._batch(), payload)) == 2

    def test_correct_payload_not_flagged(self) -> None:
        payload = {"subtitles": [
            {"line_no": 1, "text": "Vương Mai tốt bụng."},
            {"line_no": 2, "text": "Trời đẹp thật."},
            {"line_no": 3, "text": "Alipay tiền vào tài khoản."},
        ]}
        assert self._translator()._detect_batch_misalignment_payload(self._batch(), payload) == []

    def test_renumbered_payload_uses_position(self) -> None:
        """Model đánh số lại từ 1 → vẫn dò được nhờ dùng VỊ TRÍ như bước ghép."""
        payload = {"subtitles": [
            {"line_no": 101, "text": "Alipay tiền vào tài khoản."},
            {"line_no": 102, "text": "Trời đẹp thật."},
            {"line_no": 103, "text": "Vương Mai tốt bụng."},
        ]}
        assert self._translator()._detect_batch_misalignment_payload(self._batch(), payload)

    def test_malformed_payload_is_safe(self) -> None:
        translator = self._translator()
        assert translator._detect_batch_misalignment_payload(self._batch(), {}) == []
        assert translator._detect_batch_misalignment_payload(
            self._batch(), {"subtitles": ["không phải dict"]}
        ) == []


class TestSplitSentenceExemption:
    """[v3.23.522] Câu bị TÁCH ĐÔI, lệch đúng 1 dòng → đảo trật tự hợp lệ, không phải lệch.

    Tiếng Việt đảo trật tự nên từ khoá có thể rơi sang dòng kế mà bản dịch vẫn ĐÚNG. Ví dụ thật:
    ``我早就跟贺强`` / ``睡过了`` dịch thành "tôi sớm đã ngủ với" / "Hạ Cường rồi." — tên nhân vật
    chuyển dòng là tự nhiên. Đo trên dữ liệu thật: 32/40 dòng bị cờ (80%) thuộc dạng này.
    """

    def _is_split(self, sources, first, second) -> bool:
        from subtitles_extractor.application.services.alignment_check import is_split_sentence

        return is_split_sentence(sources, first, second)

    def test_line_without_terminal_mark_is_split(self) -> None:
        sources = {1: "我早就跟贺强", 2: "睡过了"}
        assert self._is_split(sources, 2, 1) is True

    def test_line_with_terminal_mark_is_not_split(self) -> None:
        sources = {1: "我早就跟贺强睡过了。", 2: "你听清楚没有"}
        assert self._is_split(sources, 2, 1) is False

    def test_distance_two_is_never_split(self) -> None:
        sources = {1: "我早就跟贺强", 2: "睡过了", 3: "另一句"}
        assert self._is_split(sources, 3, 1) is False

    def test_split_pair_not_reported_as_misaligned(self) -> None:
        sources = {1: "林逸来了", 2: "他很高兴"}
        # Tên rơi sang dòng sau do đảo trật tự — bản dịch vẫn đúng.
        translations = {1: "Anh ấy rất vui,", 2: "Lâm Dật đến rồi."}
        findings = find_misaligned_lines(sources, translations, _anchors())
        assert findings == []

    def test_real_shift_still_reported(self) -> None:
        """Lệch từ 2 dòng trở lên vẫn phải báo — không nới lỏng quá tay."""
        sources = {1: "林逸来了", 2: "天气不错", 3: "他说话"}
        translations = {1: "Anh ấy nói.", 2: "Trời đẹp.", 3: "Lâm Dật đến rồi."}
        findings = find_misaligned_lines(sources, translations, _anchors())
        assert any(item["index"] == 3 for item in findings)

    def test_terminal_mark_pair_still_reported(self) -> None:
        """Dòng trước ĐÃ kết thúc câu → hai dòng độc lập, lệch là lệch thật."""
        sources = {1: "林逸来了。", 2: "他很高兴。"}
        translations = {1: "Anh ấy rất vui.", 2: "Lâm Dật đến rồi."}
        findings = find_misaligned_lines(sources, translations, _anchors())
        assert findings != []


class TestFrequentAnchorFiltering:
    """[v3.23.523] Mốc neo xuất hiện quá dày là bằng chứng TỒI cho việc căn dòng.

    Tên nhân vật chính có mặt ở hàng trăm dòng (dữ liệu thật: "lâm dật" 253/5540 = 4,6%), nên
    "dòng bên cạnh cũng chứa tên đó" chẳng chứng minh được gì. Ví dụ báo nhầm: dòng 2578 gốc
    ``我如果告诉了小逸`` dịch "Nếu chị mà nói cho Lâm Dật biết" — hoàn toàn ĐÚNG, nhưng bị cờ vì biệt
    danh ``小逸`` không có trong roster.
    """

    def _drop(self, anchors, translations):
        from subtitles_extractor.application.services.alignment_check import (
            drop_frequent_anchors,
        )

        return drop_frequent_anchors(anchors, translations)

    def test_common_anchor_dropped(self) -> None:
        anchors = [("林逸", "lâm dật"), ("支付宝", "alipay")]
        translations = {i: "Lâm Dật nói chuyện." for i in range(200)}
        translations[200] = "Alipay nhận tiền."
        kept = dict(self._drop(anchors, translations))
        assert "林逸" not in kept
        assert "支付宝" in kept

    def test_rare_anchor_kept(self) -> None:
        anchors = [("支付宝", "alipay")]
        translations = {i: "Câu bình thường." for i in range(200)}
        translations[0] = "Alipay nhận tiền."
        assert self._drop(anchors, translations) == anchors

    def test_empty_translations_keeps_all(self) -> None:
        anchors = [("林逸", "lâm dật")]
        assert self._drop(anchors, {}) == anchors

    def test_diagnostics_delegates_to_service(self) -> None:
        """Cờ chuẩn đoán phải DÙNG CHUNG service, không giữ bản sao logic riêng."""
        import inspect

        from subtitles_extractor.application.services import translation_diagnostics

        source = inspect.getsource(translation_diagnostics._shifted_lines)
        assert "find_misaligned_lines" in source
        assert "MAX_SHIFT_DISTANCE" not in source  # không còn tự dò khoảng lệch
