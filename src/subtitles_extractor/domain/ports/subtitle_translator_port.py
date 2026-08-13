"""Port (giao diện) cho dịch vụ dịch phụ đề bằng AI.

Tách biệt logic nghiệp vụ dịch (application) khỏi nhà cung cấp cụ thể
(Gemini, OpenAI...) theo nguyên tắc Dependency Inversion. Tầng application
chỉ phụ thuộc vào ``SubtitleTranslatorPort``; adapter cụ thể được tiêm vào
qua ``ApplicationContainer``.

Lưu ý: Port này KHÁC với ``TranslatorPort`` (i18n cho chuỗi giao diện).
``SubtitleTranslatorPort`` dịch *nội dung phụ đề*, còn ``TranslatorPort``
dịch *nhãn giao diện*.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class TranslationStageKind(Enum):
    """Bốn giai đoạn dịch tuần tự, mô phỏng quy trình biên dịch chuyên nghiệp."""

    PREPROCESS = "preprocess"
    """Tiền xử lý: sửa lỗi chính tả/OCR của bản gốc, GIỮ nguyên ngôn ngữ gốc."""

    LITERAL = "literal"
    """Dịch thô sát nghĩa sang ngôn ngữ đích, gán nhãn người nói nếu nhận diện được."""

    STYLE = "style"
    """Tinh chỉnh văn phong theo thể loại/phong cách phim."""

    LOCALIZE = "localize"
    """Bản địa hoá: đơn vị đo, xưng hô, thành ngữ, quy ước văn hoá đích."""


@dataclass(frozen=True)
class TranslationStageConfig:
    """Cấu hình bất biến cho một giai đoạn dịch.

    Attributes:
        kind:          Loại giai đoạn.
        model_name:    Tên model AI dùng cho giai đoạn này.
        temperature:   Độ sáng tạo ``[0.0, 1.0]``.
        batch_size:    Số dòng mỗi lô gửi lên AI (> 0).
        context_size:  Số dòng ngữ cảnh trước/sau gửi kèm để giữ mạch (≥ 0).
        style_name:    Tên thể loại/phong cách (chỉ dùng cho ``STYLE``).
        locale_notes:  Ghi chú bản địa hoá (chỉ dùng cho ``LOCALIZE``).
        allow_retime:  Cho phép hiệu chuẩn mốc thời gian (chỉ dùng cho ``PREPROCESS``).
    """

    kind: TranslationStageKind
    model_name: str
    temperature: float = 0.1
    batch_size: int = 50
    context_size: int = 10
    style_name: str = "Trung tính"
    locale_notes: str = ""
    allow_retime: bool = False
    # ── Thinking (Gemini 3.x / 2.5.x) ─────────────────────────────────────
    enable_thinking: bool = False
    """Bật chế độ suy nghĩ nội tâm (Thinking). Cải thiện chất lượng dịch phức
    tạp (STYLE/LOCALIZE) nhưng tốn thêm thời gian và token."""
    thinking_budget: int = -1
    """Token budget cho thinking: -1 = Dynamic (model tự quyết), 0 = Tắt,
    1–32768 = giới hạn cụ thể. CHỈ áp dụng cho Gemini 2.5.x khi enable_thinking=True."""
    thinking_level: str = "low"
    """[v3.23.52] Mức suy nghĩ cho Gemini 3.x: 'low' (dịch — nhanh/rẻ, khuyến nghị),
    'medium', hoặc 'high'. Với Gemini 3.x, dùng trường này thay cho thinking_budget
    (không được dùng cả hai). Dịch thuật nên để 'low' — suy nghĩ nhiều dễ phản tác dụng."""


@dataclass(frozen=True)
class TranslationContext:
    """Ngữ cảnh toàn cục dùng chung cho mọi giai đoạn dịch.

    Attributes:
        target_lang: Ngôn ngữ đích (tên đầy đủ, vd ``"Vietnamese"``).
        source_lang: Ngôn ngữ gốc (mã ISO hoặc tên), có thể rỗng nếu chưa rõ.
        overview:    Tóm tắt cốt truyện để AI hiểu bối cảnh, có thể rỗng.
        characters:  Danh sách nhân vật (roster) ngăn cách bởi dấu phẩy, có thể rỗng.
        enable_tags: Cho phép gán nhãn người nói trong kết quả.
        include_desc: Cho phép giữ mô tả tiếng động.
    """

    target_lang: str
    source_lang: str = ""
    overview: str = ""
    characters: str = ""
    glossary: str = ""
    visual_cues: str = ""  # [v3.23.35] JSON rút gọn từ phân tích, dùng lại khi dịch
    enable_tags: bool = False
    include_desc: bool = True


@dataclass
class TranslationLine:
    """Một dòng phụ đề ở dạng trung gian cho pipeline dịch.

    Dùng ``index`` 1-based để khớp ``line_no`` mà AI phải giữ nguyên. Mốc thời
    gian lưu bằng mili-giây để khớp định dạng nội bộ của thuật toán tham khảo.

    Attributes:
        index:       Số thứ tự 1-based, BẮT BUỘC giữ nguyên qua mỗi giai đoạn.
        start_ms:    Mốc bắt đầu (mili-giây).
        end_ms:      Mốc kết thúc (mili-giây).
        text:        Nội dung dòng.
        speaker:     Tên người nói (có thể rỗng).
        description: Mô tả tiếng động (có thể rỗng).
    """

    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str = ""
    description: str = ""
    # [Anti-Chinese Whispers] Văn bản GỐC bất biến, giữ xuyên suốt mọi giai đoạn
    # để khâu Tinh chỉnh/Bản địa hoá đối chiếu, tránh "tam sao thất bản".
    original_text: str = ""
    # [Visual Cues] "Nói với ai" do AI phân tích video ghi chú (vd "Nam đệ tử"),
    # truyền xuống khâu dịch để xưng hô đúng vai vế/giới tính.
    addressee: str = ""
    # [Visual Cues] Bối cảnh + thái độ/cảm xúc người nói (vd "giận dữ trong đại điện")
    # do AI nghe-nhìn video ghi chú; truyền xuống khâu dịch để chọn đúng giọng văn.
    scene: str = ""


# Callback báo tiến độ trong khoảng [0.0, 1.0] cho MỘT giai đoạn.
StageProgressCallback = Callable[[float], None]

# Callback kiểm tra yêu cầu huỷ; trả về True để dừng sớm.
CancellationCallback = Callable[[], bool]


@dataclass(frozen=True)
class SubtitleContextAnalysis:
    """Kết quả phân tích ngữ cảnh toàn cục do AI thực hiện.

    Attributes:
        source_lang: Mã ngôn ngữ gốc xác định (vd ``"zh"``, ``"ja"``).
        characters:  Danh sách nhân vật phát hiện được, ngăn cách dấu phẩy.
        overview:    Tóm tắt súc tích cốt truyện/bối cảnh phim.
        glossary:    Bảng thuật ngữ & viết tắt để dịch nhất quán (mỗi dòng 'gốc => dịch').
        visual_cues: Gợi ý hình ảnh "ai nói/nói với ai" (JSON rút gọn), rỗng nếu không bật.
    """

    source_lang: str = ""
    characters: str = ""
    overview: str = ""
    glossary: str = ""
    visual_cues: str = ""


@dataclass(frozen=True)
class VisualCue:
    """Gợi ý hình ảnh do "Vision Director" (AI xem video) tạo cho MỘT dòng phụ đề.

    Trả lời ba câu hỏi cho mỗi dòng: Ai đang nói? Nói với ai? Bối cảnh/thái độ?
    Nhờ vậy khâu dịch xưng hô đúng vai vế/giới tính mà không cần "xem" lại video.

    Attributes:
        line_no:   Số thứ tự dòng (khớp ``TranslationLine.index``).
        speaker:   Người đang nói (vd "Lâm Côn", "Nam đệ tử").
        addressee: Người được nói tới (vd "Nữ tỳ", "Sư phụ").
        scene:     Bối cảnh + thái độ/cảm xúc ngắn gọn (vd "tức giận, trong đại điện").
    """

    line_no: int
    speaker: str = ""
    addressee: str = ""
    scene: str = ""


def apply_visual_cues_to_lines(
    lines: list[TranslationLine], cues: list[VisualCue]
) -> list[TranslationLine]:
    """[Silent Context Injection] Bơm gợi ý hình ảnh vào dòng phụ đề.

    Ghép ``speaker``/``addressee`` từ :class:`VisualCue` vào :class:`TranslationLine`
    tương ứng theo ``line_no``. Nhờ vậy ở khâu dịch, AI "thấy" bối cảnh qua chữ viết
    thay vì phải nạp lại video tốn token. Trả danh sách MỚI (thuần, không đột biến).

    Args:
        lines: Các dòng phụ đề cần làm giàu ngữ cảnh.
        cues:  Gợi ý hình ảnh do AI phân tích video.

    Returns:
        Danh sách dòng mới đã gắn speaker/addressee (cue thắng khi có giá trị).
    """
    from dataclasses import replace

    cue_by_line: dict[int, VisualCue] = {cue.line_no: cue for cue in cues}
    enriched: list[TranslationLine] = []
    for line in lines:
        cue = cue_by_line.get(line.index)
        if cue is None:
            enriched.append(line)
            continue
        enriched.append(
            replace(
                line,
                speaker=cue.speaker or line.speaker,
                addressee=cue.addressee or line.addressee,
                scene=cue.scene or line.scene,
            )
        )
    return enriched


class SubtitleTranslationError(Exception):
    """Lỗi nghiệp vụ khi dịch phụ đề (cấu hình sai, API lỗi không thể retry...)."""


class TranslationCancelledError(SubtitleTranslationError):
    """Người dùng chủ động huỷ tiến trình dịch.

    Kế thừa ``SubtitleTranslationError`` để tương thích ngược (code cũ bắt
    ``SubtitleTranslationError`` vẫn hoạt động), nhưng tách riêng để tầng trình
    bày phân biệt HUỶ (kết thúc êm, giữ phần đã dịch) với LỖI THẬT (hiện cảnh báo).
    """


class TranslatorUnavailableError(SubtitleTranslationError):
    """Bộ dịch không khả dụng (thiếu thư viện ``google-genai`` hoặc thiếu API key)."""


class SubtitleTranslatorPort(Protocol):
    """Giao diện cho một bộ dịch phụ đề đa giai đoạn.

    Adapter cụ thể (vd Gemini) chịu trách nhiệm: chia lô, dựng prompt theo
    giai đoạn, gọi API, retry, kiểm tra tính toàn vẹn batch và parse kết quả.
    Tầng application chỉ điều phối thứ tự giai đoạn và chuyển đổi dữ liệu.
    """

    def is_available(self) -> bool:
        """Trả về True nếu adapter sẵn sàng (đã cài thư viện, có cấu hình tối thiểu)."""
        ...

    def translate_stage(
        self,
        *,
        stage: TranslationStageConfig,
        context: TranslationContext,
        source_lines: list[TranslationLine],
        input_lines: list[TranslationLine],
        has_prior_translation: bool,
        progress_cb: StageProgressCallback | None = None,
        cancel_cb: CancellationCallback | None = None,
        video_refs: list[Any] | None = None,
        attach_video: bool = False,
    ) -> list[TranslationLine]:
        """Chạy một giai đoạn dịch trên toàn bộ danh sách dòng.

        Args:
            stage:                 Cấu hình giai đoạn.
            context:               Ngữ cảnh toàn cục.
            source_lines:          Bản gốc tham chiếu (mốc so sánh, không đổi).
            input_lines:           Đầu vào của giai đoạn này (đầu ra giai đoạn trước).
            has_prior_translation: True nếu đã có bản dịch từ giai đoạn trước.
            progress_cb:           Callback tiến độ ``[0.0, 1.0]`` cho giai đoạn này.
            cancel_cb:             Callback kiểm tra huỷ; trả True để dừng sớm.

        Returns:
            Danh sách dòng đã xử lý, cùng độ dài và cùng ``index`` với ``input_lines``.

        Raises:
            TranslatorUnavailableError: Khi adapter không khả dụng.
            SubtitleTranslationError:   Khi lỗi không thể khắc phục bằng retry.
        """
        ...


    def analyze_global_context(
        self,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str = "gemini-3.1-flash-lite",
        cancel_cb: CancellationCallback | None = None,
        video_refs: list[Any] | None = None,
        with_visual_cues: bool = False,
        prior_context: str = "",
    ) -> SubtitleContextAnalysis:
        """Phân tích **toàn bộ** phụ đề nguồn để trích xuất ngữ cảnh toàn cục.

        Gửi TOÀN BỘ nội dung phụ đề (không lấy mẫu) tới model AI đã chỉ định.
        Một lần gọi duy nhất bao phủ cả 3 mục tiêu:

        * **Ngôn ngữ nguồn** — mã ISO 639-1 của ngôn ngữ trong phụ đề.
        * **Danh sách nhân vật** — tên + vai trò/mô tả ngắn của các nhân vật chính.
        * **Tóm tắt đầy đủ** — bối cảnh, thế giới quan, cốt truyện chính, các
          tình tiết quan trọng (dùng ngôn ngữ ``target_lang``).

        Args:
            source_lines: Danh sách đầy đủ dòng nguồn (sẽ khử trùng lặp liên tiếp).
            target_lang:  Ngôn ngữ viết phần tóm tắt (vd ``"Vietnamese"``).
            model_name:   Model AI thực hiện phân tích (vd ``"gemini-3.1-flash-lite"``).
            cancel_cb:    Callback kiểm tra huỷ; trả True để dừng sớm.

        Returns:
            :class:`SubtitleContextAnalysis` với ``source_lang``, ``characters``,
            ``overview`` đã điền đầy đủ.
        """
        ...

    def analyze_visual_cues(
        self,
        source_lines: list[TranslationLine],
        target_lang: str,
        model_name: str = "gemini-3.1-flash-lite",
        video_refs: list[Any] | None = None,
        batch_size: int = 150,
        sleep_between_s: float = 4.0,
        cancel_cb: CancellationCallback | None = None,
        progress_cb: StageProgressCallback | None = None,
    ) -> list[VisualCue]:
        """[Vision Director] Phân tích video → gợi ý hình ảnh cho TỪNG dòng phụ đề.

        Với mỗi dòng, xác định: ai nói (speaker), nói với ai (addressee), bối cảnh +
        thái độ (scene). Dùng micro-batching + nghỉ giữa lô để chống rate-limit, và
        JSON minified để không chạm trần token output.

        Args:
            source_lines: Toàn bộ dòng phụ đề gốc.
            target_lang:  Ngôn ngữ ghi chú vai vế.
            model_name:   Model Gemini có thị giác.
            video_refs:   Các đoạn video ngữ cảnh đã tải lên.
            batch_size:   Số dòng mỗi lô (chống chạm trần output).
            sleep_between_s: Nghỉ giữa lô (giây) chống lỗi 429.
            cancel_cb:    Callback huỷ.
            progress_cb:  Callback tiến độ ∈ [0, 1].

        Returns:
            Danh sách :class:`VisualCue` theo thứ tự dòng.
        """
        ...


__all__ = [
    "TranslationStageKind",
    "TranslationStageConfig",
    "TranslationContext",
    "TranslationLine",
    "SubtitleContextAnalysis",
    "StageProgressCallback",
    "CancellationCallback",
    "SubtitleTranslationError",
    "TranslationCancelledError",
    "TranslatorUnavailableError",
    "SubtitleTranslatorPort",
]
