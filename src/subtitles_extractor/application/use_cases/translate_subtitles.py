"""Use case: dịch danh sách ``SubtitleEvent`` qua pipeline đa giai đoạn.

Điều phối thứ tự các giai đoạn (tiền xử lý → dịch thô → phong cách → bản địa hoá),
chuyển đổi giữa ``SubtitleEvent`` (domain) và ``TranslationLine`` (trung gian),
nối đầu ra giai đoạn trước làm đầu vào giai đoạn sau, và báo tiến độ tổng.

**Resume checkpoint (Stage-level)**:
Nếu ``checkpoint_dir`` được cung cấp, sau mỗi giai đoạn hoàn tất use case lưu
kết quả xuống ``{checkpoint_dir}/{key}.json``. Lần chạy tiếp theo cùng request
(cùng source + cấu hình) sẽ đọc checkpoint và bỏ qua giai đoạn đã xong — không
dịch lại từ đầu khi bị ngắt giữa chừng.

[v3.23.584] Thêm mức LÔ: trong lúc một giai đoạn chạy, mỗi lô dịch xong được ghi vào
``{key}.partial.json``; lần chạy sau (hết hạn mức ngày, mất mạng, 503) chỉ dịch các lô
còn thiếu của giai đoạn đó.

[v3.23.608] Bộ nhớ đệm được GIỮ sau khi dịch xong (xem ``services.stage_checkpoint``):
đổi cấu hình một giai đoạn thì các giai đoạn trước dùng lại; người dùng sửa tay một giai
đoạn thì giai đoạn sau chỉ dịch lại các lô có dòng đổi, và dòng sửa tay luôn được giữ.

Tầng này KHÔNG biết Gemini là gì — chỉ phụ thuộc ``SubtitleTranslatorPort``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from subtitles_extractor.application.services.stage_cache import (
    StageCacheScope,
    adopt_previous_checkpoints,
)
from subtitles_extractor.application.services.stage_checkpoint import (
    CheckpointIdentity,
    ReuseVerdict,
    StageCheckpoint,
    StageRun,
    context_fingerprint,
    json_to_lines,
    legacy_checkpoint_key,
    lines_to_json,
    prune_stage_checkpoints,
    stage_config_signature,
    stage_input_fingerprint,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    BatchDoneCallback,
    LineCharBudgetProvider,
    SubtitleTranslationError,
    SubtitleTranslatorPort,
    TranslationCancelledError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
    VisualCue,
    apply_visual_cues_to_lines,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

if TYPE_CHECKING:
    from subtitles_extractor.application.services.cue_completion import (
        SpeakerCompletionOutcome,
    )

logger = logging.getLogger(__name__)

#: [v3.23.558] Số dòng đệm mỗi bên khi dịch lại một khối lệch — đủ để model thấy câu
#: bị tách ở mép khối, mà không kéo thêm dòng đang đúng vào vòng dịch lại.
BLOCK_REPAIR_PADDING_LINES = 3

#: [v3.23.559] Dưới độ phủ này, Visual Cues từ phân tích ngữ cảnh bị coi là THIẾU và được
#: quét bổ sung cho các dòng còn trống. Đo thật: model phân tích mạnh phủ ~80%, model yếu
#: chỉ 27% (có đoạn video chỉ 1/208 dòng) — mà cues là nguồn duy nhất cho xưng hô.
MIN_CONTEXT_CUE_COVERAGE = 0.60

#: Trần số khối sửa mỗi giai đoạn — giới hạn chi phí nếu một giai đoạn hỏng hàng loạt.
MAX_BLOCK_REPAIRS_PER_STAGE = 3

#: [v3.23.608] Bộ nhớ đệm giai đoạn được giữ sau khi dịch xong: mặc định giữ 60 bộ phụ đề
#: gần nhất trong 180 ngày. Đo trên phim thật 1246 dòng: ~175 byte/dòng/giai đoạn → phim
#: 3902 dòng × 4 giai đoạn ≈ 3 MB, 60 phim ≈ 160 MB. Cấu hình qua
#: ``TranslationSettings.stage_cache_max_files`` / ``stage_cache_max_age_days``.
DEFAULT_CACHE_MAX_FILES = 60
DEFAULT_CACHE_MAX_AGE_DAYS = 180

#: [v3.23.569] Số dải lệch-so-với-bản-trước được phân xử mỗi giai đoạn (mỗi dải tốn một
#: request dịch trọng tài). Dữ liệu thật: 0-3 dải mỗi giai đoạn ở các bản gần đây.
MAX_SHIFT_ARBITRATIONS_PER_STAGE = 4
#: Dòng đệm mỗi bên khi dịch trọng tài — đủ ngữ cảnh để model không tự lệch.
SHIFT_REFERENCE_PADDING_LINES = 2
#: Dòng đệm mỗi bên khi lùi về bản trước — dòng biên hay bị gộp câu/lặp câu, và câu quá
#: ngắn ở mép dải không đủ làm bằng chứng (ca thật: dòng 3393). Lùi thừa một dòng đúng chỉ
#: mất chút trau chuốt vì bản trước đã được trọng tài xác nhận căn đúng.
SHIFT_REVERT_PADDING_LINES = 2

#: [v3.23.432] Nhận diện chữ CJK (Hán/Nhật kana/Hàn) để chống giai đoạn tinh chỉnh chèn
#: CJK ngược vào câu đã dịch sạch. Dùng cho safety-net _revert_cjk_regressions.
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")

# Callback tiến độ tổng: (phần_trăm_0_đến_1, mô_tả_giai_đoạn).
OverallProgressCallback = Callable[[float, str], None]
# Callback kiểm tra huỷ.
CancellationCallback = Callable[[], bool]

# [v3.23.350] Ngưỡng tỉ lệ dòng trùng-hệt-nguồn để coi cả lần dịch là "không tác dụng".
# Đặt cao (0.98) để KHÔNG báo oan: bản dịch tốt vẫn có thể giữ nguyên vài dòng là tên
# riêng/số/thán từ ("OK", "Alice", "911"). Chỉ khi gần như MỌI dòng trùng mới cảnh báo.
_PASSTHROUGH_WARN_RATIO = 0.98


@dataclass(frozen=True)
class TranslateSubtitlesRequest:
    """Đầu vào cho use case dịch.

    Attributes:
        events:  Danh sách phụ đề gốc cần dịch (đã sắp xếp theo thời gian).
        stages:  Danh sách giai đoạn theo đúng thứ tự thực thi (đã lọc bật/tắt).
        context: Ngữ cảnh toàn cục dùng chung.
    """

    events: list[SubtitleEvent]
    stages: list[TranslationStageConfig]
    context: TranslationContext
    # Ngữ cảnh video tuỳ chọn: danh sách đoạn đã tải lên + tập giai đoạn được đính
    # video. Phân tích ngữ cảnh xử lý riêng ở use case phân tích (gộp mọi đoạn);
    # ở đây chỉ áp cho các giai đoạn dịch (LITERAL/STYLE/LOCALIZE…) được chọn.
    video_refs: tuple[Any, ...] = ()
    attach_video_stages: frozenset[TranslationStageKind] = frozenset()
    # [v3.23.31] Visual Cues (Vision Director): nếu bật VÀ có video_refs, quét video
    # MỘT LẦN trước khi dịch để xác định ai nói / nói với ai từng dòng, rồi bơm vào
    # speaker/addressee giúp dịch xưng hô đúng vai vế. TỐN TOKEN (mỗi ~150 dòng 1 lần
    # gọi) nên là tuỳ chọn người dùng tự bật.
    enable_visual_cues: bool = False
    visual_cues_model: str = "gemini-flash-lite-latest"
    visual_cues_batch_size: int = 150
    # [v3.23.602] Model hỏi lại NGƯỜI NÓI các dòng còn thiếu (nhận diện bằng thị giác):
    # model phân tích ngữ cảnh người dùng chọn. Rỗng = dùng ``visual_cues_model``.
    speaker_model: str = ""
    # [v3.23.623] Gán ai nói / nói với ai CHỈ BẰNG CHỮ khi KHÔNG có video: đường ống xưng hô
    # (quy tắc 'tự xưng … gọi …', lưới đại từ) chỉ chạy khi dòng có speaker/addressee, mà nhãn
    # đó từng chỉ đến từ video. Đo: 3.5-flash-lite đọc cả tập một lượt đúng ~96% người nói.
    text_cues: bool = False
    text_cues_model: str = "gemini-3.5-flash-lite"
    # [v3.23.623] Video CÓ TIẾNG cạnh phụ đề: tách giọng từ âm thanh (không tải video lên) để
    # gán người nói chính xác hơn nữa. Đo: 97-99% / 99-100% so với 91-96% chỉ bằng chữ.
    voice_source_path: str = ""
    # [v3.23.616] ``False`` khi ``events`` là TẬP CON rời rạc (dịch lại dòng chọn): hai
    # dòng kề nhau trong tập con không phải hai dòng kề nhau trong phim, nên các bước sửa
    # theo dòng trước/sau (đóng câu, viết hoa đầu câu) không được chạy ở đây.
    contiguous: bool = True


@dataclass(frozen=True)
class TranslateSubtitlesResponse:
    """Kết quả dịch.

    Attributes:
        events:           Phụ đề sau khi dịch (giữ uid/timing gốc).
        completed_stages: Các giai đoạn đã chạy thành công, theo thứ tự.
        resumed_from:     Giai đoạn resume đầu tiên (None nếu chạy mới).
        elapsed_seconds:  Tổng thời gian thực thi (giây).
    """

    events: list[SubtitleEvent]
    completed_stages: list[TranslationStageKind]
    resumed_from: TranslationStageKind | None = None
    elapsed_seconds: float = 0.0
    # [v3.23.47] Kết quả văn bản sau MỖI giai đoạn (để so sánh giữa các giai đoạn).
    # Khoá = TranslationStageKind, giá trị = danh sách SubtitleEvent của giai đoạn đó.
    stage_outputs: dict[TranslationStageKind, list[SubtitleEvent]] = field(
        default_factory=dict
    )
    # [v3.23.559] Visual Cues ĐÃ BỔ SUNG (JSON) khi cues từ phân tích ngữ cảnh phủ quá thấp
    # và đã được quét thêm; rỗng nếu không bổ sung. Tầng giao diện lưu lại vào phiên để lần
    # dịch sau và file chuẩn đoán dùng bản đầy đủ, không phải quét lại.
    visual_cues: str = ""
    # [v3.23.608] Giai đoạn lấy NGUYÊN từ bộ nhớ đệm (không gọi API), và số dòng giữ lại
    # được ở các giai đoạn chỉ dịch lại phần đầu vào đã đổi — để báo người dùng.
    reused_stages: tuple[TranslationStageKind, ...] = ()
    carried_lines: dict[TranslationStageKind, int] = field(default_factory=dict)
    # [v3.23.608] Số dòng sửa tay đã áp lại lên bản dịch mới, theo giai đoạn.
    reapplied_edits: dict[TranslationStageKind, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _StageResumePlan:
    """[v3.23.608] Phần dùng lại được của một giai đoạn sắp dịch.

    Attributes:
        resume_lines: Dòng đã có (chỉ số theo đầu vào), ``None`` nếu không có.
        batch_done_cb: Callback lưu tiến độ từng lô, ``None`` khi không có bộ nhớ đệm.
        carried: Số dòng giữ từ bản đã lưu vì đầu vào dòng đó không đổi.
        partial: Số dòng của lần chạy dở.
    """

    resume_lines: list[TranslationLine] | None
    batch_done_cb: BatchDoneCallback | None
    carried: int
    partial: int

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _record_guard_event(event: str, **fields: Any) -> None:
    """[v3.23.588] Ghi việc một lưới tất định vừa làm vào nhật ký diễn biến dịch.

    File chuẩn đoán dịch chỉ chứa KẾT QUẢ sau lưới của từng giai đoạn. Lưới nào không ghi
    sự kiện thì việc nó đã sửa dòng nào chỉ còn trong nhật ký ứng dụng — đúng trường hợp
    của lưới số, lưới xưng hô theo bảng, lưới tên gọi và khâu dọn chữ Hán sót.

    Args:
        event: Mã sự kiện, vd ``"numbers_normalised"``.
        **fields: Dữ liệu kèm theo (nhãn giai đoạn, các dòng đã sửa…).
    """
    from subtitles_extractor.application.services.translation_process_log import (
        record_event,
    )

    record_event(event, **fields)


def checkpoint_identity(request: TranslateSubtitlesRequest) -> CheckpointIdentity:
    """[v3.23.608] Định danh file bộ nhớ đệm của một yêu cầu dịch.

    Gồm câu gốc, cặp ngôn ngữ và Visual Cues (làm giàu người nói/người nghe → đổi kết quả
    dịch). KHÔNG gồm cấu hình từng giai đoạn ([v3.23.606], so riêng từng giai đoạn) và từ
    v3.23.608 KHÔNG gồm model quét Visual Cues: model đó mặc định là model giai đoạn đầu,
    nên trước đây bật Visual Cues mà đổi model giai đoạn đầu (vì hết hạn mức) là đổi tên
    file → mất mọi giai đoạn đã xong — đúng điều v3.23.606 định sửa.
    """
    return CheckpointIdentity(
        texts=tuple(event.text for event in request.events),
        target_lang=request.context.target_lang,
        source_lang=request.context.source_lang or "",
        visual_cues=request.enable_visual_cues,
        visual_cues_batch_size=request.visual_cues_batch_size,
        visual_cues_model=request.visual_cues_model,
    )


def _compute_checkpoint_key(request: TranslateSubtitlesRequest) -> str:
    """Khoá file bộ nhớ đệm của một yêu cầu dịch (xem :func:`checkpoint_identity`).

    Cố ý KHÔNG đưa ngữ cảnh tự-phân-tích (nhân vật/tóm tắt) vào khoá: bảng nhân vật/thuật
    ngữ đổi giữa hai lần chạy do ``context_fingerprint`` ghi trong file đảm nhận.
    """
    return checkpoint_identity(request).key


def _compute_legacy_checkpoint_key(request: TranslateSubtitlesRequest) -> str:
    """Khoá file của v3.23.135 đến v3.23.605 (gồm cấu hình mọi giai đoạn).

    Chỉ dùng để nhận file của lần chạy dở do bản cũ ghi.
    """
    return legacy_checkpoint_key(
        checkpoint_identity(request), request.stages, request.attach_video_stages
    )


def build_stage_cache_scope(
    request: TranslateSubtitlesRequest, checkpoint_dir: Path
) -> StageCacheScope:
    """[v3.23.608] Phạm vi bộ nhớ đệm của một yêu cầu dịch (dùng chung use case và UI).

    Args:
        request: Yêu cầu dịch (phụ đề, giai đoạn, ngữ cảnh, Visual Cues).
        checkpoint_dir: Thư mục bộ nhớ đệm.

    Returns:
        :class:`StageCacheScope` để mở/xem/sửa bộ nhớ đệm.
    """
    return StageCacheScope(
        checkpoint_dir=checkpoint_dir,
        identity=checkpoint_identity(request),
        source_lines=tuple(TranslateSubtitlesUseCase._events_to_lines(request.events)),
        stages=tuple(request.stages),
        attach_video_stages=frozenset(request.attach_video_stages),
        context_fingerprint=context_fingerprint(request.context),
        visual_cues=request.context.visual_cues or "",  # [v3.23.656]
    )


#: Tên cũ (trước v3.23.608 các hàm/lớp này nằm trong mô-đun này).
_StageCheckpoint = StageCheckpoint
_lines_to_json = lines_to_json
_json_to_lines = json_to_lines


def _measure_passthrough_ratio(
    source_events: list[SubtitleEvent], translated_events: list[SubtitleEvent]
) -> tuple[int, int]:
    """Đếm số dòng KẾT QUẢ giống HỆT nguồn (đã trim) — dấu hiệu "dịch không tác dụng".

    Hàm thuần (không I/O, không trạng thái ngoài): chỉ so từng cặp ``text`` theo
    vị trí. Dùng cho guard cuối pipeline nhằm phát hiện trường hợp toàn bộ (hoặc gần
    như toàn bộ) bản dịch trùng bản gốc — thường do pipeline THIẾU khâu LITERAL,
    nguồn đã cùng ngôn ngữ đích, hoặc resume nhầm một checkpoint hỏng.

    Args:
        source_events:     Phụ đề gốc (đầu vào).
        translated_events: Phụ đề sau toàn bộ pipeline.

    Returns:
        Cặp ``(số_dòng_giữ_nguyên, tổng_số_dòng_so_được)``. Chỉ so các dòng có nội
        dung nguồn khác rỗng để tỉ lệ không bị nhiễu bởi dòng trống.
    """
    unchanged = 0
    comparable = 0
    for source, translated in zip(source_events, translated_events, strict=False):
        source_text = (source.text or "").strip()
        if not source_text:
            continue
        comparable += 1
        if source_text == (translated.text or "").strip():
            unchanged += 1
    return unchanged, comparable


class TranslateSubtitlesUseCase:
    """Điều phối dịch phụ đề đa giai đoạn qua một ``SubtitleTranslatorPort``."""

    _STAGE_LABELS = {
        TranslationStageKind.PREPROCESS: "Giai đoạn 1: Tiền xử lý bản gốc",
        TranslationStageKind.LITERAL: "Giai đoạn 2: Dịch thô sát nghĩa",
        TranslationStageKind.STYLE: "Giai đoạn 3: Tinh chỉnh phong cách",
        TranslationStageKind.LOCALIZE: "Giai đoạn 4: Bản địa hoá",
    }

    def __init__(
        self,
        translator: SubtitleTranslatorPort,
        checkpoint_dir: Path | None = None,
        *,
        cache_max_files: int = DEFAULT_CACHE_MAX_FILES,
        cache_max_age_days: int = DEFAULT_CACHE_MAX_AGE_DAYS,
        native_address: bool = False,
    ) -> None:
        """Khởi tạo use case.

        Args:
            translator: Bộ dịch (cổng ``SubtitleTranslatorPort``).
            checkpoint_dir: Thư mục bộ nhớ đệm giai đoạn; ``None`` = không lưu.
            cache_max_files: [v3.23.608] Số bộ phụ đề tối đa giữ bộ nhớ đệm.
            cache_max_age_days: [v3.23.608] Tuổi tối đa (ngày) của bộ nhớ đệm.
            native_address: [v3.23.612] Đổi tiếng gọi kiểu Hán-Việt ("Kiếm ca") của bản
                cuối sang thuần Việt ("anh Kiếm") — xem
                :mod:`subtitles_extractor.application.services.native_address`.
        """
        self._translator = translator
        self._checkpoint_dir = checkpoint_dir
        self._cache_max_files = max(1, cache_max_files)
        self._cache_max_age_days = max(1, cache_max_age_days)
        self._native_address = bool(native_address)

    def ensure_viable_key(
        self,
        model_name: str,
        needed_requests: int = 1,
        *,
        avoid_reupload: bool = False,
    ) -> str | None:
        """[v3.23.145] Chọn API key còn quota ngày TRƯỚC khi upload video (uỷ quyền adapter).

        Cùng cơ chế như phân tích: tránh upload dưới key hết quota rồi phải xoay + tải lại.

        Args:
            model_name: Model của giai đoạn có đính video (mốc xét quota).
            needed_requests: Số request dự trù cho trọn phiên dịch (>= 1).
            avoid_reupload: [v3.23.294] True khi video đã upload dưới key hiện tại và
                phiên này tái dùng nó → adapter bỏ xoay key cơ hội, tránh upload lại.
        """
        fn = getattr(self._translator, "ensure_viable_key", None)
        if fn is None:
            return None
        try:
            return fn(model_name, needed_requests, avoid_reupload=avoid_reupload)
        except TypeError:
            # Adapter cũ chưa hỗ trợ tham số mới → fallback gọi tối giản (1 tham số).
            try:
                return fn(model_name)
            except (AttributeError, TypeError, ValueError):
                return None
        except (AttributeError, ValueError):
            return None

    def set_video_reupload_callback(self, cb: Callable[..., list] | None) -> None:
        """[v3.23.482] Truyền callback tải-lại-video xuống adapter dịch → khi 403 do XOAY KEY
        (file thuộc key cũ), adapter tự tải lại đoạn bằng key hiện tại thay vì để DỊCH thất bại.
        Trước đây chỉ khâu PHÂN TÍCH set callback này; khâu dịch (có đính video) thiếu → 403 làm
        rớt cả tập dịch (app_log)."""
        fn = getattr(self._translator, "set_video_reupload_callback", None)
        if fn is not None:
            fn(cb)

    def stages_needing_model(
        self, request: TranslateSubtitlesRequest
    ) -> list[TranslationStageKind]:
        """[v3.23.609] Các giai đoạn lượt dịch này phải gọi model (không lấy nguyên từ bộ
        nhớ đệm), theo đúng quy tắc dùng lại của :meth:`execute`.

        Bộ nhớ đệm nay được giữ sau khi dịch xong, nên bấm Dịch lại (vd để áp dòng vừa sửa
        ở giai đoạn cuối) thường không cần gọi model nào — nơi gọi dùng kết quả này để
        khỏi nén + tải video (vài phút) và khỏi chặn vì hết hạn mức ngày.

        Args:
            request: Yêu cầu dịch (đã gồm ``attach_video_stages`` như lúc chạy thật).

        Returns:
            Các giai đoạn phải dịch; mọi giai đoạn nếu không có bộ nhớ đệm.
        """
        if self._checkpoint_dir is None:
            return [stage.kind for stage in request.stages]
        from subtitles_extractor.application.services.stage_cache import (
            StageAction,
            StageCacheService,
        )

        overview = StageCacheService().overview(
            build_stage_cache_scope(request, self._checkpoint_dir)
        )
        return [
            entry.kind for entry in overview.entries
            if entry.enabled and entry.action is not StageAction.REUSE
        ]

    def has_any_daily_quota(self, model_name: str) -> bool:
        """[v3.23.146] True nếu còn ÍT NHẤT một key có quota ngày (để fail nhanh nếu cạn)."""
        fn = getattr(self._translator, "has_any_daily_quota", None)
        if fn is None:
            return True
        try:
            return bool(fn(model_name))
        except (AttributeError, TypeError, ValueError):
            return True

    def _retranslation_stage(
        self,
        request: TranslateSubtitlesRequest,
        active_stage: TranslationStageConfig | None = None,
    ) -> TranslationStageConfig | None:
        """[v3.23.611] Cấu hình lượt dịch lại phụ: giai đoạn dịch thô, model còn quota.

        Xem :mod:`subtitles_extractor.application.services.retranslation_model`.

        Args:
            request: Yêu cầu dịch (danh sách giai đoạn).
            active_stage: Giai đoạn đang chạy — model của nó vừa trả lời được nên được
                thử trước các giai đoạn khác.

        Returns:
            Giai đoạn dịch thô (có thể đã đổi model); ``None`` nếu không có giai đoạn đó.
        """
        from subtitles_extractor.application.services.retranslation_model import (
            choose_retranslation_model,
        )

        literal_stage = next(
            (s for s in request.stages if s.kind is TranslationStageKind.LITERAL), None
        )
        if literal_stage is None:
            return None
        alternatives = [
            *([active_stage.model_name] if active_stage is not None else []),
            *(stage.model_name for stage in reversed(request.stages)),
        ]
        chosen = choose_retranslation_model(
            literal_stage.model_name, alternatives, self.has_any_daily_quota
        )
        if chosen == literal_stage.model_name:
            return literal_stage
        logger.info(
            "Dịch lại phụ: model dịch thô '%s' đã hết hạn mức ngày → dùng '%s' (giữ "
            "nguyên các thiết lập khác của giai đoạn dịch thô).",
            literal_stage.model_name, chosen,
        )
        return replace(literal_stage, model_name=chosen)

    def _cleanup_duplicate_adjacent(
        self,
        current_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        cancel_cb: CancellationCallback | None,
    ) -> list[TranslationLine]:
        """[v3.23.510] Dịch lại các CẶP dòng liền kề bị gán CÙNG một bản dịch.

        Vì sao xuất hiện: khi một CÂU bị tách qua hai dòng phụ đề, model dịch trọn câu rồi đặt y
        hệt bản dịch đó vào cả hai dòng (dữ liệu thật: dòng 1860 "连大家的工资都" và 1861 "发不了"
        cùng ra "không phát nổi."). Ràng buộc số phần tử ở schema (v3.23.509) khiến model buộc phải
        điền đủ số dòng, nên thay vì bỏ trống nó LẶP dòng bên cạnh — tác dụng phụ đo được là
        duplicate_adjacent tăng từ 2 lên 10.

        [v3.23.576] Cùng cơ chế còn sinh dòng NÓI LẶP: dòng này chứa trọn câu gộp, dòng
        kề vẫn giữ nửa câu của nó ("Mọi người, mọi người bình tĩnh nào," rồi "Mọi người
        bình tĩnh nào,") — không giống hệt nên trước đây lọt lưới. Xem
        :mod:`subtitles_extractor.application.services.line_echo_guard`.

        Cách xử lý: gom các NHÓM dòng đó rồi dịch lại RIÊNG từ nguồn. Lô nhỏ và chỉ chứa
        đúng phần cần chia nên model dễ tách nghĩa theo từng dòng. Mỗi nhóm được nhận hay
        bỏ RIÊNG: chỉ nhận khi nhóm THỰC SỰ hết trùng và hết nói lặp; nếu không, giữ
        nguyên bản cũ (không làm xấu thêm).

        [v3.23.590] Cùng lượt gọi đó dịch lại luôn các cặp dòng CÁCH XA nhau mang cùng
        một bản dịch dù câu gốc khác hẳn (ca thật: dòng 1141 "一二三" mang lời bài hát
        của dòng 1165). Ở đây không biết dòng nào sai nên chỉ nhận bản dịch lại của dòng
        ra KHÁC HẲN bản cũ — xem
        :mod:`subtitles_extractor.application.services.distant_duplicate_guard`.

        Args:
            current_lines: Kết quả cuối của pipeline.
            source_lines: Dòng nguồn (để dịch lại).
            request: Yêu cầu dịch (chứa context + danh sách giai đoạn).
            cancel_cb: Callback huỷ.

        Returns:
            Danh sách dòng đã dọn (giữ nguyên nếu không dọn được).
        """
        texts = {ln.index: (ln.text or "").strip() for ln in current_lines}
        source_by_index = {ln.index: ln for ln in source_lines}
        sources = {
            index: (
                source_by_index[index].original_text or source_by_index[index].text or ""
                if index in source_by_index
                else texts[index]
            )
            for index in texts
        }
        from subtitles_extractor.application.services.distant_duplicate_guard import (
            find_distant_duplicates,
            retranslations_to_take,
        )
        from subtitles_extractor.application.services.line_echo_guard import (
            group_still_repeats,
            repeated_line_groups,
        )

        groups = repeated_line_groups(sources, texts)
        adjacent = {index for group in groups for index in group}
        distant = [
            pair
            for pair in find_distant_duplicates(sources, texts)
            if not adjacent & set(pair)
        ]
        if not groups and not distant:
            return current_lines

        repeated = adjacent | {index for pair in distant for index in pair}
        literal_stage = self._retranslation_stage(request)
        subset = [ln for ln in source_lines if ln.index in repeated]
        if literal_stage is None or not subset:
            logger.warning(
                "Trùng/lặp dòng: %d dòng nói trùng nhau nhưng KHÔNG dịch lại được (thiếu "
                "giai đoạn dịch thô) — cần soi tay: %s",
                len(repeated), sorted(repeated),
            )
            return current_lines

        if groups:
            logger.warning(
                "Trùng/lặp dòng liền kề: %d nhóm (%s) → dịch lại riêng từ nguồn.",
                len(groups),
                ", ".join("-".join(map(str, (g[0], g[-1]))) for g in groups),
            )
        if distant:
            logger.warning(
                "Trùng dòng cách xa: %d cặp dòng cùng bản dịch dù câu gốc khác hẳn (%s) "
                "→ dịch lại riêng từ nguồn.",
                len(distant), ", ".join(f"{first}/{second}" for first, second in distant),
            )
        try:
            retried = self._translator.translate_stage(
                stage=literal_stage,
                context=request.context,
                source_lines=subset,
                input_lines=subset,
                has_prior_translation=False,
                progress_cb=None,
                cancel_cb=cancel_cb,
                video_refs=list(request.video_refs),
                attach_video=False,
            )
        except (RuntimeError, ValueError, OSError, SubtitleTranslationError) as exc:
            self._reraise_user_cancel(exc, cancel_cb)
            logger.warning("Trùng/lặp dòng: dịch lại thất bại (%s) — giữ nguyên.", exc)
            return current_lines

        retry_by_index = {
            ln.index: (ln.text or "").strip() for ln in retried if (ln.text or "").strip()
        }
        accepted: list[tuple[int, ...]] = []
        for group in groups:
            retried_group = {i: retry_by_index[i] for i in group if i in retry_by_index}
            trial = {**texts, **retried_group}
            if trial == texts or group_still_repeats(group, sources, trial):
                continue
            texts = trial
            accepted.append(group)

        replaced = retranslations_to_take(distant, texts, retry_by_index)
        texts = {**texts, **replaced}

        if groups:
            _record_guard_event(
                "repeated_lines_retranslated",
                groups=[list(group) for group in groups],
                fixed=[list(group) for group in accepted],
            )
            if accepted:
                logger.info(
                    "Trùng/lặp dòng liền kề: sửa được %d/%d nhóm sau khi dịch lại.",
                    len(accepted), len(groups),
                )
            else:
                logger.info(
                    "Trùng dòng liền kề: dịch lại KHÔNG cải thiện — giữ nguyên bản cũ."
                )
        if distant:
            _record_guard_event(
                "distant_duplicates_retranslated",
                pairs=[list(pair) for pair in distant],
                replaced=sorted(replaced),
            )
            logger.info(
                "Trùng dòng cách xa: thay %d dòng có bản dịch lại khác hẳn bản cũ (%s); "
                "các dòng còn lại dịch lại vẫn ra như cũ nên giữ nguyên.",
                len(replaced), ", ".join(map(str, sorted(replaced))) or "không dòng nào",
            )
        fixed = {index for group in accepted for index in group} | set(replaced)
        if not fixed:
            return current_lines
        return [
            replace(line, text=texts[line.index]) if line.index in fixed else line
            for line in current_lines
        ]

    def _patch_under_translated(
        self,
        current_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        cancel_cb: CancellationCallback | None,
    ) -> list[TranslationLine]:
        """[v3.23.632] Vá RIÊNG các dòng còn bị nghi DỊCH THIẾU ở bản cuối.

        Lưới ``under_translation_guard`` chỉ cảnh báo, còn cơ chế "bổ sung" của giai đoạn kế
        không cứu được: log thật báo lại CÙNG một dòng ở cả ba giai đoạn rồi để nguyên trong bản
        cuối (xem :mod:`~subtitles_extractor.application.services.under_translation_patch`).
        Ở đây dịch lại riêng các dòng đó, kèm vài dòng kề làm ngữ cảnh, và chỉ nhận bản dịch lại
        khi nó thật sự tốt hơn; không tốt hơn thì giữ nguyên bản cũ.

        Args:
            current_lines: Kết quả cuối của pipeline.
            source_lines: Dòng nguồn (để dịch lại).
            request: Yêu cầu dịch (chứa context + danh sách giai đoạn).
            cancel_cb: Callback huỷ.

        Returns:
            Danh sách dòng đã vá (chính ``current_lines`` nếu không vá được dòng nào).
        """
        from subtitles_extractor.application.services.under_translation_patch import (
            accept_patches,
            patch_targets,
            patch_window,
        )

        source_by_index = {ln.index: ln for ln in source_lines}
        pairs = [
            (
                ln.index,
                (source_by_index[ln.index].original_text or source_by_index[ln.index].text or "")
                if ln.index in source_by_index else (ln.text or ""),
                (ln.text or "").strip(),
            )
            for ln in current_lines
        ]
        targets = patch_targets(pairs)
        if not targets:
            return current_lines
        literal_stage = self._retranslation_stage(request)
        window = set(patch_window(targets, [index for index, _, _ in pairs]))
        subset = [ln for ln in source_lines if ln.index in window]
        wanted = sorted(target.index for target in targets)
        if literal_stage is None or not subset:
            logger.warning(
                "Dịch thiếu: %d dòng còn bị nghi dịch thiếu (%s) nhưng KHÔNG dịch lại được "
                "(thiếu giai đoạn dịch thô) — cần soi tay.", len(wanted), wanted,
            )
            return current_lines

        logger.warning(
            "Dịch thiếu: %d dòng còn bị nghi dịch thiếu ở bản cuối (%s) → dịch lại riêng kèm "
            "ngữ cảnh (%d dòng gửi).", len(wanted), wanted, len(subset),
        )
        try:
            retried = self._translator.translate_stage(
                stage=literal_stage,
                context=request.context,
                source_lines=subset,
                input_lines=subset,
                has_prior_translation=False,
                progress_cb=None,
                cancel_cb=cancel_cb,
                video_refs=list(request.video_refs),
                attach_video=False,
            )
        except (RuntimeError, ValueError, OSError, SubtitleTranslationError) as exc:
            self._reraise_user_cancel(exc, cancel_cb)
            logger.warning("Dịch thiếu: dịch lại thất bại (%s) — giữ nguyên bản cũ.", exc)
            return current_lines

        accepted = accept_patches(
            pairs, targets, {ln.index: (ln.text or "").strip() for ln in retried},
        )
        still = sorted(set(wanted) - set(accepted))
        _record_guard_event(
            "under_translated_patched", patched=sorted(accepted), still_suspect=still,
        )
        logger.info(
            "Dịch thiếu: vá được %d/%d dòng%s.", len(accepted), len(wanted),
            f"; còn nghi: {still}" if still else "",
        )
        if not accepted:
            return current_lines
        return [
            replace(line, text=accepted[line.index]) if line.index in accepted else line
            for line in current_lines
        ]

    def _cleanup_residual_cjk(
        self,
        current_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        cancel_cb: CancellationCallback | None,
    ) -> list[TranslationLine]:
        """[v3.23.433] Dọn CJK còn sót thật: thử dịch LẠI riêng các dòng còn chữ Hán.

        Áp dụng khi giai đoạn LITERAL dịch MỘT PHẦN, để lại chữ ghép/đuôi dạng CJK (vd
        'hỗn账', 'Phi Phi姐', 'thuần dương之气') — safety-net revert (v432) không cứu được vì
        MỌI giai đoạn đều còn CJK (không có bản sạch để lùi). Chiến lược:

        1. Tìm các dòng bản cuối còn chữ CJK.
        2. Dịch LẠI riêng các dòng đó từ NGUỒN qua giai đoạn LITERAL (lấy mẫu mới — model
           thường dịch trọn ở lần thử khác).
        3. CHỈ thay khi bản dịch lại SẠCH CJK (nếu vẫn sót → giữ nguyên, không làm tệ hơn).
        4. Log rõ dòng nào còn sót sau cùng để người dùng soi (kể cả khi không có key/không
           gọi lại được API → fallback an toàn, giữ nguyên bản cũ).

        Args:
            current_lines: Kết quả cuối của pipeline.
            source_lines: Dòng nguồn (để dịch lại).
            request: Yêu cầu dịch (chứa context + danh sách giai đoạn).
            cancel_cb: Callback huỷ.

        Returns:
            Danh sách dòng đã dọn (giữ nguyên nếu không dọn được).
        """
        residual = {
            ln.index for ln in current_lines if _CJK_CHAR_RE.search(ln.text or "")
        }
        if not residual:
            return current_lines

        # [v3.23.501] BƯỚC 1 — thay tất định bằng bảng tên/thuật ngữ ĐÃ DUYỆT. Model hay dịch nửa
        # vời, để lại chữ lai ("Lâm先生", "Cường哥") hoặc bỏ nguyên tên ("秦雅"); dịch lại cả dòng
        # thường cho ra đúng kết quả cũ nên không cứu được. Thay theo bảng thì không thể bịa nội
        # dung và luôn nhất quán với phần còn lại của bản dịch.
        from subtitles_extractor.application.services.roster_format import (
            cjk_substitution_map,
            substitute_known_cjk,
        )

        context = getattr(request, "context", None)
        substitution_map = cjk_substitution_map(
            str(getattr(context, "characters", "") or ""),
            str(getattr(context, "glossary", "") or ""),
        )
        if substitution_map:
            patched: list[TranslationLine] = []
            fixed_count = 0
            for line in current_lines:
                if line.index in residual:
                    candidate = substitute_known_cjk(line.text or "", substitution_map)
                    if candidate and not _CJK_CHAR_RE.search(candidate):
                        patched.append(replace(line, text=candidate))
                        fixed_count += 1
                        continue
                patched.append(line)
            if fixed_count:
                logger.info(
                    "Dọn CJK sót: %d dòng đã sạch nhờ thay theo bảng tên/thuật ngữ.", fixed_count
                )
                _record_guard_event(
                    "residual_cjk_cleaned",
                    how="substitution",
                    lines=sorted(
                        ln.index
                        for ln in patched
                        if ln.index in residual and not _CJK_CHAR_RE.search(ln.text or "")
                    ),
                )
                current_lines = patched
                residual = {
                    ln.index for ln in current_lines if _CJK_CHAR_RE.search(ln.text or "")
                }
                if not residual:
                    return current_lines

        literal_stage = self._retranslation_stage(request)
        source_subset = [ln for ln in source_lines if ln.index in residual]

        retry_by_index: dict[int, str] = {}
        if literal_stage is not None and source_subset:
            logger.warning(
                "Dọn CJK sót: %d dòng còn chữ Hán (%s) → thử dịch lại từ nguồn.",
                len(residual), sorted(residual),
            )
            try:
                retried = self._translator.translate_stage(
                    stage=literal_stage,
                    context=request.context,
                    source_lines=source_subset,
                    input_lines=source_subset,
                    has_prior_translation=False,
                    progress_cb=None,
                    cancel_cb=cancel_cb,
                    video_refs=list(request.video_refs),
                    attach_video=False,
                )
                retry_by_index = {ln.index: ln.text for ln in retried}
            except (RuntimeError, ValueError, OSError, SubtitleTranslationError) as exc:
                self._reraise_user_cancel(exc, cancel_cb)
                logger.warning("Dọn CJK sót: dịch lại thất bại (%s) — giữ nguyên bản cũ.", exc)

        result: list[TranslationLine] = []
        still_residual: list[int] = []
        for line in current_lines:
            if line.index in residual:
                candidate = retry_by_index.get(line.index)
                if candidate and not _CJK_CHAR_RE.search(candidate):
                    result.append(replace(line, text=candidate))
                    continue
                still_residual.append(line.index)
            result.append(line)
        if still_residual:
            logger.warning(
                "Dọn CJK sót: %d dòng VẪN còn chữ Hán sau khi thử lại: %s — cần soi/sửa tay "
                "(cờ QC 'cjk_leftover' đã đánh dấu).",
                len(still_residual), still_residual,
            )
        _record_guard_event(
            "residual_cjk_cleaned",
            how="retranslation",
            lines=sorted(residual - set(still_residual)),
            still_residual=still_residual,
        )
        return result

    @staticmethod
    def _close_cleanup_sentences(
        before: dict[int, str],
        current_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        context: TranslationContext,
    ) -> list[TranslationLine]:
        """[v3.23.603] Đóng câu ở dòng vừa dịch lại mà kết bằng dấu phẩy ở ranh giới câu.

        Xem :mod:`subtitles_extractor.application.services.retranslation_punctuation`.

        Args:
            before: ``index → bản dịch`` trước bước dọn cuối.
            current_lines: Kết quả sau bước dọn cuối.
            source_lines: Câu gốc (mốc thời gian).
            context: Ngữ cảnh dịch (Visual Cues cho người nói).

        Returns:
            Danh sách dòng đã sửa; chính ``current_lines`` nếu không có gì để sửa.
        """
        from subtitles_extractor.application.services.retranslation_punctuation import (
            close_new_trailing_commas,
        )
        from subtitles_extractor.application.services.visual_cue_serializer import (
            deserialize_visual_cues,
        )

        after = {line.index: line.text or "" for line in current_lines}
        if after == before:
            return current_lines
        spans = {
            line.index: (line.start_ms / 1000.0, line.end_ms / 1000.0)
            for line in source_lines
        }
        speakers = {
            cue.line_no: cue.speaker.strip()
            for cue in deserialize_visual_cues(str(context.visual_cues or ""))
            if cue.speaker.strip()
        }
        fixed = close_new_trailing_commas(before, after, spans, speakers)
        if not fixed:
            return current_lines
        logger.info(
            "Bước dọn cuối: %d dòng dịch lại kết bằng dấu phẩy ở chỗ đã hết câu → đóng "
            "câu: %s",
            len(fixed), ", ".join(str(index) for index in sorted(fixed)[:10]),
        )
        _record_guard_event("retranslated_endings_closed", lines=sorted(fixed))
        return [
            replace(line, text=fixed[line.index]) if line.index in fixed else line
            for line in current_lines
        ]

    @staticmethod
    def _capitalize_sentence_starts(
        current_lines: list[TranslationLine], context: TranslationContext
    ) -> list[TranslationLine]:
        """[v3.23.616] Viết hoa chữ đầu dòng mở câu mới sau dòng đã kết câu.

        Xem :mod:`subtitles_extractor.application.services.sentence_case_guard`.

        Args:
            current_lines: Bản dịch cuối theo thứ tự thời gian.
            context: Ngữ cảnh dịch (ngôn ngữ đích).

        Returns:
            Danh sách dòng đã sửa; chính ``current_lines`` nếu không có gì để sửa.
        """
        from subtitles_extractor.application.services.sentence_case_guard import (
            capitalize_sentence_starts,
            supports_sentence_case,
        )

        if not supports_sentence_case(context.target_lang):
            return current_lines
        fixed = capitalize_sentence_starts([line.text or "" for line in current_lines])
        if not fixed:
            return current_lines
        indices = sorted(current_lines[position].index for position in fixed)
        logger.info(
            "Viết hoa chữ đầu %d dòng mở câu mới (dòng trước đã kết câu): %s",
            len(indices), ", ".join(str(index) for index in indices[:10]),
        )
        _record_guard_event("sentence_starts_capitalized", lines=indices)
        return [
            replace(line, text=fixed[position]) if position in fixed else line
            for position, line in enumerate(current_lines)
        ]

    @staticmethod
    def _apply_native_address(
        current_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        context: TranslationContext,
    ) -> list[TranslationLine]:
        """[v3.23.612] Đổi tiếng gọi Hán-Việt của bản cuối sang thuần Việt.

        Từ vai vế theo cách người nói (Visual Cues) gọi người đó trong bảng nhân vật —
        xem :mod:`subtitles_extractor.application.services.native_address`.

        Args:
            current_lines: Bản cuối.
            source_lines: Câu gốc (phải có chữ Hán của hậu tố thì mới đổi).
            context: Ngữ cảnh dịch (bảng nhân vật, Visual Cues).

        Returns:
            Danh sách dòng đã đổi; chính ``current_lines`` nếu không có gì để đổi.
        """
        from subtitles_extractor.application.services.native_address import (
            describe_changes,
            nativize_lines,
        )
        from subtitles_extractor.application.services.visual_cue_serializer import (
            deserialize_visual_cues,
        )

        cues = deserialize_visual_cues(str(context.visual_cues or ""))
        changed, changes = nativize_lines(
            {line.index: line.text or "" for line in current_lines},
            {line.index: line.original_text or line.text or "" for line in source_lines},
            str(context.characters or ""),
            speakers={cue.line_no: cue.speaker for cue in cues if cue.speaker},
            listeners={cue.line_no: cue.addressee for cue in cues if cue.addressee},
        )
        if not changed:
            return current_lines
        logger.info(
            "Xưng hô thuần Việt: đổi %d tiếng gọi ở %d dòng — %s.",
            len(changes), len(changed), describe_changes(changes),
        )
        _record_guard_event("native_address_applied", lines=sorted(changed))
        return [
            replace(line, text=changed[line.index]) if line.index in changed else line
            for line in current_lines
        ]

    @staticmethod
    def _process_log_mark() -> int:
        """[v3.23.592] Mốc hiện tại của nhật ký diễn biến (0 nếu không dùng được)."""
        try:
            from subtitles_extractor.application.services.translation_process_log import (
                get_process_log,
            )

            return get_process_log().count()
        except (ImportError, AttributeError, TypeError):  # nhật ký là việc phụ
            return 0

    @staticmethod
    def _process_log_events_since(mark: int) -> list[dict[str, Any]] | None:
        """[v3.23.592] Sự kiện ghi sau ``mark``; ``None`` nếu nhật ký không dùng được."""
        try:
            from subtitles_extractor.application.services.translation_process_log import (
                get_process_log,
            )

            return get_process_log().events_since(mark)
        except (ImportError, AttributeError, TypeError):  # nhật ký là việc phụ
            return None

    @staticmethod
    def _replay_stage_events(events: list[dict[str, Any]]) -> None:
        """[v3.23.592] Phát lại nhật ký đã lưu của giai đoạn được bỏ qua khi chạy tiếp."""
        if not events:
            return
        try:
            from subtitles_extractor.application.services.translation_process_log import (
                get_process_log,
            )

            get_process_log().replay(events)
        except (ImportError, AttributeError, TypeError):  # nhật ký là việc phụ
            pass

    @staticmethod
    def _record_stage_completed(
        stage: Any,
        stage_label: str,
        stage_output: list[TranslationLine],
        attach_video: bool = False,
    ) -> None:
        """[v3.23.549] Ghi mốc "giai đoạn xong" vào nhật ký diễn biến dịch.

        Nhờ mốc này, file chuẩn đoán của một lần dịch SẠCH vẫn tự chứng minh là nhật ký có chạy
        và cho biết mỗi giai đoạn dùng model nào, ra bao nhiêu dòng — thay vì một danh sách rỗng
        không nói lên điều gì.

        [v3.23.591] Ghi cả việc giai đoạn có ĐÍNH VIDEO hay không: bản dịch khác hẳn giữa
        hai chế độ (model dịch theo lời nghe được), mà trước đây muốn biết phải mở nhật ký
        ứng dụng đếm từng lời gọi.

        Args:
            stage: Cấu hình giai đoạn vừa hoàn tất.
            stage_label: Nhãn tiếng Việt của giai đoạn.
            stage_output: Các dòng giai đoạn trả về.
            attach_video: Giai đoạn có gửi kèm video cho từng lô.
        """
        try:
            from subtitles_extractor.application.services.translation_process_log import (
                record_event,
            )

            record_event(
                "stage_completed",
                stage=getattr(getattr(stage, "kind", None), "value", str(stage)),
                label=stage_label,
                model=getattr(stage, "model_name", ""),
                lines=len(stage_output),
                attach_video=bool(attach_video),
            )
        except Exception:  # noqa: BLE001 — nhật ký là việc phụ, không được cản việc dịch
            pass

    def _repair_misaligned_blocks(
        self,
        stage_output: list[TranslationLine],
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        cancel_cb: CancellationCallback | None,
        stage_label: str,
        *,
        active_stage: TranslationStageConfig | None = None,
    ) -> list[TranslationLine]:
        """[v3.23.558] Dò và sửa KHỐI dòng lệch trên TOÀN BỘ đầu ra của một giai đoạn.

        Vì sao cần, dù lúc dịch đã có lưới lệch khối: lưới đó chạy trên từng LÔ 60 dòng,
        còn khối lệch có thể VẮT QUA ranh giới lô. Ca thật v3.23.557: khối 3407-3421 nằm ở
        đuôi lô 3361-3420 — chỉ còn 5 cửa sổ trượt, thiếu 1 so với ngưỡng 6, nên lưới lúc
        dịch mù hẳn. Hậu quả: từ dòng 3409 mỗi câu mang bản dịch của câu KẾ TIẾP
        ("好嘞" ra "Thế còn Từ Tuấn Hào thì sao?"), lỗi đi thẳng tới bản cuối.

        Nhìn trên toàn bộ đầu ra thì không còn ranh giới lô. Đo trên 10 bộ chuẩn đoán
        thật: bắt đúng mọi khối lệch thật đã biết, không báo nhầm ở lần chạy nào.

        Cách sửa: dịch lại riêng vùng lệch (kèm :data:`BLOCK_REPAIR_PADDING_LINES` dòng
        đệm) từ CÂU GỐC bằng giai đoạn dịch thô — dịch từ giai đoạn trước chỉ chép lại cái
        lệch. Chỉ nhận kết quả khi khối THỰC SỰ biến mất; nếu không thì giữ nguyên.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            source_lines: Câu gốc.
            request: Yêu cầu dịch (context + danh sách giai đoạn).
            cancel_cb: Callback huỷ.
            stage_label: Nhãn giai đoạn (để log).
            active_stage: [v3.23.611] Giai đoạn đang chạy — model của nó được thử trước
                khi model dịch thô đã hết hạn mức ngày.

        Returns:
            Danh sách dòng đã sửa khối lệch; ``stage_output`` nếu không có gì để sửa.
        """
        from subtitles_extractor.application.services.alignment_check import (
            find_low_correlation_blocks,
        )

        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        current = {ln.index: (ln.text or "") for ln in stage_output}
        blocks = find_low_correlation_blocks(sources, current)
        if not blocks:
            return stage_output
        literal_stage = self._retranslation_stage(request, active_stage)
        if literal_stage is None:
            logger.warning(
                "%s: có %d khối lệch dòng nhưng KHÔNG dịch lại được (thiếu giai đoạn "
                "dịch thô) — cần soi tay: %s",
                stage_label, len(blocks),
                ", ".join(f"{b['from_index']}-{b['to_index']}" for b in blocks),
            )
            return stage_output

        ordered = sorted(sources)
        source_by_index = {ln.index: ln for ln in source_lines}
        repaired: list[str] = []
        for block in blocks[:MAX_BLOCK_REPAIRS_PER_STAGE]:
            start = ordered.index(block["from_index"])
            end = ordered.index(block["to_index"])
            span = ordered[
                max(0, start - BLOCK_REPAIR_PADDING_LINES):
                end + BLOCK_REPAIR_PADDING_LINES + 1
            ]
            subset = [source_by_index[i] for i in span if i in source_by_index]
            logger.warning(
                "%s: KHỐI LỆCH DÒNG %d-%d (%d dòng, tương quan %.2f) — lưới theo lô "
                "không thấy vì khối vắt qua ranh giới lô → dịch lại riêng từ câu gốc.",
                stage_label, block["from_index"], block["to_index"], block["lines"],
                block["min_correlation"],
            )
            try:
                retried = self._translator.translate_stage(
                    stage=literal_stage,
                    context=request.context,
                    source_lines=subset,
                    input_lines=subset,
                    has_prior_translation=False,
                    progress_cb=None,
                    cancel_cb=cancel_cb,
                    video_refs=list(request.video_refs),
                    attach_video=False,
                )
            except (RuntimeError, ValueError, OSError, SubtitleTranslationError) as exc:
                # Huỷ giữa lúc sửa khối mà nuốt lỗi thì giai đoạn được LƯU khi chưa sửa
                # xong và lần sau dùng lại nguyên bản đó.
                self._reraise_user_cancel(exc, cancel_cb)
                logger.warning("%s: dịch lại khối lệch thất bại (%s) — giữ nguyên.",
                               stage_label, exc)
                continue
            candidate = dict(current)
            candidate.update(
                {ln.index: ln.text for ln in retried if (ln.text or "").strip()}
            )
            still_bad = any(
                other["from_index"] <= span[-1] and other["to_index"] >= span[0]
                for other in find_low_correlation_blocks(sources, candidate)
            )
            if still_bad:
                logger.warning(
                    "%s: dịch lại khối %d-%d vẫn lệch — giữ nguyên bản cũ, cần soi tay.",
                    stage_label, block["from_index"], block["to_index"],
                )
                continue
            current = candidate
            repaired.append(f"{block['from_index']}-{block['to_index']}")

        if not repaired:
            return stage_output
        logger.info("%s: đã sửa %d khối lệch dòng: %s",
                    stage_label, len(repaired), ", ".join(repaired))
        return [replace(line, text=current[line.index]) for line in stage_output]

    def _repair_stage_shifts(
        self,
        stage_output: list[TranslationLine],
        previous_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        cancel_cb: CancellationCallback | None,
        stage_label: str,
        *,
        active_stage: TranslationStageConfig | None = None,
    ) -> list[TranslationLine]:
        """[v3.23.569] Lùi về bản trước những dải dòng giai đoạn tinh chỉnh đẩy lệch.

        Ca thật v3.23.568: bản dịch thô căn đúng 3395-3421, bản tinh chỉnh phong cách đẩy
        lệch hai dòng; bản cuối còn 9 dòng lệch và mất một câu. Xem
        :mod:`subtitles_extractor.application.services.stage_shift_guard`.

        Mỗi dải nghi lệch được dịch TRỌNG TÀI từ câu gốc (giai đoạn dịch thô, kèm dòng
        đệm).
        Chỉ lùi khi trọng tài xác nhận bản trước mới là bản căn đúng — nhiều khi chính bản
        trước lệch và giai đoạn này đã sửa lại, khi đó giữ nguyên.

        [v3.23.589] Dải bị VIẾT LẠI khác hẳn bản trước (không giống cả dòng kề nên bộ dò
        lệch không thấy) đi qua cùng trọng tài. Ca thật phim "灭门惨案II" dòng 68-77: giai
        đoạn phong cách đổi "Đừng giận." thành "Mở cửa!" theo tiếng đám đông; bản cuối
        có dòng bị viết lại trong dải này ở 8/14 lượt dịch, bản thô thì đúng cả 14 lượt.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            previous_lines: Bản trước (đầu vào của giai đoạn này).
            source_lines: Câu gốc.
            request: Yêu cầu dịch.
            cancel_cb: Callback huỷ.
            stage_label: Nhãn giai đoạn (để log).
            active_stage: [v3.23.611] Giai đoạn đang chạy — model của nó được thử trước
                khi model dịch thô đã hết hạn mức ngày.

        Returns:
            Danh sách dòng đã lùi các dải lệch; ``stage_output`` nếu không có gì để sửa.

        Raises:
            TranslationCancelledError: Khi người dùng huỷ trong lúc dịch trọng tài.
        """
        from subtitles_extractor.application.services.stage_shift_guard import (
            VERDICT_PREVIOUS,
            find_stage_rewrites,
            find_stage_shifts,
            judge_shift,
            shift_span,
        )

        previous = {ln.index: (ln.text or "") for ln in previous_lines}
        current = {ln.index: (ln.text or "") for ln in stage_output}
        shifts = find_stage_shifts(previous, current)
        shifts = sorted(
            [*shifts, *find_stage_rewrites(previous, current, shifts)],
            key=lambda shift: shift.from_index,
        )
        if not shifts:
            return stage_output
        literal_stage = self._retranslation_stage(request, active_stage)
        if literal_stage is None:
            logger.warning(
                "%s: %d dải dòng lệch so với bản trước nhưng KHÔNG có giai đoạn dịch thô "
                "để phân xử — cần soi tay: %s",
                stage_label, len(shifts),
                ", ".join(f"{s.from_index}-{s.to_index}" for s in shifts),
            )
            return stage_output

        ordered = sorted(current)
        source_by_index = {ln.index: ln for ln in source_lines}
        reverted: list[str] = []
        for shift in shifts[:MAX_SHIFT_ARBITRATIONS_PER_STAGE]:
            span = shift_span(ordered, shift, SHIFT_REFERENCE_PADDING_LINES)
            subset = [source_by_index[i] for i in span if i in source_by_index]
            try:
                reference_lines = self._translator.translate_stage(
                    stage=literal_stage,
                    context=request.context,
                    source_lines=subset,
                    input_lines=subset,
                    has_prior_translation=False,
                    progress_cb=None,
                    cancel_cb=cancel_cb,
                    video_refs=list(request.video_refs),
                    attach_video=False,
                )
            except TranslationCancelledError:
                if self._user_cancelled(cancel_cb):
                    raise
                continue
            except (RuntimeError, ValueError, OSError, SubtitleTranslationError) as exc:
                logger.warning(
                    "%s: dịch trọng tài cho dải %d-%d thất bại (%s) — giữ nguyên.",
                    stage_label, shift.from_index, shift.to_index, exc,
                )
                continue
            reference = {ln.index: (ln.text or "") for ln in reference_lines}
            run = [i for i in ordered if shift.from_index <= i <= shift.to_index]
            verdict = judge_shift(run, previous, current, reference)
            logger.warning(
                "%s: dải %d-%d %s so với bản trước — trọng tài: %s.",
                stage_label, shift.from_index, shift.to_index,
                "bị VIẾT LẠI khác hẳn" if shift.is_rewrite
                else f"mang câu của dòng lệch {shift.offset:+d}",
                {
                    VERDICT_PREVIOUS: "bản trước căn đúng → lùi về bản trước",
                    "current": "bản mới căn đúng (bản trước lệch) → giữ",
                }.get(verdict, "chưa phân định → giữ"),
            )
            self._record_stage_shift(stage_label, shift, verdict)
            if verdict != VERDICT_PREVIOUS:
                continue
            # Dải lệch kéo theo dòng biên (gộp/lặp câu) nên lùi rộng thêm; dải viết lại
            # thì dòng ngoài dải vẫn là bản tinh chỉnh đúng — không đụng tới.
            padding = 0 if shift.is_rewrite else SHIFT_REVERT_PADDING_LINES
            for index in shift_span(ordered, shift, padding):
                current[index] = previous.get(index, current[index])
            reverted.append(f"{shift.from_index}-{shift.to_index}")

        if not reverted:
            return stage_output
        logger.info("%s: đã lùi %d dải lệch dòng về bản trước: %s",
                    stage_label, len(reverted), ", ".join(reverted))
        return [replace(line, text=current[line.index]) for line in stage_output]

    def _revert_lone_rewrites(
        self,
        stage_output: list[TranslationLine],
        previous_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        cancel_cb: CancellationCallback | None,
        stage_label: str,
        *,
        active_stage: TranslationStageConfig | None = None,
    ) -> list[TranslationLine]:
        """[v3.23.592] Lùi dòng LẺ mà giai đoạn tinh chỉnh viết lại sai về bản trước.

        Ca thật v3.23.591: dòng 68 "不要生氣", bản thô "Đừng giận chứ.", giai đoạn địa
        hoá đổi thành "Mau mở cửa!" (lời đám đông nghe được trong video); hai dòng kề
        không đổi nên lưới dải viết lại không thấy. Mọi ứng viên của giai đoạn được dịch
        trọng tài trong MỘT lượt gọi từ câu gốc, không kèm video; chỉ lùi dòng nào trọng
        tài đứng hẳn về phía bản trước. Xem
        :mod:`subtitles_extractor.application.services.lone_rewrite_guard`.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy (đã qua lưới dải lệch).
            previous_lines: Bản trước (đầu vào của giai đoạn này).
            source_lines: Câu gốc.
            request: Yêu cầu dịch.
            cancel_cb: Callback huỷ.
            stage_label: Nhãn giai đoạn (để log).
            active_stage: [v3.23.611] Giai đoạn đang chạy — model của nó được thử trước
                khi model dịch thô đã hết hạn mức ngày.

        Returns:
            Danh sách dòng đã lùi các dòng lẻ viết sai; ``stage_output`` nếu không có gì
            để sửa hoặc không dịch trọng tài được.

        Raises:
            TranslationCancelledError: Khi người dùng huỷ trong lúc dịch trọng tài.
        """
        from subtitles_extractor.application.services.lone_rewrite_guard import (
            MAX_LONE_REWRITES_PER_STAGE,
            find_lone_rewrites,
            reference_lines_needed,
            rewrites_to_revert,
        )

        previous = {ln.index: (ln.text or "") for ln in previous_lines}
        current = {ln.index: (ln.text or "") for ln in stage_output}
        candidates = find_lone_rewrites(previous, current)
        if not candidates:
            return stage_output
        literal_stage = self._retranslation_stage(request, active_stage)
        if literal_stage is None:
            return stage_output
        if len(candidates) > MAX_LONE_REWRITES_PER_STAGE:
            logger.warning(
                "%s: %d dòng lẻ bị viết lại khác hẳn bản trước — chỉ phân xử %d dòng "
                "đầu.",
                stage_label, len(candidates), MAX_LONE_REWRITES_PER_STAGE,
            )
            candidates = candidates[:MAX_LONE_REWRITES_PER_STAGE]

        wanted = set(reference_lines_needed(candidates, sorted(current)))
        subset = [ln for ln in source_lines if ln.index in wanted]
        try:
            reference_lines = self._translator.translate_stage(
                stage=literal_stage,
                context=request.context,
                source_lines=subset,
                input_lines=subset,
                has_prior_translation=False,
                progress_cb=None,
                cancel_cb=cancel_cb,
                video_refs=list(request.video_refs),
                attach_video=False,
            )
        except TranslationCancelledError:
            if self._user_cancelled(cancel_cb):
                raise
            return stage_output
        except (RuntimeError, ValueError, OSError, SubtitleTranslationError) as exc:
            logger.warning(
                "%s: dịch trọng tài cho %d dòng lẻ bị viết lại thất bại (%s) — giữ "
                "nguyên.", stage_label, len(candidates), exc,
            )
            return stage_output

        reference = {ln.index: (ln.text or "") for ln in reference_lines}
        reverted = rewrites_to_revert(candidates, previous, current, reference)
        _record_guard_event(
            "lone_rewrites_judged",
            label=stage_label,
            candidates=candidates,
            reverted=reverted,
        )
        if not reverted:
            logger.info(
                "%s: %d dòng lẻ bị viết lại khác hẳn bản trước — trọng tài không đứng "
                "về bản trước ở dòng nào → giữ.", stage_label, len(candidates),
            )
            return stage_output
        logger.warning(
            "%s: %d/%d dòng lẻ bị viết lại khác hẳn bản trước mà trọng tài xác nhận bản "
            "trước mới đúng → lùi về bản trước: %s",
            stage_label, len(reverted), len(candidates),
            ", ".join(map(str, reverted[:20])) + ("…" if len(reverted) > 20 else ""),
        )
        chosen = set(reverted)
        return [
            replace(line, text=previous[line.index]) if line.index in chosen else line
            for line in stage_output
        ]

    @staticmethod
    def _drop_contradicted_scan_cues(
        request: TranslateSubtitlesRequest, source_lines: list[TranslationLine]
    ) -> TranslateSubtitlesRequest:
        """[v3.23.581] Bỏ nhãn quét bổ sung mâu thuẫn với cuộc hội thoại quanh nó.

        Phiên đã lưu không ghi nguồn gốc nhãn; nhãn của bộ quét nhận ra vì luôn kèm mô tả
        bối cảnh, còn nhãn của khâu phân tích thì không. Xem
        :mod:`subtitles_extractor.application.services.cue_contradiction_guard`.

        Args:
            request: Yêu cầu dịch.
            source_lines: Câu gốc kèm mốc thời gian (số dòng = ``id`` của cues).

        Returns:
            Yêu cầu mới đã bỏ các nhãn đó; chính ``request`` nếu không có gì để bỏ.
        """
        raw = (request.context.visual_cues or "").strip()
        if not raw or not source_lines:
            return request
        from subtitles_extractor.application.services.cue_contradiction_guard import (
            described_scan_lines,
        )
        from subtitles_extractor.application.services.visual_cue_serializer import (
            deserialize_visual_cues,
            serialize_visual_cues,
        )

        cues = deserialize_visual_cues(raw)
        untrusted = described_scan_lines(cues)
        if not untrusted:
            return request
        characters = request.context.characters or ""
        kept, dropped = TranslateSubtitlesUseCase._without_contradicted(
            cues, untrusted, source_lines, characters
        )
        # [v3.23.602] Nhãn tên quét/hỏi lại đã lưu mà người đó không có mặt trong cảnh →
        # hạ xuống nhãn giọng (phiên lưu từ bản trước lưới fill_support_guard).
        kept, demoted = TranslateSubtitlesUseCase._without_unsupported(kept, characters)
        if not dropped and not demoted:
            return request
        new_context = replace(request.context, visual_cues=serialize_visual_cues(kept))
        return replace(request, context=new_context)

    @staticmethod
    def _without_unsupported(
        cues: list[VisualCue], characters: str
    ) -> tuple[list[VisualCue], list[int]]:
        """[v3.23.602] Hạ nhãn tên quét/hỏi lại không có chỗ dựa trong cảnh (ghi nhật ký).

        Args:
            cues: Visual Cues đã lưu.
            characters: Bảng nhân vật.

        Returns:
            ``(cues, các dòng đã hạ nhãn)``.
        """
        from subtitles_extractor.application.services.cue_label_canonicalizer import (
            canonicalize_cue_labels,
        )
        from subtitles_extractor.application.services.fill_support_guard import (
            demote_unsupported_scan_labels,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        canonical = canonicalize_cue_labels(cues, characters)
        demoted_cues, findings = demote_unsupported_scan_labels(canonical, characters)
        if not findings:
            return cues, []
        lines = [finding.line_no for finding in findings]
        logger.warning(
            "Visual Cues quét/hỏi lại đã lưu: %d dòng mang tên người KHÔNG có mặt trong "
            "cảnh (không xuất hiện trong nhãn ±10 dòng) → hạ xuống nhãn giọng: %s",
            len(lines),
            ", ".join(f"{f.line_no} ({f.speaker})" for f in findings[:12]),
        )
        record_event("scan_cues_demoted", lines=lines)
        return demoted_cues, lines

    @staticmethod
    def _without_contradicted(
        cues: list[VisualCue],
        untrusted: set[int],
        source_lines: list[TranslationLine],
        characters: str,
    ) -> tuple[list[VisualCue], list[int]]:
        """Bỏ nhãn không tin cậy mâu thuẫn với hai nhãn tin cậy bao quanh (ghi nhật ký).

        Args:
            cues: Visual Cues.
            untrusted: Số dòng mang nhãn quét bổ sung.
            source_lines: Câu gốc kèm mốc thời gian.
            characters: Bảng nhân vật.

        Returns:
            ``(cues còn lại, các dòng đã bỏ nhãn)``.
        """
        from subtitles_extractor.application.services.cue_contradiction_guard import (
            contradicted_cue_lines,
        )
        from subtitles_extractor.application.services.cue_label_canonicalizer import (
            canonicalize_cue_labels,
        )
        from subtitles_extractor.application.services.cue_run_filler import LineSpan
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        spans = {
            ln.index: LineSpan(ln.index, ln.start_ms, ln.end_ms) for ln in source_lines
        }
        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        dropped = contradicted_cue_lines(
            canonicalize_cue_labels(cues, characters),
            untrusted,
            spans,
            sources,
            characters,
        )
        if not dropped:
            return cues, []
        logger.warning(
            "Visual Cues quét bổ sung: bỏ nhãn ở %d dòng mâu thuẫn với cuộc hội thoại "
            "quanh nó (hai dòng bao quanh cùng một cặp nhân vật khác hẳn): %s",
            len(dropped), ", ".join(str(line) for line in dropped[:12]),
        )
        record_event("scan_cues_dropped", lines=dropped)
        removed = set(dropped)
        return [cue for cue in cues if cue.line_no not in removed], dropped

    def _complete_speaker_cues(
        self,
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        checkpoint: StageCheckpoint | None,
        cancel_cb: CancellationCallback | None,
        progress_cb: OverallProgressCallback | None = None,
    ) -> TranslateSubtitlesRequest:
        """[v3.23.600] Hỏi lại người nói cho mọi dòng có lời còn thiếu (cue_completion).

        Lượt dịch thật phim "he thong sung the": nhãn phủ 97,7% nên không quét bổ sung,
        125/5540 dòng dịch và lồng tiếng mà không biết ai nói. Nay hỏi lại ĐÚNG các dòng
        thiếu (kèm nhãn đã biết làm mạch hội thoại), tối đa hai lượt; nhãn đã có luôn giữ.
        Nhãn mới cũng qua lưới "lạc cảnh" như nhãn quét bổ sung. Kết quả hỏi được cộng vào
        cache Visual Cues của checkpoint để dịch lại không phải hỏi lại.

        Args:
            source_lines: Toàn bộ dòng gốc.
            request: Yêu cầu dịch.
            checkpoint: Checkpoint của lượt dịch.
            cancel_cb: Callback huỷ.
            progress_cb: Callback tiến độ tổng.

        Returns:
            Yêu cầu mới với cues đã điền; chính ``request`` nếu không có gì để điền.
        """
        from subtitles_extractor.application.services.cue_completion import (
            fill_missing_speakers,
            lines_missing_speaker,
            speaker_coverage,
        )
        from subtitles_extractor.application.services.cue_label_canonicalizer import (
            canonicalize_cue_labels,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )
        from subtitles_extractor.application.services.visual_cue_serializer import (
            deserialize_visual_cues,
            serialize_visual_cues,
        )

        raw = (request.context.visual_cues or "").strip()
        if (
            not raw
            or getattr(self._translator, "analyze_visual_cues", None) is None
            or not source_lines
        ):
            return request
        characters = request.context.characters or ""
        cues = canonicalize_cue_labels(deserialize_visual_cues(raw), characters)
        before = speaker_coverage(cues, source_lines)
        if not before.missing:
            return request
        cached = checkpoint.get_visual_cues() if checkpoint is not None else None
        if cached:
            cues, _from_cache = fill_missing_speakers(
                cues,
                canonicalize_cue_labels(
                    deserialize_visual_cues(json.dumps(cached)), characters
                ),
            )
        missing = lines_missing_speaker(cues, source_lines)
        if missing:
            # [v3.23.602] Nhận ra người nói là việc của model THỊ GIÁC mạnh (model phân
            # tích ngữ cảnh), không phải model dịch thô: lượt thật dùng
            # gemini-3.5-flash-lite gán luân phiên ba tên nhân vật cho cả đám dân làng.
            # Model phân tích hỏng/hết hạn mức thì lùi về model của khâu dịch.
            candidates = (request.speaker_model, request.visual_cues_model)
            models = list(dict.fromkeys(name for name in candidates if name))
            logger.warning(
                "%d/%d dòng có lời chưa có NGƯỜI NÓI → hỏi lại đúng các dòng đó bằng "
                "'%s' "
                "(lồng tiếng cần người nói cho mọi dòng).",
                len(missing), before.spoken_lines, models[0],
            )
            scanned_all: list[VisualCue] = []
            outcome = self._complete_with_models(
                models, cues, source_lines, request, cancel_cb, progress_cb, scanned_all
            )
            cues = outcome.cues
            if outcome.dropped:
                logger.warning(
                    "Hỏi lại người nói: bỏ nhãn ở %d dòng mâu thuẫn với cuộc hội thoại "
                    "quanh nó (lạc cảnh): %s",
                    len(outcome.dropped),
                    ", ".join(str(line) for line in outcome.dropped[:12]),
                )
                record_event("scan_cues_dropped", lines=list(outcome.dropped))
            if scanned_all and checkpoint is not None:
                checkpoint.save_visual_cues(json.loads(serialize_visual_cues(
                    fill_missing_speakers(
                        deserialize_visual_cues(json.dumps(cached or [])), scanned_all
                    )[0]
                )))
        after = speaker_coverage(cues, source_lines)
        record_event(
            "speaker_completion",
            before=before.with_speaker, after=after.with_speaker,
            spoken=after.spoken_lines, voice_only=after.voice_only,
            missing=list(after.missing),
        )
        if after.with_speaker == before.with_speaker:
            return request
        new_context = replace(request.context, visual_cues=serialize_visual_cues(cues))
        return replace(request, context=new_context)

    def _complete_with_models(
        self,
        models: list[str],
        cues: list[VisualCue],
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        cancel_cb: CancellationCallback | None,
        progress_cb: OverallProgressCallback | None,
        scanned_all: list[VisualCue],
    ) -> SpeakerCompletionOutcome:
        """[v3.23.602] Hỏi lại người nói bằng model đầu tiên dùng được trong ``models``.

        Model trước mà MỌI lượt hỏi đều lỗi (hết hạn mức, quá tải) thì thử model sau;
        model cuối cùng lỗi thì giữ nhãn sẵn có như trước.

        Args:
            models: Model theo thứ tự ưu tiên (không rỗng).
            cues: Visual Cues hiện có (đã quy nhãn).
            source_lines: Toàn bộ dòng gốc.
            request: Yêu cầu dịch.
            cancel_cb: Callback huỷ.
            progress_cb: Callback tiến độ tổng.
            scanned_all: Nơi nhận mọi nhãn model trả về (cho cache checkpoint).

        Returns:
            Kết quả hỏi lại.

        Raises:
            TranslationCancelledError: Người dùng huỷ.
        """
        from subtitles_extractor.application.use_cases.complete_speakers import (
            CompleteSpeakersUseCase,
        )

        completer = CompleteSpeakersUseCase(self._translator)
        report = (
            (lambda _fraction, message: progress_cb(0.0, message))
            if progress_cb is not None else None
        )
        for position, model in enumerate(models):
            is_last = position == len(models) - 1
            try:
                return completer.complete(
                    cues, source_lines,
                    characters=request.context.characters or "",
                    target_lang=request.context.target_lang,
                    model_name=model,
                    video_refs=list(request.video_refs),
                    batch_size=request.visual_cues_batch_size,
                    cancel_cb=cancel_cb,
                    progress_cb=report,
                    raise_when_all_failed=not is_last,
                    scanned_sink=scanned_all,
                )
            except (RuntimeError, ValueError, OSError, SubtitleTranslationError) as exc:
                if self._user_cancelled(cancel_cb):
                    raise
                logger.warning(
                    "Hỏi lại người nói bằng '%s' thất bại (%s) → thử bằng '%s'.",
                    model, exc, models[position + 1],
                )
        raise AssertionError("unreachable: model cuối cùng không ném lỗi")

    @staticmethod
    def _fill_cue_runs(
        request: TranslateSubtitlesRequest, source_lines: list[TranslationLine]
    ) -> TranslateSubtitlesRequest:
        """[v3.23.572] Điền nhãn cues cho dòng nói tiếp khi model chỉ ghi đầu lượt lời.

        Ca thật v3.23.571: gemini-3.8-flash chỉ ghi cue ở dòng đầu lượt lời ở 8/22 đoạn
        video — 25% số dòng dịch không có người nói/người nghe. Xem
        :mod:`subtitles_extractor.application.services.cue_run_filler`.

        Args:
            request: Yêu cầu dịch.
            source_lines: Câu gốc kèm mốc thời gian (số dòng = ``id`` của cues).

        Returns:
            Yêu cầu mới với cues đã điền; chính ``request`` nếu không có gì để điền.
        """
        raw = (request.context.visual_cues or "").strip()
        if not raw or not source_lines:
            return request
        from subtitles_extractor.application.services.cue_run_filler import (
            LineSpan,
            fill_cue_runs,
        )
        from subtitles_extractor.application.services.visual_cue_serializer import (
            deserialize_visual_cues,
            serialize_visual_cues,
        )

        cues = deserialize_visual_cues(raw)
        spans = [LineSpan(ln.index, ln.start_ms, ln.end_ms) for ln in source_lines]
        filled_cues, filled_lines = fill_cue_runs(cues, spans)
        if not filled_lines:
            return request
        logger.info(
            "Visual Cues chỉ ghi dòng ĐẦU lượt lời ở vùng thưa → điền nhãn cho %d dòng "
            "nói tiếp (%d → %d dòng có cue).",
            len(filled_lines), len(cues), len(filled_cues),
        )
        new_context = replace(
            request.context, visual_cues=serialize_visual_cues(filled_cues)
        )
        return replace(request, context=new_context)

    def _voice_kwargs(self, voice_source_path: str, source_lang: str) -> dict[str, str]:
        """Tham số tách giọng cho ``analyze_text_cues`` (rỗng nếu không có video hoặc adapter cũ)."""
        if not voice_source_path or not hasattr(self._translator, "label_voices"):
            return {}
        return {"voice_source": voice_source_path, "source_lang": source_lang or ""}

    @staticmethod
    def _cues_usable(request: TranslateSubtitlesRequest) -> bool:
        """Có nhãn người nói/người nghe để bơm vào dòng dịch không (video hoặc cues bằng chữ)."""
        if request.enable_visual_cues and request.video_refs:
            return True
        return bool(request.text_cues and (request.context.visual_cues or "").strip())

    def _ensure_text_cues(
        self,
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        checkpoint: StageCheckpoint | None,
        cancel_cb: CancellationCallback | None,
        progress_cb: OverallProgressCallback | None,
    ) -> TranslateSubtitlesRequest:
        """[v3.23.623] Có cues bằng chữ cho cả tập trước khi dịch (không cần video).

        Thứ tự ưu tiên: cues đã có trong ngữ cảnh (từ phân tích / người dùng đã sửa) → cues
        lưu trong checkpoint → gọi ``analyze_text_cues`` MỘT lần. Lỗi (trừ huỷ) được nuốt:
        dịch vẫn chạy như trước, chỉ thiếu gợi ý người nói.

        Returns:
            Yêu cầu mới có ``context.visual_cues``; chính ``request`` nếu không có gì đổi.
        """
        from subtitles_extractor.application.services.cue_label_canonicalizer import (
            canonicalize_cue_labels,
        )
        from subtitles_extractor.application.services.visual_cue_serializer import (
            serialize_visual_cues,
        )

        if (request.context.visual_cues or "").strip():
            return request
        if checkpoint is not None:
            cached = checkpoint.get_visual_cues()
            if cached:
                logger.info("Dùng lại %d cues gán bằng chữ từ checkpoint.", len(cached))
                return replace(request, context=replace(
                    request.context, visual_cues=json.dumps(cached, ensure_ascii=False)))
        analyze = getattr(self._translator, "analyze_text_cues", None)
        if analyze is None:
            return request
        models = list(dict.fromkeys(
            m for m in (request.text_cues_model, request.visual_cues_model) if m))
        if progress_cb is not None:
            progress_cb(0.0, "Gán người nói / người nghe từ phụ đề")
        for model in models:
            try:
                cues = analyze(
                    source_lines, request.context.target_lang, model_name=model,
                    characters=request.context.characters or "", cancel_cb=cancel_cb,
                    **self._voice_kwargs(request.voice_source_path, request.context.source_lang),
                )
            except TranslationCancelledError:
                if self._user_cancelled(cancel_cb):
                    raise
                continue
            except SubtitleTranslationError as exc:
                logger.warning(
                    "Gán người nói bằng chữ thất bại với '%s' — %s", model, exc)
                continue
            if not cues:
                continue
            cues = canonicalize_cue_labels(cues, request.context.characters or "")
            serialized = serialize_visual_cues(cues)
            if checkpoint is not None:
                checkpoint.save_visual_cues(json.loads(serialized))
            logger.info(
                "Đã gán người nói/người nghe từ phụ đề cho %d/%d dòng (model '%s').",
                sum(1 for c in cues if c.speaker), len(source_lines), model)
            return replace(request, context=replace(request.context, visual_cues=serialized))
        return request

    @staticmethod
    def _fix_vocative_cues(
        request: TranslateSubtitlesRequest, source_lines: list[TranslationLine]
    ) -> TranslateSubtitlesRequest:
        """[v3.23.570] Sửa nhãn cues sai ở dòng gọi tên (câu gốc chỉ là một tên).

        Ca thật: dòng 2532 gốc ``柳慕云`` có nhãn ``spk=柳慕云, to=沈明月`` — model tin
        nhãn, dịch tiếng gọi thành "Thẩm Minh Nguyệt,". [v3.23.571] Dòng ``宋钱弟弟`` có
        nhãn người nói gọi Tống Tiền là "anh" → cả lượt lời ra "Anh Tống Tiền, anh đi
        trước đi!". Xem :mod:`subtitles_extractor.application.services.vocative_guard`.

        Args:
            request: Yêu cầu dịch.
            source_lines: Câu gốc (số dòng = ``id`` của cues).

        Returns:
            Yêu cầu mới với cues đã sửa; chính ``request`` nếu không có gì để sửa.
        """
        raw = (request.context.visual_cues or "").strip()
        characters = request.context.characters or ""
        if not raw or not characters:
            return request
        from subtitles_extractor.application.services.visual_cue_serializer import (
            deserialize_visual_cues,
            serialize_visual_cues,
        )
        from subtitles_extractor.application.services.vocative_guard import (
            fix_vocative_cue_labels,
        )

        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        original_cues = deserialize_visual_cues(raw)
        fixes = fix_vocative_cue_labels(original_cues, sources, characters)
        if not fixes.changed:
            return request
        from subtitles_extractor.application.services.visual_cue_serializer import keep_user_cues

        # [v3.23.653] Người nói người dùng chọn (vd sửa ở trang TTS) không bị lưới tiếng gọi đảo lại.
        kept = keep_user_cues(fixes.cues, original_cues)
        if kept == sorted(original_cues, key=lambda cue: cue.line_no):
            return request
        fixes = replace(fixes, cues=kept)
        if fixes.kin_relabeled:
            logger.info(
                "Đổi người nói ở %d tiếng gọi kèm vai vế có nhãn trái vai: %s",
                len(fixes.kin_relabeled),
                ", ".join(str(i) for i in fixes.kin_relabeled[:10]),
            )
        if fixes.name_answers:
            logger.info(
                "Sửa người nói ở %d dòng TRẢ LỜI tên (sau câu hỏi tên / lời tự giới "
                "thiệu) đang ghi là lời người khác: %s",
                len(fixes.name_answers),
                ", ".join(str(i) for i in fixes.name_answers[:10]),
            )
        if fixes.swapped:
            logger.info(
                "Đảo nhãn người nói/người nghe ở %d dòng gọi tên bị gán ngược: %s",
                len(fixes.swapped), ", ".join(str(i) for i in fixes.swapped[:10]),
            )
        new_context = replace(
            request.context, visual_cues=serialize_visual_cues(fixes.cues)
        )
        return replace(request, context=new_context)

    @staticmethod
    def _enforce_vocative_names(
        stage_output: list[TranslationLine],
        source_lines: list[TranslationLine],
        context: TranslationContext,
        stage_label: str,
    ) -> list[TranslationLine]:
        """[v3.23.570] Dòng gọi tên X bị dịch thành tên người khác → sửa về tên X.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            source_lines: Câu gốc.
            context: Ngữ cảnh (bảng nhân vật).
            stage_label: Nhãn giai đoạn (để log).

        Returns:
            Danh sách dòng đã sửa; chính ``stage_output`` nếu không có gì để sửa.
        """
        characters = context.characters or ""
        if not characters:
            return stage_output
        from subtitles_extractor.application.services.vocative_guard import (
            find_vocative_name_fixes,
        )

        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        translations = {ln.index: (ln.text or "") for ln in stage_output}
        fixes = find_vocative_name_fixes(sources, translations, characters)
        if not fixes:
            return stage_output
        logger.info(
            "%s: sửa tên ở %d dòng gọi tên: %s",
            stage_label, len(fixes),
            ", ".join(
                f"{fix.index} ({fix.wrong_name} \u2192 {fix.right_name})"
                for fix in fixes[:8]
            ),
        )
        _record_guard_event(
            "vocative_names_fixed",
            label=stage_label,
            fixes=[
                {"index": fix.index, "before": fix.wrong_name, "after": fix.right_name}
                for fix in fixes
            ],
        )
        patched = {fix.index: fix.text_after for fix in fixes}
        return [
            replace(line, text=patched[line.index]) if line.index in patched else line
            for line in stage_output
        ]

    @staticmethod
    def _strip_glossary_notes(
        stage_output: list[TranslationLine],
        source_lines: list[TranslationLine],
        glossary: str,
        stage_label: str,
    ) -> list[TranslationLine]:
        """[v3.23.586] Gỡ chú thích của bảng thuật ngữ bị chép vào lời thoại.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            source_lines: Câu gốc.
            glossary: Bảng thuật ngữ của lượt dịch.
            stage_label: Nhãn giai đoạn (để log và ghi nhật ký diễn biến).

        Returns:
            Danh sách dòng đã gỡ; chính ``stage_output`` nếu không có gì để gỡ.
        """
        from subtitles_extractor.application.services.glossary_note_guard import (
            find_note_leaks,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        translations = {ln.index: (ln.text or "") for ln in stage_output}
        leaks = find_note_leaks(sources, translations, glossary)
        if not leaks:
            return stage_output
        logger.info(
            "%s: gỡ chú thích thuật ngữ bị chép vào lời thoại ở %d dòng: %s",
            stage_label, len(leaks),
            ", ".join(f"{leak.index} ({'; '.join(leak.notes)})" for leak in leaks[:8]),
        )
        record_event(
            "glossary_notes_stripped",
            label=stage_label,
            lines=[leak.index for leak in leaks],
        )
        patched = {leak.index: leak.text_after for leak in leaks}
        return [
            replace(line, text=patched[line.index]) if line.index in patched else line
            for line in stage_output
        ]

    @staticmethod
    def _record_stage_shift(stage_label: str, shift: Any, verdict: str) -> None:
        """[v3.23.569] Ghi dải lệch và phán quyết vào nhật ký diễn biến dịch."""
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        record_event(
            "stage_rewrite" if shift.is_rewrite else "stage_shift",
            label=stage_label,
            from_index=shift.from_index,
            to_index=shift.to_index,
            offset=shift.offset,
            verdict=verdict,
        )

    @staticmethod
    def _enforce_number_consistency(
        stage_output: list[TranslationLine],
        source_lines: list[TranslationLine],
        stage_label: str,
        previous_lines: list[TranslationLine] | None = None,
    ) -> list[TranslationLine]:
        """[v3.23.556] Sửa con số dịch SAI GIÁ TRỊ hoặc lệch lối đọc so với chính bộ phim.

        Ca thật v3.23.555: ``百万年薪`` (lương năm MỘT triệu) ra "lương năm triệu tệ" —
        người nghe hiểu thành 5 triệu; cùng ``一千万`` khi thì "mười triệu", khi thì
        "một nghìn vạn".
        Xem :mod:`subtitles_extractor.application.services.number_consistency_guard`.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            source_lines: Câu gốc (để tính giá trị con số).
            stage_label: Nhãn giai đoạn (để log).

        Returns:
            Danh sách dòng đã chuẩn hoá số; chính ``stage_output`` nếu không có gì để sửa.
        """
        from subtitles_extractor.application.services.number_consistency_guard import (
            find_number_fixes,
        )

        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        translations = {ln.index: (ln.text or "") for ln in stage_output}
        # [v3.23.569] Cách đọc của bản TRƯỚC cũng là cách đọc của chính bộ phim: ca thật,
        # dịch thô đọc đúng "mười triệu" cả 6 lần, tinh chỉnh đổi thành "một nghìn vạn"
        # (4) và "một triệu" (2, SAI GIÁ TRỊ) — không còn bản đúng nào để lưới chọn theo.
        prior = (
            {ln.index: (ln.text or "") for ln in previous_lines}
            if previous_lines is not None
            else None
        )
        fixes = find_number_fixes(sources, translations, prior_translations=prior)
        if not fixes:
            return stage_output
        patched = {fix.index: fix.text_after for fix in fixes}
        logger.info(
            "%s: chuẩn hoá con số ở %d dòng: %s",
            stage_label,
            len(fixes),
            ", ".join(
                f"{fix.index} ({fix.reason}: {fix.phrase_before} → {fix.phrase_after})"
                for fix in fixes[:8]
            ),
        )
        # [v3.23.588] Vào file chuẩn đoán: lần lưới số tự đổi 5 dòng ĐÚNG sang lối đọc
        # khác (v3.23.587) chỉ lộ ra nhờ nhật ký ứng dụng, file chuẩn đoán không có dấu.
        _record_guard_event(
            "numbers_normalised",
            label=stage_label,
            fixes=[
                {
                    "index": fix.index,
                    "reason": fix.reason,
                    "before": fix.phrase_before,
                    "after": fix.phrase_after,
                }
                for fix in fixes
            ],
        )
        return [
            replace(line, text=patched[line.index]) if line.index in patched else line
            for line in stage_output
        ]

    @staticmethod
    def _enforce_address_rules(
        stage_output: list[TranslationLine],
        source_lines: list[TranslationLine],
        context: TranslationContext,
        stage_label: str,
    ) -> list[TranslationLine]:
        """[v3.23.554] Ép đại từ tự xưng về đúng quy tắc trong bảng nhân vật.

        Chuẩn đoán thật v3.23.553: 51 dòng xưng sai quy tắc, 50 trong số đó sai NGAY Ở
        GIAI ĐOẠN DỊCH THÔ — gắn quy tắc vào lô dịch (v3.23.551) đã tới hạn, phải ép
        bằng mã. Lưới cố ý sửa ÍT mà CHẮC; xem
        :mod:`subtitles_extractor.application.services.address_rule_guard`.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            source_lines: Câu gốc (để xác định ngôi nhân xưng trong nguyên văn).
            context: Ngữ cảnh dịch (cần ``characters`` và ``visual_cues``).
            stage_label: Nhãn giai đoạn (để log).

        Returns:
            Danh sách dòng đã ép xưng hô; chính ``stage_output`` nếu không có gì để sửa.
        """
        characters = str(getattr(context, "characters", "") or "")
        visual_cues = str(getattr(context, "visual_cues", "") or "")
        if not characters.strip() or not visual_cues.strip():
            return stage_output
        from subtitles_extractor.application.services.address_rule_guard import (
            find_address_rule_fixes,
        )

        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        translations = {ln.index: (ln.text or "") for ln in stage_output}
        fixes = find_address_rule_fixes(sources, translations, visual_cues, characters)
        if not fixes:
            return stage_output
        patched = {fix.index: fix.text_after for fix in fixes}
        logger.info(
            "%s: ép xưng hô theo bảng nhân vật ở %d dòng: %s",
            stage_label,
            len(fixes),
            ", ".join(
                f"{fix.index} ({fix.pronoun_found}→{fix.pronoun_expected})"
                for fix in fixes[:8]
            ),
        )
        _record_guard_event(
            "address_rules_enforced",
            label=stage_label,
            fixes=[
                {
                    "index": fix.index,
                    "before": fix.pronoun_found,
                    "after": fix.pronoun_expected,
                }
                for fix in fixes
            ],
        )
        return [
            replace(line, text=patched[line.index]) if line.index in patched else line
            for line in stage_output
        ]

    @staticmethod
    def _stage_resume(
        checkpoint: StageCheckpoint | None,
        kind: TranslationStageKind,
        input_lines: list[TranslationLine],
        stage_label: str,
        run: StageRun | None = None,
    ) -> _StageResumePlan:
        """Phần dùng lại được của một giai đoạn sắp dịch + callback lưu tiến độ từng lô.

        Hai nguồn, dòng của lần chạy dở thắng khi trùng chỉ số (mới hơn):

        * [v3.23.584] Lô đã dịch ở lần chạy DỞ của giai đoạn (cùng đầu vào).
        * [v3.23.608] Dòng của bản đã lưu mà đầu vào dòng đó không đổi — người dùng sửa
          vài dòng giai đoạn trước thì giai đoạn này chỉ dịch lại các lô có dòng đổi.

        Args:
            checkpoint: Bộ nhớ đệm của lượt dịch; ``None`` = không lưu tiến độ.
            kind: Giai đoạn sắp chạy.
            input_lines: Đầu vào của giai đoạn.
            stage_label: Nhãn giai đoạn (để log).
            run: [v3.23.606] Cấu hình và model của lần chạy này — chỉ dùng lại khi cùng
                cấu hình và dịch bằng model mạnh hơn hoặc ngang.

        Returns:
            Kế hoạch tiếp tục (dòng dùng lại, callback lưu lô, số dòng theo nguồn).
        """
        if checkpoint is None:
            return _StageResumePlan(None, None, 0, 0)
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        fingerprint = stage_input_fingerprint(kind, input_lines)
        partial_model = checkpoint.partial_model(kind, fingerprint)
        resumed = checkpoint.get_partial(kind, fingerprint, run) or []
        carried = checkpoint.carry_over(kind, run, input_lines) if run is not None else []
        merged = {line.index: line for line in carried}
        merged.update({line.index: line for line in resumed})
        if resumed:
            logger.info(
                "%s: tiếp tục lần chạy dở — đã có %d/%d dòng, chỉ dịch phần còn lại.",
                stage_label, len(resumed), len(input_lines),
            )
            if run is not None and partial_model and partial_model != run.model_name:
                logger.info(
                    "%s: các lô đã có được dịch bằng '%s' (mạnh hơn hoặc ngang '%s' đang "
                    "chọn) — giữ, chỉ dịch phần còn lại bằng '%s'.",
                    stage_label, partial_model, run.model_name, run.model_name,
                )
            record_event(
                "stage_resumed_partial",
                label=stage_label,
                lines=len(resumed),
                total=len(input_lines),
            )
        carried_only = len(merged) - len(resumed)
        if carried_only:
            logger.info(
                "%s: đầu vào đổi %d/%d dòng so với bản đã lưu → giữ %d dòng đã dịch, chỉ "
                "dịch lại các lô có dòng đổi.",
                stage_label, len(input_lines) - len(carried), len(input_lines),
                carried_only,
            )
            record_event(
                "stage_carried_over",
                label=stage_label,
                kept=carried_only,
                total=len(input_lines),
            )

        def _save_batch(lines: list[TranslationLine]) -> None:
            checkpoint.add_partial(kind, fingerprint, lines, run)

        resume_lines = [merged[index] for index in sorted(merged)] or None
        return _StageResumePlan(resume_lines, _save_batch, carried_only, len(resumed))

    @staticmethod
    def _reusable_stage(
        checkpoint: StageCheckpoint,
        stage: TranslationStageConfig,
        run: StageRun,
        stage_label: str,
        *,
        chain_intact: bool,
    ) -> list[TranslationLine] | None:
        """[v3.23.606] Kết quả đã lưu của giai đoạn nếu dùng lại NGUYÊN được (log lý do).

        Args:
            checkpoint: Bộ nhớ đệm của lượt dịch.
            stage: Cấu hình giai đoạn sắp chạy.
            run: Đầu vào, cấu hình và model của lần chạy này.
            stage_label: Nhãn giai đoạn (để log).
            chain_intact: Mọi giai đoạn trước đều dùng lại kết quả đã lưu.

        Returns:
            Các dòng đã lưu, hoặc None khi phải dịch giai đoạn này.
        """
        verdict = checkpoint.reuse_verdict(stage.kind, run, chain_intact=chain_intact)
        stored = checkpoint.stored_run(stage.kind)
        if verdict is ReuseVerdict.REUSE:
            if stored is not None and stored.model_name != stage.model_name:
                logger.info(
                    "%s: dùng lại bản đã dịch bằng '%s' (mạnh hơn hoặc ngang '%s' "
                    "đang chọn) thay vì dịch lại. Muốn dịch lại bằng model đang chọn: "
                    "xoá giai đoạn này trong 'Quản lý giai đoạn'.",
                    stage_label, stored.model_name, stage.model_name,
                )
                _record_guard_event(
                    "stage_reused_other_model",
                    label=stage_label,
                    stored_model=stored.model_name,
                    requested_model=stage.model_name,
                )
            return checkpoint.get_stage(stage.kind)
        if verdict is ReuseVerdict.WEAKER_MODEL and stored is not None:
            logger.info(
                "%s: bản đã lưu dịch bằng '%s' — yếu hơn '%s' đang chọn → dịch lại bằng "
                "model mới.", stage_label, stored.model_name, stage.model_name,
            )
        elif verdict is ReuseVerdict.CONFIG_CHANGED:
            logger.info(
                "%s: cấu hình giai đoạn đã đổi so với bản đã lưu → dịch lại.", stage_label
            )
        elif verdict is ReuseVerdict.INPUT_CHANGED:
            logger.info(
                "%s: đầu vào khác bản đã lưu (giai đoạn trước vừa được sửa hoặc dịch "
                "lại).", stage_label,
            )
        return None

    def _prune_cache(self, keep_key: str) -> None:
        """[v3.23.608] Dọn bộ nhớ đệm cũ (giữ lượt đang chạy) theo số lượng và tuổi."""
        if self._checkpoint_dir is None:
            return
        try:
            prune_stage_checkpoints(
                self._checkpoint_dir,
                max_files=self._cache_max_files,
                max_age_days=self._cache_max_age_days,
                keep={keep_key},
            )
        except OSError as exc:
            logger.warning("Không dọn được bộ nhớ đệm dịch cũ: %s", exc)

    @staticmethod
    def _restore_address_regressions(
        stage_output: list[TranslationLine],
        input_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        context: TranslationContext,
        stage_label: str,
    ) -> list[TranslationLine]:
        """[v3.23.583] Khôi phục đại từ mà giai đoạn tinh chỉnh làm hỏng (bản trước đúng).

        Chỉ thay đúng MỘT từ, giữ văn phong mới của câu. Xem
        :mod:`subtitles_extractor.application.services.address_regression_guard`.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            input_lines: Đầu vào của giai đoạn (kết quả giai đoạn trước).
            source_lines: Câu gốc (ngôi nhân xưng quyết định vai của đại từ).
            context: Ngữ cảnh dịch (cần ``characters`` và ``visual_cues``).
            stage_label: Nhãn giai đoạn (để log).

        Returns:
            Danh sách dòng đã khôi phục; chính ``stage_output`` nếu không có gì để sửa.
        """
        characters = str(getattr(context, "characters", "") or "")
        visual_cues = str(getattr(context, "visual_cues", "") or "")
        if not characters.strip() or not visual_cues.strip():
            return stage_output
        from subtitles_extractor.application.services.address_regression_guard import (
            find_address_regressions,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        previous = {ln.index: (ln.text or "") for ln in input_lines}
        current = {ln.index: (ln.text or "") for ln in stage_output}
        regressions = find_address_regressions(
            sources, previous, current, visual_cues, characters
        )
        if not regressions:
            return stage_output
        logger.warning(
            "%s: %d dòng bị ĐỔI XƯNG HÔ khỏi cách đúng bảng nhân vật mà giai đoạn trước "
            "đã dùng → khôi phục đại từ: %s",
            stage_label,
            len(regressions),
            ", ".join(
                f"{item.index} ("
                + ", ".join(f"{wrong}→{right}" for wrong, right in item.swaps)
                + ")"
                for item in regressions[:8]
            ),
        )
        record_event(
            "address_regression_restored",
            label=stage_label,
            lines=[item.index for item in regressions],
        )
        patched = {item.index: item.text_after for item in regressions}
        return [
            replace(line, text=patched[line.index]) if line.index in patched else line
            for line in stage_output
        ]

    @staticmethod
    def _revert_name_regressions(
        stage_output: list[TranslationLine],
        input_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        glossary: str,
        stage_label: str,
    ) -> list[TranslationLine]:
        """[v3.23.552] Lùi các dòng bị giai đoạn tinh chỉnh làm MẤT tên riêng về bản trước.

        Chỉ áp cho giai đoạn có bản dịch trước (không áp cho dịch thô — đầu vào của nó là nguồn).
        Chỉ xét mục thuật ngữ loại TÊN RIÊNG; thành ngữ được phép diễn đạt lại. Xem
        :mod:`subtitles_extractor.application.services.name_regression_guard`.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            input_lines: Đầu vào của giai đoạn (kết quả giai đoạn trước).
            source_lines: Câu gốc.
            glossary: Bảng thuật ngữ.
            stage_label: Nhãn giai đoạn (để log).

        Returns:
            Danh sách dòng đã khôi phục các ca mất tên riêng.
        """
        if not glossary.strip():
            return stage_output
        from subtitles_extractor.application.services.name_regression_guard import (
            find_name_regressions,
        )

        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        previous = {ln.index: (ln.text or "") for ln in input_lines}
        current = {ln.index: (ln.text or "") for ln in stage_output}
        regressions = dict(find_name_regressions(sources, previous, current, glossary))
        if not regressions:
            return stage_output
        logger.warning(
            "%s: %d dòng bị LÀM MẤT tên riêng đã dịch đúng ở giai đoạn trước → đã lùi về bản "
            "trước: %s",
            stage_label, len(regressions),
            ", ".join(f"{index} ({term})" for index, term in list(regressions.items())[:8]),
        )
        _record_guard_event(
            "name_regressions_reverted", label=stage_label, lines=sorted(regressions)
        )
        return [
            replace(line, text=previous[line.index])
            if line.index in regressions and line.index in previous
            else line
            for line in stage_output
        ]

    @staticmethod
    def _normalize_roster_register(request: TranslateSubtitlesRequest) -> TranslateSubtitlesRequest:
        """[v3.23.637] Sửa tất định bảng xưng hô LƯU TỪ BẢN CŨ sai thế giới phim.

        Phân tích đã lưu được dùng lại ở mọi lần dịch sau; bảng sinh trước khi prompt dặn dùng
        đại từ cổ phong còn ``tôi``/``cậu``/``cháu`` trong phim tu tiên (tệp chuẩn đoán dịch 86
        tập: 7 mệnh đề). Xem :mod:`subtitles_extractor.application.services.roster_register`.
        """
        from subtitles_extractor.application.services.roster_register import (
            normalize_roster_register,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        context = request.context
        characters, fixes = normalize_roster_register(
            context.characters or "", context.overview or "")
        if not fixes:
            return request
        logger.info(
            "Sửa %d mệnh đề xưng hô trong bảng nhân vật đã lưu (đại từ hiện đại trong phim cổ "
            "phong / người dưới xưng 'ta' với bậc trên).", fixes,
        )
        record_event("roster_register_fixed", clauses=fixes)
        return replace(request, context=replace(context, characters=characters))

    @staticmethod
    def _warn_missing_address_rules(characters: str) -> None:
        """[v3.23.578] Báo khi bảng nhân vật không có quy tắc xưng hô nào.

        Không tự phân tích lại (tốn lượt gọi video và có thể đè bảng người dùng đã sửa) —
        chỉ ghi cảnh báo và sự kiện ``roster_without_address_rules`` vào nhật ký để file
        chuẩn đoán nói rõ vì sao mọi số liệu ``address_rule_*`` bằng 0. Xem
        ``roster_format.lacks_address_rules``.

        Args:
            characters: Bảng nhân vật của lần dịch.
        """
        from subtitles_extractor.application.services.roster_format import (
            lacks_address_rules,
            parse_roster,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        if not lacks_address_rules(characters):
            return
        count = len(parse_roster(characters))
        logger.warning(
            "Bảng nhân vật (%d người) KHÔNG có quy tắc xưng hô nào — thường do dùng "
            "lại phân tích từ phiên bản cũ. Khâu dịch sẽ tự chọn xưng hô từng lô (dễ "
            "đổi giữa chừng); nên bấm Phân tích lại để chốt xưng hô.",
            count,
        )
        record_event("roster_without_address_rules", characters=count)

    @staticmethod
    def _separate_monologues(
        source_lines: list[TranslationLine], characters: str
    ) -> list[TranslationLine]:
        """[v3.23.577] Tách độc thoại khỏi nhãn "người nói = người nghe" bị gán nhầm.

        Ca thật: khâu dịch nhận ``speaker: 宋钱, addressee: 宋钱`` ở dòng nghĩ thầm và
        dịch "Nhưng em có truyền thừa của Thần Tiên tỷ tỷ" — mang cách xưng của câu thoại
        trước. Dòng độc thoại giữ nguyên nhãn (khâu dịch tự gắn quy tắc "không có người
        nghe"); lượt nhãn tự-nói mà có ``你``/``我们``/tiếng gọi tên là nói với NGƯỜI
        KHÁC — bỏ nhãn người nghe để không bị coi là độc thoại. Xem
        :mod:`subtitles_extractor.application.services.monologue_guard`.

        Args:
            source_lines: Dòng nguồn đã gắn nhãn cues.
            characters: Bảng nhân vật (để nhận tiếng gọi tên).

        Returns:
            Danh sách dòng mới; chính ``source_lines`` nếu không có nhãn tự-nói nào.
        """
        from subtitles_extractor.application.services.monologue_guard import (
            self_addressed_runs,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        cues = [
            VisualCue(line_no=ln.index, speaker=ln.speaker, addressee=ln.addressee)
            for ln in source_lines
            if ln.speaker
        ]
        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        runs = self_addressed_runs(cues, sources, characters)
        if not runs:
            return source_lines
        monologue = sum(len(run.lines) for run in runs if run.monologue)
        misread = {index for run in runs if not run.monologue for index in run.lines}
        logger.info(
            "Nhãn tự-nói: %d dòng độc thoại (báo khâu dịch không có người nghe); %d dòng "
            "thực ra nói với người khác → bỏ nhãn người nghe.",
            monologue, len(misread),
        )
        record_event("monologue_lines", monologue=monologue, cleared=sorted(misread))
        if not misread:
            return source_lines
        return [
            replace(line, addressee="") if line.index in misread else line
            for line in source_lines
        ]

    @staticmethod
    def _revert_echo_regressions(
        stage_output: list[TranslationLine],
        input_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        stage_label: str,
    ) -> list[TranslationLine]:
        """[v3.23.576] Lùi dòng mà giai đoạn tinh chỉnh làm NÓI LẶP dòng kề về bản trước.

        Ca thật: bản dịch thô tách đúng "Ông nội, cháu gái" / "biết sai rồi", giai đoạn
        tinh chỉnh gộp thành "Ông nội, cháu gái biết sai rồi." / "Biết sai rồi." — giọng
        thuyết minh đọc "biết sai rồi" hai lần. Xem
        :mod:`subtitles_extractor.application.services.line_echo_guard`.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            input_lines: Đầu vào của giai đoạn (kết quả giai đoạn trước).
            source_lines: Câu gốc.
            stage_label: Nhãn giai đoạn (để log).

        Returns:
            Danh sách dòng đã lùi; chính ``stage_output`` nếu không có gì để lùi.
        """
        from subtitles_extractor.application.services.line_echo_guard import (
            find_echo_reverts,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        previous = {ln.index: (ln.text or "") for ln in input_lines}
        current = {ln.index: (ln.text or "") for ln in stage_output}
        reverts = find_echo_reverts(sources, previous, current)
        if not reverts:
            return stage_output
        logger.warning(
            "%s: %d dòng bị gộp câu của dòng kề rồi NÓI LẶP → lùi về bản trước: %s",
            stage_label, len(reverts), ", ".join(str(index) for index in sorted(reverts)),
        )
        record_event("echo_revert", label=stage_label, lines=sorted(reverts))
        return [
            replace(line, text=reverts[line.index]) if line.index in reverts else line
            for line in stage_output
        ]

    @staticmethod
    def _revert_unspoken_regressions(
        stage_output: list[TranslationLine],
        input_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        stage_label: str,
    ) -> list[TranslationLine]:
        """[v3.23.605] Lùi dòng thoại bị giai đoạn tinh chỉnh làm mất hết lời.

        Xem :mod:`subtitles_extractor.application.services.spoken_line_guard`.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            input_lines: Đầu vào của giai đoạn (kết quả giai đoạn trước).
            source_lines: Câu gốc.
            stage_label: Nhãn giai đoạn (để log).

        Returns:
            Danh sách dòng đã lùi; chính ``stage_output`` nếu không có gì để lùi.
        """
        from subtitles_extractor.application.services.spoken_line_guard import (
            find_unspoken_regressions,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        previous = {ln.index: (ln.text or "") for ln in input_lines}
        current = {ln.index: (ln.text or "") for ln in stage_output}
        lost = set(find_unspoken_regressions(sources, previous, current))
        if not lost:
            return stage_output
        logger.warning(
            "%s: %d dòng thoại bị thay bằng tag/chú thích không lời (%s) — phụ đề lẫn "
            "lồng tiếng sẽ mất dòng → lùi về bản trước.",
            stage_label, len(lost),
            ", ".join(f"{index} '{current[index]}'" for index in sorted(lost)),
        )
        record_event("unspoken_revert", label=stage_label, lines=sorted(lost))
        return [
            replace(line, text=previous[line.index]) if line.index in lost else line
            for line in stage_output
        ]

    def _revert_padding_regressions(
        self,
        stage_output: list[TranslationLine],
        input_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        stage_label: str,
    ) -> list[TranslationLine]:
        """[v3.23.605] Lùi dòng chỉ chèn thêm từ đệm mà vượt ngân sách đọc về bản trước.

        Ngân sách lấy từ chính bộ dịch (cùng ``max_chars`` đã gửi model); bộ dịch không có
        năng lực này thì bỏ qua. Xem
        :mod:`subtitles_extractor.application.services.padding_regression_guard`.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            input_lines: Đầu vào của giai đoạn (kết quả giai đoạn trước).
            source_lines: Câu gốc (mốc thời gian để tính ngân sách).
            stage_label: Nhãn giai đoạn (để log).

        Returns:
            Danh sách dòng đã lùi; chính ``stage_output`` nếu không có gì để lùi.
        """
        if not isinstance(self._translator, LineCharBudgetProvider):
            return stage_output
        from subtitles_extractor.application.services.padding_regression_guard import (
            find_padding_regressions,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        budgets = self._translator.line_char_budgets(source_lines)
        previous = {ln.index: (ln.text or "") for ln in input_lines}
        current = {ln.index: (ln.text or "") for ln in stage_output}
        padded = set(find_padding_regressions(previous, current, budgets))
        if not padded:
            return stage_output
        logger.info(
            "%s: %d dòng chỉ chèn thêm từ đệm mà vượt ngân sách đọc → giữ bản trước "
            "(giọng đọc khỏi nén): %s",
            stage_label, len(padded),
            ", ".join(str(index) for index in sorted(padded)),
        )
        record_event("padding_revert", label=stage_label, lines=sorted(padded))
        return [
            replace(line, text=previous[line.index]) if line.index in padded else line
            for line in stage_output
        ]

    def _revert_person_flips(
        self,
        stage_output: list[TranslationLine],
        input_lines: list[TranslationLine],
        source_lines: list[TranslationLine],
        stage_label: str,
    ) -> list[TranslationLine]:
        """[v3.23.633] Lùi dòng bị giai đoạn tinh chỉnh ĐẢO NGÔI (你 → "ta") về bản trước.

        Xem :mod:`subtitles_extractor.application.services.person_flip_guard`.

        Args:
            stage_output: Kết quả giai đoạn vừa chạy.
            input_lines: Đầu vào của giai đoạn (kết quả giai đoạn trước).
            source_lines: Câu gốc (chuẩn để xác định ngôi).
            stage_label: Nhãn giai đoạn (để log).

        Returns:
            Danh sách dòng đã lùi; chính ``stage_output`` nếu không có gì để lùi.
        """
        from subtitles_extractor.application.services.person_flip_guard import (
            find_person_flips,
        )
        from subtitles_extractor.application.services.translation_process_log import (
            record_event,
        )

        sources = {ln.index: (ln.original_text or ln.text or "") for ln in source_lines}
        previous = {ln.index: (ln.text or "") for ln in input_lines}
        current = {ln.index: (ln.text or "") for ln in stage_output}
        flipped = find_person_flips(sources, previous, current)
        if not flipped:
            return stage_output
        logger.info(
            "%s: %d dòng bị ĐẢO NGÔI (câu gốc nói với người nghe nhưng bản mới thành "
            "ta/tôi) → giữ bản trước: %s",
            stage_label, len(flipped), ", ".join(str(index) for index in sorted(flipped)),
        )
        record_event("person_flip_revert", label=stage_label, lines=sorted(flipped))
        return [
            replace(line, text=previous[line.index]) if line.index in flipped else line
            for line in stage_output
        ]

    def _revert_cjk_regressions(
        self,
        stage_output: list[TranslationLine],
        input_lines: list[TranslationLine],
        stage_label: str,
    ) -> list[TranslationLine]:
        """[v3.23.432] Lùi các dòng bị giai đoạn tinh chỉnh CHÈN CJK về bản trước đã sạch.

        Với mỗi dòng: nếu bản INPUT (giai đoạn trước) KHÔNG có chữ CJK mà OUTPUT lại CÓ → giai
        đoạn này đã chèn chữ Hán/Nhật/Hàn vào câu vốn sạch (vd 'tinh'→'星') → khôi phục dòng đó
        về bản input. Giai đoạn LITERAL không bị ảnh hưởng vì input của nó (nguồn) vốn đầy CJK
        nên điều kiện "input sạch" không thoả.

        Args:
            stage_output: Kết quả giai đoạn hiện tại.
            input_lines: Đầu vào của giai đoạn (kết quả giai đoạn trước).
            stage_label: Nhãn giai đoạn (để log).

        Returns:
            Danh sách dòng đã khôi phục các ca chèn CJK (giữ nguyên phần còn lại).
        """
        input_text_by_index = {ln.index: ln.text for ln in input_lines}
        reverted: list[int] = []
        emptied: list[int] = []
        result: list[TranslationLine] = []
        for line in stage_output:
            prior_text = input_text_by_index.get(line.index)
            current_text = line.text or ""
            if (
                prior_text is not None
                and _CJK_CHAR_RE.search(current_text)
                and not _CJK_CHAR_RE.search(prior_text)
            ):
                result.append(replace(line, text=prior_text))
                reverted.append(line.index)
            elif prior_text is not None and prior_text.strip() and not current_text.strip():
                # [v3.23.436] Guard chống LÀM-RỖNG: giai đoạn tinh chỉnh làm rỗng một dòng VỐN
                # có nội dung (AI misalignment/bỏ dòng — E01#144: 'Sao ông già này…' → ''). Lùi
                # về bản trước để không mất nội dung. Không áp cho LITERAL vì input là nguồn.
                result.append(replace(line, text=prior_text))
                emptied.append(line.index)
            else:
                result.append(line)
        if emptied:
            logger.warning(
                "%s: %d dòng bị LÀM RỖNG (vốn có nội dung) → đã lùi về bản trước "
                "(chống mất nội dung do AI bỏ dòng/lệch dòng).",
                stage_label, len(emptied),
            )
            _record_guard_event(
                "emptied_lines_reverted", label=stage_label, lines=emptied
            )
        if reverted:
            logger.warning(
                "%s: %d dòng bị chèn chữ CJK vào câu đã sạch → đã lùi về bản trước "
                "(chống revert Hán-Việt→chữ Hán).",
                stage_label, len(reverted),
            )
            _record_guard_event(
                "cjk_regressions_reverted", label=stage_label, lines=reverted
            )
        return result

    def execute(
        self,
        request: TranslateSubtitlesRequest,
        progress_cb: OverallProgressCallback | None = None,
        cancel_cb: CancellationCallback | None = None,
    ) -> TranslateSubtitlesResponse:
        """Chạy toàn bộ pipeline dịch, tự động resume từ checkpoint nếu có.

        Args:
            request:     Đầu vào (events, stages, context).
            progress_cb: Callback tiến độ tổng ``(0..1, mô tả)``.
            cancel_cb:   Callback kiểm tra huỷ; trả True để dừng sớm.

        Returns:
            ``TranslateSubtitlesResponse`` với danh sách event đã dịch.
        """
        if not request.events:
            raise SubtitleTranslationError("Không có phụ đề nào để dịch.")
        if not request.stages:
            raise SubtitleTranslationError("Chưa bật giai đoạn dịch nào.")

        # [v3.23.519] Xoá diễn biến của lần dịch TRƯỚC để file chuẩn đoán chỉ chứa lần này.
        # [v3.23.578] Xoá NGAY ĐẦU lần chạy: trước đây xoá sau các bước chuẩn bị nhãn nên
        # sự kiện của chúng ("monologue_lines") bị mất khỏi file chuẩn đoán mọi lần chạy.
        # [v3.23.579] Gắn nhật ký với phụ đề của lần chạy để lúc xuất chuẩn đoán không
        # đính nhầm diễn biến của phim khác.
        try:
            from subtitles_extractor.application.services.translation_process_log import (
                get_process_log,
                run_fingerprint,
            )

            get_process_log().start_run(
                run_fingerprint(event.text for event in request.events)
            )
        except Exception:  # noqa: BLE001 — nhật ký là việc phụ
            pass
        self._warn_missing_address_rules(request.context.characters or "")
        request = self._normalize_roster_register(request)

        # ── Khởi tạo checkpoint ──────────────────────────────────────────
        checkpoint: StageCheckpoint | None = None
        if self._checkpoint_dir is not None:
            scope = build_stage_cache_scope(request, self._checkpoint_dir)
            self._prune_cache(scope.key)
            checkpoint = StageCheckpoint(
                self._checkpoint_dir,
                scope.key,
                scope.context_fingerprint,
                meta={
                    "lines": len(request.events),
                    "texts": scope.identity.texts_fingerprint,
                    "target": request.context.target_lang,
                    "source": request.context.source_lang or "",
                    "visual_cues": request.enable_visual_cues,
                    # [v3.23.623] Chỉ thêm khoá khi bật → danh tính checkpoint cũ không đổi.
                    **({"text_cues": True} if request.text_cues else {}),
                },
            )
            adopt_previous_checkpoints(
                checkpoint, scope, context=scope.context_fingerprint
            )
            checkpoint.touch()

        source_lines = self._events_to_lines(request.events)

        # [v3.23.31] Visual Cues (tuỳ chọn): quét video MỘT LẦN để xác định ai nói/nói
        # với ai từng dòng, bơm vào speaker/addressee giúp dịch xưng hô đúng vai vế.
        # Chỉ chạy khi bật VÀ có video; lỗi/huỷ ở bước này KHÔNG làm hỏng dịch (bỏ qua).
        topped_up_cues = ""
        if request.text_cues and not request.video_refs:
            ensured = self._ensure_text_cues(
                source_lines, request, checkpoint, cancel_cb, progress_cb
            )
            if ensured is not request:
                request = ensured
                topped_up_cues = request.context.visual_cues
        if request.enable_visual_cues and request.video_refs:
            # [v3.23.559] Cues từ phân tích ngữ cảnh phủ quá thấp → quét bổ sung các dòng
            # còn thiếu, rồi dùng bản đã gộp cho MỌI khâu phía sau (kể cả lưới xưng hô, vốn
            # đọc ``request.context.visual_cues``).
            topped = self._top_up_context_cues(
                source_lines, request, checkpoint, cancel_cb, progress_cb
            )
            if topped is not request:
                request = topped
                topped_up_cues = request.context.visual_cues
        # [v3.23.581] Nhãn quét bổ sung "lạc cảnh" đã LƯU trong phiên (phiên cũ, trước khi
        # bước quét tự lọc) → bỏ nhãn, trước khi bước điền lượt lời dùng tới chúng.
        cleaned_request = self._drop_contradicted_scan_cues(request, source_lines)
        if cleaned_request is not request:
            request = cleaned_request
            topped_up_cues = request.context.visual_cues
        # [v3.23.572] Model chỉ ghi cue ở dòng ĐẦU lượt lời → điền nhãn cho dòng nói tiếp.
        # Chạy SAU bước quét bổ sung: cues quá thưa thì quét lại vẫn tốt hơn điền.
        filled_request = self._fill_cue_runs(request, source_lines)
        if filled_request is not request:
            request = filled_request
            topped_up_cues = request.context.visual_cues
        # [v3.23.600] Dòng có lời mà vẫn chưa có NGƯỜI NÓI → hỏi lại đúng các dòng đó,
        # để lồng tiếng có người nói cho mọi dòng (trước đây chỉ quét khi phủ < 60%).
        if request.enable_visual_cues and request.video_refs:
            speakers_request = self._complete_speaker_cues(
                source_lines, request, checkpoint, cancel_cb, progress_cb
            )
            if speakers_request is not request:
                request = speakers_request
                topped_up_cues = request.context.visual_cues
        # [v3.23.570] Dòng GỌI TÊN mà nhãn cues ghi ngược người nói/người nghe → đảo lại
        # TRƯỚC khi bơm nhãn vào dòng dịch và trước lưới xưng hô.
        fixed_request = self._fix_vocative_cues(request, source_lines)
        if fixed_request is not request:
            request = fixed_request
            topped_up_cues = request.context.visual_cues
        if self._cues_usable(request):
            source_lines = self._enrich_with_visual_cues(
                source_lines, request, checkpoint, progress_cb, cancel_cb
            )
            # [v3.23.577] Nhãn "người nói = người nghe": độc thoại thì giữ để khâu dịch
            # báo "không có người nghe"; lượt thực ra nói với người khác thì bỏ nhãn.
            source_lines = self._separate_monologues(
                source_lines, request.context.characters or ""
            )

        reference_lines = source_lines
        current_lines = source_lines
        completed: list[TranslationStageKind] = []
        stage_outputs: dict[TranslationStageKind, list[SubtitleEvent]] = {}
        has_prior_translation = False
        # [v3.23.504] Đánh dấu ĐÃ CÓ khâu hiệu đính phía trước, để giai đoạn dịch thô biết là
        # dòng nguồn nó nhận KHÔNG còn nguyên bản và cần đối chiếu với bản gốc thô.
        source_was_preprocessed = False
        total_stages = len(request.stages)
        resumed_from: TranslationStageKind | None = None
        _start_time = time.monotonic()
        # [v3.23.606] Mọi giai đoạn trước đều dùng lại kết quả đã lưu (chưa dịch lại gì).
        chain_intact = True
        # [v3.23.608] Để báo người dùng phần nào lấy từ bộ nhớ đệm.
        reused_stages: list[TranslationStageKind] = []
        carried_lines: dict[TranslationStageKind, int] = {}
        reapplied_edits: dict[TranslationStageKind, int] = {}

        for stage_position, stage in enumerate(request.stages):
            if cancel_cb is not None and cancel_cb():
                raise TranslationCancelledError("Người dùng đã huỷ tiến trình dịch.")

            stage_label = self._STAGE_LABELS.get(stage.kind, stage.kind.value)
            # [v3.23.653] Người nói NGƯỜI DÙNG chọn là một phần đầu vào của dòng (vân tay bộ nhớ đệm).
            from subtitles_extractor.application.services.visual_cue_serializer import tag_user_cues

            current_lines = tag_user_cues(current_lines, request.context.visual_cues)
            stage_run = StageRun.for_stage(
                stage, current_lines,
                attach_video=stage.kind in request.attach_video_stages,
            )

            # ── Thử resume từ checkpoint ─────────────────────────────────
            if checkpoint is not None:
                cached = self._reusable_stage(
                    checkpoint, stage, stage_run, stage_label, chain_intact=chain_intact
                )
                if cached is not None and len(cached) == len(source_lines):
                    logger.info("Resume checkpoint: bỏ qua %s (đã có %d dòng).",
                                stage_label, len(cached))
                    # [v3.23.592] Giai đoạn bị bỏ qua không sinh lại sự kiện nào → phát
                    # lại nhật ký đã lưu để file chuẩn đoán vẫn kể đủ cả lượt dịch.
                    self._replay_stage_events(checkpoint.get_stage_events(stage.kind))
                    stage_output = cached
                    reused_stages.append(stage.kind)
                    if resumed_from is None:
                        resumed_from = stage.kind
                    if progress_cb is not None:
                        overall = (stage_position + 1) / total_stages
                        progress_cb(overall, f"[Resume] {stage_label}")
                    # [v3.23.134] Giai đoạn RESUME cũng phải ghi vào stage_outputs như
                    # giai đoạn chạy thường — nếu không, màn "so sánh từng giai đoạn"
                    # (cần >=2 giai đoạn) sẽ THIẾU giai đoạn đã resume, mất khả năng đối
                    # chiếu trước/sau khi tiếp tục dịch sau khi huỷ/treo.
                    stage_outputs[stage.kind] = self._lines_to_events(
                        request.events, stage_output, request.context
                    )
                    current_lines = stage_output
                    completed.append(stage.kind)
                    if stage.kind is TranslationStageKind.PREPROCESS:
                        reference_lines = stage_output
                        source_was_preprocessed = True
                    else:
                        has_prior_translation = True
                    continue

            # ── Dịch bình thường ─────────────────────────────────────────
            chain_intact = False
            logger.info("Bắt đầu %s với model '%s'.", stage_label, stage.model_name)
            # [v3.23.592] Mốc nhật ký: sự kiện ghi từ đây thuộc về giai đoạn này.
            stage_event_mark = self._process_log_mark()

            def _stage_progress(
                fraction: float, _pos: int = stage_position, _label: str = stage_label
            ) -> None:
                if progress_cb is not None:
                    overall = (_pos + max(0.0, min(1.0, fraction))) / total_stages
                    progress_cb(overall, _label)

            resume_plan = self._stage_resume(
                checkpoint, stage.kind, current_lines, stage_label, stage_run
            )
            if resume_plan.carried:
                carried_lines[stage.kind] = resume_plan.carried
            stage_output = self._translator.translate_stage(
                stage=stage,
                context=request.context,
                source_lines=reference_lines,
                input_lines=current_lines,
                has_prior_translation=has_prior_translation,
                source_was_preprocessed=source_was_preprocessed,
                progress_cb=_stage_progress,
                cancel_cb=cancel_cb,
                video_refs=list(request.video_refs),
                attach_video=stage.kind in request.attach_video_stages,
                resume_lines=resume_plan.resume_lines,
                batch_done_cb=resume_plan.batch_done_cb,
            )

            # [v3.23.558] Sửa KHỐI lệch trên TOÀN BỘ đầu ra giai đoạn — chạy TRƯỚC các lưới
            # theo dòng để những lưới đó làm việc trên bản đã căn đúng dòng.
            if stage.kind is not TranslationStageKind.PREPROCESS:
                stage_output = self._repair_misaligned_blocks(
                    stage_output, source_lines, request, cancel_cb, stage_label,
                    active_stage=stage,
                )

            # [v3.23.569] Giai đoạn tinh chỉnh ĐẨY LỆCH DÒNG so với bản trước mà tương
            # quan độ dài không thấy (câu ngắn dài xấp xỉ nhau) — dò bằng chính bản trước,
            # phân xử bằng bản dịch trọng tài từ câu gốc.
            if has_prior_translation:
                stage_output = self._repair_stage_shifts(
                    stage_output, current_lines, source_lines, request, cancel_cb,
                    stage_label, active_stage=stage,
                )
                # [v3.23.592] Dòng LẺ bị viết lại khác hẳn (dải từ ba dòng đã được lưới
                # trên xử lý): gom mọi dòng như vậy vào MỘT lượt dịch trọng tài.
                stage_output = self._revert_lone_rewrites(
                    stage_output, current_lines, source_lines, request, cancel_cb,
                    stage_label, active_stage=stage,
                )

            # [v3.23.586] Gỡ chú thích thuật ngữ bị chép vào lời thoại ("120 (cấp cứu y
            # tế)") TRƯỚC các lưới so sánh với bản trước, để chúng so trên bản đã sạch.
            if stage.kind is not TranslationStageKind.PREPROCESS:
                stage_output = self._strip_glossary_notes(
                    stage_output, source_lines, request.context.glossary or "",
                    stage_label,
                )

            # [v3.23.432] Safety-net chống REVERT CJK: dữ liệu thật cho thấy giai đoạn tinh
            # chỉnh (style/localize) đôi khi thay từ Hán-Việt đã dịch ĐÚNG ngược lại thành chữ
            # Hán gốc (E04 'tinh'→'星'; E07 'phấn đồ cường'→'奋图强'), làm hỏng câu vốn sạch. Nếu
            # dòng INPUT (bản trước) đã sạch CJK mà OUTPUT lại chèn CJK → lùi dòng đó về bản
            # trước. Tự động bỏ qua giai đoạn LITERAL (input là nguồn còn CJK nên không "sạch").
            stage_output = self._revert_cjk_regressions(
                stage_output, current_lines, stage_label
            )
            # [v3.23.552] Cùng khuôn: giai đoạn tinh chỉnh làm MẤT tên riêng đã dịch đúng ở giai
            # đoạn trước ("chị dâu Đào Hoa" → "vợ Đào Hoa") → lùi dòng đó về bản trước.
            if has_prior_translation:
                stage_output = self._revert_name_regressions(
                    stage_output, current_lines, source_lines,
                    str(getattr(request.context, "glossary", "") or ""), stage_label,
                )
                # [v3.23.576] Cùng khuôn: giai đoạn tinh chỉnh gộp câu của dòng kề vào một
                # dòng rồi vẫn giữ dòng kề → nói lặp. Chạy SAU lưới lệch dòng (dải lệch đã
                # được lùi) để chỉ còn các ca gộp lẻ.
                stage_output = self._revert_echo_regressions(
                    stage_output, current_lines, source_lines, stage_label
                )
                # [v3.23.605] Cùng khuôn: dòng thoại bị biến thành chỉ còn tag ("[cười]")
                # → mất khỏi cả phụ đề lẫn lồng tiếng.
                stage_output = self._revert_unspoken_regressions(
                    stage_output, current_lines, source_lines, stage_label
                )
                # [v3.23.605] Cùng khuôn: chỉ chèn từ đệm mà làm câu vượt ngân sách đọc.
                stage_output = self._revert_padding_regressions(
                    stage_output, current_lines, source_lines, stage_label
                )
                # [v3.23.633] Cùng khuôn: 你 (người nghe) bị biến thành "ta/tôi" do nhãn người
                # nói sai (đo: lỗi nặng nhất của chuỗi 3 giai đoạn khi chấm mù).
                stage_output = self._revert_person_flips(
                    stage_output, current_lines, source_lines, stage_label
                )
                # [v3.23.583] Cùng khuôn: giai đoạn tinh chỉnh đổi đại từ mà bản trước đã
                # dùng ĐÚNG bảng nhân vật ("tớ … cậu" → "chị … em" giữa hai người bạn).
                # Chạy TRƯỚC lưới ép quy tắc để lưới đó làm việc trên bản đã khôi phục.
                stage_output = self._restore_address_regressions(
                    stage_output, current_lines, source_lines, request.context,
                    stage_label,
                )

            # [v3.23.554] Ép ĐẠI TỪ TỰ XƯNG về đúng bảng nhân vật. Chạy cho MỌI giai đoạn
            # (kể cả dịch thô): đo thật cho thấy 50/51 dòng sai đã sai ngay từ dịch thô.
            stage_output = self._enforce_address_rules(
                stage_output, source_lines, request.context, stage_label
            )

            # [v3.23.570] Dòng chỉ gọi một cái tên phải dịch ra ĐÚNG tên đó.
            stage_output = self._enforce_vocative_names(
                stage_output, source_lines, request.context, stage_label
            )

            # [v3.23.556] Chuẩn hoá CON SỐ: sai giá trị tiền bạc là lỗi người xem nhận ra
            # ngay, mà không cờ nào khác bắt được (câu vẫn trôi chảy, vẫn đủ dài).
            stage_output = self._enforce_number_consistency(
                stage_output, source_lines, stage_label,
                previous_lines=current_lines if has_prior_translation else None,
            )

            # [v3.23.608] Dòng người dùng sửa tay ở giai đoạn này luôn thắng bản model mới
            # (áp SAU các lưới để không lưới nào sửa đè câu người dùng đã chốt).
            if checkpoint is not None:
                stage_output, reapplied = checkpoint.apply_edits(
                    stage.kind, stage_output
                )
                if reapplied:
                    reapplied_edits[stage.kind] = len(reapplied)
                    logger.info(
                        "%s: áp lại %d dòng sửa tay lên bản dịch mới.",
                        stage_label, len(reapplied),
                    )
                    _record_guard_event(
                        "manual_edits_reapplied", label=stage_label, lines=reapplied
                    )

            # [v3.23.549] Ghi MỐC GIAI ĐOẠN kể cả khi không có lỗi nào. Trước đây nhật ký chỉ ghi
            # sự kiện LỖI, nên một lần dịch sạch cho ``events: []`` — người đọc file chuẩn đoán
            # không phân biệt được "không có lỗi" với "cơ chế ghi nhật ký hỏng" (chính tôi đã
            # tưởng nhầm là hỏng khi soi file thật).
            # [v3.23.592] Ghi TRƯỚC khi lưu checkpoint để mốc này nằm trong nhật ký của
            # giai đoạn được lưu kèm.
            self._record_stage_completed(
                stage,
                stage_label,
                stage_output,
                attach_video=bool(
                    stage.kind in request.attach_video_stages and request.video_refs
                ),
            )

            # ── Lưu checkpoint sau khi giai đoạn hoàn tất ────────────────
            if checkpoint is not None:
                checkpoint.save_stage(
                    stage.kind,
                    stage_output,
                    events=self._process_log_events_since(stage_event_mark),
                    run=stage_run,
                    input_lines=current_lines,
                )

            # [v3.23.47] Giữ kết quả giai đoạn (dạng events) để so sánh sau khi dịch xong.
            stage_outputs[stage.kind] = self._lines_to_events(
                request.events, stage_output, request.context
            )

            current_lines = stage_output
            completed.append(stage.kind)

            if stage.kind is TranslationStageKind.PREPROCESS:
                reference_lines = stage_output
                source_was_preprocessed = True
            else:
                has_prior_translation = True

        before_cleanup = {line.index: line.text or "" for line in current_lines}
        # [v3.23.433] Dọn CJK còn sót THẬT (literal dịch một phần, để trailing CJK — safety-net
        # revert không cứu được vì mọi giai đoạn đều còn CJK). Hướng 3: thử dịch LẠI riêng các
        # dòng đó từ nguồn (lấy mẫu mới thường dịch trọn); Hướng 2: log rõ dòng nào còn sót.
        current_lines = self._cleanup_residual_cjk(
            current_lines, source_lines, request, cancel_cb
        )

        # [v3.23.510] Dọn CẶP dòng liền kề bị gán cùng một bản dịch (câu tách đôi bị model gán
        # trùng). Chạy SAU vòng dọn CJK để không dịch lại hai lần cùng một dòng.
        current_lines = self._cleanup_duplicate_adjacent(
            current_lines, source_lines, request, cancel_cb
        )
        # [v3.23.632] Dòng còn bị nghi DỊCH THIẾU ở bản cuối: dịch lại riêng kèm ngữ cảnh.
        current_lines = self._patch_under_translated(
            current_lines, source_lines, request, cancel_cb
        )
        # [v3.23.603] Dòng dịch lại một mình hay kết bằng dấu phẩy dù đã hết câu.
        if request.contiguous:
            current_lines = self._close_cleanup_sentences(
                before_cleanup, current_lines, source_lines, request.context
            )
        # [v3.23.612] Tuỳ chọn xưng hô thuần Việt: chạy SAU mọi bước dịch lại để không
        # bước nào đưa "Kiếm ca" trở lại bản cuối.
        if self._native_address:
            current_lines = self._apply_native_address(
                current_lines, source_lines, request.context
            )

        # [v3.23.586] Hai bước dọn ở trên có dịch lại vài dòng → gỡ chú thích lần cuối.
        if completed and completed[-1] is not TranslationStageKind.PREPROCESS:
            current_lines = self._strip_glossary_notes(
                current_lines, source_lines, request.context.glossary or "", "Bản cuối"
            )

        # [v3.23.616] Viết hoa chữ đầu dòng mở câu mới — sau mọi bước có thể sửa chữ.
        if request.contiguous:
            current_lines = self._capitalize_sentence_starts(
                current_lines, request.context
            )

        translated_events = self._lines_to_events(
            request.events, current_lines, request.context
        )
        if progress_cb is not None:
            progress_cb(1.0, "Hoàn tất")

        # [v3.23.350] GUARD "dịch không tác dụng": trước đây khi toàn bộ bản dịch
        # trùng bản gốc (pipeline thiếu khâu LITERAL, nguồn đã cùng ngôn ngữ đích,
        # hoặc resume nhầm checkpoint hỏng), các khâu vẫn "thành công" và chỉ để lại
        # log INFO/WARNING rải rác → người dùng thấy "trả về bản gốc mà không báo lỗi".
        # Ở đây so KẾT QUẢ CUỐI với nguồn MỘT LẦN và cảnh báo NỔI BẬT kèm gợi ý.
        unchanged, comparable = _measure_passthrough_ratio(
            request.events, translated_events
        )
        if comparable > 0 and unchanged / comparable >= _PASSTHROUGH_WARN_RATIO:
            has_literal = any(
                s.kind is TranslationStageKind.LITERAL for s in request.stages
            )
            translating_stages = [
                s.kind for s in request.stages
                if s.kind is not TranslationStageKind.PREPROCESS
            ]
            if not translating_stages:
                likely_cause = (
                    "pipeline KHÔNG có khâu dịch nào (chỉ có Tiền xử lý)"
                )
            elif not has_literal:
                likely_cause = (
                    "pipeline THIẾU khâu 'Dịch thô sát nghĩa' (LITERAL) — các khâu "
                    "Tinh chỉnh/Bản địa hoá coi việc giữ nguyên bản nháp là HỢP LỆ nên "
                    "không đổi ngôn ngữ"
                )
            elif resumed_from is not None:
                likely_cause = (
                    f"đã RESUME từ checkpoint (khâu '{resumed_from.value}') — "
                    "checkpoint có thể lưu kết quả hỏng từ lần chạy trước; hãy xoá "
                    "checkpoint rồi dịch lại"
                )
            else:
                likely_cause = (
                    "model có thể liên tục trả về nguyên văn nguồn (xem WARNING "
                    "'giữ text gốc' ở tầng dịch phía trên)"
                )
            logger.warning(
                "DỊCH KHÔNG TÁC DỤNG: %d/%d dòng (%.0f%%) TRÙNG HỆT bản gốc. "
                "Nguyên nhân khả dĩ: %s.",
                unchanged, comparable, 100.0 * unchanged / comparable, likely_cause,
            )

        # [v3.23.608] Bộ nhớ đệm được GIỮ (trước đây xoá ở đây): đổi cấu hình, sửa tay hay
        # thêm một giai đoạn thì lần sau chỉ dịch phần thật sự đổi.
        if checkpoint is not None:
            logger.info(
                "Giữ bộ nhớ đệm %d giai đoạn của lượt dịch (%s) để dùng lại / sửa trong "
                "'Quản lý giai đoạn'.",
                len(checkpoint.stage_kinds()), checkpoint.key,
            )

        return TranslateSubtitlesResponse(
            events=translated_events,
            completed_stages=completed,
            resumed_from=resumed_from,
            elapsed_seconds=time.monotonic() - _start_time,
            stage_outputs=stage_outputs,
            visual_cues=topped_up_cues,
            reused_stages=tuple(reused_stages),
            carried_lines=carried_lines,
            reapplied_edits=reapplied_edits,
        )

    # ── Visual Cues (Vision Director) ────────────────────────────────────
    @classmethod
    def _reraise_user_cancel(
        cls, exc: BaseException, cancel_cb: CancellationCallback | None
    ) -> None:
        """[v3.23.609] Ném lại lỗi "đã huỷ" khi người dùng THẬT SỰ bấm Huỷ.

        Các bước dịch lại phụ bắt chung ``SubtitleTranslationError`` để lỗi của chúng
        không làm đổ cả lượt dịch — nhưng ``TranslationCancelledError`` là lớp con của
        nó, nên lệnh Huỷ của người dùng từng bị nuốt (lượt dịch chạy tiếp và báo "Dịch
        hoàn tất"). Lỗi "đã huỷ" do mạng chập chờn (người dùng không bấm) vẫn bỏ qua.

        Args:
            exc: Lỗi vừa bắt.
            cancel_cb: Callback huỷ của lượt dịch.

        Raises:
            TranslationCancelledError: Khi ``exc`` là lỗi huỷ và người dùng đã bấm Huỷ.
        """
        if isinstance(exc, TranslationCancelledError) and cls._user_cancelled(cancel_cb):
            raise exc

    @staticmethod
    def _user_cancelled(cancel_cb: CancellationCallback | None) -> bool:
        """[v3.23.560] Người dùng có THẬT SỰ yêu cầu huỷ không.

        Bước quét Visual Cues là phụ trợ: một ``TranslationCancelledError`` mà người dùng
        không hề bấm huỷ (ca thật: lỗi đảo điều kiện trong bộ quét làm MỌI lần nghỉ giữa
        các lô ném "đã huỷ") không được phép làm đổ cả lượt dịch — đã xảy ra 3 lần liền.

        Args:
            cancel_cb: Callback huỷ của lượt dịch.

        Returns:
            True nếu callback tồn tại và đang báo huỷ.
        """
        return cancel_cb is not None and bool(cancel_cb())

    def _top_up_context_cues(
        self,
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        checkpoint: StageCheckpoint | None,
        cancel_cb: CancellationCallback | None,
        progress_cb: OverallProgressCallback | None = None,
    ) -> TranslateSubtitlesRequest:
        """[v3.23.559] Quét BỔ SUNG Visual Cues cho dòng mà phân tích ngữ cảnh bỏ trống.

        Ca thật v3.23.558: phân tích bằng gemini-3.6-flash chạy đủ 22 đoạn KHÔNG lỗi,
        nhưng chỉ trả cues cho 27% số dòng (đoạn 19: 1/208 dòng). Trước đây khâu dịch
        thấy "đã có cues" là dùng luôn, bỏ qua bộ quét riêng — nên 73% số dòng dịch mà
        không biết ai nói với ai, và mọi lưới xưng hô cũng mù ở đó.

        Cues có sẵn LUÔN được ưu tiên (người dùng có thể đã sửa tay); bản quét chỉ lấp chỗ
        trống. Kết quả quét được lưu vào checkpoint để resume không phải quét lại. Lỗi khi
        quét (trừ huỷ) được nuốt: dịch vẫn chạy với cues sẵn có.

        [v3.23.562] "Có cues" nghĩa là người nói/nghe là nhân vật TRONG BẢNG (xem
        ``cue_label_canonicalizer.usable_cue_lines``). Ca thật v3.23.561: cues "phủ"
        99,7% nhưng 74% chỉ mang nhãn chung chung ("Nữ tỳ", "Nam chính") do bộ quét không
        được đưa bảng nhân vật — lưới xưng hô không dùng được. Nay bộ quét nhận bảng nhân
        vật, nhãn được quy về tên chuẩn, và dòng chỉ có nhãn chung chung được quét lại.

        Args:
            source_lines: Toàn bộ dòng gốc.
            request: Yêu cầu dịch.
            checkpoint: Checkpoint của lần dịch (để cache kết quả quét).
            cancel_cb: Callback huỷ.
            progress_cb: Callback tiến độ tổng — hiện "đang quét x%" vì bước này có thể
                kéo dài hàng chục lượt gọi; thiếu nó giao diện trông như treo.

        Returns:
            Yêu cầu MỚI với ``context.visual_cues`` đã gộp; chính ``request`` nếu không
            cần hoặc không thể bổ sung.
        """
        from subtitles_extractor.application.services.cue_label_canonicalizer import (
            canonicalize_cue_labels,
            usable_cue_lines,
        )
        from subtitles_extractor.application.services.visual_cue_serializer import (
            deserialize_visual_cues,
            serialize_visual_cues,
        )

        characters = request.context.characters or ""
        existing = canonicalize_cue_labels(
            deserialize_visual_cues((request.context.visual_cues or "").strip()),
            characters,
        )
        if not existing or not source_lines:
            return request
        covered = usable_cue_lines(existing, characters)
        coverage = len(covered) / len(source_lines)
        analyze = getattr(self._translator, "analyze_visual_cues", None)
        if coverage >= MIN_CONTEXT_CUE_COVERAGE or analyze is None:
            return request

        missing = [line for line in source_lines if line.index not in covered]
        scanned = []
        if checkpoint is not None:
            cached = checkpoint.get_visual_cues()
            if cached:
                scanned = deserialize_visual_cues(json.dumps(cached))
        if not scanned:
            logger.warning(
                "Visual Cues dùng được (nhãn khớp bảng nhân vật) chỉ phủ %.0f%% số dòng "
                "(ngưỡng %.0f%%) → quét bổ sung %d dòng còn thiếu bằng '%s'.",
                coverage * 100, MIN_CONTEXT_CUE_COVERAGE * 100, len(missing),
                request.visual_cues_model,
            )
            def _report(fraction: float) -> None:
                if progress_cb is not None:
                    label = f"Quét bổ sung gợi ý hình ảnh (ai nói với ai): {fraction:.0%}"
                    progress_cb(0.0, label)

            _report(0.0)
            try:
                scanned = analyze(
                    missing,
                    request.context.target_lang,
                    model_name=request.visual_cues_model,
                    video_refs=list(request.video_refs),
                    batch_size=request.visual_cues_batch_size,
                    cancel_cb=cancel_cb,
                    progress_cb=_report,
                    characters=characters,
                    # [v3.23.581] Dòng xen kẽ ĐÃ có nhãn đi kèm lô làm mạch hội thoại —
                    # quét riêng các dòng thiếu thì model không biết cuộc nói chuyện nào.
                    known_cues=[cue for cue in existing if cue.line_no in covered],
                    context_lines=source_lines,
                )
            except TranslationCancelledError as exc:
                if self._user_cancelled(cancel_cb):
                    raise
                logger.warning(
                    "Quét bổ sung Visual Cues báo 'đã huỷ' dù người dùng KHÔNG huỷ — "
                    "bỏ qua, dịch với cues sẵn có: %s", exc,
                )
                return request
            except (RuntimeError, ValueError, OSError, SubtitleTranslationError) as exc:
                logger.warning(
                    "Quét bổ sung Visual Cues thất bại — dịch với cues sẵn có: %s", exc
                )
                return request
            if scanned and checkpoint is not None:
                checkpoint.save_visual_cues(json.loads(serialize_visual_cues(scanned)))

        added = [
            cue for cue in canonicalize_cue_labels(scanned, characters)
            if cue.line_no not in covered and (cue.speaker or cue.addressee)
        ]
        if added:
            # [v3.23.581] Nhãn quét "lạc cảnh" (mâu thuẫn với hai nhãn sẵn có bao quanh).
            kept, _dropped = self._without_contradicted(
                [cue for cue in existing if cue.line_no in covered] + added,
                {cue.line_no for cue in added},
                source_lines,
                characters,
            )
            kept_lines = {cue.line_no for cue in kept}
            added = [cue for cue in added if cue.line_no in kept_lines]
        if not added:
            return request
        replaced = {cue.line_no for cue in added}
        # Cue cũ chỉ có nhãn chung chung được giữ lại cho dòng mà bản quét mới bỏ trống.
        merged = sorted(
            [cue for cue in existing if cue.line_no not in replaced] + added,
            key=lambda cue: cue.line_no,
        )
        logger.info(
            "Đã bổ sung Visual Cues: %d → %d dòng có cue dùng được (%.0f%% → %.0f%%).",
            len(covered), len(usable_cue_lines(merged, characters)), coverage * 100,
            100 * len(usable_cue_lines(merged, characters)) / len(source_lines),
        )
        new_context = replace(request.context, visual_cues=serialize_visual_cues(merged))
        return replace(request, context=new_context)

    def _enrich_with_visual_cues(
        self,
        source_lines: list[TranslationLine],
        request: TranslateSubtitlesRequest,
        checkpoint: StageCheckpoint | None,
        progress_cb: OverallProgressCallback | None,
        cancel_cb: CancellationCallback | None,
    ) -> list[TranslationLine]:
        """Quét video để xác định ai nói/nói với ai, bơm vào speaker/addressee.

        An toàn-trước-tiên: mọi lỗi (trừ huỷ) đều được nuốt và trả về dòng gốc, để
        bước phụ trợ này KHÔNG bao giờ làm hỏng tiến trình dịch chính.
        """
        analyze = getattr(self._translator, "analyze_visual_cues", None)

        # [v3.23.35] ƯU TIÊN CAO NHẤT: dùng Visual Cues đã phân tích chung với ngữ cảnh
        # toàn cục (người dùng có thể đã xem/sửa) — không quét lại video.
        ctx_cues = (request.context.visual_cues or "").strip()
        if ctx_cues:
            from subtitles_extractor.application.services.visual_cue_serializer import (
                deserialize_visual_cues,
            )
            cues = deserialize_visual_cues(ctx_cues)
            if cues:
                logger.info(
                    "Dùng %d Visual Cues từ phân tích ngữ cảnh (đã có sẵn, không quét "
                    "lại video).", len(cues),
                )
                return apply_visual_cues_to_lines(source_lines, cues)

        if analyze is None:
            return source_lines

        # Kế đến: dùng cache trong checkpoint (cùng source+video) nếu có.
        if checkpoint is not None:
            cached = checkpoint.get_visual_cues()
            if cached is not None:
                from subtitles_extractor.application.services.visual_cue_serializer import (
                    deserialize_visual_cues,
                )
                cues = deserialize_visual_cues(json.dumps(cached))
                if cues:
                    logger.info(
                        "Dùng lại %d Visual Cues từ checkpoint (bỏ qua quét video).",
                        len(cues),
                    )
                    return apply_visual_cues_to_lines(source_lines, cues)

        if progress_cb is not None:
            progress_cb(0.0, "Phân tích hình ảnh (ai nói với ai)")
        try:
            cues = analyze(
                source_lines,
                request.context.target_lang,
                model_name=request.visual_cues_model,
                video_refs=list(request.video_refs),
                batch_size=request.visual_cues_batch_size,
                cancel_cb=cancel_cb,
                characters=request.context.characters or "",
            )
        except TranslationCancelledError as exc:
            if self._user_cancelled(cancel_cb):
                raise
            logger.warning(
                "Phân tích Visual Cues báo 'đã huỷ' dù người dùng KHÔNG huỷ — bỏ qua, "
                "dịch không có gợi ý hình ảnh: %s", exc,
            )
            return source_lines
        except SubtitleTranslationError as exc:
            logger.warning(
                "Phân tích Visual Cues thất bại — bỏ qua, dịch không có gợi ý hình "
                "ảnh: %s", exc,
            )
            return source_lines
        if not cues:
            return source_lines
        from subtitles_extractor.application.services.cue_label_canonicalizer import (
            canonicalize_cue_labels,
        )

        # [v3.23.562] Nhãn khớp bảng nhân vật → tên chuẩn, để bảng quy đổi tên dùng được.
        cues = canonicalize_cue_labels(cues, request.context.characters or "")
        # Cache lại để lần dịch sau (hoặc resume) không phải quét video lại.
        if checkpoint is not None:
            from subtitles_extractor.application.services.visual_cue_serializer import (
                serialize_visual_cues,
            )
            checkpoint.save_visual_cues(json.loads(serialize_visual_cues(cues)))
        enriched = apply_visual_cues_to_lines(source_lines, cues)
        logger.info(
            "Đã bơm gợi ý hình ảnh cho %d/%d dòng (ai nói / nói với ai).",
            sum(1 for c in cues if c.speaker or c.addressee), len(source_lines),
        )
        return enriched

    # ── Chuyển đổi domain ↔ trung gian ───────────────────────────────────
    @staticmethod
    def _events_to_lines(events: list[SubtitleEvent]) -> list[TranslationLine]:
        return [
            TranslationLine(
                index=position,
                start_ms=int(round(event.start_sec * 1000)),
                end_ms=int(round(event.end_sec * 1000)),
                text=event.text,
            )
            for position, event in enumerate(events, start=1)
        ]

    @staticmethod
    def _lines_to_events(
        original_events: list[SubtitleEvent],
        translated_lines: list[TranslationLine],
        context: TranslationContext,
    ) -> list[SubtitleEvent]:
        result: list[SubtitleEvent] = []
        last_speaker = ""  # [v3.23.20] theo dõi người nói để chỉ tag khi ĐỔI người
        # [v3.23.25] Ánh xạ tên CJK → phiên âm (Hán Việt…) lấy từ roster phân tích,
        # để tag tên người nói CJK được phiên âm sang tiếng đích, Latin giữ nguyên.
        roster_map = TranslateSubtitlesUseCase._build_roster_pronunciation_map(
            context.characters or ""
        )
        # [v3.23.84] Map tên CHUẨN từ roster để chuẩn hoá tag người nói nhất quán
        # (đồng nhất cách viết + ánh xạ alias CJK -> tên đích).
        canonical_map = TranslateSubtitlesUseCase._build_canonical_name_map(
            context.characters or ""
        )
        for position, original in enumerate(original_events, start=1):
            line = translated_lines[position - 1] if position - 1 < len(translated_lines) else None
            new_text = line.text if line is not None else original.text
            display_text, last_speaker = TranslateSubtitlesUseCase._compose_display_text(
                line, new_text, context, last_speaker, roster_map, canonical_map
            )
            result.append(
                SubtitleEvent(
                    index=original.index,
                    text=display_text,
                    interval=TimeInterval(original.start_sec, original.end_sec),
                    confidence=original.confidence,
                    frame_count=original.frame_count,
                    position=original.position,
                    bounding_box=original.bounding_box,
                    uid=original.uid,
                )
            )
        return TranslateSubtitlesUseCase._suppress_repeated_speaker_tags(result)

    @classmethod
    def _suppress_repeated_speaker_tags(
        cls, events: list[SubtitleEvent]
    ) -> list[SubtitleEvent]:
        """[v3.23.137] Gỡ nhãn '[Người nói:]' ở ĐẦU dòng khi TRÙNG người nói dòng ngay
        trước (theo khoá chuẩn hoá), BẤT KỂ nhãn do app gắn hay do model tự chèn vào text.

        Vì model lite gán người nói KHÔNG nhất quán giữa các giai đoạn (lúc dùng trường
        'speaker', lúc chèn thẳng '[Tên:]' vào text), khử trùng chỉ dựa trên trường speaker
        (``_compose_display_text``) không đủ → nhãn bị lặp mỗi dòng dù cùng một người. Lượt
        hợp nhất này bảo đảm: chuỗi dòng liên tiếp CÙNG người chỉ hiện nhãn ở dòng đầu; dòng
        hiệu ứng âm thanh/nhạc NGẮT mạch (người nói kế sau được tag lại).
        """
        tag_re = re.compile(r"^\s*\[([^\]]+?):\]\s*")
        last_key = ""
        out: list[SubtitleEvent] = []
        for ev in events:
            text = ev.text
            if cls._is_sound_effect(text):
                last_key = ""  # ngắt mạch người nói qua hiệu ứng âm thanh/nhạc
                out.append(ev)
                continue
            match = tag_re.match(text)
            if match:
                key = cls._normalize_name_key(match.group(1))
                if key and key == last_key:
                    stripped = text[match.end():].lstrip()
                    # Chỉ gỡ khi PHẦN CÒN LẠI không rỗng (tránh tạo dòng trống).
                    if stripped:
                        out.append(replace(ev, text=stripped))
                        continue
                else:
                    last_key = key
            out.append(ev)
        return out

    # Người nói "rỗng" không nên tag: N/A, unknown, dấu chấm hỏi…
    _EMPTY_SPEAKERS = frozenset({
        "", "n/a", "na", "unknown", "không rõ", "khong ro", "?", "-", "none",
        "người nói", "speaker", "nhân vật", "narrator unknown",
    })

    # [v3.23.23] Dịch các nhãn người nói CHUNG sang tiếng Việt. Tên riêng (John,
    # Tiến sĩ Teven…) KHÔNG dịch. Khoá là dạng thường, đã bỏ ngoặc/dấu hai chấm.
    _GENERIC_SPEAKER_VI: dict[str, str] = {
        "man": "Người đàn ông", "woman": "Người phụ nữ",
        "boy": "Cậu bé", "girl": "Cô bé", "child": "Đứa trẻ", "kid": "Đứa trẻ",
        "narrator": "Người dẫn chuyện", "announcer": "Phát thanh viên",
        "reporter": "Phóng viên", "interviewer": "Người phỏng vấn",
        "host": "Người dẫn chương trình", "presenter": "Người dẫn chương trình",
        "crowd": "Đám đông", "audience": "Khán giả", "voice": "Giọng nói",
        "voiceover": "Lời dẫn", "male voice": "Giọng nam", "female voice": "Giọng nữ",
        "old man": "Ông lão", "old woman": "Bà lão",
        "doctor": "Bác sĩ", "teacher": "Giáo viên", "student": "Học sinh",
        "soldier": "Người lính", "officer": "Sĩ quan", "nurse": "Y tá",
        "all": "Mọi người", "both": "Cả hai", "everyone": "Mọi người",
        "computer": "Máy tính", "radio": "Đài", "tv": "Ti-vi", "phone": "Điện thoại",
        # [v3.23.28] Bổ sung từ phân tích bản dịch thực tế (PBS NOVA…).
        "astronaut": "Phi hành gia", "pilot": "Phi công",
        "nasa announcer": "Phát thanh viên NASA", "mission control": "Trung tâm điều khiển",
        "controller": "Nhân viên điều phối", "scientist": "Nhà khoa học",
        "engineer": "Kỹ sư", "commander": "Chỉ huy", "captain": "Đội trưởng",
        "automated voice": "Giọng tự động", "computer voice": "Giọng máy tính",
        "robot": "Người máy", "ai": "Trí tuệ nhân tạo",
        "operator": "Nhân viên trực tổng đài", "news anchor": "Phát thanh viên",
        "translator": "Phiên dịch", "interpreter": "Phiên dịch",
        "boy and girl": "Cậu bé và cô bé", "men": "Những người đàn ông",
        "women": "Những người phụ nữ", "group": "Nhóm người", "people": "Mọi người",
    }

    # [v3.23.90] Chức danh/kính ngữ CJK phổ biến trong phim cổ trang -> Hán-Việt. Dùng
    # làm FALLBACK cho tag người nói khi thuật ngữ không có trong roster, tránh rò ký tự
    # Trung vào phụ đề tiếng Việt (vd cue trả về "大臣"/"皇上/陛下").
    _CJK_ROLE_VI: dict[str, str] = {
        "皇上": "Hoàng thượng", "陛下": "Bệ hạ", "朕": "Trẫm", "皇帝": "Hoàng đế",
        "太后": "Thái hậu", "皇后": "Hoàng hậu", "皇贵妃": "Hoàng quý phi",
        "贵妃": "Quý phi", "娘娘": "Nương nương", "公主": "Công chúa",
        "太子": "Thái tử", "王爷": "Vương gia", "世子": "Thế tử", "殿下": "Điện hạ",
        "大臣": "Đại thần", "丞相": "Thừa tướng", "宰相": "Tể tướng",
        "将军": "Tướng quân", "大将军": "Đại tướng quân", "大人": "Đại nhân",
        "将士": "Tướng sĩ", "士兵": "Binh sĩ", "士卒": "Binh sĩ", "侍卫": "Thị vệ",
        "太监": "Thái giám", "公公": "Công công", "宫女": "Cung nữ", "婢女": "Tỳ nữ",
        "老奴": "Lão nô", "奴婢": "Nô tỳ", "百姓": "Bách tính", "大夫": "Đại phu",
        "钦差": "Khâm sai", "王妃": "Vương phi", "侧妃": "Trắc phi", "嫔妃": "Tần phi",
    }

    # [v3.23.28] Chú thích KÊNH THOẠI đi kèm tên (vd 'on computer', 'on radio').
    # Tách riêng để dịch chú thích nhưng GIỮ tên chính + dấu ngoặc đúng.
    _SPEAKER_CHANNEL_VI: dict[str, str] = {
        "on computer": "trên máy tính", "on radio": "qua radio",
        "on phone": "qua điện thoại", "on tv": "trên ti-vi",
        "on screen": "trên màn hình", "on speaker": "qua loa",
        "on intercom": "qua bộ đàm", "on monitor": "trên màn hình",
        "voiceover": "lời dẫn", "v.o.": "lời dẫn", "o.s.": "ngoài hình",
        "off screen": "ngoài hình", "on recording": "trong bản ghi",
        "via video": "qua video", "on video": "qua video",
        "translated": "đã dịch", "in english": "bằng tiếng Anh",
    }

    @staticmethod
    def _has_cjk(text: str) -> bool:
        """True nếu chuỗi chứa ký tự CJK (Trung/Nhật Kanji/Hàn Hanja)."""
        for ch in text:
            o = ord(ch)
            if (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
                    or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF):
                return True
        return False

    @staticmethod
    def _normalize_name_key(name: str) -> str:
        """Khoá chuẩn hoá để so tên: hạ hoa-thường + gộp khoảng trắng."""
        return " ".join(name.strip().lower().split())

    @classmethod
    def _build_canonical_name_map(cls, characters: str) -> dict[str, str]:
        """Map ``normalized(alias) -> TÊN CHUẨN`` từ roster, để chuẩn hoá tag người nói.

        Mỗi dòng roster có dạng ``Tên (alias)`` — trong đó MỘT bên là tên đích
        (Latinh/Việt) và bên kia là tên gốc CJK; thứ tự có thể đảo tuỳ model. Hàm tự nhận
        diện phần CJK để lấy phần KHÔNG-CJK làm TÊN CHUẨN, và đăng ký:
          * chính tên chuẩn (chuẩn hoá hoa-thường/khoảng trắng) -> tên chuẩn
          * mỗi alias CJK (tách theo '/', '、', ',') -> tên chuẩn

        CHỈ khớp CHÍNH XÁC sau chuẩn hoá — KHÔNG gộp mờ (không trộn nhân vật khác nhau).
        """
        mapping: dict[str, str] = {}
        if not characters:
            return mapping
        for raw_line in characters.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^(.+?)\s*[(（]\s*([^)）]+?)\s*[)）]", line)
            if not match:
                continue
            left = match.group(1).strip()
            right = match.group(2).strip()
            if not left or not right:
                continue
            # Tên CHUẨN = phần KHÔNG phải CJK (Latinh/Việt); alias = phần CJK.
            if cls._has_cjk(left) and not cls._has_cjk(right):
                canonical, cjk_part = right, left
            elif cls._has_cjk(right) and not cls._has_cjk(left):
                canonical, cjk_part = left, right
            else:
                # Không phân biệt được (cả hai cùng loại) -> dùng phần trái làm chuẩn.
                canonical, cjk_part = left, right
            aliases = [canonical, *re.split(r"[/、,]", cjk_part)]
            for alias in aliases:
                key = cls._normalize_name_key(alias)
                if key and key not in mapping:
                    mapping[key] = canonical
        return mapping

    @staticmethod
    def _build_roster_pronunciation_map(characters: str) -> dict[str, str]:
        """Trích ánh xạ tên-gốc-CJK → tên-phiên-âm từ roster phân tích.

        Roster thường có dạng '林昆 (Lâm Côn), 王语嫣 (Vương Ngữ Yên)'. Hàm tách cặp
        'CJK (phiên âm)' để khi gặp tag tên CJK ta thay bằng bản phiên âm tiếng đích.
        """
        mapping: dict[str, str] = {}
        if not characters:
            return mapping
        # Bắt mẫu 'TÊN_GỐC (phiên âm)' hoặc 'TÊN_GỐC（phiên âm）' (ngoặc CJK).
        for match in re.finditer(r"([^,;()（）]+?)\s*[(（]\s*([^,;()（）]+?)\s*[)）]", characters):
            original = match.group(1).strip()
            romanized = match.group(2).strip()
            if original and romanized:
                mapping[original] = romanized
        return mapping

    @classmethod
    def _localize_one_name(
        cls,
        name: str,
        roster_map: dict[str, str] | None,
        canonical_map: dict[str, str] | None = None,
    ) -> str:
        """Chuẩn hoá MỘT tên người nói (không gồm chú thích).

        0. [v3.23.84] Nếu khớp CHÍNH XÁC một alias trong roster (canonical_map) -> trả về
           TÊN CHUẨN của roster (đồng nhất cách viết + ánh xạ alias CJK sang tên đích).
        1. Nhãn CHUNG (Woman/Astronaut…) → dịch tiếng Việt.
        2. Tên CJK → phiên âm từ roster (khớp chính xác rồi khớp dài nhất); roster
           không có → giữ nguyên CJK.
        3. Tên Latin → GIỮ NGUYÊN.
        """
        cleaned = name.strip().strip("[]():").strip()
        if not cleaned:
            return ""
        # [v3.23.84] Ưu tiên tên chuẩn từ roster (khớp chính xác sau chuẩn hoá).
        if canonical_map:
            canonical = canonical_map.get(cls._normalize_name_key(cleaned))
            if canonical:
                return canonical
        key = cleaned.lower()
        if key in cls._GENERIC_SPEAKER_VI:
            return cls._GENERIC_SPEAKER_VI[key]
        if cls._has_cjk(cleaned) and roster_map:
            if cleaned in roster_map:
                return roster_map[cleaned]
            best_original = ""
            for original in roster_map:
                if original and (original in cleaned or cleaned in original):
                    if len(original) > len(best_original):
                        best_original = original
            if best_original:
                return roster_map[best_original]
        # [v3.23.90] Fallback chức danh/kính ngữ CJK phổ biến (cổ trang) -> Hán-Việt,
        # tránh rò ký tự Trung vào tag tiếng Việt khi không có trong roster. Xử lý cả
        # dạng ghép "皇上/陛下" bằng cách thử từng phần (canonical map trước, rồi map).
        if cls._has_cjk(cleaned):
            for part in re.split(r"[/、,]", cleaned):
                part = part.strip()
                if not part:
                    continue
                if canonical_map:
                    mapped = canonical_map.get(cls._normalize_name_key(part))
                    if mapped:
                        return mapped
                if part in cls._CJK_ROLE_VI:
                    return cls._CJK_ROLE_VI[part]
        return cleaned

    @classmethod
    def _localize_speaker(
        cls,
        speaker: str,
        roster_map: dict[str, str] | None = None,
        canonical_map: dict[str, str] | None = None,
    ) -> str:
        """Chuẩn hoá nhãn người nói cho tag, GIỮ chú thích kênh thoại.

        Xử lý dạng 'TÊN (chú thích)' (vd 'MAN (on computer)', 'ASTRONAUT (on radio)'):
        - Dịch TÊN qua :meth:`_localize_one_name` (nhãn chung sang VI, CJK phiên âm).
        - Dịch CHÚ THÍCH kênh thoại qua bảng :attr:`_SPEAKER_CHANNEL_VI` nếu nhận ra,
          ngược lại giữ nguyên; luôn trả về đúng định dạng 'Tên (chú thích)'.
        Ví dụ: 'MAN (on computer)' → 'Người đàn ông (trên máy tính)'.
        """
        raw = speaker.strip().strip("[]:").strip()
        # Tách 'TÊN (chú thích)' — chú thích là cụm trong ngoặc đơn cuối chuỗi.
        match = re.match(r"^(.*?)\s*[(（]\s*(.*?)\s*[)）]\s*$", raw)  # noqa: RUF001
        if match:
            name_part = match.group(1).strip()
            note_part = match.group(2).strip()
            localized_name = cls._localize_one_name(name_part, roster_map, canonical_map)
            # [v3.23.86] Bỏ chú thích khi nó chỉ là ECHO của TÊN (model hay xuất
            # 'Tần Chính (Tần Chính)' hoặc 'Tần Chính (秦政)') — tránh tag lặp tên.
            if cls._is_redundant_name_note(
                name_part, note_part, localized_name, canonical_map
            ):
                return localized_name
            note_key = note_part.lower().strip()
            localized_note = cls._SPEAKER_CHANNEL_VI.get(note_key, note_part)
            if localized_name and localized_note:
                return f"{localized_name} ({localized_note})"
            return localized_name or localized_note
        return cls._localize_one_name(raw, roster_map, canonical_map)

    @classmethod
    def _is_redundant_name_note(
        cls,
        name_part: str,
        note_part: str,
        localized_name: str,
        canonical_map: dict[str, str] | None,
    ) -> bool:
        """True nếu chú thích trong ngoặc chỉ là echo của TÊN (không phải kênh thoại).

        Bắt các trường hợp: trùng tên gốc/đã địa phương hoá; là tên CJK gốc (chú thích
        kênh thoại không bao giờ là CJK); hoặc cùng quy về MỘT tên chuẩn trong roster.
        """
        if not note_part:
            return False
        note_key = cls._normalize_name_key(note_part)
        # 1. Trùng tên gốc hoặc tên đã địa phương hoá.
        if note_key in {
            cls._normalize_name_key(name_part),
            cls._normalize_name_key(localized_name),
        }:
            return True
        # 2. Chú thích là CJK -> echo tên gốc (kênh thoại luôn ở tiếng đích).
        if cls._has_cjk(note_part):
            return True
        # 3. Cùng quy về một tên chuẩn trong roster.
        if canonical_map:
            note_canon = canonical_map.get(note_key)
            name_canon = canonical_map.get(cls._normalize_name_key(localized_name))
            if note_canon and note_canon == name_canon:
                return True
        return False

    @staticmethod
    def _is_sound_effect(text: str) -> bool:
        """True nếu dòng CHỈ là hiệu ứng âm thanh/nhạc (toàn bộ trong ngoặc đơn).

        Vd '(nhạc kịch tính)', '(tiếng chuông)', '[âm thanh nền]'. Các dòng này KHÔNG
        gắn với người nói nên không bao giờ tag tên.
        """
        stripped = text.strip()
        if not stripped:
            return False
        return (
            (stripped.startswith("(") and stripped.endswith(")"))
            or (stripped.startswith("[") and stripped.endswith("]"))
            or (stripped.startswith("♪") or stripped.endswith("♪"))
        )

    @classmethod
    def _compose_display_text(
        cls,
        line: TranslationLine | None,
        fallback_text: str,
        context: TranslationContext,
        last_speaker: str = "",
        roster_map: dict[str, str] | None = None,
        canonical_map: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """Ghép text hiển thị, trả về (text, người_nói_hiện_tại để theo dõi tiếp).

        [v3.23.20] Quy tắc tag tên người nói:
          * KHÔNG tag dòng hiệu ứng âm thanh/nhạc (toàn bộ trong ngoặc).
          * KHÔNG tag người nói rỗng/N/A/không xác định.
          * CHỈ tag khi người nói ĐỔI so với dòng liền trước (cùng người → bỏ tag).
          * [v3.23.25] Nhãn chung dịch sang VI; tên CJK phiên âm qua roster; Latin giữ.
        """
        if line is None:
            return fallback_text, last_speaker

        body = line.text
        speaker = line.speaker.strip()
        speaker_norm = speaker.lower()
        is_sfx = cls._is_sound_effect(body)

        # Hiệu ứng âm thanh → reset người nói (lời thoại tiếp theo coi như đổi người),
        # không tag, vẫn giữ phần mô tả nếu bật.
        if is_sfx:
            prefix = ""
            if context.include_desc and line.description.strip():
                prefix += f"({line.description.strip()}) "
            return (f"{prefix}{body}".strip() or fallback_text), ""

        prefix = ""
        valid_speaker = bool(speaker) and speaker_norm not in cls._EMPTY_SPEAKERS
        # [v3.23.136] So sánh "đổi người nói" theo tên ĐÃ CHUẨN HOÁ (localized), không
        # theo chuỗi thô. Cùng một người dưới biến thể tên (tên CJK vs phiên âm, hoa/
        # thường, alias trong roster) trước đây bị coi là người KHÁC -> lặp nhãn thừa.
        localized = (
            cls._localize_speaker(speaker, roster_map, canonical_map)
            if valid_speaker
            else ""
        )
        current_speaker = localized
        # Chỉ tag khi có người nói hợp lệ VÀ khác người nói trước đó (theo tên chuẩn hoá).
        if context.enable_tags and valid_speaker and localized != last_speaker:
            prefix += f"[{localized}:] "
        if context.include_desc and line.description.strip():
            prefix += f"({line.description.strip()}) "
        composed = f"{prefix}{body}".strip() or fallback_text
        composed = cls._dedupe_leading_speaker_tag(composed)
        return composed, current_speaker

    @staticmethod
    def _dedupe_leading_speaker_tag(text: str) -> str:
        """[v3.23.128] Gộp nhãn người nói '[X:] [X:]' bị lặp liền nhau thành một.

        Nguồn có nhãn người nói INLINE (vd 'NARRATOR:') khiến model vừa dịch nhãn vào
        text, vừa điền trường speaker → khi ghép prefix bị lặp '[Tên:] [Tên:]'. Hàm này
        gộp các nhãn dạng '[...:]' GIỐNG HỆT đứng liền đầu câu về một nhãn duy nhất.
        """
        if not text:
            return text
        return re.sub(r"^(\[[^\]]+:\]\s*)\1+", r"\1", text)


__all__ = [
    "TranslateSubtitlesRequest",
    "TranslateSubtitlesResponse",
    "TranslateSubtitlesUseCase",
    "OverallProgressCallback",
    "_compute_checkpoint_key",
    "_compute_legacy_checkpoint_key",
    "_StageCheckpoint",
    "_events_to_lines_helper",
    "_lines_to_events_helper",
    "StageRun",
    "build_stage_cache_scope",
    "checkpoint_identity",
    "context_fingerprint",
    "stage_config_signature",
    "stage_input_fingerprint",
]


def _events_to_lines_helper(events: list[SubtitleEvent]) -> list[TranslationLine]:
    """Helper module-level để tầng ngoài chuyển đổi event → line."""
    return TranslateSubtitlesUseCase._events_to_lines(events)


def _lines_to_events_helper(
    events: list[SubtitleEvent],
    lines: list[TranslationLine],
    context: TranslationContext,
) -> list[SubtitleEvent]:
    """[v3.23.608] Dòng dịch của một giai đoạn → phụ đề hiển thị (như kết quả dịch).

    Args:
        events: Phụ đề gốc (giữ thời gian/vị trí).
        lines: Dòng của giai đoạn.
        context: Ngữ cảnh (bảng nhân vật cho nhãn người nói, cờ bật nhãn).

    Returns:
        Phụ đề đã dịch.
    """
    return TranslateSubtitlesUseCase._lines_to_events(events, lines, context)
