"""Dependency Injection container — composition root."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from subtitles_extractor.application.use_cases.reocr import ReOcrUseCase
    from subtitles_extractor.application.use_cases.extract_embedded_subtitles import (
        ExtractEmbeddedSubtitlesUseCase,
    )
    from subtitles_extractor.application.use_cases.load_sidecar_subtitles import (
        LoadSidecarSubtitlesUseCase,
    )
    from subtitles_extractor.application.use_cases.transcribe_speech import (
        TranscribeSpeechUseCase,
    )

from subtitles_extractor.application.dtos.extract_subtitles_dto import (
    SubtitleBuilderConfig,
)
from subtitles_extractor.application.services.subtitle_builder import SubtitleBuilder
from subtitles_extractor.application.services.subtitle_editor_service import (
    SubtitleEditorService,
)
from subtitles_extractor.application.use_cases.detect_auto_roi import DetectAutoRoiUseCase
from subtitles_extractor.application.use_cases.detect_hardsub import DetectHardsubUseCase
from subtitles_extractor.application.use_cases.export_subtitles import (
    ExportSubtitlesUseCase,
)
from subtitles_extractor.application.use_cases.extract_subtitles import (
    ExtractSubtitlesUseCase,
)
from subtitles_extractor.application.use_cases.import_subtitles import (
    ImportSubtitlesUseCase,
)
from subtitles_extractor.application.use_cases.load_video_metadata import (
    LoadVideoMetadataUseCase,
)
from subtitles_extractor.domain.exceptions import OcrModelLoadError
from subtitles_extractor.domain.ports.auto_roi_detector_port import AutoRoiDetectorPort
from subtitles_extractor.domain.ports.frame_sampler_port import FrameSamplerPort
from subtitles_extractor.domain.ports.hardsub_detector_port import HardsubDetectorPort
from subtitles_extractor.domain.ports.ocr_engine_port import (
    OcrEngineConfig,
    OcrEnginePort,
)
from subtitles_extractor.domain.ports.settings_repository_port import (
    SettingsRepositoryPort,
)
from subtitles_extractor.domain.ports.subtitle_exporter_port import SubtitleExporterPort
from subtitles_extractor.domain.ports.subtitle_importer_port import SubtitleImporterPort
from subtitles_extractor.domain.ports.subtitle_repository_port import (
    SubtitleRepositoryPort,
)
from subtitles_extractor.domain.ports.translator_port import TranslatorPort
from subtitles_extractor.domain.ports.video_metadata_reader_port import (
    VideoMetadataReaderPort,
)
from subtitles_extractor.domain.ports.video_state_repository_port import (
    VideoStateRepositoryPort,
)
from subtitles_extractor.domain.value_objects.device_kind import DeviceKind, PrecisionMode
from subtitles_extractor.infrastructure.database.sqlite_subtitle_repository import (
    SqliteSubtitleRepository,
)
from subtitles_extractor.infrastructure.database.sqlite_video_state_repository import (
    SqliteVideoStateRepository,
)
from subtitles_extractor.infrastructure.database.sqlite_project_repository import (
    SqliteProjectRepository,
)
from subtitles_extractor.domain.ports.project_repository_port import (
    ProjectRepositoryPort,
)
from subtitles_extractor.infrastructure.logging.qt_log_handler import QtLogBridge
from subtitles_extractor.infrastructure.i18n.json_translator import JsonTranslator
from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import PaddleOcrAdapter
from subtitles_extractor.infrastructure.settings.settings_service import SettingsService
from subtitles_extractor.infrastructure.subtitle.exporters.ass_exporter import AssExporter
from subtitles_extractor.infrastructure.subtitle.exporters.srt_exporter import SrtExporter
from subtitles_extractor.infrastructure.subtitle.importers.ass_importer import AssImporter
from subtitles_extractor.infrastructure.subtitle.importers.srt_importer import SrtImporter
from subtitles_extractor.infrastructure.video.decoders.mpv_frame_sampler import (
    MpvFrameSampler,
)
from subtitles_extractor.infrastructure.video.decoders.opencv_frame_sampler import (
    OpenCvFrameSampler,
)
from subtitles_extractor.infrastructure.video.gradient_hardsub_detector import (
    GradientHardsubDetector,
)
from subtitles_extractor.infrastructure.video.mpv_metadata_reader import MpvMetadataReader
from subtitles_extractor.infrastructure.video.mpv_options_builder import build_mpv_kwargs
from subtitles_extractor.infrastructure.video.ocr_based_auto_roi_detector import (
    OcrBasedAutoRoiDetector,
)
from subtitles_extractor.infrastructure.video.opencv_metadata_reader import (
    OpenCvMetadataReader,
)

logger = logging.getLogger(__name__)

class ApplicationContainer:
    def __init__(
        self,
        settings_repository: SettingsRepositoryPort,
        i18n_data_dir: Path,
        default_locale: str = "vi",
        user_data_dir: Path | None = None,
    ) -> None:
        self._settings_repository = settings_repository
        self._i18n_data_dir = i18n_data_dir
        self._default_locale = default_locale
        self._user_data_dir = user_data_dir or Path.home() / ".subtitles_extractor"

        self._settings_service: SettingsService | None = None
        self._translator: TranslatorPort | None = None
        self._metadata_reader: VideoMetadataReaderPort | None = None
        self._frame_sampler: FrameSamplerPort | None = None
        self._ocr_engine: OcrEnginePort | None = None
        self._exporters: dict[str, SubtitleExporterPort] | None = None
        self._importers: dict[str, SubtitleImporterPort] | None = None
        self._hardsub_detector: HardsubDetectorPort | None = None
        self._auto_roi_detector: AutoRoiDetectorPort | None = None
        self._video_state_repo: VideoStateRepositoryPort | None = None
        self._subtitle_repo: SubtitleRepositoryPort | None = None
        self._project_repo: ProjectRepositoryPort | None = None
        self._log_bridge: QtLogBridge | None = None

        import threading
        self._preload_thread: threading.Thread | None = None
        self._configure_nlp_adapter()

    @property
    def user_data_dir(self) -> Path:
        """Thư mục dữ liệu người dùng (checkpoints, logs, database)."""
        return self._user_data_dir

    @property
    def settings_service(self) -> SettingsService:
        if self._settings_service is None:
            self._settings_service = SettingsService(self._settings_repository)
        return self._settings_service

    @property
    def video_state_repository(self) -> VideoStateRepositoryPort:
        if self._video_state_repo is None:
            db_path = self._user_data_dir / "app_state.db"
            self._video_state_repo = SqliteVideoStateRepository(db_path)
        return self._video_state_repo

    @property
    def subtitle_repository(self) -> SubtitleRepositoryPort:
        if self._subtitle_repo is None:
            db_path = self._user_data_dir / "app_state.db"
            self._subtitle_repo = SqliteSubtitleRepository(db_path)
        return self._subtitle_repo

    @property
    def project_repository(self) -> ProjectRepositoryPort:
        """Kho dự án Auto-Dubbing (theo hash video) — liên thông các khâu."""
        if self._project_repo is None:
            db_path = self._user_data_dir / "app_state.db"
            self._project_repo = SqliteProjectRepository(db_path)
        return self._project_repo

    @property
    def translation_session_store(self):
        """[v3.23.15] Phiên dịch theo video: phân tích, bản dịch từng giai đoạn, file
        cloud — để mở lại video là khôi phục, dịch tiếp/dịch lại từng khâu được."""
        existing = getattr(self, "_translation_session_store", None)
        if existing is None:
            from subtitles_extractor.infrastructure.database.sqlite_translation_session_store import (
                SqliteTranslationSessionStore,
            )

            db_path = self._user_data_dir / "app_state.db"
            existing = SqliteTranslationSessionStore(db_path)
            self._translation_session_store = existing
        return existing

    @property
    def translation_memory_store(self):
        """[v3.23.55] Bộ nhớ dịch theo phim bộ (Translation Memory) — tích luỹ cặp câu đã
        dịch qua các tập để truy hồi làm tham chiếu, giữ nhất quán tên riêng/thuật ngữ."""
        existing = getattr(self, "_translation_memory_store", None)
        if existing is None:
            from subtitles_extractor.infrastructure.database.sqlite_translation_memory_store import (
                SqliteTranslationMemoryStore,
            )

            db_path = self._user_data_dir / "app_state.db"
            existing = SqliteTranslationMemoryStore(db_path)
            self._translation_memory_store = existing
        return existing

    @property
    def log_bridge(self) -> QtLogBridge:
        """Cầu nối log Loguru→Qt cho trang Nhật ký (khởi tạo một lần)."""
        if self._log_bridge is None:
            self._log_bridge = QtLogBridge()
        return self._log_bridge

    @property
    def translator(self) -> TranslatorPort:
        if self._translator is None:
            current_locale = self.settings_service.current.ui.locale
            self._translator = JsonTranslator(
                data_dir=self._i18n_data_dir,
                default_locale=current_locale or self._default_locale,
            )
        return self._translator

    @property
    def metadata_reader(self) -> VideoMetadataReaderPort:
        if self._metadata_reader is None:
            self._metadata_reader = self._create_metadata_reader()
        return self._metadata_reader

    @property
    def frame_sampler(self) -> FrameSamplerPort:
        if self._frame_sampler is None:
            self._frame_sampler = self._create_frame_sampler()
        return self._frame_sampler

    @property
    def ocr_engine(self) -> OcrEnginePort:
        if self._ocr_engine is None:
            self._ocr_engine = PaddleOcrAdapter(self._build_ocr_config())
        return self._ocr_engine

    def preload_ocr_engine_async(self) -> None:
        import threading
        if self._ocr_engine is not None and self._ocr_engine.is_initialized:
            logger.debug("OCR engine đã được preload — skip.")
            return

        if getattr(self, "_preload_thread", None) is not None:
            existing_thread = self._preload_thread
            if existing_thread is not None and existing_thread.is_alive():
                logger.debug("Preload OCR engine đang chạy — skip.")
                return

        # [v3.23.1] KHÔNG "mồi" import paddle ở main thread nữa. Trước đây mồi để
        # tránh import đồng thời, nhưng nếu mồi FAIL giữa chừng (vd lỗi cuDNN),
        # PaddleX (PDX) đã init một phần → bg thread initialize() lại báo "PDX has
        # already been initialized". HEAVY_IMPORT_LOCK đã đủ chống import đồng thời.
        # Chỉ import MỘT lần trong bg thread.

        def _bg_preload() -> None:
            try:
                engine = self.ocr_engine
                if not engine.is_initialized:
                    engine.initialize()
                logger.info("Preload mô hình OCR thành công ở background thread.")
            except OcrModelLoadError as exc:
                logger.warning(
                    "Preload OCR engine thất bại: %s — sẽ thử lại lúc cần.",
                    exc,
                )
            except OSError as exc:
                exc_msg = str(exc)
                # [Windows] Lỗi nạp DLL của paddle (cuDNN/cuBLAS/shm) — thường do
                # xung đột với torch CUDA đã chiếm cuDNN trong process, hoặc paddle
                # tìm cuDNN sai chỗ. Thử lại MỘT lần sau khi dọn module paddle hỏng.
                # [v3.23] KHÔNG import torch ở đây (sẽ nạp cuDNN của torch → càng xung
                # đột với paddle). Process chính phải sạch torch.
                if (
                    "shm.dll" in exc_msg
                    or "WinError 127" in exc_msg
                    or "cudnn" in exc_msg.lower()
                    or "cublas" in exc_msg.lower()
                    or ".dll" in exc_msg
                ):
                    logger.info(
                        "Paddle nạp DLL lỗi (%s). Dọn module hỏng & thử lại OCR…",
                        "cuDNN/cuBLAS" if "cud" in exc_msg.lower() else "Windows DLL",
                    )
                    # [v3.22.8] Nếu paddle bị nửa-khởi-tạo trong sys.modules, phải XOÁ
                    # nó đi rồi import lại, nếu không retry vẫn thấy module hỏng.
                    import sys as _sys

                    for _mod_name in list(_sys.modules):
                        if _mod_name == "paddle" or _mod_name.startswith("paddle."):
                            del _sys.modules[_mod_name]

                    # Retry OCR khởi tạo
                    try:
                        engine = self.ocr_engine
                        if not engine.is_initialized:
                            engine.initialize()
                        logger.info("Preload OCR engine thành công sau retry.")
                    except Exception as retry_exc:  # noqa: BLE001
                        logger.warning(
                            "OCR engine không preload được sau retry: %s — sẽ thử lại lúc cần. "
                            "Nếu cài torch CUDA: đảm bảo cuDNN của paddle khớp (xem README).",
                            retry_exc,
                        )
                else:
                    logger.warning("Preload OCR engine lỗi hệ thống: %s.", exc)
            except (RuntimeError, ImportError) as exc:
                logger.warning("Preload OCR engine lỗi: %s.", exc)

        thread = threading.Thread(
            target=_bg_preload, name="OcrPreloadThread", daemon=True
        )
        self._preload_thread = thread
        thread.start()

    @property
    def exporters(self) -> dict[str, SubtitleExporterPort]:
        if self._exporters is None:
            self._exporters = {"srt": SrtExporter(), "ass": AssExporter()}
        return self._exporters

    @property
    def importers(self) -> dict[str, SubtitleImporterPort]:
        if self._importers is None:
            _ass_importer = AssImporter()
            # [v3.23.168] .ssa dùng chung parser với .ass -> map cùng importer.
            self._importers = {
                "srt": SrtImporter(), "ass": _ass_importer, "ssa": _ass_importer,
            }
        return self._importers

    @property
    def hardsub_detector(self) -> HardsubDetectorPort:
        if self._hardsub_detector is None:
            self._hardsub_detector = GradientHardsubDetector()
        return self._hardsub_detector

    @property
    def auto_roi_detector(self) -> AutoRoiDetectorPort:
        if self._auto_roi_detector is None:
            self._auto_roi_detector = OcrBasedAutoRoiDetector(
                self.ocr_engine,
                self.frame_sampler,
                analyzer_kwargs_provider=self._build_auto_roi_analyzer_kwargs,
            )
        return self._auto_roi_detector

    def _build_auto_roi_analyzer_kwargs(self) -> dict[str, Any]:
        """[v3.19] Đọc tham số tinh chỉnh Auto-ROI từ Settings (tại thời điểm gọi).

        Người dùng đổi cài đặt → lần dò ROI kế tiếp dùng giá trị mới ngay, không
        cần khởi động lại ứng dụng.
        """
        roi_settings = self.settings_service.current.roi
        return {
            "enable_band_refinement": roi_settings.auto_enable_band_refinement,
            "band_keep_ratio": roi_settings.auto_band_keep_ratio,
            "band_extend_ratio": roi_settings.auto_band_extend_ratio,
            "bottom_padding_factor": roi_settings.auto_bottom_padding_factor,
            "heatmap_threshold_multiplier": roi_settings.auto_sensitivity_multiplier,
        }

    def build_mpv_player_kwargs(self) -> dict[str, Any]:
        return build_mpv_kwargs(self.settings_service.current.mpv, role="player")

    def _configure_nlp_adapter(self) -> None:
        try:
            from subtitles_extractor.infrastructure.nlp.fastembed_adapter import (
                FastEmbedAdapter,
            )
            adapter = FastEmbedAdapter()
            settings = self.settings_service.current.nlp
            adapter.configure(
                enabled=settings.enable_vector_embeddings,
                model_name=settings.model_name,
                mode=settings.similarity_mode
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.debug("Bỏ qua nạp NLP adapter do thiếu module: %s.", exc)
        except (AttributeError, ValueError, RuntimeError) as exc:
            logger.warning(
                "Cấu hình NLP adapter thất bại — vẫn tiếp tục: %s.", exc,
            )

    def reset_database(self) -> dict[str, list[str]]:
        """[v3.23.88] Dọn DB về "như mới tạo": xoá sạch dữ liệu mọi bảng.

        Vô hiệu hoá các store đang cache (để lần sau tạo mới, xoá cache in-memory) rồi
        xoá dữ liệu trong ``app_state.db`` và cache tải video ``gemini_video_uploads.db``.
        GIỮ schema để tránh xung đột tệp với kết nối đang mở.

        Returns:
            Map tên tệp DB -> danh sách bảng đã xoá dữ liệu.
        """
        from subtitles_extractor.infrastructure.database.maintenance import (
            reset_database as _reset_db,
        )

        # Bỏ tham chiếu store đang cache -> lần truy cập sau khởi tạo lại (cache sạch).
        self._video_state_repo = None
        self._subtitle_repo = None
        self._project_repo = None
        self._translation_session_store = None
        self._translation_memory_store = None

        cleared: dict[str, list[str]] = {}
        for db_name in ("app_state.db", "gemini_video_uploads.db"):
            db_path = self._user_data_dir / db_name
            if db_path.exists():
                cleared[db_name] = _reset_db(db_path)
        return cleared

    def apply_settings_changes(self) -> None:
        self._metadata_reader = None
        self._frame_sampler = None
        self._auto_roi_detector = None

        snapshot = self.settings_service.current
        self._configure_nlp_adapter()

        if snapshot.advanced.disable_paddle_network_check:
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        else:
            os.environ.pop("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", None)

        if self._ocr_engine is not None:
            new_ocr_cfg = self._build_ocr_config()
            apply_cfg = getattr(self._ocr_engine, "apply_config", None) or getattr(
                self._ocr_engine, "update_preprocess_config", None
            )
            if callable(apply_cfg):
                apply_cfg(new_ocr_cfg)
            else:
                logger.debug(
                    "OCR adapter %s không hỗ trợ apply_config — "
                    "tái khởi tạo từ đầu khi cần.",
                    type(self._ocr_engine).__name__,
                )
                self._ocr_engine = None

    def make_extract_subtitles_use_case(self) -> ExtractSubtitlesUseCase:
        builder = SubtitleBuilder(config=self._build_subtitle_builder_config())
        return ExtractSubtitlesUseCase(
            metadata_reader=self.metadata_reader, frame_sampler=self.frame_sampler,
            ocr_engine=self.ocr_engine, builder=builder, exporters=self.exporters,
        )

    def make_reocr_use_case(self) -> ReOcrUseCase:
        from subtitles_extractor.application.use_cases.reocr import ReOcrUseCase
        return ReOcrUseCase(
            extract_use_case=self.make_extract_subtitles_use_case(),
            load_metadata_use_case=self.make_load_video_metadata_use_case(),
        )

    def make_export_subtitles_use_case(self) -> ExportSubtitlesUseCase:
        return ExportSubtitlesUseCase(exporters=self.exporters)

    def make_import_subtitles_use_case(self) -> ImportSubtitlesUseCase:
        return ImportSubtitlesUseCase(importers=self.importers)

    def make_load_sidecar_subtitles_use_case(self) -> "LoadSidecarSubtitlesUseCase":
        """[v3.23.168] Use case dò & nạp phụ đề rời cùng tên cạnh video."""
        from subtitles_extractor.application.use_cases.load_sidecar_subtitles import (
            LoadSidecarSubtitlesUseCase,
        )
        return LoadSidecarSubtitlesUseCase(
            import_use_case=self.make_import_subtitles_use_case()
        )

    def make_extract_embedded_use_case(
        self, ocr_paddle_lang: str | None = None
    ) -> "ExtractEmbeddedSubtitlesUseCase":
        """[v3.21] Use case trích phụ đề NHÚNG (text + bitmap OCR qua PaddleOCR).

        Args:
            ocr_paddle_lang: Mã ngôn ngữ PaddleOCR cho OCR ảnh (vd ``"en"``, ``"japan"``).
                ``None``/rỗng -> dùng engine OCR mặc định của người dùng. Nếu khác ngôn ngữ
                mặc định -> dựng engine riêng đúng ngôn ngữ (tránh nhiễu □ do sai model).
        """
        from subtitles_extractor.application.use_cases.extract_embedded_subtitles import (
            ExtractEmbeddedSubtitlesUseCase,
        )
        from subtitles_extractor.infrastructure.media import find_ffmpeg, find_ffprobe
        from subtitles_extractor.infrastructure.video.ffmpeg_embedded_subtitle_adapter import (
            FfmpegEmbeddedSubtitleAdapter,
        )

        ocr_engine = self._get_embedded_ocr_engine(ocr_paddle_lang)
        # [v3.23.297] Ưu tiên ffmpeg/ffprobe ĐÃ NHÚNG (bundle-first) để trích phụ đề
        # nhúng chạy trên máy standalone không có ffmpeg trên PATH.
        return ExtractEmbeddedSubtitlesUseCase(
            embedded_port=FfmpegEmbeddedSubtitleAdapter(
                ffprobe_binary=find_ffprobe() or "ffprobe",
                ffmpeg_binary=find_ffmpeg() or "ffmpeg",
            ),
            ocr_engine=ocr_engine,
            ocr_batch_size=self._build_ocr_config().batch_size,
        )

    def _get_embedded_ocr_engine(self, ocr_paddle_lang: str | None):
        """Trả engine OCR cho ngôn ngữ yêu cầu; cache theo ngôn ngữ.

        Nếu ngôn ngữ rỗng hoặc trùng ngôn ngữ engine mặc định -> tái dùng ``self.ocr_engine``.
        Ngược lại dựng :class:`PaddleOcrAdapter` riêng (model nhận dạng theo ngôn ngữ).
        """
        base_config = self._build_ocr_config()
        from subtitles_extractor.application.services.embedded_ocr_language import (
            is_covered_by_unified_model,
        )

        # Tái dùng engine chính (PP-OCRv6) nếu: không chỉ định / trùng ngôn ngữ mặc định /
        # ngôn ngữ đã nằm trong model hợp nhất (Trung/Nhật/Anh/Latin/Việt).
        if (
            not ocr_paddle_lang
            or ocr_paddle_lang == base_config.language
            or is_covered_by_unified_model(ocr_paddle_lang)
        ):
            return self.ocr_engine

        if not hasattr(self, "_lang_ocr_engines"):
            self._lang_ocr_engines: dict[str, Any] = {}
        cached = self._lang_ocr_engines.get(ocr_paddle_lang)
        if cached is not None:
            return cached

        from dataclasses import replace

        from subtitles_extractor.infrastructure.ocr.paddle_ocr_adapter import (
            PaddleOcrAdapter,
        )

        # Xoá tên model det/rec để PaddleOCR tự chọn model đúng theo 'lang'.
        lang_config = replace(
            base_config,
            language=ocr_paddle_lang,
            detection_model_name="",
            recognition_model_name="",
        )
        logger.info(
            "Dựng engine OCR riêng cho phụ đề nhúng: lang=%s.", ocr_paddle_lang
        )
        engine = PaddleOcrAdapter(lang_config)
        self._lang_ocr_engines[ocr_paddle_lang] = engine
        return engine

    def make_transcribe_speech_use_case(self) -> "TranscribeSpeechUseCase":
        """[v3.21] Use case phiên âm giọng nói (WhisperX) thành phụ đề."""
        from subtitles_extractor.application.use_cases.transcribe_speech import (
            TranscribeSpeechUseCase,
        )
        from subtitles_extractor.infrastructure.stt.whisperx_adapter import WhisperXAdapter

        return TranscribeSpeechUseCase(stt_engine=WhisperXAdapter())

    def make_load_video_metadata_use_case(self) -> LoadVideoMetadataUseCase:
        return LoadVideoMetadataUseCase(reader=self.metadata_reader)

    def make_detect_hardsub_use_case(self) -> DetectHardsubUseCase:
        return DetectHardsubUseCase(detector=self.hardsub_detector)

    def make_detect_auto_roi_use_case(self) -> DetectAutoRoiUseCase:
        return DetectAutoRoiUseCase(detector=self.auto_roi_detector)

    def make_subtitle_editor_service(self) -> SubtitleEditorService:
        return SubtitleEditorService()

    def make_subtitle_translator(
        self, api_key: str, retry_count: int | None = None,
        request_timeout_s: float | None = None,
    ):
        """Tạo adapter dịch phụ đề bằng Gemini.

        Import lazy để ứng dụng không phụ thuộc cứng vào ``google-genai``:
        nếu thiếu thư viện, adapter vẫn khởi tạo nhưng ``is_available()`` = False.

        Args:
            api_key:           Khoá API Gemini.
            retry_count:       Số lần retry tối đa; None → lấy từ Cài đặt.
            request_timeout_s: Thời gian chờ tối đa mỗi request (giây); None → từ Cài đặt.
        """
        from subtitles_extractor.infrastructure.translation import GeminiSubtitleTranslator

        # [v3.23.39] Lấy retry/timeout từ Cài đặt khi caller không chỉ định.
        tr = self.settings_service.current.translation
        if retry_count is None:
            retry_count = tr.retry_count
        if request_timeout_s is None:
            request_timeout_s = float(tr.request_timeout_sec)
        return GeminiSubtitleTranslator(
            api_key=api_key, retry_count=retry_count, request_timeout_s=request_timeout_s,
            quota_manager=self.gemini_quota_manager,
            analysis_media_resolution=getattr(tr, "analysis_media_resolution", "medium"),
            analysis_thinking_level=getattr(tr, "analysis_thinking_level", "medium"),
            parallel_batches=getattr(tr, "translation_parallel_batches", 1),
        )

    @property
    def gemini_quota_manager(self):
        """Bộ điều tiết quota Gemini DÙNG CHUNG (đặt chỗ token, tránh vượt rate-limit).

        Dùng chung một thể hiện cho mọi adapter/giai đoạn để cửa sổ RPM/TPM/RPD được
        đếm thống nhất trong cả phiên dịch.
        """
        existing = getattr(self, "_gemini_quota_manager", None)
        if existing is None:
            from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
                GeminiQuotaManager,
            )
            existing = GeminiQuotaManager(
                state_path=self._user_data_dir / "gemini_quota_state.json"
            )
            self._load_quota_limits(existing)
            self._gemini_quota_manager = existing
        return existing

    @property
    def quota_limits_path(self) -> "Path":
        """Đường dẫn tệp JSON lưu giới hạn quota tuỳ chỉnh của người dùng."""
        return self._user_data_dir / "gemini_quota_limits.json"

    def _load_quota_limits(self, manager) -> None:
        """[v3.23.122] Nạp giới hạn quota tuỳ chỉnh (nếu có) từ JSON khi tạo manager."""
        import json
        path = self.quota_limits_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Không đọc được giới hạn quota tuỳ chỉnh: %s", exc)
            return
        if isinstance(data, dict):
            manager.replace_limits_dict(data)
            logger.info("Đã nạp %d giới hạn quota tuỳ chỉnh.", len(data))

    def save_quota_limits(self, limits: dict) -> None:
        """[v3.23.122] Lưu giới hạn quota tuỳ chỉnh ra JSON VÀ áp ngay vào manager dùng chung."""
        import json
        path = self.quota_limits_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(limits, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.gemini_quota_manager.replace_limits_dict(limits)
        logger.info("Đã lưu %d giới hạn quota tuỳ chỉnh.", len(limits))

    def make_video_context_provider(self, api_key: str):
        """Tạo provider chuẩn bị + tải đoạn video ngữ cảnh lên Gemini.

        Cache tải lên (hash→tên file cloud) lưu cùng thư mục dữ liệu người dùng để tái
        sử dụng đoạn đã tải giữa các phiên.
        """
        from subtitles_extractor.infrastructure.translation.gemini_video_context import (
            GeminiVideoContextProvider,
            video_tokens_per_sec,
        )

        cache_db = self._user_data_dir / "gemini_video_uploads.db"
        # [v3.23.39] Đọc tham số video ngữ cảnh từ Cài đặt (người dùng chỉnh được):
        # độ phân giải/fps/chất lượng nén + token mỗi đoạn. Phân tích chạy TUẦN TỰ nên
        # giữ TẤT CẢ đoạn (phủ kín toàn phim); mỗi đoạn ≤ tokens_per_chunk/request.
        vc = self.settings_service.current.video_context
        tr = self.settings_service.current.translation
        # [v3.23.141] Số token/giây PHẢI khớp media_resolution dùng khi PHÂN TÍCH — nếu
        # không, đoạn cắt theo mức 100 (low) nhưng gửi ở medium (300) sẽ vượt TPM -> 429.
        # Chia đoạn theo token/giây thực tế để mỗi đoạn nằm gọn trong ngân sách TPM.
        analysis_res = getattr(tr, "analysis_media_resolution", "low")
        res_tokens_per_sec = max(
            int(getattr(vc, "tokens_per_second", 100)),
            video_tokens_per_sec(analysis_res),
        )
        # [v3.23.141] Gắn ĐỘ PHÂN GIẢI NÉN theo media_resolution. media_resolution quyết
        # định token/frame (low 66, medium/high 258) nhưng KHÔNG tự nâng chi tiết ảnh: nếu
        # gửi 360p mà chọn high thì model cấp nhiều token cho frame NGHÈO chi tiết -> phí.
        # Dùng dạng SÀN (tôn trọng nếu người dùng đặt cao hơn): low 360p, medium 540p,
        # high 720p. Nâng độ phân giải KHÔNG tăng token (token theo media_resolution), chỉ
        # tăng dung lượng upload -> chỉ nâng khi media_resolution thực sự cần chi tiết.
        res_floor = {"low": 360, "medium": 540, "high": 720}.get(analysis_res, 360)
        effective_height = max(int(vc.resolution_height), res_floor)
        return GeminiVideoContextProvider(
            api_key=api_key,
            cache_db_path=cache_db,
            max_tokens_per_chunk=vc.tokens_per_chunk,
            max_chunk_minutes=vc.max_chunk_minutes,
            max_total_tokens=max(vc.tokens_per_chunk, 950_000),
            resolution_height=effective_height,
            fps=vc.fps,
            nvenc_cq=vc.nvenc_cq,
            cpu_crf=vc.cpu_crf,
            tokens_per_second=res_tokens_per_sec,
            chunk_cache_max_total_mb=getattr(vc, "chunk_cache_max_total_mb", 4096),
            chunk_cache_max_age_hours=getattr(vc, "chunk_cache_max_age_hours", 72),
        )

    def make_translate_subtitles_use_case(self, api_key: str, retry_count: int = 5):
        """Tạo use case dịch phụ đề với adapter Gemini đã cấu hình.

        Checkpoint được lưu tại ``{user_data_dir}/translation_checkpoints/`` để
        cho phép resume khi bị ngắt giữa chừng.
        """
        from subtitles_extractor.application.use_cases.translate_subtitles import (
            TranslateSubtitlesUseCase,
        )

        translator = self.make_subtitle_translator(api_key=api_key, retry_count=retry_count)
        checkpoint_dir = self._user_data_dir / "translation_checkpoints"
        return TranslateSubtitlesUseCase(translator=translator, checkpoint_dir=checkpoint_dir)

    def make_analyze_context_use_case(self, api_key: str):
        """Tạo use case phân tích ngữ cảnh toàn cục bằng Gemini."""
        from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
            AnalyzeSubtitleContextUseCase,
        )

        translator = self.make_subtitle_translator(api_key=api_key, retry_count=3)
        return AnalyzeSubtitleContextUseCase(translator=translator)

    # ── TTS ──────────────────────────────────────────────────────────────────

    def make_edge_tts_adapter(self):
        """Tạo adapter Edge TTS (online — Tiếng Việt và 300+ giọng khác)."""
        from subtitles_extractor.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter
        return EdgeTTSAdapter()

    def make_gemini_tts_adapter(self, api_key: str = ""):
        """Tạo adapter Gemini TTS (online — 30+ voices, 40+ ngôn ngữ)."""
        from subtitles_extractor.infrastructure.tts.gemini_tts_adapter import GeminiTTSAdapter
        return GeminiTTSAdapter(api_key=api_key)

    def make_vieneu_tts_adapter(
        self, mode: str = "standard", emotion: str = "natural", force_cpu: bool = True
    ):
        """Tạo adapter VieNeu-TTS (offline — TTS tiếng Việt chất lượng cao, voice cloning).

        Args:
            mode: Chế độ engine ("standard"/"turbo"/"v3turbo").
            emotion: Sắc thái giọng ("natural"/"storytelling").
            force_cpu: True (mặc định) ép chạy CPU/ONNX, né lỗi PyTorch/cuDNN.
        """
        from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
            VieNeuTtsAdapter,
        )
        return VieNeuTtsAdapter(mode=mode, emotion=emotion, force_cpu=force_cpu)

    def make_generate_tts_use_case(self, adapter=None):
        """Tạo use case TTS. Nếu không truyền adapter, dùng Edge TTS."""
        from subtitles_extractor.application.use_cases.generate_tts import GenerateTTSUseCase
        if adapter is None:
            adapter = self.make_edge_tts_adapter()
        return GenerateTTSUseCase(adapter=adapter)

    def shutdown(self) -> None:
        """Giải phóng tất cả resource: OCR engine, SQLite, settings."""
        if self._ocr_engine is not None:
            try:
                self._ocr_engine.release()
            except (RuntimeError, OSError) as exc:
                logger.warning("Lỗi khi release OCR engine: %s.", exc)
            self._ocr_engine = None

        # Đóng SQLite repos explicit để tránh dựa vào __del__.
        for repo_attr in ("_video_state_repo", "_subtitle_repo", "_project_repo"):
            repo = getattr(self, repo_attr, None)
            if repo is not None and hasattr(repo, "close"):
                try:
                    repo.close()
                except (RuntimeError, OSError) as exc:
                    logger.warning(
                        "Lỗi khi đóng SQLite repo %s: %s.", repo_attr, exc,
                    )
                setattr(self, repo_attr, None)

        if self._settings_service is not None:
            try:
                self._settings_service.flush()
            except (OSError, RuntimeError) as exc:
                logger.warning("Lỗi khi flush settings: %s.", exc)

    @staticmethod
    def _is_module_available(module_name: str) -> bool:
        from importlib.util import find_spec

        try:
            if find_spec(module_name) is None:
                return False
        except (ValueError, ModuleNotFoundError):
            return False

        # [v3.23.287] PyNvVideoCodec: find_spec trả True (có VersionCheck.pyd) nhưng import
        # THẬT có thể lỗi nếu thiếu .pyd chính (version-specific). Thử import thật để tránh
        # tạo sampler rồi crash runtime — cho phép fallback mpv/PyAV hoạt động đúng.
        if module_name == "PyNvVideoCodec":
            try:
                import PyNvVideoCodec  # noqa: F401
            except (ImportError, OSError, RuntimeError):
                return False
        return True

    def _create_metadata_reader(self) -> VideoMetadataReaderPort:
        snapshot = self.settings_service.current
        backend = snapshot.hardware.metadata_reader_backend

        if backend == "mpv" and self._is_module_available("mpv"):
            return MpvMetadataReader(
                mpv_options=build_mpv_kwargs(snapshot.mpv, role="metadata")
            )
        if backend in {"mpv", "pyav"} and self._is_module_available("av"):
            from subtitles_extractor.infrastructure.video.pyav_metadata_reader import (
                PyAvMetadataReader,
            )
            return PyAvMetadataReader()
        return OpenCvMetadataReader()

    def _create_frame_sampler(self) -> FrameSamplerPort:
        snapshot = self.settings_service.current
        backend = snapshot.hardware.frame_decoder_backend

        # [v3.23.288] NVDEC GPU-First CAN CuPy de xu ly frame tren GPU (from_dlpack voi CUDA
        # stream). Thieu CuPy -> np.from_dlpack(stream=0) gay loi "ExternalBuffer::dlpack
        # Passed value of 0". Chi chon PyNvVideoCodec khi co CA hai; nguoc lai fallback PyAV
        # (van dung NVDEC qua FFmpeg neu co, on dinh hon).
        if (
            backend == "pynvvideocodec"
            and self._is_module_available("PyNvVideoCodec")
            and self._is_module_available("cupy")
        ):
            from subtitles_extractor.infrastructure.video.decoders.pynvvideocodec_frame_sampler import (
                PyNvVideoCodecFrameSampler,
            )
            return PyNvVideoCodecFrameSampler(gpu_id=0)

        # [v3.23.289] Khi NVDEC khong kha dung (thieu CuPy) VA backend != 'mpv' tuong minh,
        # uu tien PyAV hon mpv cho sampling headless — PyAV on dinh hon (mpv vo=null co the
        # timeout voi vai codec/duong dan). Chi dung mpv khi nguoi dung chon 'mpv' tuong minh.
        if backend == "mpv" and self._is_module_available("mpv"):
            return MpvFrameSampler(
                mpv_options=build_mpv_kwargs(snapshot.mpv, role="sampler")
            )

        if self._is_module_available("av"):
            from subtitles_extractor.infrastructure.video.decoders.pyav_frame_sampler import (
                PyAvFrameSampler,
            )
            return PyAvFrameSampler()

        if backend in {"pynvvideocodec", "mpv"} and self._is_module_available("mpv"):
            return MpvFrameSampler(
                mpv_options=build_mpv_kwargs(snapshot.mpv, role="sampler")
            )

        return OpenCvFrameSampler()

    def _build_ocr_config(self) -> OcrEngineConfig:
        snapshot = self.settings_service.current
        from subtitles_extractor.application.services.ocr_model_resolver import (
            resolve_ocr_model_names,
        )
        from subtitles_extractor.domain.ports.ocr_engine_port import PreprocessConfig
        det_model, rec_model = resolve_ocr_model_names(snapshot.ocr)
        pre = snapshot.preprocess
        return OcrEngineConfig(
            device=DeviceKind(snapshot.hardware.device.value),
            detection_model_name=det_model,
            recognition_model_name=rec_model,
            language=snapshot.ocr.language,
            limit_side_len=snapshot.ocr.limit_side_len,
            limit_type=snapshot.ocr.limit_type,
            det_thresh=snapshot.ocr.det_thresh,
            det_box_thresh=snapshot.ocr.det_box_thresh,
            det_unclip_ratio=snapshot.ocr.det_unclip_ratio,
            score_threshold=snapshot.ocr.score_threshold,
            enable_mkldnn=snapshot.hardware.enable_mkldnn,
            use_tensorrt=snapshot.hardware.use_tensorrt,
            precision=PrecisionMode(snapshot.hardware.precision.value),
            batch_size=snapshot.hardware.batch_size_ocr,
            parallel_workers=snapshot.hardware.workers,
            use_textline_orientation=snapshot.ocr.use_textline_orientation,
            use_doc_orientation_classify=snapshot.ocr.use_doc_orientation_classify,
            use_doc_unwarping=snapshot.ocr.use_doc_unwarping,
            preprocess=PreprocessConfig(
                upscale_small_text=pre.upscale_small_text, upscale_target_height_px=pre.upscale_target_height_px,
                add_white_border=pre.add_white_border, border_thickness_px=pre.border_thickness_px,
                apply_sharpen=pre.apply_sharpen, apply_contrast_boost=pre.apply_contrast_boost,
                contrast_factor=pre.contrast_factor,
                apply_clahe=pre.apply_clahe,
                clahe_clip_limit=pre.clahe_clip_limit,
                clahe_tile_size=pre.clahe_tile_size,
            ),
        )

    def _build_subtitle_builder_config(self) -> SubtitleBuilderConfig:
        snapshot = self.settings_service.current
        post = snapshot.post_process
        threshold = snapshot.threshold

        return SubtitleBuilderConfig(
            similarity_threshold=min(post.similarity_threshold, threshold.text_similarity),
            min_duration_sec=post.min_duration_sec, max_duration_sec=post.max_duration_sec,
            min_confidence=threshold.ocr_min_confidence, merge_gap_sec=post.merge_gap_sec,
            use_viterbi=post.use_viterbi, viterbi_open_penalty=post.viterbi_open_penalty,
            min_text_chars=threshold.drop_short_text_chars, line_similarity_threshold=threshold.line_similarity,
            sample_step_sec=snapshot.frame.sample_step_sec,
            temporal_padding_sec=post.temporal_padding_sec,
            y_clustering_tolerance_ratio=post.y_clustering_tolerance_ratio,
            y_clustering_tolerance_min_px=post.y_clustering_tolerance_min_px,
            alignment_center_tolerance_ratio=post.alignment_center_tolerance_ratio,
            alignment_margin_tolerance_ratio=post.alignment_margin_tolerance_ratio,
            alignment_tolerance_min_px=post.alignment_tolerance_min_px,
        )

__all__ = ["ApplicationContainer"]
