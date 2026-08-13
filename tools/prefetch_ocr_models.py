"""Prefetch model PaddleOCR về một thư mục staging để nhúng vào bản đóng gói.

MỤC ĐÍCH
========
Cho phép OCR chạy **offline ngay lần đầu**: script này tải sẵn model
detection/recognition (+ textline orientation) của **một phiên bản OCR** cho
**toàn bộ ngôn ngữ mà UI phơi bày**, đặt vào thư mục staging theo đúng cấu trúc
``<staging>/official_models/<model>/`` mà PaddleX kỳ vọng. ``SubtitlesExtractor.spec``
sẽ nhúng thư mục này; lúc chạy, ``infrastructure/ocr/bundled_models.py`` trỏ
``PADDLE_PDX_CACHE_HOME`` về đó nên PaddleX dùng model local, không tải, không
chạm mạng.

CÁCH DÙNG (trong build_env đã cài paddleocr + có mạng)
------------------------------------------------------
    python tools/prefetch_ocr_models.py --staging models/paddle

Tùy chọn:
    --version PP-OCRv6_medium      Phiên bản OCR cần prefetch (mặc định: khớp
                                   OcrSettings.version của app).
    --languages ch en japan ...    Ghi đè danh sách ngôn ngữ (mặc định: toàn bộ
                                   ngôn ngữ trong UI_LANGUAGE_CHOICES của app).
    --device cpu|gpu               Thiết bị khởi tạo prefetch (mặc định cpu — chỉ
                                   cần tải file, không cần GPU).

GHI CHÚ THIẾT KẾ
----------------
* Chỉ prefetch **một** phiên bản (mặc định = bản mặc định của app, PP-OCRv6_medium
  là model hợp nhất gọn nhẹ phủ Trung/Nhật/Anh + 46 ngôn ngữ Latin). Các phiên
  bản nặng khác (PP-OCRv5 server…) cố ý KHÔNG prefetch để giữ dung lượng bundle
  trong tầm ~500MB–1GB; chúng vẫn tải theo yêu cầu khi người dùng đổi Cài đặt.
* Idempotent: PaddleX tự bỏ qua tải nếu model đã tồn tại trong staging.
* Không dùng ``print`` — mọi thứ qua ``logging`` để build log nhất quán.

Script này KHÔNG phải một phần runtime của app; nó là công cụ build độc lập.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("prefetch_ocr_models")

# Nạp muộn (sau khi _load_app_defaults() chèn src/ vào sys.path) — xem main().
SCRIPT_GROUP_OCR_VERSIONS: dict[str, str] = {}

# Thư mục con bắt buộc mà PaddleX kỳ vọng dưới cache home (khớp bundled_models.py).
_OFFICIAL_MODELS_SUBDIR: str = "official_models"
_PADDLE_CACHE_HOME_ENV: str = "PADDLE_PDX_CACHE_HOME"


class PrefetchError(RuntimeError):
    """Lỗi nghiệp vụ khi prefetch model (thiếu paddleocr, tải hỏng…)."""


def _configure_logging(verbose: bool) -> None:
    """Cấu hình logging tối giản cho công cụ build."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_app_defaults() -> tuple[str, list[str]]:
    """Đọc phiên bản OCR mặc định + danh sách ngôn ngữ UI từ chính mã nguồn app.

    Returns:
        Cặp ``(default_version, ui_language_codes)``. ``ui_language_codes`` đã
        loại bỏ mục "tự động" (mã rỗng) và khử trùng lặp, giữ nguyên thứ tự.

    Raises:
        PrefetchError: Khi không import được module app (chạy sai thư mục gốc).
    """
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from subtitles_extractor.application.services.embedded_ocr_language import (
            UI_LANGUAGE_CHOICES,
        )
        from subtitles_extractor.infrastructure.ocr.script_group_versions import (
            SCRIPT_GROUP_OCR_VERSIONS as _script_group_versions,
        )
        from subtitles_extractor.infrastructure.settings.application_settings import (
            OcrSettings,
        )
    except ImportError as import_error:  # noqa: TRY003 — thông điệp cụ thể hữu ích
        raise PrefetchError(
            "Không import được module app. Hãy chạy script từ thư mục gốc dự án "
            f"(nơi có thư mục 'src/'). Chi tiết: {import_error}"
        ) from import_error

    # [v3.23.303] Dùng CHUNG bảng ánh xạ với runtime để prefetch và lúc chạy khớp nhau.
    SCRIPT_GROUP_OCR_VERSIONS.update(_script_group_versions)

    default_version = OcrSettings().version

    seen: set[str] = set()
    language_codes: list[str] = []
    for _label, code in UI_LANGUAGE_CHOICES:
        if code and code not in seen:
            seen.add(code)
            language_codes.append(code)

    return default_version, language_codes


def _resolve_version_model_names(version: str) -> tuple[str, str]:
    """Suy tên model detection + recognition theo tên phiên bản OCR.

    Args:
        version: Ví dụ ``"PP-OCRv6_medium"``, ``"PP-OCRv5_server"``.

    Returns:
        Cặp ``(detection_model_name, recognition_model_name)`` theo quy ước đặt
        tên của PaddleOCR (``<version>_det`` / ``<version>_rec``).
    """
    return f"{version}_det", f"{version}_rec"


def _prefetch_single_language(
    *,
    paddle_ocr_cls: type,
    language: str,
    device: str,
) -> None:
    """Khởi tạo một pipeline PaddleOCR cho ``language`` để kích hoạt tải model.

    Việc khởi tạo ``PaddleOCR(lang=...)`` khiến PaddleX phân giải và tải bộ model
    detection + recognition đúng ngôn ngữ vào ``PADDLE_PDX_CACHE_HOME`` hiện hành.

    [v3.23.303] Với các NHÓM HỆ CHỮ VIẾT (latin/cyrillic/arabic/devanagari), phiên
    bản mặc định không có model nên lần đầu sẽ lỗi::

        No models are available for lang='latin' and ocr_version=None

    Khi đó thử lại với ``ocr_version`` tường minh (các nhóm này tồn tại dưới
    PP-OCRv5 dạng ``<nhóm>_PP-OCRv5_mobile_rec``).

    Args:
        paddle_ocr_cls: Lớp ``PaddleOCR`` đã import.
        language: Mã ngôn ngữ PaddleOCR (vd ``"ch"``, ``"japan"``).
        device: ``"cpu"`` hoặc ``"gpu"``.

    Raises:
        PrefetchError: Khi mọi cách khởi tạo đều thất bại.
    """
    logger.info("→ Prefetch model cho ngôn ngữ '%s'…", language)

    # use_textline_orientation=True để tải luôn model xoay dòng chữ (app bật mặc
    # định). doc_orientation/unwarping tắt — app không dùng, tránh tải dư.
    base_kwargs: dict[str, object] = {
        "lang": language,
        "device": device,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
    }

    attempts: list[dict[str, object]] = [dict(base_kwargs)]
    fallback_version = SCRIPT_GROUP_OCR_VERSIONS.get(language)
    if fallback_version:
        attempts.append({**base_kwargs, "ocr_version": fallback_version})

    last_error: Exception | None = None
    for attempt_index, kwargs in enumerate(attempts, start=1):
        try:
            paddle_ocr_cls(**kwargs)
        except Exception as init_error:  # noqa: BLE001 — build tool: gói lại thành lỗi rõ
            last_error = init_error
            if attempt_index < len(attempts):
                logger.info(
                    "  ↻ '%s' chưa có model ở phiên bản mặc định — thử lại với "
                    "ocr_version=%s.",
                    language,
                    attempts[attempt_index].get("ocr_version"),
                )
            continue
        else:
            logger.info("  ✓ Xong '%s'.", language)
            return

    raise PrefetchError(f"Không prefetch được ngôn ngữ '{language}': {last_error}")


def prefetch_models(
    *,
    staging_dir: Path,
    version: str,
    languages: list[str],
    device: str,
) -> Path:
    """Prefetch toàn bộ ngôn ngữ của một phiên bản OCR vào ``staging_dir``.

    Args:
        staging_dir: Thư mục gốc cache staging (sẽ chứa ``official_models/``).
        version: Phiên bản OCR (chỉ để log/kiểm chứng; PaddleOCR chọn model theo
            cấu hình mặc định của bản paddleocr đã cài).
        languages: Danh sách mã ngôn ngữ cần tải.
        device: ``"cpu"`` hoặc ``"gpu"``.

    Returns:
        Đường dẫn ``staging_dir/official_models`` sau khi hoàn tất.

    Raises:
        PrefetchError: Khi thiếu paddleocr hoặc bất kỳ ngôn ngữ nào tải hỏng.
    """
    staging_dir = staging_dir.resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Trỏ cache PaddleX về staging TRƯỚC khi import paddleocr (nó đọc env lúc import).
    os.environ[_PADDLE_CACHE_HOME_ENV] = str(staging_dir)
    logger.info("%s = %s", _PADDLE_CACHE_HOME_ENV, staging_dir)

    detection_model, recognition_model = _resolve_version_model_names(version)
    logger.info(
        "Phiên bản OCR mục tiêu: %s (det=%s, rec=%s).",
        version,
        detection_model,
        recognition_model,
    )

    try:
        from paddleocr import PaddleOCR
    except ImportError as import_error:
        raise PrefetchError(
            "Chưa cài paddleocr trong môi trường build. Chạy: pip install paddleocr"
        ) from import_error

    failures: list[str] = []
    for language in languages:
        try:
            _prefetch_single_language(
                paddle_ocr_cls=PaddleOCR, language=language, device=device
            )
        except PrefetchError as prefetch_error:
            # Ghi nhận và tiếp tục — một ngôn ngữ hỏng không nên chặn phần còn lại.
            logger.warning("%s", prefetch_error)
            failures.append(language)

    official_models_dir = staging_dir / _OFFICIAL_MODELS_SUBDIR
    if not official_models_dir.is_dir():
        raise PrefetchError(
            f"Sau prefetch không thấy thư mục '{official_models_dir}'. "
            "Kiểm tra mạng và phiên bản paddleocr."
        )

    downloaded = sorted(p.name for p in official_models_dir.iterdir() if p.is_dir())
    logger.info(
        "Hoàn tất: %d bộ model trong %s.", len(downloaded), official_models_dir
    )
    for name in downloaded:
        logger.debug("  • %s", name)

    if failures:
        logger.warning(
            "Có %d ngôn ngữ KHÔNG prefetch được (%s) — bản đóng gói sẽ tự tải "
            "chúng theo yêu cầu khi cần.",
            len(failures),
            ", ".join(failures),
        )

    return official_models_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    """Dựng parser tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Prefetch model PaddleOCR để nhúng offline vào bản đóng gói."
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=Path("models") / "paddle",
        help="Thư mục kho model paddle (mặc định: models/paddle).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="Phiên bản OCR (mặc định: OcrSettings.version của app).",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Danh sách mã ngôn ngữ (mặc định: toàn bộ ngôn ngữ UI của app).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Thiết bị khởi tạo prefetch (mặc định cpu — chỉ cần tải file).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In log DEBUG chi tiết.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Điểm vào CLI. Trả về mã thoát (0 = thành công)."""
    args = _build_arg_parser().parse_args(argv)
    _configure_logging(verbose=args.verbose)

    try:
        default_version, default_languages = _load_app_defaults()
    except PrefetchError as load_error:
        logger.error("%s", load_error)
        return 2

    version = args.version or default_version
    languages = args.languages if args.languages else default_languages

    logger.info(
        "Bắt đầu prefetch — version=%s, %d ngôn ngữ: %s",
        version,
        len(languages),
        ", ".join(languages),
    )

    try:
        prefetch_models(
            staging_dir=args.staging,
            version=version,
            languages=languages,
            device=args.device,
        )
    except PrefetchError as prefetch_error:
        logger.error("%s", prefetch_error)
        return 1

    logger.info("✓ Prefetch hoàn tất. Staging sẵn sàng để .spec nhúng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
