"""Prefetch model HuggingFace về kho tập trung ``models/huggingface`` (offline-first).

MỤC ĐÍCH
========
Các engine dùng ``huggingface_hub`` (VieNeu-TTS ONNX, fastembed embedding) mặc định
tải model từ Internet ở LẦN CHẠY ĐẦU. Script này tải sẵn chúng vào ``models/huggingface``
theo đúng bố cục cache của ``huggingface_hub``; lúc chạy,
:mod:`infrastructure.model_store` trỏ ``HF_HOME`` về đó nên thư viện dùng model local.

CÁCH DÙNG
---------
    # Mặc định: tải model VieNeu-TTS v3 Turbo (bản ONNX int8 — engine mặc định của app)
    python tools/prefetch_hf_models.py

    # Chỉ định repo khác / thêm repo
    python tools/prefetch_hf_models.py --repos pnnbao-ump/VieNeu-TTS-v3-Turbo BAAI/bge-small-zh-v1.5

    # Đổi thư mục kho
    python tools/prefetch_hf_models.py --store models/huggingface

GHI CHÚ
-------
* Idempotent: ``huggingface_hub`` tự bỏ qua file đã có (kiểm tra hash/etag).
* Không dùng ``print`` — mọi thứ qua ``logging`` để build log nhất quán.
* Script này KHÔNG phải một phần runtime của app; nó là công cụ build độc lập.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("prefetch_hf_models")

# Biến môi trường gốc cache huggingface_hub (đọc lúc import -> phải set trước).
_HF_HOME_ENV: str = "HF_HOME"

# Repo mặc định: model của engine TTS offline mặc định (VieNeu-TTS v3 Turbo).
_DEFAULT_REPOS: tuple[str, ...] = ("pnnbao-ump/VieNeu-TTS-v3-Turbo",)


class HuggingFacePrefetchError(RuntimeError):
    """Lỗi nghiệp vụ khi prefetch model HuggingFace (thiếu thư viện, tải hỏng…)."""


def _configure_logging(verbose: bool) -> None:
    """Cấu hình logging tối giản cho công cụ build."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def prefetch_repositories(
    *, store_dir: Path, repo_ids: list[str], allow_patterns: list[str] | None
) -> Path:
    """Tải các repo HuggingFace về ``store_dir`` theo bố cục cache chuẩn.

    Args:
        store_dir: Gốc kho HF (sẽ trở thành ``HF_HOME``; hub cache là ``<store>/hub``).
        repo_ids: Danh sách repo id (vd ``["pnnbao-ump/VieNeu-TTS-v3-Turbo"]``).
        allow_patterns: Lọc file cần tải (vd ``["*onnx_int8*"]``); ``None`` = tải hết.

    Returns:
        Đường dẫn ``store_dir`` sau khi hoàn tất.

    Raises:
        HuggingFacePrefetchError: Khi thiếu ``huggingface_hub`` hoặc mọi repo đều lỗi.
    """
    store_dir = store_dir.resolve()
    store_dir.mkdir(parents=True, exist_ok=True)

    # Set HF_HOME TRƯỚC khi import huggingface_hub (nó đọc env lúc import).
    os.environ[_HF_HOME_ENV] = str(store_dir)
    logger.info("%s = %s", _HF_HOME_ENV, store_dir)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as import_error:
        raise HuggingFacePrefetchError(
            "Chưa cài huggingface_hub trong môi trường build. "
            "Chạy: pip install huggingface_hub"
        ) from import_error

    failures: list[str] = []
    for repo_id in repo_ids:
        logger.info("→ Tải repo '%s'…", repo_id)
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow_patterns,
            )
        except Exception as download_error:  # noqa: BLE001 — build tool: gói lại thành lỗi rõ
            logger.warning("Không tải được '%s': %s", repo_id, download_error)
            failures.append(repo_id)
            continue
        logger.info("  ✓ Xong '%s' → %s", repo_id, local_path)

    if len(failures) == len(repo_ids):
        raise HuggingFacePrefetchError(
            f"Không tải được repo nào ({', '.join(failures)}). Kiểm tra mạng/repo id."
        )
    if failures:
        logger.warning(
            "Có %d repo KHÔNG tải được (%s) — app sẽ tải chúng theo yêu cầu khi cần.",
            len(failures),
            ", ".join(failures),
        )

    return store_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    """Dựng parser tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Prefetch model HuggingFace vào kho tập trung models/huggingface."
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("models") / "huggingface",
        help="Thư mục kho HF (mặc định: models/huggingface).",
    )
    parser.add_argument(
        "--repos",
        nargs="*",
        default=None,
        help=f"Repo id cần tải (mặc định: {' '.join(_DEFAULT_REPOS)}).",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Chỉ tải file khớp mẫu (vd '*onnx_int8*'). Mặc định: tải toàn bộ repo.",
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

    repo_ids = args.repos if args.repos else list(_DEFAULT_REPOS)
    logger.info("Bắt đầu prefetch %d repo: %s", len(repo_ids), ", ".join(repo_ids))

    try:
        prefetch_repositories(
            store_dir=args.store,
            repo_ids=repo_ids,
            allow_patterns=args.allow_patterns,
        )
    except HuggingFacePrefetchError as prefetch_error:
        logger.error("%s", prefetch_error)
        return 1

    logger.info("✓ Prefetch hoàn tất. Kho models/huggingface sẵn sàng để .spec nhúng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
