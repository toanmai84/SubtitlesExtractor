"""[v3.23.373] Kiểm thử gợi ý khắc phục khi GPU worker VieNeu chết do lệch phiên bản."""
from __future__ import annotations

import importlib.util
from pathlib import Path

# Nạp module qua đường dẫn để tránh kéo theo PySide6 (không có trong môi trường test).
_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/subtitles_extractor/infrastructure/tts/vieneu_tts_adapter.py"
)


def _load_hint_fn():
    # Đọc thẳng hàm thuần từ source để không import cả adapter (vốn cần torch/PySide6).
    source = _MODULE_PATH.read_text(encoding="utf-8")
    start = source.index("def _gpu_worker_failure_hint")
    end = source.index("\n\n\n", start)
    snippet = (
        "from typing import Final\n"
        '_COMPATIBLE_TRANSFORMERS_SPEC: Final[str] = "transformers<4.56"\n'
        + source[start:end]
    )
    namespace: dict = {}
    exec(compile(snippet, str(_MODULE_PATH), "exec"), namespace)  # noqa: S102
    return namespace["_gpu_worker_failure_hint"]


_hint = _load_hint_fn()


def test_scalingtype_error_gives_repair_hint() -> None:
    reason = (
        "tiến trình GPU không báo sẵn sàng — ImportError: cannot import name "
        "'ScalingType' from 'torch.nn.functional'"
    )
    hint = _hint(reason)
    assert "transformers<4.56" in hint
    assert "whisperx_env" in hint


def test_generic_cannot_import_torch_symbol_hint() -> None:
    assert "transformers<4.56" in _hint("cannot import name 'RMSNorm' from torch import")


def test_unrelated_reason_returns_empty() -> None:
    assert _hint("bản đóng gói thiếu vieneu_gpu_subprocess.py") == ""
    assert _hint("không khởi chạy được: WinError 2") == ""


def test_hint_prefixed_with_space_for_log_concat() -> None:
    hint = _hint("ImportError: cannot import name 'ScalingType' from torch")
    assert hint.startswith(" ")
