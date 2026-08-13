"""[v3.23.220] Kỷ luật phân tầng TTS: module dùng chung KHÔNG được phụ thuộc adapter.

Bối cảnh: các nguyên hàm DSP, toán thời lượng và bộ tiền xử lý văn bản vốn nằm rải trong
``edge_tts_adapter`` / ``vieneu_tts_adapter`` (adapter cụ thể) nhưng lại bị module hạ tầng
dùng chung (``audio_mastering``) và hai engine còn lại import ngược -> vòng tròn phụ thuộc
+ mầm mống lệch parity (đúng dạng bug đã trả giá suốt v215-v219).

Bộ test này KHOÁ chiều phụ thuộc lại để không tái phạm, đồng thời chứng minh việc gộp bản
sao KHÔNG đổi hành vi (zero regression).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from subtitles_extractor.infrastructure.tts import (
    audio_mastering,
    audio_utils,
    dsp_primitives,
    edge_tts_adapter,
    text_prep,
    timing_math,
    vieneu_tts_adapter,
)

_TTS_DIR = Path(audio_mastering.__file__).parent

# Module hạ tầng THUẦN — không được import bất kỳ adapter nào.
_SHARED_MODULES = (
    "audio_mastering.py",
    "audio_utils.py",
    "dsp_primitives.py",
    "text_prep.py",
    "time_stretch.py",
    "timing_math.py",
)
_ADAPTER_MODULES = (
    "edge_tts_adapter",
    "gemini_tts_adapter",
    "vieneu_tts_adapter",
)


def _imported_modules(source_path: Path) -> set[str]:
    """Tập tên module được import trong tệp (kể cả import lười trong thân hàm)."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


# ── Chiều phụ thuộc: shared -/-> adapter ─────────────────────────────────────
@pytest.mark.parametrize("module_file", _SHARED_MODULES)
def test_shared_module_khong_import_adapter(module_file: str) -> None:
    imported = _imported_modules(_TTS_DIR / module_file)
    offenders = [
        name
        for name in imported
        for adapter in _ADAPTER_MODULES
        if adapter in name
    ]
    assert not offenders, (
        f"{module_file} import ngược từ adapter: {offenders}. Module dùng chung phải "
        f"độc lập với mọi engine cụ thể."
    )


def test_gemini_khong_phu_thuoc_vieneu() -> None:
    # Adapter không được phụ thuộc adapter khác (trước v220: Gemini import VieNeu 6 chỗ).
    imported = _imported_modules(_TTS_DIR / "gemini_tts_adapter.py")
    assert not [n for n in imported if "vieneu_tts_adapter" in n]


def test_vieneu_khong_phu_thuoc_edge() -> None:
    imported = _imported_modules(_TTS_DIR / "vieneu_tts_adapter.py")
    assert not [n for n in imported if "edge_tts_adapter" in n]


def test_edge_khong_phu_thuoc_vieneu() -> None:
    imported = _imported_modules(_TTS_DIR / "edge_tts_adapter.py")
    assert not [n for n in imported if "vieneu_tts_adapter" in n]


# ── Tương thích ngược: alias cũ vẫn trỏ đúng hàm dùng chung ──────────────────
def test_alias_edge_tro_dung_ham_dung_chung() -> None:
    assert (
        edge_tts_adapter._true_peak_chunked_overlap
        is dsp_primitives.true_peak_chunked_overlap
    )
    assert edge_tts_adapter._loudness_gain_linear is dsp_primitives.loudness_gain_linear
    assert edge_tts_adapter._inter_node_soft_clip is dsp_primitives.inter_node_soft_clip
    assert edge_tts_adapter._clamp_smoothed_gain is dsp_primitives.clamp_smoothed_gain
    assert edge_tts_adapter._fit_length_no_silence is dsp_primitives.fit_length_no_silence
    assert (
        edge_tts_adapter._noise_gate_threshold_linear
        is dsp_primitives.noise_gate_threshold_linear
    )
    assert (
        edge_tts_adapter._gated_loudness_from_kweighted
        is dsp_primitives.gated_loudness_from_kweighted
    )
    assert edge_tts_adapter._preprocess_tts_text is text_prep.preprocess_tts_text
    assert edge_tts_adapter._skip_from_request is text_prep.skip_from_request
    assert edge_tts_adapter._SkipOptions is text_prep.SkipOptions
    assert edge_tts_adapter._has_speakable_content is text_prep.has_speakable_content


def test_alias_vieneu_tro_dung_ham_dung_chung() -> None:
    assert vieneu_tts_adapter.lead_in_seconds is timing_math.lead_in_seconds
    assert vieneu_tts_adapter.total_speed_ratio is timing_math.total_speed_ratio
    assert vieneu_tts_adapter.master_length_samples is timing_math.master_length_samples
    assert vieneu_tts_adapter.fit_limit_samples is timing_math.fit_limit_samples
    assert (
        vieneu_tts_adapter.effective_available_seconds
        is timing_math.effective_available_seconds
    )
    assert vieneu_tts_adapter.is_effectively_silent is audio_utils.is_effectively_silent
    assert vieneu_tts_adapter.trim_edge_silence is audio_utils.trim_edge_silence
    assert vieneu_tts_adapter.resample_audio is audio_utils.resample_audio
    assert vieneu_tts_adapter.has_speakable_content is text_prep.has_speakable_content


def test_hang_so_tran_chat_luong_van_giu_nguyen_gia_tri() -> None:
    # Trần 2.0 là kết luận đo đạc ở v201/v216 — không được đổi khi dời module.
    assert timing_math.QUALITY_STRETCH_CAP == 2.0
    assert vieneu_tts_adapter._QUALITY_STRETCH_CAP == 2.0
    assert timing_math.MAX_LEAD_IN_S == 0.25
    assert timing_math.MASTER_TAIL_PAD_S == 5.0


# ── Gộp bản sao limiter: hành vi Edge PHẢI y hệt bản dùng chung ──────────────
def test_edge_true_peak_limit_bang_ban_dung_chung() -> None:
    rng = np.random.default_rng(7)
    signal = (rng.standard_normal(24_000) * 0.4).astype(np.float32)
    signal[5_000:5_010] = 1.8  # đỉnh nhọn vượt trần
    from_edge = edge_tts_adapter.EdgeTTSAdapter._true_peak_limit(
        signal.copy(), 24_000, ceiling=0.95
    )
    from_shared = audio_mastering.true_peak_limit(signal.copy(), 24_000, ceiling=0.95)
    np.testing.assert_allclose(from_edge, from_shared, rtol=0, atol=0)


def test_edge_soft_limit_bang_ban_dung_chung() -> None:
    signal = np.linspace(-1.5, 1.5, 4_096, dtype=np.float32)
    from_edge = edge_tts_adapter.EdgeTTSAdapter._soft_limit_master(
        signal.copy(), threshold=0.85, ceiling=0.92
    )
    from_shared = audio_mastering.soft_limit(
        signal.copy(), threshold=0.85, ceiling=0.92
    )
    np.testing.assert_allclose(from_edge, from_shared, rtol=0, atol=0)
    assert float(np.max(np.abs(from_edge))) <= 0.92 + 1e-6


# ── Hợp nhất has_speakable_content: tương đương hai bản cài đặt cũ ───────────
@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "□",
        "□□□",
        "♪",
        "...",
        "—",
        "-",
        "（）",  # noqa: RUF001 (ngoặc CJK — đúng dữ liệu OCR thực tế)
        "_",
        "xin chào",
        "Đông Hải",
        "你好",
        "ㅁ",
        "123",
        "٣",
        "[Nam:]",
    ],
)
def test_has_speakable_content_tuong_duong_ban_cu_cua_vieneu(text: str) -> None:
    # Bản cũ của VieNeu: any(ch.isalnum()). Bản hợp nhất dùng regex [^\W_].
    assert text_prep.has_speakable_content(text) == any(ch.isalnum() for ch in text)


# ── Hàm chết đã bị xoá (không còn call site) ─────────────────────────────────
def test_ham_chet_da_duoc_xoa() -> None:
    assert not hasattr(edge_tts_adapter.EdgeTTSAdapter, "_kweighting_filter")
    assert not hasattr(edge_tts_adapter.EdgeTTSAdapter, "_normalize_to_lufs")


# ── Không còn hai bản cài đặt song song trong mã nguồn Edge ──────────────────
def test_edge_khong_con_ban_sao_thuat_toan_dsp() -> None:
    source = (_TTS_DIR / "edge_tts_adapter.py").read_text(encoding="utf-8")
    # Các nguyên hàm chỉ được PHÉP xuất hiện dưới dạng alias/gọi, không định nghĩa lại.
    for name in (
        "def _true_peak_chunked_overlap(",
        "def _gated_loudness_from_kweighted(",
        "def _loudness_gain_linear(",
        "def _inter_node_soft_clip(",
        "def _noise_gate_threshold_linear(",
        "def _clamp_smoothed_gain(",
        "def _fit_length_no_silence(",
        "def _preprocess_tts_text(",
    ):
        assert name not in source, f"{name} vẫn được định nghĩa lại trong Edge"
