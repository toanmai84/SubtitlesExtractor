"""Test đường chạy VieNeu trên GPU qua worker môi trường riêng — v3.23.344.

LỖ HỔNG ĐƯỢC LẤP: v3.23.343 thêm nút cài ``vieneu`` vào ``whisperx_env``, nhưng adapter
vẫn nạp VieNeu bằng ``from vieneu import Vieneu`` NGAY TRONG tiến trình chính — nơi
``torch`` cố ý không được đóng gói. Nên VieNeu gặp ``ImportError`` rồi **âm thầm** lùi về
ONNX/CPU: người dùng cài đủ gói, giao diện báo sẵn sàng, mà GPU vẫn không được dùng.

Nút cài đặt vô dụng nếu thiếu worker này.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _worker_path() -> Path:
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[2]
    return root / "infrastructure" / "tts" / "vieneu_gpu_subprocess.py"


def _adapter_source() -> str:
    import subtitles_extractor.infrastructure.tts.vieneu_tts_adapter as module

    return Path(module.__file__).read_text(encoding="utf-8")


def _run_worker(job: dict) -> subprocess.CompletedProcess[bytes]:
    """Chạy worker thật với một công việc, trả kết quả tiến trình."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        worker = temp / "worker.py"
        worker.write_bytes(_worker_path().read_bytes())
        job_path = temp / "job.json"
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(worker), "--job", str(job_path)],
            capture_output=True,
        )


# ── Worker: phân biệt nguyên nhân bằng mã thoát ──────────────────────────────
def test_worker_exists() -> None:
    assert _worker_path().is_file()


def test_worker_is_valid_python() -> None:
    ast.parse(_worker_path().read_text(encoding="utf-8"))


def test_empty_items_is_bad_args() -> None:
    """Công việc rỗng là lỗi tham số (mã 2), không phải lỗi chạy."""
    assert _run_worker({"items": []}).returncode == 2


def test_missing_torch_reports_library_missing() -> None:
    """Không có torch -> mã 3, để adapter biết TẮT hẳn đường GPU thay vì thử lại."""
    result = _run_worker(
        {"voice_preset": "x", "items": [{"text": "a", "output": "/tmp/x.wav"}]}
    )
    # Mã 3 = thiếu thư viện, mã 4 = có torch nhưng không có GPU. Cả hai đều là lỗi
    # CỐ ĐỊNH nên adapter phải xử lý giống nhau.
    assert result.returncode in (3, 4)


def test_worker_stderr_is_utf8_readable() -> None:
    """Thông điệp chẩn đoán phải đọc được (Windows mặc định cp1252 làm hỏng chữ Việt)."""
    result = _run_worker({"items": []})
    text = result.stderr.decode("utf-8", "strict")  # 'strict' -> lỗi nếu không phải UTF-8
    assert "VIENEU_GPU_BAD_JOB" in text


@pytest.mark.parametrize(
    "marker",
    ["VIENEU_GPU_NO_TORCH", "VIENEU_GPU_UNAVAILABLE", "VIENEU_GPU_NO_VIENEU"],
)
def test_worker_has_distinct_error_markers(marker: str) -> None:
    """Mỗi nguyên nhân có dấu hiệu riêng — để chẩn đoán được từ nhật ký."""
    assert marker in _worker_path().read_text(encoding="utf-8")


def test_worker_forces_pytorch_backend() -> None:
    """ĐIỂM MẤU CHỐT: để ``backend="auto"`` thì VieNeu vẫn có thể chọn ONNX.

    [v3.23.364] Worker nay thử NHIỀU chữ ký (fallback khi SDK đổi API); chữ ký ĐẦU TIÊN
    vẫn là pytorch+cuda nên ưu tiên GPU-PyTorch được giữ nguyên.
    """
    source = _worker_path().read_text(encoding="utf-8")
    assert '"backend": "pytorch"' in source
    assert '"device": "cuda"' in source


def test_worker_resolves_voice_itself() -> None:
    """Giải quyết giọng TRONG worker — không truyền embedding qua ranh giới tiến trình."""
    source = _worker_path().read_text(encoding="utf-8")
    assert "def _resolve_voice" in source
    assert "encode_reference" in source


def test_worker_survives_one_bad_item() -> None:
    """Một câu lỗi không được phá cả lô."""
    source = _worker_path().read_text(encoding="utf-8")
    index = source.index("for index, item in enumerate(items)")
    window = source[index : index + 1400]
    assert "except Exception" in window
    assert "results.append" in window


# ── Adapter: rẽ nhánh sang worker ────────────────────────────────────────────
def test_adapter_routes_to_gpu_worker_when_not_forcing_cpu() -> None:
    """Đây là mảnh ghép còn thiếu ở v3.23.343."""
    source = _adapter_source()
    assert "_infer_via_gpu_worker" in source
    index = source.index("if not self._force_cpu:")
    window = source[index : index + 300]
    assert "_infer_via_gpu_worker" in window


def test_adapter_falls_back_to_in_process() -> None:
    """Worker trả ``None`` -> vẫn phải chạy được bằng ONNX/CPU."""
    source = _adapter_source()
    index = source.index("if not self._force_cpu:")
    window = source[index : index + 400]
    assert "engine.infer(**kwargs)" in window


def test_adapter_disables_gpu_after_fixed_error() -> None:
    """Lỗi CỐ ĐỊNH phải tắt hẳn đường GPU — thử lại 55 câu chỉ tốn thời gian.

    [v3.23.349] Cơ chế đổi: trước đây xét mã thoát 3/4 của tiến trình chạy-một-lần. Nay
    tiến trình THƯỜNG TRÚ nạp model một lần, nên lỗi cố định lộ ra ngay ở bước BẮT TAY
    (worker không báo "ready"). Hành vi giữ nguyên, chỉ đổi chỗ phát hiện.
    """
    source = _adapter_source()
    assert "def _disable_gpu_worker" in source
    assert "không báo sẵn sàng" in source


def test_disable_logs_only_once() -> None:
    """55 câu × cùng một cảnh báo làm nhật ký vô dụng."""
    source = _adapter_source()
    index = source.index("def _disable_gpu_worker")
    window = source[index : index + 700]
    assert "if self._gpu_worker_disabled:" in window
    assert "return" in window


def test_adapter_records_voice_source_for_worker() -> None:
    """Worker cần biết giọng nào; adapter phải ghi lại lúc giải quyết giọng."""
    source = _adapter_source()
    assert "self._gpu_voice_preset" in source
    assert "self._gpu_reference_wav" in source


def test_gpu_state_initialised_in_constructor() -> None:
    """Thiếu khởi tạo -> AttributeError ngay câu đầu tiên."""
    source = _adapter_source()
    for attribute in (
        "_gpu_worker_disabled", "_gpu_voice_preset", "_gpu_reference_wav"
    ):
        assert f"self.{attribute}: " in source


# ── Đóng gói ─────────────────────────────────────────────────────────────────
def test_worker_is_bundled() -> None:
    """Thiếu trong bundle -> TTS âm thầm lùi về CPU, đúng lỗi vừa sửa."""
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    spec = root / "SubtitlesExtractor.spec"
    if not spec.is_file():
        pytest.skip("Không tìm thấy spec")
    assert "vieneu_gpu_subprocess.py" in spec.read_text(encoding="utf-8")


def test_bundle_checker_requires_worker() -> None:
    """Công cụ kiểm bundle phải canh cả worker này."""
    import subtitles_extractor.domain.entities.project_record as anchor

    root = Path(anchor.__file__).resolve().parents[4]
    tool = root / "tools" / "check_bundle.py"
    if not tool.is_file():
        pytest.skip("Không tìm thấy check_bundle.py")
    assert "vieneu_gpu_subprocess.py" in tool.read_text(encoding="utf-8")
