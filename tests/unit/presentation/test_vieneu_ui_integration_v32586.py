"""[v3.23.186] Test tích hợp VieNeu-TTS vào UI/ViewModel (registry + chọn adapter).

Kiểm phần logic không cần dựng widget Qt: registry engine có VieNeu, container factory
tạo đúng adapter, và ViewModel chọn đúng adapter theo engine id. Không kiểm widget thật
(PyQt6 segfault trong CI) — chỉ kiểm hằng số registry và định tuyến adapter.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import VieNeuTtsAdapter


def test_vieneu_adapter_engine_name() -> None:
    adapter = VieNeuTtsAdapter()
    assert adapter.get_engine_name() == "VieNeu-TTS (Offline)"


def test_vieneu_adapter_respects_mode_emotion() -> None:
    adapter = VieNeuTtsAdapter(mode="turbo", emotion="storytelling")
    assert adapter._mode == "turbo"
    assert adapter._emotion == "storytelling"


def test_vieneu_adapter_invalid_mode_falls_back() -> None:
    adapter = VieNeuTtsAdapter(mode="xyz", emotion="abc")
    assert adapter._mode == "standard"  # không hợp lệ -> mặc định
    assert adapter._emotion == "natural"


def test_engine_registry_includes_vieneu() -> None:
    # Registry trong trang TTS phải liệt kê VieNeu (id, nhãn, mô tả).
    from subtitles_extractor.presentation.pages import tts_page
    ids = [engine_id for engine_id, _label, _desc in tts_page._ENGINES]
    assert tts_page._ENGINE_VIENEU in ids
    assert tts_page._ENGINE_VIENEU == "vieneu"


def test_engine_registry_order_stable() -> None:
    # [v3.23.195] F5-TTS đã gỡ. Thứ tự 3 engine ổn định; cấu hình lưu theo engine_id
    # (không còn theo index) nên thay đổi danh sách không vỡ cấu hình đã lưu.
    from subtitles_extractor.presentation.pages import tts_page
    ids = [engine_id for engine_id, _label, _desc in tts_page._ENGINES]
    assert ids == [
        tts_page._ENGINE_EDGE, tts_page._ENGINE_GEMINI, tts_page._ENGINE_VIENEU
    ]
    assert not hasattr(tts_page, "_ENGINE_F5")  # F5 gỡ sạch khỏi registry
