"""[v3.23.248] UI chỉnh nhiệt độ Gemini TTS + lưu/khôi phục settings.

Thêm ô ``QDoubleSpinBox`` "Nhiệt độ" vào trang TTS (khối Gemini). Dùng
``specialValueText``: khi ở min (-0.1) hiển thị "Tự động" và map sang ``None`` = dùng
mặc định của model.

Rà bug đi kèm: ``_save_settings``/``_restore_settings`` phải lưu cả temperature (nếu quên,
giá trị mất mỗi lần khởi động lại app — đúng lớp bug đã có với các widget khác).
"""

from __future__ import annotations

import pathlib

_PAGE_SRC = pathlib.Path(
    "src/subtitles_extractor/presentation/pages/tts_page.py"
).read_text(encoding="utf-8")


def test_widget_temperature_tồn_tại() -> None:
    assert "self._gemini_temperature = QDoubleSpinBox()" in _PAGE_SRC
    assert "setRange(-0.1, 2.0)" in _PAGE_SRC


def test_special_value_là_tự_động() -> None:
    # Ở min -> hiển thị "Tự động", map sang None (mặc định model).
    assert 'setSpecialValueText("Tự động (mặc định model)")' in _PAGE_SRC


def test_map_âm_thành_none() -> None:
    # value < 0 -> None; ngược lại truyền giá trị thật (kể cả 0.0).
    assert "temp_val if temp_val >= 0.0 else None" in _PAGE_SRC
    assert "gemini_temperature=gemini_temp" in _PAGE_SRC


def test_lưu_settings() -> None:
    assert 's.setValue("gemini_temperature"' in _PAGE_SRC


def test_khôi_phục_settings() -> None:
    assert 's.contains("gemini_temperature")' in _PAGE_SRC
    assert 's.value("gemini_temperature", type=float)' in _PAGE_SRC


def test_logic_map_none_đúng() -> None:
    # Mô phỏng đúng biểu thức trong page: temp_val >= 0.0 else None.
    def map_temp(temp_val: float) -> float | None:
        return temp_val if temp_val >= 0.0 else None

    assert map_temp(-0.1) is None  # "Tự động"
    assert map_temp(0.0) == 0.0  # 0.0 vẫn là giá trị hợp lệ, KHÁC None
    assert map_temp(0.7) == 0.7
    assert map_temp(2.0) == 2.0
