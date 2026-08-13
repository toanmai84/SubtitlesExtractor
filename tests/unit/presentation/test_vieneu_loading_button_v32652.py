"""[v3.23.252] Khoá nút Tổng hợp khi VieNeu đang nạp giọng (sửa bug bấm chạy -> lỗi).

**Bug (Toan báo):** VieNeu nạp giọng ở thread nền (~9s, hiển thị "(Đang nạp giọng…)").
Trong lúc đó nút "Tổng hợp tiếng nói" VẪN bật — người dùng bấm chạy sẽ lỗi ngay vì giọng
chưa sẵn sàng.

**Sửa — nhiều lớp bảo vệ:**
1. ``_load_vieneu_voices`` khoá nút ngay khi bắt đầu nạp (nếu engine hiện tại là VieNeu).
2. Bật lại nút khi nạp XONG (``_on_vieneu_voices_loaded``) hoặc LỖI/RỖNG
   (``_reset_vieneu_voice_combo``) qua ``_update_action_states``.
3. ``_update_action_states`` khoá nút nếu đang chọn VieNeu và loader còn chạy (hàm trung
   tâm — mọi lần cập nhật trạng thái đều tôn trọng).
4. ``_on_generate`` chặn sớm nếu loader còn chạy (lớp phòng thủ cuối — kể cả nút lỡ bật).
5. Đổi engine RỜI VieNeu -> ``_update_action_states`` bật lại nút (engine khác không cần
   giọng VieNeu).

Chỉ khoá khi engine HIỆN TẠI là VieNeu — loader chạy nền không được khoá oan nút khi người
dùng đã chuyển sang Edge/Gemini.
"""

from __future__ import annotations

import pathlib

_PAGE_SRC = pathlib.Path(
    "src/subtitles_extractor/presentation/pages/tts_page.py"
).read_text(encoding="utf-8")


def test_khoá_nút_khi_bắt_đầu_nạp() -> None:
    # _load_vieneu_voices phải khoá nút khi engine hiện tại là VieNeu.
    idx = _PAGE_SRC.find("def _load_vieneu_voices")
    end = _PAGE_SRC.find("def _on_vieneu_voices_loaded")
    doan = _PAGE_SRC[idx:end]
    assert "_btn_gen.setEnabled(False)" in doan
    assert "_ENGINE_VIENEU" in doan


def test_on_generate_chặn_khi_đang_nạp() -> None:
    # Lớp phòng thủ: _on_generate chặn sớm nếu loader còn chạy.
    idx = _PAGE_SRC.find("def _on_generate")
    end = _PAGE_SRC.find("def ", idx + 10)
    doan = _PAGE_SRC[idx:end]
    assert "_vieneu_voice_loader" in doan
    assert "isRunning()" in doan
    assert "Đang nạp giọng" in doan


def test_update_action_states_tôn_trọng_loader() -> None:
    idx = _PAGE_SRC.find("def _update_action_states")
    end = _PAGE_SRC.find("def ", idx + 10)
    doan = _PAGE_SRC[idx:end]
    # Chỉ khoá khi engine hiện tại là VieNeu VÀ loader đang chạy.
    assert "_ENGINE_VIENEU" in doan
    assert "isRunning()" in doan


def test_bật_lại_nút_khi_nạp_xong() -> None:
    # _on_vieneu_voices_loaded gọi _update_action_states sau khi nạp xong.
    idx = _PAGE_SRC.find("def _on_vieneu_voices_loaded")
    end = _PAGE_SRC.find("def _on_vieneu_voices_failed")
    doan = _PAGE_SRC[idx:end]
    assert "_update_action_states()" in doan


def test_bật_lại_nút_khi_nạp_lỗi() -> None:
    # _reset_vieneu_voice_combo (dùng khi lỗi/rỗng) gọi _update_action_states.
    idx = _PAGE_SRC.find("def _reset_vieneu_voice_combo")
    end = _PAGE_SRC.find("def ", idx + 10)
    doan = _PAGE_SRC[idx:end]
    assert "_update_action_states()" in doan


def test_đổi_engine_cập_nhật_nút() -> None:
    # _on_engine_changed: nhánh else (rời VieNeu) cập nhật lại trạng thái nút.
    idx = _PAGE_SRC.find("def _on_engine_changed")
    end = _PAGE_SRC.find("def _sync_edge_only_controls")
    doan = _PAGE_SRC[idx:end]
    assert "_update_action_states()" in doan


def test_chỉ_khoá_khi_engine_là_vieneu() -> None:
    # Bảo vệ then chốt: không khoá oan nút khi loader chạy nền mà đã chuyển engine.
    # _update_action_states + _on_busy_changed đều phải kiểm engine hiện tại.
    for func in ("_update_action_states", "_on_busy_changed"):
        idx = _PAGE_SRC.find(f"def {func}")
        end = _PAGE_SRC.find("def ", idx + 10)
        doan = _PAGE_SRC[idx:end]
        # Có kiểm engine == VieNeu trước khi khoá vì loader.
        if "isRunning()" in doan:
            assert "_ENGINE_VIENEU" in doan, f"{func} phải kiểm engine trước khi khoá"
