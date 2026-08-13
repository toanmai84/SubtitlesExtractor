# Kế hoạch đóng gói (Build) SubtitlesExtractor

Tài liệu hướng dẫn đóng gói ứng dụng thành file thực thi chạy độc lập trên Windows.

## 1. Tổng quan & lựa chọn công cụ

Ứng dụng là GUI desktop (PySide6 + fluent_compat tự xây) với các thành phần nặng: PaddleOCR,
PyAV, python-mpv, các engine TTS. Đây là ứng dụng phức tạp nên lựa chọn công cụ cần cân
nhắc kỹ.

| Công cụ | Ưu | Nhược | Khuyến nghị |
|---|---|---|---|
| **PyInstaller** | Phổ biến nhất, hỗ trợ Qt/paddle tốt, cộng đồng lớn | File lớn | ✅ **Chọn** |
| Nuitka | Chạy nhanh hơn (biên dịch C) | Build lâu, khó debug paddle | Cân nhắc sau |
| cx_Freeze | Nhẹ | Ít hỗ trợ Qt phức tạp | Không |

**Quyết định:** dùng **PyInstaller** ở chế độ `--onedir` (thư mục, KHÔNG `--onefile`).
Lý do chọn onedir:
- Khởi động nhanh hơn nhiều (onefile phải giải nén vào temp mỗi lần chạy — với paddle
  nặng có thể mất 30-60s mỗi lần khởi động).
- Dễ chẩn đoán khi thiếu DLL/model.
- PaddleOCR tải model runtime — onedir cho phép cache model cạnh app.

## 2. Chiến lược xử lý các phụ thuộc nặng

### 2.1. PaddleOCR + paddlepaddle (thách thức lớn nhất)
- paddlepaddle chứa nhiều native `.dll`/`.so` + model tải runtime từ mạng.
- **Model KHÔNG đóng gói vào exe** — để app tải lần đầu chạy (như hiện tại). Giữ thư mục
  cache model cạnh app (`.paddleocr/` hoặc `PADDLE_HOME`).
- Cần `--collect-all paddle`, `--collect-all paddleocr`, `--collect-all paddlex`.

### 2.2. PySide6 (thay PyQt6/qfluentwidgets)
- [v3.23.267] Đã chuyển sang PySide6 (LGPL) + bỏ qfluentwidgets (GPL), tự xây fluent_compat
  bằng Qt thuần. PyInstaller có hook sẵn cho PySide6.
- Cần hidden imports PySide6.QtCore/QtGui/QtWidgets + shiboken6 (xem spec).

### 2.3. whisperx (TÙY CHỌN — KHÔNG đóng gói)
- whisperx cần torch + CUDA, rất nặng (vài GB), chạy ở **subprocess riêng**.
- **Chiến lược:** KHÔNG bundle whisperx vào exe. Nếu người dùng cần STT, cài Python riêng
  + `pip install whisperx` trên máy; app gọi qua subprocess (đã có sẵn cơ chế).
- Điều này giảm kích thước exe đáng kể và tránh xung đột CUDA.

### 2.4. python-mpv / libmpv
- python-mpv là wrapper thuần Python; native `libmpv-2.dll` KHÔNG có trong pip.
- App đã tự tải `libmpv-2.dll` từ mirror lúc chạy (xem `mpv_dll_manager.py`).
- **Không cần** bundle libmpv — app tự xử lý. Đảm bảo thư mục ghi được cạnh app.

### 2.5. PyNvVideoCodec / cupy (TÙY CHỌN — KHÔNG đóng gói)
- Chỉ dùng khi có GPU NVIDIA. App tự phát hiện & fallback PyAV/OpenCV nếu vắng.
- KHÔNG bundle. Người dùng có GPU tự cài nếu muốn tăng tốc.

### 2.6. torch_import_blocker
- App set `sys.modules["torch"] = None` để ép ONNX cho VieNeu. Cơ chế này chạy runtime,
  KHÔNG ảnh hưởng build. Nhưng: nếu bundle torch (cho whisperx) thì blocker vẫn hoạt động
  đúng vì nó chỉ chặn theo ngữ cảnh. Vì ta KHÔNG bundle whisperx/torch → không lo.

## 3. Hidden imports cần khai báo

Các import động (`importlib`, `__import__`) PyInstaller không tự phát hiện:
- (Không còn qfluentwidgets — đã thay bằng fluent_compat nội bộ)
- `paddleocr`, `paddlex` submodules (paddle tải động nhiều module)
- `soundfile`, `av`, `scipy.signal`, `scipy.special` (numpy/scipy C-ext)
- `pydantic`, `pydantic_settings`
- `rjieba`, `json_repair`
- Các adapter engine TTS (edge_tts, vieneu nếu bundle — xem mục 4)

## 4. Quyết định về engine TTS

| Engine | Bundle? | Lý do |
|---|---|---|
| Edge TTS | ✅ Có | Thuần Python + gọi API online, nhẹ |
| Gemini TTS | ✅ Có | google-genai thuần Python, nhẹ |
| VieNeu | ⚠️ Tùy | Nặng (onnxruntime + model). Có thể bundle onnxruntime nhưng model tải runtime |

**Khuyến nghị:** bundle Edge + Gemini mặc định. VieNeu: bundle `onnxruntime` + `sea-g2p`
nhưng để model VieNeu tải runtime (như paddle). Nếu muốn exe nhỏ, để VieNeu là tùy chọn
cài thêm.

## 5. Quy trình build (tóm tắt)

```
1. Tạo môi trường ảo SẠCH (tránh bundle thừa):
   python -m venv build_env && build_env\Scripts\activate
2. Cài đúng dependencies runtime (KHÔNG cài dev/test/whisperx):
   pip install -r requirements.txt
3. Cài PyInstaller:
   pip install pyinstaller>=6.0
4. Chạy build bằng spec file:
   pyinstaller SubtitlesExtractor.spec --noconfirm
5. Kết quả ở dist/SubtitlesExtractor/SubtitlesExtractor.exe
6. Test khởi động + smoke test các chức năng chính.
7. (Tùy chọn) Đóng gói installer bằng Inno Setup.
```

## 6. Kiểm thử sau build (checklist)

- [ ] App khởi động không lỗi DLL.
- [ ] Mở được video (thử cả PyAV và OpenCV backend).
- [ ] OCR chạy (model paddle tải được lần đầu).
- [ ] Dịch Gemini hoạt động (cần API key).
- [ ] TTS Edge phát được (online).
- [ ] TTS VieNeu (nếu bundle) tải model + phát được.
- [ ] Ghi file output (docx/srt/audio) vào thư mục ghi được.
- [ ] Không cần Python cài sẵn trên máy đích.

## 7. Rủi ro & giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| paddle thiếu DLL runtime | `--collect-all paddle` + test kỹ; thêm `--paths` nếu cần |
| Antivirus chặn exe không ký | Ký số (code signing) hoặc hướng dẫn whitelist |
| File quá lớn (>2GB) | Không bundle whisperx/torch/cupy; nén UPX cẩn thận (paddle dễ hỏng với UPX) |
| Model tải chậm lần đầu | Kèm hướng dẫn / cho phép trỏ thư mục model có sẵn |
| Đường dẫn ghi (log/output) | Dùng thư mục người dùng, không ghi cạnh exe (Program Files chặn ghi) |

## 8. Lưu ý quan trọng

- **KHÔNG bật UPX cho paddle** — UPX nén DLL paddle thường gây crash. Đặt `upx=False` hoặc
  loại trừ paddle khỏi UPX.
- **Đường dẫn ghi:** exe trong `Program Files` không ghi được. Log/output/cache model phải
  vào `%LOCALAPPDATA%` hoặc thư mục người dùng chọn.
- **freeze_support()** đã có trong main.py — bắt buộc cho multiprocessing khi đóng gói.
- Build trên **cùng OS đích** (Windows build cho Windows). PyInstaller không cross-compile.
