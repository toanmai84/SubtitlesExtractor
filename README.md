# Subtitles Extractor — v3.23

Ứng dụng desktop **trích xuất phụ đề hardsub/embedded → dịch bằng AI → lồng tiếng (TTS)**
cho video, viết hoàn toàn bằng Python (PyQt6) theo kiến trúc **Clean Architecture / Hexagonal**.

Quy trình hoàn chỉnh trong một ứng dụng: **Trích xuất → Biên tập → Dịch → Lồng tiếng**.

---

## ✨ Tính năng chính

### 🎯 Trích xuất phụ đề
- Hỗ trợ MP4/MKV/AVI/MOV/FLV/WEBM → SRT/ASS.
- **OCR hardsub** bằng PaddleOCR (GPU + TensorRT FP16 khi có CUDA, tự fallback CPU).
- **STT** (nhận dạng tiếng nói) bằng WhisperX cho video không có phụ đề cứng.
- Trích xuất phụ đề embedded có sẵn trong video.
- Lấy mẫu khung hình thông minh: bước thời gian + loại trùng pHash + so sánh pixel.
- Hai thuật toán gộp câu: **Greedy** (mặc định, nhanh) và **Viterbi** (tối ưu khi OCR nhiễu).
- **Tự phát hiện ROI** vùng phụ đề bằng clustering Connected Components, lấy mẫu phân tầng theo thời gian.
- Nhận biết và xử lý riêng văn bản **CJK** (Trung/Nhật/Hàn) và Latin xuyên suốt pipeline.

### ✏️ Biên tập phụ đề
- Mở/sửa `.srt`/`.ass`; bảng sửa trực tiếp; chèn/xoá/tách/gộp dòng.
- Re-OCR từng dòng/khung hình để sửa nhận dạng sai.
- Undo/Redo (Memento). Waveform âm thanh với phím tắt chuẩn DAW.
- Xem trước video đồng bộ (MPV/PyAV/OpenCV).

### 🌐 Dịch bằng AI (Gemini, đa giai đoạn)
- Pipeline dịch nhiều giai đoạn: **Tiền xử lý → Dịch thô → Tinh chỉnh văn phong → Bản địa hoá**.
- **Phân tích ngữ cảnh toàn cục**: tự nhận diện ngôn ngữ, nhân vật, tóm tắt cốt truyện, bảng thuật ngữ.
- **Phân tích hình ảnh (visual cues)**: đính video để AI hiểu bối cảnh, xưng hô, người nói.
- `thinking_level` tối ưu cho Gemini 3.x (nhanh/rẻ cho dịch), `thinking_budget` cho Gemini 2.5.
- Length-hint (`max_chars`) cho bản dịch súc tích, vừa thời lượng đọc TTS.
- **Tinh chỉnh chuyên dụng**: dịch lại dòng đã chọn, sửa bản dịch tại chỗ, so sánh các giai đoạn,
  kiểm tra nhất quán thuật ngữ. Mọi chỉnh sửa được lưu bền vững và khôi phục khi mở lại.

### 📚 Translation Memory cho phim bộ
- Tự **nhớ câu đã dịch** theo từng phim bộ (gom theo thư mục) và truy hồi làm tham chiếu
  (RAG grounding) khi dịch tập mới → nhất quán tên riêng/thuật ngữ/xưng hô xuyên suốt.
- **Chia sẻ bảng thuật ngữ + roster nhân vật** tự động giữa các tập (gộp thông minh, giữ
  cách dịch đã thống nhất).
- Giao diện quản lý: xem/xoá bộ nhớ dịch theo từng phim bộ.

### 🎙️ Lồng tiếng (TTS)
- Tổng hợp giọng nói bằng EdgeTTS, đồng bộ thời gian với phụ đề.
- Bỏ qua nội dung không đọc được; tag người nói được giữ trong file phụ đề nhưng không đọc thành tiếng.
- Cân chỉnh tốc độ/time-stretch để khớp thời lượng.

### 🛠️ Cấu hình & Đa ngôn ngữ
- Validate runtime bằng `pydantic-settings`; override qua biến môi trường (`SE_HW_*`,
  `SE_OCR_*`, `SE_VIDCTX_*`, `SE_TRANS_*`...).
- Lưu tự động qua QSettings. Giao diện tiếng Việt, theme Sáng/Tối/Tự động.

---

## 🏛️ Kiến trúc

```
src/subtitles_extractor/
├── domain/          # Entities, value objects, ports — Python thuần
├── application/     # Use cases + services — điều phối nghiệp vụ
│   ├── use_cases/       Extract, Translate, Export, Import, DetectAutoRoi...
│   └── services/        SubtitleBuilder, TranslationMemory, GlossaryConsistency...
├── infrastructure/  # Adapters cho thế giới ngoài
│   ├── ocr/             PaddleOcrAdapter
│   ├── stt/             WhisperxAdapter
│   ├── translation/     GeminiSubtitleTranslator, GeminiVideoContext
│   ├── tts/             EdgeTtsAdapter
│   ├── video/           Metadata, FrameSampler, AutoRoiDetector
│   ├── subtitle/        SRT/ASS exporter & importer
│   └── database/        SQLite: project, session, translation memory
├── presentation/    # Qt UI (MVVM)
│   ├── pages/           Extract, Editor, Translate, TTS, Debug, Projects, Log, Settings
│   ├── view_models/     theo từng trang
│   ├── theme/           token màu thích ứng theme (sáng/tối)
│   └── widgets/         waveform, video canvas, ROI review...
└── composition/     # DI container + bootstrap
```

**Quy tắc phụ thuộc**: `presentation → application → domain ← infrastructure`. Domain thuần
Python, không phụ thuộc Qt/Paddle/OpenCV; các tầng giao tiếp qua `Protocol` (Port).

---

## 🚀 Cài đặt & Chạy

### Yêu cầu
- Python ≥ **3.11** (khuyến nghị 3.12).
- (Tuỳ chọn) GPU NVIDIA + CUDA cho OCR/STT nhanh.
- API Key Gemini (miễn phí tại https://aistudio.google.com/apikey) để dùng tính năng dịch.

### Cài đặt
```bash
git clone <repo>
cd subtitles-extractor
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tuỳ chọn — cho phát triển:
```bash
pip install -r requirements-dev.txt
```

### Chạy ứng dụng
```bash
python main.py
```
`main.py` tự inject `src/` vào path — không cần export `PYTHONPATH`.

### Chạy tests
```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m pytest tests/unit -q
```

---

## 📖 Hướng dẫn nhanh: dịch một phim bộ nhiều tập

1. Đặt tất cả các tập của phim trong **cùng một thư mục**.
2. **Trích xuất** phụ đề tập 1 (OCR/STT) → tinh chỉnh ở **Biên tập** nếu cần.
3. Sang trang **Dịch**: dán API Key, nhấn *Phân tích toàn bộ phụ đề* để tự sinh ngữ cảnh +
   bảng thuật ngữ + nhân vật, rồi bật các giai đoạn dịch và chạy.
4. Tinh chỉnh: *Dịch lại* dòng chưa ưng, *Sửa bản dịch* tại chỗ, *So sánh giai đoạn*,
   *Kiểm tra thuật ngữ*.
5. Các tập sau (cùng thư mục): ứng dụng **tự kế thừa** câu dịch + thuật ngữ + nhân vật để
   nhất quán — chỉ cần bổ sung khi có nội dung mới.
6. (Tuỳ chọn) Sang trang **TTS** để lồng tiếng bản dịch.

---

## 🔧 Tuỳ chỉnh qua biến môi trường (ví dụ)

```bash
export SE_HW_DEVICE=cpu                 # ép CPU
export SE_HW_USE_TENSORRT=true
export SE_OCR_SCORE_THRESHOLD=0.5
export SE_POST_USE_VITERBI=true         # bật Viterbi grouper
export SE_TRANS_DEFAULT_BATCH_SIZE=50   # cỡ lô dịch mặc định
export SE_VIDCTX_FPS=1.0                # fps lấy mẫu cho phân tích video
python main.py
```

---

## 📋 Quy tắc code

- **PEP 8** + **Black** (line length 90) + **Ruff**.
- **Type hints** đầy đủ; tránh `except Exception` trống (bắt ngoại lệ cụ thể).
- Dùng `logging` thay cho `print`. Identifier tiếng Anh, docstring/log/UI tiếng Việt.
- Mỗi thay đổi đi kèm test; pytest phải xanh trước khi đóng gói.

---

## 📝 Giấy phép

MIT.
