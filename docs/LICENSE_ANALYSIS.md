# Phân tích License & Giải pháp cho Build Thương mại

**Ngày rà soát:** 2026-07-18
**Phạm vi:** toàn bộ dependencies runtime của SubtitlesExtractor.

> ⚠️ **MIỄN TRỪ:** Tài liệu này là phân tích kỹ thuật để tham khảo, KHÔNG phải tư vấn
> pháp lý. Trước khi phát hành thương mại, hãy nhờ luật sư sở hữu trí tuệ rà soát lại.
> Người viết là kỹ sư, không phải luật sư.

---

## 1. Tóm tắt điều hành

Rà soát phát hiện **3 thư viện GPL** gây rủi ro pháp lý cho ứng dụng thương mại đóng
(closed-source). GPL yêu cầu: nếu phân phối ứng dụng có dùng thư viện GPL, PHẢI công khai
toàn bộ mã nguồn ứng dụng theo GPL — điều thường không chấp nhận được với sản phẩm thương mại.

| Thư viện | License | Rủi ro | Bắt buộc xử lý? |
|---|---|---|---|
| **PyQt6** | GPL v3 / Thương mại | 🔴 CAO | ✅ Bắt buộc |
| **PyQt6-Fluent-Widgets** | GPL v3 / Thương mại | 🔴 CAO | ✅ Bắt buộc |
| **pedalboard** | GPL v3 | 🔴 CAO | ✅ Bắt buộc |
| python-mpv | GPL v2+/LGPL v2.1+ | 🟡 TRUNG BÌNH | ⚠️ Cần kiểm |
| FFmpeg (qua PyAV) | LGPL/GPL tuỳ build | 🟡 TRUNG BÌNH | ⚠️ Cần kiểm |

Các thư viện còn lại (paddle, numpy, scipy, pydantic, rapidfuzz, google-genai...) đều là
license dễ chịu (Apache 2.0 / BSD / MIT / ISC) — **an toàn cho thương mại**.

---

## 2. Chi tiết các thư viện GPL (bắt buộc xử lý)

### 2.1. 🔴 PyQt6 (GPL v3) — VẤN ĐỀ LỚN NHẤT

**Vấn đề:** PyQt6 chỉ có GPL v3 hoặc license thương mại (KHÔNG có LGPL). Nếu phân phối ứng
dụng đóng dùng PyQt6, phải công khai mã nguồn theo GPL, hoặc mua license thương mại từ
Riverbank Computing (~550 USD/lập trình viên/năm).

**Giải pháp — 2 lựa chọn:**

#### Lựa chọn A (KHUYẾN NGHỊ): Chuyển sang **PySide6** (LGPL v3)
- PySide6 là binding Qt chính thức của The Qt Company, license **LGPL v3**.
- LGPL cho phép dùng trong ứng dụng đóng MÀ KHÔNG cần công khai mã nguồn, miễn là:
  - Link động tới thư viện (mặc định khi đóng gói Python — thoả mãn).
  - Cho phép người dùng thay thế phiên bản Qt (cung cấp cách relink — với Python thường
    thoả mãn tự nhiên).
- API PySide6 **99% giống PyQt6** — chỉ khác vài điểm nhỏ:
  - `pyqtSignal` → `Signal`, `pyqtSlot` → `Slot`
  - `PyQt6.QtCore` → `PySide6.QtCore` (đổi import)
  - Enum access đôi chỗ khác (PySide6 khoan dung hơn).
- **Công sức chuyển đổi:** trung bình. Chủ yếu là đổi import + tên signal/slot. Có công cụ
  hỗ trợ tự động một phần.

#### Lựa chọn B: Mua license thương mại PyQt6
- Giữ nguyên code, mua license ~550 USD/dev/năm từ Riverbank.
- Phù hợp nếu chi phí chuyển đổi kỹ thuật > chi phí license, hoặc cần giữ nguyên codebase.

### 2.2. 🔴 PyQt6-Fluent-Widgets (GPL v3) — GẮN LIỀN VỚI QUYẾT ĐỊNH 2.1

**Vấn đề:** qfluentwidgets cũng GPL v3 cho phi thương mại; thương mại phải mua license.

**Giải pháp:**
- Thư viện này CÓ bản **PySide6-Fluent-Widgets** (cùng tác giả zhiyiYo) — nếu chuyển sang
  PySide6 (Lựa chọn A ở trên), dùng bản PySide6 tương ứng.
- **NHƯNG** bản PySide6-Fluent-Widgets VẪN là GPL v3 cho phi thương mại → vẫn cần **mua
  license thương mại** từ tác giả (qfluentwidgets.com) nếu bán sản phẩm đóng.
- **Chi phí:** mua commercial license của QFluentWidgets (giá trên trang chủ, thường rẻ hơn
  nhiều so với build lại toàn bộ UI).
- **Phương án thay thế nếu không mua:** tự xây UI bằng widget Qt thuần (PySide6 core, LGPL)
  — công sức lớn vì phải thay toàn bộ SectionCard/FluentIcon/theme. KHÔNG khuyến nghị trừ
  khi ngân sách bằng 0.

### 2.3. 🔴 pedalboard (GPL v3) — DỄ THAY THẾ NHẤT

**Vấn đề:** pedalboard (Spotify) là GPL v3 (do nhúng JUCE/VST3 SDK GPL). App dùng nó cho
time-stretch giữ cao độ (nén audio TTS cho khớp khung phụ đề).

**Giải pháp — ĐÃ CÓ SẴN fallback trong code:**
- App **đã có** 3 nhánh time-stretch: pedalboard → librosa → WSOLA nội bộ (xem
  `time_stretch.py::vocal_time_stretch`). Chỉ cần **bỏ pedalboard**, app tự dùng librosa.
- **librosa** = ISC license (thoả mãn thương mại). Chất lượng time-stretch của librosa
  (phase-vocoder) tốt, chỉ chậm hơn pedalboard chút.
- **Công sức:** THẤP — chỉ cần không cài pedalboard trong môi trường build; app tự fallback.
  Có thể thêm cờ tắt hẳn nhánh pedalboard để chắc chắn.
- Nếu cần chất lượng/tốc độ cao hơn librosa: dùng **Rubber Band CLI** (có license thương mại
  riêng) hoặc **soundtouch** (LGPL) qua pydub, nhưng librosa thường là đủ.

---


### 2.4. ✅ chardet (LGPL 2.1) — ĐÃ XỬ LÝ (v3.23.271) — phát hiện ở rà soát lượt 2

**Vấn đề:** ``chardet`` (bản ổn định ≤6.x) dùng **LGPL 2.1** (copyleft). Bị ``paddlex`` kéo
vào (app KHÔNG dùng trực tiếp). Khi đóng gói PyInstaller, thư viện Python thuần bị nhúng
TĨNH → LGPL khó thoả mãn (yêu cầu cho phép thay thế). Lưu ý: chardet 7.x đổi sang MIT nhưng
đang có tranh chấp pháp lý về tính hợp lệ (rewrite bằng LLM từ mã LGPL) — nên KHÔNG dựa vào.

**Giải pháp (đã làm):** shim ``chardet_shim.py`` — cung cấp module ``chardet`` giả (API
``detect``/``detect_all``) nội bộ gọi **charset-normalizer** (MIT). ``install_chardet_shim()``
gọi sớm trong main.py (trước khi paddle import) đăng ký vào sys.modules. Spec loại chardet
khỏi bundle + thêm charset_normalizer. paddlex nhận bản MIT, không cần chardet LGPL.


## 3. Các thư viện TRUNG BÌNH (cần kiểm, có thể chấp nhận)

### 3.1. ✅ python-mpv + libmpv — LGPL (v3.23.270, xác nhận nguồn)
- python-mpv (wrapper) là GPL v2+/LGPL v2.1+. App liên kết ĐỘNG libmpv-2.dll (tải runtime).
- **libmpv-2.dll:** app tải từ mirror **SubtitleEdit/support-files** — bản build LGPL
  (xem mpv_dll_manager.py). Liên kết động + bản LGPL → thoả mãn LGPL, an toàn thương mại.
- **Script kiểm (tools/check_media_licenses.py):** nếu có mpv CLI, đọc license từ
  `mpv --version`; nếu chỉ có .dll runtime thì xác nhận nguồn mirror LGPL.
- **Khuyến nghị:** giữ nguyên. Đảm bảo tải libmpv từ nguồn LGPL đã biết (không đổi sang
  bản build GPL).

### 3.2. ✅ FFmpeg (qua PyAV) — ĐÃ XÁC NHẬN LGPL (v3.23.270)
- **Kết quả kiểm (tools/check_media_licenses.py):** PyAV 18.0.0 nhúng FFmpeg 8.1.2, TẤT CẢ
  7 thư viện (libavutil/avcodec/avformat/avdevice/avfilter/swscale/swresample) đều báo
  license "LGPL version 3 or later", configuration KHÔNG có `--enable-gpl`/`--enable-nonfree`.
- Có tham chiếu libx264/libx265 trong metadata nhưng KHÔNG `--enable-gpl` → không biên dịch
  vào → an toàn.
- **Kết luận:** FFmpeg trong PyAV là bản LGPL — AN TOÀN thương mại. App chỉ decode.
- ⚠️ Lưu ý: nếu đổi wheel PyAV khác, chạy lại script kiểm để xác nhận.

---


### 3.3. ✅ soxr (LGPL v2.1+) — phụ thuộc VieNeu, xử lý như LGPL khác (v3.23.292)

**Bối cảnh:** bundle VieNeu-TTS mặc định kéo theo ``soxr`` (python-soxr, resampling voice
cloning). soxr là **LGPL v2.1+** (theo libsoxr).

**Đánh giá:** LGPL cho phép dùng trong sản phẩm thương mại đóng nếu liên kết ĐỘNG. soxr là
binary ``.pyd`` riêng (không nhúng tĩnh vào mã nguồn app) → thoả mãn LGPL như soundfile,
libmpv, FFmpeg đã xử lý. Kèm toàn văn LGPL trong bản phân phối (THIRD_PARTY_LICENSES.md).

**Kết luận:** AN TOÀN thương mại. Không cần thay thế — chỉ giữ liên kết động + kèm license.


## 4. Các thư viện AN TOÀN (license dễ chịu — không cần xử lý)

| Thư viện | License | An toàn thương mại |
|---|---|---|
| paddlepaddle, paddleocr, paddlex | Apache 2.0 | ✅ |
| opencv-python | Apache 2.0 | ✅ |
| numpy | BSD 3-Clause | ✅ |
| scipy | BSD 3-Clause | ✅ |
| soundfile | BSD 3-Clause | ✅ |
| librosa | ISC | ✅ |
| pydantic, pydantic-settings | MIT | ✅ |
| rapidfuzz | MIT | ✅ |
| google-genai | Apache 2.0 | ✅ |
| json-repair | MIT | ✅ |
| loguru | MIT | ✅ |
| rjieba | MIT | ✅ |
| pydub | MIT | ✅ |
| edge-tts | GPL v3 ⚠️ | Xem mục 5 |
| VieNeu | Apache 2.0 | ✅ |
| whisperx | BSD-4 ⚠️ | Không bundle (subprocess) |

---

## 5. Engine TTS — kiểm riêng

- **edge-tts:** license GPL v3 (⚠️). ✅ **ĐÃ XỬ LÝ (v3.23.268):** cách ly vào subprocess
  riêng (``edge_tts_subprocess.py``). Tiến trình chính KHÔNG import edge_tts (dùng
  ``find_spec`` để kiểm tồn tại). edge-tts KHÔNG đóng gói kèm exe (excludes trong spec) —
  chạy bằng Python NGOÀI như công cụ độc lập. Nhờ ranh giới tiến trình + không phân phối
  kèm, edge-tts là "công cụ ngoài" không lan GPL vào app.
- **VieNeu:** Apache 2.0 ✅ — an toàn, giữ làm engine TTS chính offline.
- **Gemini TTS:** google-genai Apache 2.0 ✅ — an toàn (client API).

---

## 6. Kế hoạch hành động đề xuất (theo thứ tự ưu tiên)

### Bắt buộc trước khi bán:
1. **pedalboard → librosa** (dễ nhất, làm ngay): bỏ pedalboard khỏi build, app tự fallback
   librosa (ISC). ✅ Không tốn tiền, công sức thấp.
2. **PyQt6 → PySide6** (LGPL) HOẶC mua license PyQt6 thương mại. Đây là quyết định lớn nhất.
   - Khuyến nghị: chuyển PySide6 nếu muốn miễn phí license; mua PyQt6 nếu muốn giữ code.
3. **qfluentwidgets:** mua commercial license (dù PyQt6 hay PySide6). Đây là chi phí khó
   tránh nếu giữ giao diện Fluent. Rẻ hơn tự xây UI.
4. **edge-tts:** cô lập subprocess hoặc bỏ, thay bằng Gemini/VieNeu.

### Cần xác nhận (kiểm build flags):
5. Xác nhận **libmpv** dùng bản LGPL.
6. Xác nhận **PyAV/FFmpeg** wheel là LGPL (không kèm x264 GPL).

### Không cần làm gì:
7. Toàn bộ thư viện Apache/BSD/MIT/ISC — giữ nguyên.

---

## 7. Ước tính chi phí (tham khảo, cần xác nhận giá thực tế)

| Hạng mục | Chi phí | Ghi chú |
|---|---|---|
| pedalboard → librosa | 0đ | Đã có fallback |
| PySide6 (thay PyQt6) | 0đ | Công sức kỹ thuật thay import |
| HOẶC PyQt6 commercial | ~550 USD/dev/năm | Giữ nguyên code |
| qfluentwidgets commercial | Xem qfluentwidgets.com | Thường vài trăm CNY/USD |
| Luật sư rà soát | Tuỳ | Khuyến nghị mạnh |

**Con đường rẻ nhất:** pedalboard→librosa (0đ) + PyQt6→PySide6 (0đ công sức) + mua chỉ
qfluentwidgets commercial license. Tổng chi phí license = chỉ qfluentwidgets.
