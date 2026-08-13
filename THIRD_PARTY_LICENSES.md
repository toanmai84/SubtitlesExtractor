# Thông báo License Thư viện Bên thứ ba (Third-Party Licenses)

> ## ⚠️ THAY ĐỔI ĐỊNH HƯỚNG (v3.23.312)
>
> Dự án **KHÔNG còn hướng tới phân phối thương mại mã đóng**. Ràng buộc "chỉ dùng
> LGPL/Apache/MIT/ISC" đã được gỡ bỏ; các thành phần **GPL** nay được chấp nhận để
> đổi lấy chất lượng tốt hơn.
>
> **Hệ quả pháp lý cần biết:** khi phân phối ứng dụng có kèm thành phần GPL
> (libmpv, pedalboard, edge-tts), toàn bộ tác phẩm phải được phát hành theo **GPL** —
> nghĩa là phải **kèm mã nguồn** và giữ nguyên các thông báo bản quyền dưới đây.
> Dùng cho mục đích cá nhân/nội bộ thì không phát sinh nghĩa vụ này.

---



Ứng dụng **SubtitlesExtractor** sử dụng các thư viện mã nguồn mở bên thứ ba dưới đây. Bản
phân phối này giữ nguyên các thông báo bản quyền và license theo yêu cầu của từng thư viện.

> Tài liệu này liệt kê license để tuân thủ yêu cầu phân phối. Không phải tư vấn pháp lý —
> hãy nhờ luật sư rà soát trước khi phát hành thương mại. Toàn văn license của mỗi thư viện
> có trong thư mục cài đặt tương ứng và tại trang chủ dự án.

---

## 1. Thư viện giao diện (GUI)

### PySide6 / shiboken6
- **License:** LGPL v3 (LGPL-3.0-only)
- **Bản quyền:** © The Qt Company Ltd.
- **Trang chủ:** https://www.qt.io/qt-for-python
- **Ghi chú tuân thủ LGPL:** ứng dụng liên kết ĐỘNG tới PySide6/Qt. Người dùng có quyền
  thay thế thư viện Qt bằng phiên bản khác — với ứng dụng Python đóng gói, thư viện Qt nằm
  ở dạng file .dll/.so riêng có thể thay thế, thoả mãn điều kiện LGPL. Toàn văn LGPL v3
  kèm trong bản phân phối (xem thư mục PySide6).

---

## 2. Thư viện OCR & Thị giác máy tính

### PaddlePaddle / PaddleOCR / PaddleX
- **License:** Apache License 2.0
- **Bản quyền:** © PaddlePaddle Authors, Baidu, Inc.
- **Trang chủ:** https://github.com/PaddlePaddle/PaddleOCR

### OpenCV (opencv-python / opencv-contrib-python)
- **License:** Apache License 2.0
- **Bản quyền:** © OpenCV team
- **Trang chủ:** https://opencv.org

---

## 3. Thư viện khoa học dữ liệu & âm thanh

### NumPy
- **License:** BSD 3-Clause
- **Bản quyền:** © 2005-2025, NumPy Developers
- **Trang chủ:** https://numpy.org

### SciPy
- **License:** BSD 3-Clause
- **Bản quyền:** © 2001-2002 Enthought, Inc.; © 2003-2025 SciPy Developers
- **Trang chủ:** https://scipy.org

### soundfile (PySoundFile)
- **License:** BSD 3-Clause
- **Bản quyền:** © Bastian Bechtold
- **Trang chủ:** https://github.com/bastibe/python-soundfile
- **Ghi chú:** đóng gói libsndfile (LGPL v2.1+, © Erik de Castro Lopo).

### librosa
- **License:** ISC License
- **Bản quyền:** © librosa development team
- **Trang chủ:** https://librosa.org
- **Ghi chú:** dùng cho time-stretch giữ cao độ (thay thế pedalboard GPL).

### PyAV (av)
- **License:** BSD 3-Clause
- **Bản quyền:** © Mike Boers và cộng sự
- **Trang chủ:** https://github.com/PyAV-Org/PyAV
- **Ghi chú FFmpeg:** PyAV nhúng FFmpeg. Bản phân phối này dùng FFmpeg build **LGPL** (chỉ
  decode, không kèm encoder GPL như x264/x265). Xem mục 6.

### pydub
- **License:** MIT
- **Bản quyền:** © James Robert
- **Trang chủ:** https://github.com/jiaaro/pydub

---

## 4. Thư viện tiện ích & cấu hình

### pydantic / pydantic-settings
- **License:** MIT
- **Bản quyền:** © Pydantic Services Inc. và cộng sự
- **Trang chủ:** https://pydantic.dev

### rapidfuzz
- **License:** MIT
- **Bản quyền:** © Max Bachmann
- **Trang chủ:** https://github.com/rapidfuzz/RapidFuzz

### loguru
- **License:** MIT
- **Bản quyền:** © Delgan
- **Trang chủ:** https://github.com/Delgan/loguru

### json-repair
- **License:** MIT
- **Bản quyền:** © Stefano Baccianella
- **Trang chủ:** https://github.com/mangiucugna/json_repair

### rjieba
- **License:** MIT
- **Bản quyền:** © messense
- **Trang chủ:** https://github.com/messense/rjieba-py

---

## 5. Thư viện AI / Dịch & TTS

### google-genai (Google Gen AI SDK)
- **License:** Apache License 2.0
- **Bản quyền:** © Google LLC
- **Trang chủ:** https://github.com/googleapis/python-genai
- **Ghi chú:** client gọi API Gemini (dịch + TTS). Yêu cầu API key của người dùng.

### VieNeu-TTS (bundle mặc định từ v3.23.292)
- **License:** Apache License 2.0 — © Phạm Nguyễn Ngọc Bảo (pnnbao97)
- **Trang chủ:** https://github.com/pnnbao97/VieNeu-TTS
- **Ghi chú:** engine TTS tiếng Việt offline (ONNX Runtime, torch-free trên CPU). Model ONNX
  tải runtime lần đầu từ HuggingFace (KHÔNG bundle kèm — nặng). An toàn thương mại.

### sea-g2p (phụ thuộc VieNeu)
- **License:** Apache License 2.0 — © pnnbao97
- **Trang chủ:** https://github.com/pnnbao97/sea-g2p
- **Ghi chú:** chuyển tự vị→âm vị tiếng Việt/Anh (Rust, torch-free).

### soxr (python-soxr, phụ thuộc VieNeu)
- **License:** LGPL v2.1+ — © dofuuz (wrapper), libsoxr © chirlu
- **Trang chủ:** https://github.com/dofuuz/python-soxr
- **Ghi chú:** resampling cho voice cloning. LGPL — liên kết động qua .pyd riêng (thoả mãn
  LGPL như soundfile/libmpv). Kèm toàn văn LGPL trong bản phân phối.

### tokenizers (phụ thuộc VieNeu)
- **License:** Apache License 2.0 — © HuggingFace
- **Trang chủ:** https://github.com/huggingface/tokenizers

### onnxruntime (phụ thuộc VieNeu)
- **License:** MIT — © Microsoft
- **Trang chủ:** https://github.com/microsoft/onnxruntime

### sea-g2p
- **License:** Apache License 2.0 (kiểm xác nhận tại trang chủ)
- **Trang chủ:** https://github.com/pnnbao97/sea-g2p
- **Ghi chú:** phụ thuộc của VieNeu (chuyển tự vị → âm vị).

---

## 6. Thành phần chạy NGOÀI tiến trình chính (không liên kết vào app)

Các thành phần dưới đây KHÔNG được liên kết tĩnh/đóng gói vào ứng dụng. Chúng chạy như
CÔNG CỤ ĐỘC LẬP qua ranh giới tiến trình (subprocess) hoặc tải runtime, nên license của
chúng không áp lên mã nguồn ứng dụng.

### edge-tts (tùy chọn — cách ly subprocess)
- **License:** GPL v3
- **Trang chủ:** https://github.com/rany2/edge-tts
- **Cách dùng:** chạy trong tiến trình RIÊNG (``edge_tts_subprocess.py``) bằng Python cài
  ngoài. KHÔNG đóng gói kèm ứng dụng. Người dùng tự cài nếu muốn dùng engine Edge TTS.

### WhisperX (tùy chọn — cách ly subprocess)
- **License:** BSD-4-Clause
- **Trang chủ:** https://github.com/m-bain/whisperX
- **Cách dùng:** chạy subprocess với Python cài ngoài (cần torch + CUDA). KHÔNG đóng gói.

### libmpv (tùy chọn — tải runtime)
- **License:** LGPL v2.1+ (bản phân phối LGPL) — xác nhận build flags
- **Trang chủ:** https://mpv.io
- **Cách dùng:** ``libmpv-2.dll`` tải runtime từ mirror, dùng cho phát video. Liên kết động
  qua python-mpv → thoả mãn LGPL. Dùng bản build LGPL (không kèm thành phần GPL).
- **python-mpv (wrapper):** GPL v2+/LGPL v2.1+ — © Sebastian Götte.

### FFmpeg (qua PyAV)
- **License:** LGPL v2.1+ (bản build LGPL) — © FFmpeg developers
- **Trang chủ:** https://ffmpeg.org
- **Cách dùng:** decode video/audio. Bản phân phối dùng FFmpeg build LGPL (không --enable-gpl,
  không kèm x264/x265). Xác nhận build flags của wheel PyAV được dùng.

---

## 7. Thành phần TÙY CHỌN không đóng gói (GPU / NVIDIA)

Chỉ cài khi người dùng có phần cứng phù hợp; KHÔNG đóng gói kèm.

- **PyNvVideoCodec** — © NVIDIA. License MIT (kiểm xác nhận). Chỉ dùng khi có GPU NVIDIA.
- **CuPy** — MIT License. © Preferred Networks, Inc. & Preferred Infrastructure, Inc.

---

## 8. Ghi chú tuân thủ khi phân phối

1. **LGPL (PySide6, libmpv, FFmpeg, libsndfile):** giữ liên kết động; kèm toàn văn LGPL;
   cho phép người dùng thay thế thư viện. Ứng dụng Python đóng gói thoả mãn tự nhiên vì
   thư viện nằm ở file .dll/.so riêng.
2. **Apache 2.0 (paddle, opencv, google-genai, VieNeu):** kèm file LICENSE + NOTICE nếu có;
   ghi nhận thay đổi nếu có sửa đổi mã nguồn thư viện (ứng dụng không sửa).
3. **BSD/MIT/ISC:** giữ thông báo bản quyền + văn bản license (tài liệu này thực hiện điều đó).
4. **GPL (edge-tts, whisperx):** KHÔNG đóng gói kèm; chạy như công cụ ngoài qua subprocess.
5. **Kèm tài liệu này** trong bản phân phối (thư mục cài đặt hoặc menu "Giới thiệu → License").

---

*Cập nhật: 2026-07-18. Danh sách phiên bản chính xác xem requirements.txt.*


---

## FFmpeg (binary CLI trong vendor/ffmpeg/)

- **Giấy phép:** GNU Lesser General Public License v3 (LGPL v3)
- **Nguồn:** BtbN/FFmpeg-Builds, biến thể `win64-lgpl` (static)
- **Build:** `--extra-version=20260723`
- **Xác minh:** chuỗi cấu hình nhúng trong binary KHÔNG có `--enable-gpl` và
  KHÔNG có `--enable-nonfree`; có `--enable-version3`. Các thành phần GPL/nonfree
  (x264, x265, xvid, rubberband, frei0r, vidstab, fdk-aac, avisynth) đều bị tắt.
- **Văn bản giấy phép:** `vendor/ffmpeg/LICENSE-ffmpeg-LGPLv3.txt` (phân phối kèm).
- **Nghĩa vụ LGPL:** ứng dụng gọi ffmpeg qua **tiến trình riêng** (subprocess), không
  liên kết tĩnh vào mã ứng dụng — đây là hình thức sử dụng an toàn nhất với LGPL.
  Người dùng có thể thay thế binary này bằng bản khác của riêng họ trong thư mục
  `vendor/ffmpeg/`.


---

## Thành phần GPL được bật từ v3.23.312

| Thành phần | Giấy phép | Vai trò | Phương án dự phòng nếu thiếu |
|---|---|---|---|
| libmpv | GPL v2+ | Trình phát video chính | Trình phát PyAV (LGPL) — đã có sẵn |
| pedalboard | GPL v3 | Kéo giãn thời gian chất lượng cao | librosa (ISC) → WSOLA |
| edge-tts | GPL v3 | Giọng đọc neural Microsoft | VieNeu-TTS / Gemini TTS |

Các gói dưới đây **không phải GPL** — trước đây loại chỉ vì dung lượng:

| Thành phần | Giấy phép | Vai trò |
|---|---|---|
| fastembed | Apache 2.0 | Nhúng ngữ nghĩa cho Bộ nhớ Dịch |
| scikit-learn | BSD-3 | Gom cụm DBSCAN cho tự động dò ROI |
| WhisperX (tùy chọn) | BSD-4 / MIT | Nhận dạng giọng nói để canh thời gian |
