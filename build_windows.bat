@echo off
REM ==========================================================================
REM  Build SubtitlesExtractor thanh MOT file .exe Windows (che do one-file, mac dinh).
REM  Xem docs\BUILD_PLAN.md de hieu cac quyet dinh dong goi.
REM
REM  Cach dung:  build_windows.bat
REM  Ket qua:    dist\SubtitlesExtractor.exe   (MOT file duy nhat)
REM
REM  Cong tac moi truong (dat TRUOC khi chay):
REM    set SUBEXT_ONEDIR=1            -> che do thu muc (dist\SubtitlesExtractor\) thay vi 1 file
REM    set SUBEXT_BUNDLE_PADDLE=1     -> nhung san loi paddle ~810MB (mac dinh: tai luc chay)
REM    set SUBEXT_BUNDLE_CUDA=1       -> nhung san CUDA ~2.3GB (mac dinh: tai luc chay)
REM    set SUBEXT_PREFETCH_MODELS=1   -> nhung san model OCR/TTS (mac dinh: tai luc chay)
REM    set SUBEXT_ENABLE_WHISPERX=1   -> tao moi truong rieng whisperx_env (~3GB)
REM ==========================================================================
setlocal enabledelayedexpansion

REM --- Chuyen vao thu muc chua file .bat (du chay tu o dia / thu muc khac) ---
cd /d "%~dp0"

echo.
echo ========================================================
echo   BUILD SubtitlesExtractor (one-file mac dinh)
echo ========================================================
echo.
echo [INFO] Thu muc lam viec: %CD%
echo.

REM --- Buoc 0a: KIEM TRA DUNG LUONG DIA (v3.23.303) ---------------------------
REM  Build v3.23.302 that bai o buoc COLLECT voi loi:
REM      OSError: [Errno 28] No space left on device
REM  Ban GPU can rat nhieu cho trong TREN CUNG O DIA voi du an:
REM      build_env\      ~10 GB  (paddlepaddle-gpu 810MB + CUDA nvidia-* ~2.3GB
REM                               + PySide6 ~250MB + phan con lai)
REM      models\         ~2  GB  (PaddleOCR ~250MB + VieNeu-TTS ~1.7GB)
REM      build\          ~8  GB  (thu muc trung gian PyInstaller)
REM      dist\           ~8  GB  (ban giao cuoi cung - COLLECT chep tu build\)
REM  => Can khoang 35 GB trong. Kiem tra truoc de bao som thay vi chet o phut 90.
set "MIN_FREE_GB=35"
set "FREE_GB="
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "[math]::Floor((Get-PSDrive -Name ((Get-Location).Drive.Name)).Free/1GB)" 2^>nul`) do set "FREE_GB=%%A"

if not defined FREE_GB echo [CANH BAO] Khong do duoc dung luong dia - bo qua kiem tra.
if not defined FREE_GB goto :disk_check_done

echo [INFO] Dung luong trong o dia hien tai: %FREE_GB% GB ^(can toi thieu %MIN_FREE_GB% GB^).
if %FREE_GB% GEQ %MIN_FREE_GB% goto :disk_check_done

echo.
echo [LOI] KHONG DU DUNG LUONG DIA. Con %FREE_GB% GB, can toi thieu %MIN_FREE_GB% GB.
echo.
echo       Cach giai phong nhanh:
echo         1. Xoa build_env, build, dist cu cua lan build truoc.
echo         2. Xoa cache pip:  pip cache purge
echo         3. Xoa cache HuggingFace cu trong %%USERPROFILE%%\.cache\huggingface
echo         4. Chuyen du an sang o dia khac con nhieu cho hon.
echo.
exit /b 1

:disk_check_done
echo.

REM --- Buoc 0: Kiem tra Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python. Hay cai Python 3.11+ va them vao PATH.
    exit /b 1
)
echo [OK] Python da san sang.

REM --- Kiem tra dang o dung thu muc du an (co requirements.txt + spec) ---
if not exist "requirements.txt" (
    echo [LOI] Khong tim thay requirements.txt trong: %CD%
    echo       Hay dat build_windows.bat trong THU MUC GOC du an,
    echo       cung cho voi requirements.txt va SubtitlesExtractor.spec
    exit /b 1
)
if not exist "SubtitlesExtractor.spec" (
    echo [LOI] Khong tim thay SubtitlesExtractor.spec trong: %CD%
    echo       Hay dat build_windows.bat trong THU MUC GOC du an.
    exit /b 1
)
echo [OK] Da tim thay requirements.txt va spec file.

REM --- Buoc 1: Tao moi truong ao SACH (tranh bundle thu vien thua) ---
if exist build_env (
    echo [INFO] Xoa moi truong build cu...
    rmdir /s /q build_env
)
echo [INFO] Tao moi truong ao build_env...
python -m venv build_env
if errorlevel 1 (
    echo [LOI] Khong tao duoc moi truong ao.
    exit /b 1
)
REM Goi truc tiep python trong build_env (chac chan hon activate + PATH)
set "PYEXE=build_env\Scripts\python.exe"

REM --- Buoc 2: Cai dependencies runtime (KHONG cai dev/test/whisperx) ---
echo [INFO] Nang cap pip...
"%PYEXE%" -m pip install --upgrade pip >nul

REM --- Chon file requirements: uu tien ban GPU neu co ---
set "REQFILE=requirements.txt"
if exist "requirements-gpu.txt" (
    set "REQFILE=requirements-gpu.txt"
    echo [INFO] Dung requirements-gpu.txt - ban GPU + paddlepaddle-gpu.
) else (
    echo [INFO] Dung requirements.txt - ban CPU.
)

echo [INFO] Cai dependencies runtime (co the mat vai phut)...
REM [v3.23.302] TACH 2 BUOC de tranh WARNING ReadTimeout tu paddlepaddle.org.cn:
REM   Neu de --extra-index-url trong requirements, pip se hoi CA HAI index cho MOI
REM   goi -> may chu Paddle (Trung Quoc) cham/timeout tu VN -> retry 15s x nhieu goi.
REM   Giai phap: cai phan lon tu PyPI (khong extra-index), rieng paddlepaddle-gpu moi
REM   dung index Paddle. Nhanh hon nhieu va het WARNING.
"%PYEXE%" -m pip install -r "%REQFILE%" --index-url https://pypi.org/simple --retries 3 --timeout 60
if errorlevel 1 (
    echo [LOI] Cai requirements that bai.
    exit /b 1
)

REM Rieng paddlepaddle-gpu: chi goi nay moi can index Paddle (cu129).
if "%REQFILE%"=="requirements-gpu.txt" (
    echo [INFO] Cai paddlepaddle-gpu tu kho Paddle cu129 - goi lon ~810MB, kien nhan...
    "%PYEXE%" -m pip install "paddlepaddle-gpu==3.3.1" --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ --extra-index-url https://pypi.org/simple --retries 5 --timeout 120
    if errorlevel 1 (
        echo [LOI] Cai paddlepaddle-gpu that bai. Kiem tra mang / thu lai.
        exit /b 1
    )
)

REM --- (TUY CHON) WhisperX: nhan dang giong noi de canh thoi gian phu de -------
REM  [v3.23.333] CAI VAO MOI TRUONG RIENG `whisperx_env`, KHONG cai chung build_env.
REM  Ly do (da tra metadata that cua whisperx 3.8.6):
REM    1. whisperx ghim `huggingface-hub<1.0.0` nhung ung dung dang dung 1.24.0.
REM       Cai chung se HA CAP goi nay -> co the lam hong VieNeu-TTS va PaddleOCR.
REM    2. torch nap CUDA rieng, de xung dot DLL voi paddle (adapter von da chay
REM       subprocess chinh vi ly do nay).
REM    3. Bundle se phinh them ~3GB neu gom torch vao.
REM  Adapter tu tim `whisperx_env` canh thu muc du an, nen KHONG can gom vao bundle.
REM
REM  De BAT:
REM      set SUBEXT_ENABLE_WHISPERX=1
REM      build_windows.bat
if /i not "%SUBEXT_ENABLE_WHISPERX%"=="1" goto :skip_whisperx
echo.
echo [INFO] SUBEXT_ENABLE_WHISPERX=1 - tao moi truong RIENG cho WhisperX (~3GB)...
if exist whisperx_env goto :whisperx_env_ready
python -m venv whisperx_env
if errorlevel 1 echo [CANH BAO] Khong tao duoc whisperx_env - bo qua WhisperX. & goto :skip_whisperx
:whisperx_env_ready
echo [INFO] Cai torch/torchaudio/torchvision ban CUDA 12.9 (khop paddle cu129)...
whisperx_env\Scripts\python.exe -m pip install --upgrade pip --quiet
whisperx_env\Scripts\python.exe -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu129 --retries 3 --timeout 300
if errorlevel 1 echo [CANH BAO] Cai torch CUDA that bai - WhisperX se khong dung duoc GPU. & goto :skip_whisperx
echo [INFO] Cai whisperx...
whisperx_env\Scripts\python.exe -m pip install whisperx --retries 3 --timeout 300
if errorlevel 1 echo [CANH BAO] Cai whisperx that bai - tinh nang STT se tat.
echo [INFO] Xong. Ung dung se tu tim whisperx_env khi can nhan dang giong noi.
:skip_whisperx

REM --- (TUY CHON) TensorRT cho paddle -----------------------------------------
REM  Log build hien canh bao:
REM      Failed to collect submodules for 'paddle.tensorrt' -> No module named 'tensorrt'
REM  Day chi la CANH BAO vo hai (paddle van chay GPU binh thuong qua cuDNN/cuBLAS).
REM  App DA co san tuy chon "use_tensorrt" trong Cai dat > Phan cung, nhung no chi co
REM  tac dung neu goi 'tensorrt' duoc cai trong build_env.
REM
REM  ** MAC DINH TAT ** vi 3 ly do (theo uu tien license-clean thuong mai cua du an):
REM    1. LICENSE: TensorRT la phan mem DOC QUYEN cua NVIDIA, co EULA rieng — KHONG
REM       phai giay phep mo nhu phan con lai cua stack (LGPL/Apache/MIT). Muon phan
REM       phoi thuong mai phai doc va tuan thu dieu khoan phan phoi lai cua NVIDIA.
REM    2. DUNG LUONG: them khoang 1-2 GB nua vao ban dong goi (da ~8GB).
REM    3. DO ON DINH: phai khop phien ban TensorRT <-> paddle <-> CUDA; lech phien ban
REM       de gay crash. Lan chay dau con ton them thoi gian dung "engine" TRT.
REM
REM  De BAT: dat bien moi truong truoc khi chay build:
REM      set SUBEXT_ENABLE_TENSORRT=1
REM      build_windows.bat
if /i not "%SUBEXT_ENABLE_TENSORRT%"=="1" goto :skip_tensorrt
echo.
echo [INFO] SUBEXT_ENABLE_TENSORRT=1 - cai TensorRT (tuy chon, goi lon)...
echo [CANH BAO] TensorRT la phan mem doc quyen NVIDIA - kiem tra EULA truoc khi phan phoi.
"%PYEXE%" -m pip install "tensorrt-cu12" --retries 3 --timeout 120
if errorlevel 1 echo [CANH BAO] Cai TensorRT that bai - build tiep, paddle chay khong co TRT.
:skip_tensorrt

REM (Tuy chon) Cai VieNeu neu muon bundle engine offline. Bo REM de bat:
REM echo [INFO] Cai VieNeu (tuy chon)...
REM "%PYEXE%" -m pip install vieneu sea-g2p onnxruntime

echo [INFO] Cai PyInstaller...
"%PYEXE%" -m pip install "pyinstaller>=6.0"
if errorlevel 1 (
    echo [LOI] Cai PyInstaller that bai.
    exit /b 1
)

REM [v3.23.391] EP setuptools<80 (SAU khi cai paddlepaddle-gpu, phong khi no keo lai 84).
REM paddlepaddle-gpu 3.3.1 import setuptools.command.easy_install (bi xoa o setuptools>=80).
REM Khong ep thi 'import paddle' loi "No module named 'setuptools.command.easy_install'".
echo [INFO] Ghim setuptools^<80 (paddle can easy_install)...
"%PYEXE%" -m pip install "setuptools<80"
if errorlevel 1 (
    echo [CANH BAO] Khong ha duoc setuptools^<80 - paddle co the loi easy_install luc chay.
)

REM --- Buoc 3: Don dep ban build cu ---
if exist build (
    echo [INFO] Xoa thu muc build cu...
    rmdir /s /q build
)
if exist dist (
    echo [INFO] Xoa thu muc dist cu...
    rmdir /s /q dist
)

REM --- Buoc 3b: Prefetch model PaddleOCR de OCR OFFLINE lan dau (v3.23.293) ---
REM Tai san model detection/recognition moi ngon ngu UI cua phien ban mac dinh
REM vao models\paddle. Spec se tu nhung neu thu muc nay ton tai.
REM Bo qua an toan neu that bai (khi do app se tai model theo yeu cau nhu cu).
REM [v3.23.391] MAC DINH BO QUA prefetch de ban one-file NHO (model tai luc chay vao
REM models\ canh exe). Dat SUBEXT_PREFETCH_MODELS=1 neu muon nhung san model (OCR offline
REM ngay lan dau, nhung file .exe to hon).
if /i not "%SUBEXT_PREFETCH_MODELS%"=="1" (
    echo [INFO] Bo qua prefetch model ^(ban nho - model tai luc chay^). Dat SUBEXT_PREFETCH_MODELS=1 de nhung san.
    goto :skip_prefetch
)
echo.
echo [INFO] Prefetch model OCR (offline lan dau)... co the mat vai phut + can mang.
"%PYEXE%" tools\prefetch_ocr_models.py --staging models\paddle --device cpu
if errorlevel 1 (
    echo [CANH BAO] Prefetch model OCR that bai — build tiep, app se tai model khi chay.
)

echo [INFO] Prefetch model HuggingFace (VieNeu-TTS)... can mang.
"%PYEXE%" tools\prefetch_hf_models.py --store models\huggingface
if errorlevel 1 (
    echo [CANH BAO] Prefetch model HF that bai - build tiep, app se tai model khi chay.
)
:skip_prefetch

REM --- Buoc 3a: KIEM CU PHAP CHINH TEP BATCH NAY ------------------------------
REM  [v3.23.337] Su co that: mot khoi cu chua xoa khien `pip install whisperx` chay
REM  VO DIEU KIEN vao moi truong chinh -> ha cap huggingface-hub 1.25 xuong 0.36 va
REM  lam hong gradio. Phep kiem cu chi xet ngoac + goto nen khong thay gi.
echo.
echo [INFO] Kiem cu phap tep batch...
"%PYEXE%" tools\check_batch_syntax.py
if errorlevel 1 echo [CANH BAO] Tep batch co van de - doc bao cao o tren.

REM --- Buoc 3b: KIEM XUNG DOT PHU THUOC (thong tin) ---------------------------
REM  [v3.23.334] Sau su co whisperx (ghim huggingface-hub<1.0 trong khi app dung 1.24),
REM  buoc nay kiem TRUOC khi build xem co goi nao se lam DOI phien ban goi dang dung.
REM  Nguy hiem nhat la truong hop pip HA CAP am tham: paddlex/vieneu khong rang buoc
REM  phien ban huggingface-hub nen pip cho phep ha cap ma khong bao loi.
echo.
echo [INFO] Kiem xung dot phu thuoc...
"%PYEXE%" tools\check_dependency_conflicts.py
if errorlevel 1 (
    echo.
    echo [CANH BAO] Co goi se lam doi phien ban goi dang dung - doc bao cao o tren.
    echo            Can nhac cai goi do vao moi truong rieng nhu whisperx_env.
    echo.
)

REM --- Buoc 3c: GHI NHAN LICENSE (thong tin, KHONG chan build) ----------------
REM  [v3.23.312] Du an KHONG con huong toi phan phoi thuong mai nen GPL duoc chap nhan.
REM  Van chay kiem de GHI LAI thanh phan dang dung — huu ich khi chia se ma nguon
REM  (GPL yeu cau kem ma nguon + ghi nhan ban quyen).
echo.
echo [INFO] Ghi nhan license cac thanh phan media...
"%PYEXE%" tools\check_media_licenses.py
echo [INFO] Da ghi nhan. Du an theo huong GPL - xem THIRD_PARTY_LICENSES.md.
echo.

REM [v3.23.397] Nhung san Python EMBEDDABLE (~15-25MB) de .exe TU LAP tai paddle/CUDA ma
REM KHONG can Python cai san tren may nguoi dung. Idempotent (bo qua neu da co). That bai thi
REM build tiep - app se lui ve Python he thong nhu cu.
echo.
echo [INFO] Chuan bi Python embeddable (de .exe tu lap, khong can Python he thong)...
"%PYEXE%" tools\setup_embedded_python.py
if errorlevel 1 (
    echo [CANH BAO] Khong chuan bi duoc Python embeddable - build tiep. Khi chay, viec tai
    echo            paddle/CUDA se can Python cai san tren may.
)

REM --- Buoc 4: Chay PyInstaller voi spec file ---
REM [v3.23.392] TIEN-KIEM TRA moi truong build TRUOC khi ton 10-30 phut dong goi.
REM Bat som cac loi da tung khien mat ca chu ky build (setuptools>=80 thieu easy_install,
REM thieu paddle/paddleocr...). DUNG build neu co muc BAT BUOC chua dat.
echo.
echo [INFO] Tien-kiem tra moi truong build...
"%PYEXE%" tools\preflight_build_check.py
if errorlevel 1 (
    echo.
    echo [LOI] Moi truong build chua dat - xem bao cao o tren. Dung build.
    exit /b 1
)

echo.
echo [INFO] Bat dau build (co the mat 10-30 phut voi paddle)...
echo.
"%PYEXE%" -m PyInstaller SubtitlesExtractor.spec --noconfirm --clean
if errorlevel 1 (
    echo.
    echo [LOI] Build that bai. Xem log ben tren de chan doan.
    echo       Cac loi thuong gap: thieu DLL paddle, hidden import.
    exit /b 1
)

REM --- Buoc 5: Kiem tra ket qua (onefile vs onedir) ---
REM [v3.23.391] one-file: dist\SubtitlesExtractor.exe ; onedir: dist\SubtitlesExtractor\SubtitlesExtractor.exe
if /i "%SUBEXT_ONEDIR%"=="1" (
    set "EXE_PATH=dist\SubtitlesExtractor\SubtitlesExtractor.exe"
) else (
    set "EXE_PATH=dist\SubtitlesExtractor.exe"
)
if not exist "!EXE_PATH!" goto :build_failed

echo.
echo ========================================================
echo   BUILD THANH CONG
echo   File: !EXE_PATH!
echo ========================================================
echo.

REM --- KIEM BAN DONG GOI (v3.23.341) ------------------------------------------
REM  check_bundle.py chi kiem cau truc thu muc onedir. one-file goi het vao 1 .exe
REM  nen bo qua buoc nay.
if /i "%SUBEXT_ONEDIR%"=="1" (
    echo [INFO] Kiem ban dong goi...
    "%PYEXE%" tools\check_bundle.py
    if errorlevel 1 (
        echo.
        echo [CANH BAO] Ban dong goi THIEU thanh phan - doc bao cao o tren.
        echo            Sua tep .spec roi build lai truoc khi phan phoi.
        echo.
    )
) else (
    echo [INFO] Che do one-file - bo qua check_bundle ^(khong co thu muc de kiem^).
)

REM [v3.23.303] Thu hoi dung luong: build\ chi la trung gian, xoa an toan.
echo [INFO] Don thu muc trung gian build\ de thu hoi dung luong...
if exist build rmdir /s /q build

REM [v3.23.391] Bao cao kich thuoc ban giao (onefile: 1 file; onedir: ca thu muc).
set "DIST_MB="
if /i "%SUBEXT_ONEDIR%"=="1" (
    for /f "usebackq delims=" %%S in (`powershell -NoProfile -Command "$t=0; foreach($f in [System.IO.Directory]::EnumerateFiles('%CD%\dist\SubtitlesExtractor','*','AllDirectories')){$t+=(New-Object System.IO.FileInfo $f).Length}; [math]::Round($t/1MB)" 2^>nul`) do set "DIST_MB=%%S"
) else (
    for /f "usebackq delims=" %%S in (`powershell -NoProfile -Command "[math]::Round((New-Object System.IO.FileInfo '%CD%\dist\SubtitlesExtractor.exe').Length/1MB)" 2^>nul`) do set "DIST_MB=%%S"
)
if defined DIST_MB echo [INFO] Kich thuoc ban dong goi: %DIST_MB% MB.

echo.
echo   Buoc tiep theo:
echo   1. Chay thu: !EXE_PATH!
echo   2. Lan dau chay: app se moi TAI LOI OCR (paddle ~810MB) - bam Co de tai.
echo      Muon GPU: Cai dat -^> Bat tang toc GPU (OCR) -^> tai CUDA.
echo   3. Kiem tra checklist trong docs\BUILD_PLAN.md muc 6.
echo   4. Tuy chon: dong goi installer bang Inno Setup.
echo.
goto :build_done

:build_failed
echo [LOI] Khong tim thay file exe sau build. Kiem tra log.
exit /b 1

:build_done

endlocal
