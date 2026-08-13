@echo off
REM ==========================================================================
REM  Build lai NHANH + SACH HOAN TOAN moi loai cache.
REM  Giu build_env (khong cai lai dependencies) nhung xoa:
REM   - build/ + dist/ (cache phan tich PyInstaller)
REM   - __pycache__ trong project (bytecode .pyc cu co the stale)
REM   - cache toan cuc PyInstaller (qua --clean)
REM ==========================================================================
setlocal

cd /d "%~dp0"

echo.
echo ========================================================
echo   BUILD LAI SACH (xoa moi cache, giu build_env)
echo ========================================================
echo [INFO] Thu muc: %CD%
echo.

if not exist build_env (
    echo [LOI] Chua co build_env. Hay chay build_windows.bat truoc.
    exit /b 1
)
if not exist "SubtitlesExtractor.spec" (
    echo [LOI] Khong tim thay SubtitlesExtractor.spec.
    exit /b 1
)

set "PYEXE=build_env\Scripts\python.exe"
if not exist "%PYEXE%" (
    echo [LOI] Khong tim thay %PYEXE%. Hay chay lai build_windows.bat.
    exit /b 1
)

"%PYEXE%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [INFO] build_env chua co PyInstaller. Dang cai...
    "%PYEXE%" -m pip install "pyinstaller>=6.0"
    if errorlevel 1 (
        echo [LOI] Cai PyInstaller that bai.
        exit /b 1
    )
)

REM --- Xoa build/ + dist/ ---
if exist build (
    echo [INFO] Xoa cache build cu...
    rmdir /s /q build
)
if exist dist (
    echo [INFO] Xoa dist cu...
    rmdir /s /q dist
)

REM --- Xoa MOI __pycache__ trong project (bytecode stale) ---
echo [INFO] Xoa __pycache__ cu trong src...
for /d /r "src" %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r "." %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

REM --- KIEM CHUNG: in so metadata spec se bundle (xac nhan copy_metadata co tac dung) ---
echo.
echo [KIEM CHUNG] Dem metadata paddleocr recursive se bundle...
"%PYEXE%" -c "from PyInstaller.utils.hooks import copy_metadata; print('  -> paddleocr metadata:', len(copy_metadata('paddleocr', recursive=True)), 'entries (phai > 1)')" 2>nul
echo.

REM --- Build voi --clean (xoa ca cache toan cuc PyInstaller) ---
echo [INFO] Build lai (--clean, sach hoan toan)...
echo.
"%PYEXE%" -m PyInstaller SubtitlesExtractor.spec --noconfirm --clean
if errorlevel 1 (
    echo [LOI] Build that bai. Xem log ben tren.
    exit /b 1
)

if exist "dist\SubtitlesExtractor\SubtitlesExtractor.exe" (
    echo.
    echo ========================================================
    echo   BUILD LAI THANH CONG!
    echo   File: dist\SubtitlesExtractor\SubtitlesExtractor.exe
    echo ========================================================
    echo.
    echo [KIEM CHUNG] Metadata da bundle trong ban build:
    dir /b "dist\SubtitlesExtractor\_internal\paddleocr*" 2>nul
    dir /b "dist\SubtitlesExtractor\_internal\paddlex*" 2>nul
    echo.
) else (
    echo [LOI] Khong tim thay exe sau build.
    exit /b 1
)

endlocal
