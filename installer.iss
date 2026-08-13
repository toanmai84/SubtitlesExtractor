; ============================================================================
;  Inno Setup script - tao installer Windows cho SubtitlesExtractor.
;
;  Yeu cau:
;    1. Da build xong bang build_windows.bat (co dist\SubtitlesExtractor\).
;    2. Cai Inno Setup: https://jrsoftware.org/isdl.php
;
;  Cach dung:
;    - Mo file nay bang Inno Setup Compiler, bam Build (Ctrl+F9).
;    - Hoac: iscc installer.iss
;
;  Ket qua: Output\SubtitlesExtractor_Setup.exe
; ============================================================================

#define MyAppName "SubtitlesExtractor"
#define MyAppVersion "3.23.292"
#define MyAppPublisher "SubtitlesExtractor Team"
#define MyAppExeName "SubtitlesExtractor.exe"

[Setup]
AppId={{A7F3C2E1-8B4D-4E6A-9C1F-2D5B8E7A3F90}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
; Cho phep cai vao thu muc nguoi dung (khong can quyen admin neu chon vay).
PrivilegesRequiredOverridesAllowed=dialog
OutputBaseFilename=SubtitlesExtractor_Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Hien thi thong bao license ben thu ba khi cai dat.
InfoBeforeFile=THIRD_PARTY_LICENSES.md
; App ghi du lieu vao %APPDATA% (da xu ly trong code) - khong ghi vao thu muc cai.

[Languages]
Name: "vietnamese"; MessagesFile: "compiler:Languages\Vietnamese.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Tao bieu tuong tren Desktop"; GroupDescription: "Bieu tuong bo sung:"

[Files]
; Toan bo thu muc onedir tu PyInstaller.
Source: "dist\SubtitlesExtractor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Go cai dat {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Chay {#MyAppName} ngay"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Khong xoa du lieu nguoi dung o %APPDATA% khi go cai (giu lai cau hinh/lich su).
; Neu muon xoa sach, bo comment dong duoi:
; Type: filesandordirs; Name: "{userappdata}\SubtitlesExtractor"
