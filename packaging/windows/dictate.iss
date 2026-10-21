; Inno Setup script for the Windows installer.
; Build: iscc /DAppVersion=0.7.0 /DSourceDir=<dir with dictate.exe and DLLs> packaging\windows\dictate.iss
; Writes dictate-windows-x86_64-setup.exe to OutputDir (default: target\installer).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\target\x86_64-pc-windows-msvc\release"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\target\installer"
#endif

[Setup]
AppId={{6F0D3B52-8C1E-4F7A-9B2D-5E4C1A7D9E30}
AppName=Dictate
AppVersion={#AppVersion}
AppVerName=Dictate {#AppVersion}
AppPublisher=Dictate
AppPublisherURL=https://github.com/saostad/dictate
AppSupportURL=https://github.com/saostad/dictate/issues
DefaultDirName={autopf}\Dictate
DefaultGroupName=Dictate
DisableProgramGroupPage=yes
; Per-user install by default (no admin prompt); the dialog lets the user
; install for all users instead.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
SetupIconFile=dictate.ico
UninstallDisplayIcon={app}\dictate.ico
UninstallDisplayName=Dictate
OutputDir={#OutputDir}
OutputBaseFilename=dictate-windows-x86_64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=force

[Tasks]
Name: "autostart"; Description: "Start Dictate when I sign in"
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\dictate.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "dictate.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; "daemon --background" runs without a console window and shows the tray icon.
Name: "{group}\Dictate"; Filename: "{app}\dictate.exe"; Parameters: "daemon --background"; IconFilename: "{app}\dictate.ico"
Name: "{autodesktop}\Dictate"; Filename: "{app}\dictate.exe"; Parameters: "daemon --background"; IconFilename: "{app}\dictate.ico"; Tasks: desktopicon

[Registry]
; The same entry the app's own "Start at login" setting writes.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "dictate"; ValueData: """{app}\dictate.exe"" daemon --background"; Tasks: autostart
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: none; ValueName: "dictate"; Flags: uninsdeletevalue dontcreatekey

[Run]
Filename: "{app}\dictate.exe"; Parameters: "daemon --background"; Description: "{cm:LaunchProgram,Dictate}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/F /IM dictate.exe"; Flags: runhidden; RunOnceId: "StopDictate"

[Code]
// The speech model (~660 MB) is not in the installer; dictate cannot start
// without it. Download it now, in a console window that shows the progress.
// An already downloaded model is kept.
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then
    Exit;
  WizardForm.StatusLabel.Caption := 'Downloading the speech model (about 660 MB)...';
  if not Exec(ExpandConstant('{app}\dictate.exe'), 'download', '', SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
    SuppressibleMsgBox('Dictate could not download its speech model. Check your internet connection, then run this installer again.', mbError, MB_OK, IDOK);
end;
