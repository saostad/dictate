# dictate

Superwhisper-like voice typing in pure Rust: press a hotkey, speak, release,
and the text is typed into the focused input. No Tauri, webview or Electron.

Round 1 covers the foundation: download the speech model and transcribe a WAV
file from the command line.

## Install

Download the archive for your OS from
[the latest release](https://github.com/saostad/dictate/releases/latest):

| OS | CPU | File |
|---|---|---|
| Linux | x86_64 | `dictate-linux-x86_64.tar.gz` |
| Linux (CPUs without AVX2) | x86_64 | `dictate-linux-x86_64-system-ort.tar.gz` |
| Linux | aarch64 | `dictate-linux-aarch64.tar.gz` |
| macOS | Apple Silicon | `dictate-macos-aarch64.tar.gz` |
| Windows | x86_64 | `dictate-windows-x86_64.zip` |

There is no Intel macOS build (ONNX Runtime ships no prebuilt library for it).

The Linux binaries need glibc 2.39+ (Ubuntu 24.04+, Fedora 40+, Arch).

1. Unpack the archive:

   ```sh
   tar -xzf dictate-linux-x86_64.tar.gz
   cd dictate-linux-x86_64
   ```

   On Windows, right-click → Extract All (or
   `Expand-Archive dictate-windows-x86_64.zip .` in PowerShell).

2. Download the speech model (~660 MB):

   ```sh
   ./dictate download
   ```

3. Start the daemon (it keeps the model loaded):

   ```sh
   ./dictate daemon
   ```

   Then dictate — see [Use it](#use-it) and
   [Global hotkey](#global-hotkey).

### macOS notes

The binary is unsigned, so remove the quarantine flag after unpacking:

```sh
xattr -d com.apple.quarantine ./dictate
```

Grant Microphone and Accessibility to the terminal (or app) that runs
`dictate daemon` (System Settings → Privacy & Security).

### Windows notes

On first run, SmartScreen may warn about the unsigned binary: click
"More info → Run anyway". Socket control commands (`dictate toggle`,
`start`/`stop`) are not available on Windows yet — dictate with the
global hotkey (Ctrl+Shift+Space by default).

### Linux notes

On Wayland, bind `dictate toggle` (or `start`/`stop`) in your compositor;
see [Use it](#use-it).

Most x86_64 PCs want `dictate-linux-x86_64.tar.gz`. If your CPU has no
AVX2, that binary exits with:

```text
This CPU has no AVX2, but this build bundles an ONNX Runtime that needs AVX2. Install ONNX Runtime from your package manager (Arch: sudo pacman -S onnxruntime-cpu) and rebuild with: cargo build --release --no-default-features --features system-ort
```

In that case use `dictate-linux-x86_64-system-ort.tar.gz` instead,
together with a system ONNX Runtime:

```sh
sudo pacman -S onnxruntime-cpu
```

or download ONNX Runtime and set ORT_DYLIB_PATH to the full path of the
onnxruntime library file.

## Build

```sh
cargo build --release
```

## Usage

```sh
dictate download
dictate transcribe x.wav
dictate listen
```

## Use it

Start the daemon once (it keeps the model loaded), then trigger recordings
with the control commands:

```sh
dictate daemon
dictate toggle   # start/stop recording
dictate status   # idle, recording or transcribing
```

The daemon plays a short tone when recording starts and stops; set `DICTATE_SOUNDS=0` to silence it.

Bind keys to the control commands in your compositor or desktop. Hyprland
(`~/.config/hypr/hyprland.conf`):

```conf
bind = SUPER, D, exec, dictate toggle
```

Push-to-talk:

```conf
bind = SUPER, D, exec, dictate start
bindr = SUPER, D, exec, dictate stop
```

On GNOME or KDE, bind the same commands in Settings → Keyboard → Custom
Shortcuts.

## Global hotkey

Outside a Wayland session, `dictate daemon` also registers a global hotkey:
**Ctrl+Shift+Space** by default, or whatever `DICTATE_HOTKEY` holds:

```sh
DICTATE_HOTKEY=Alt+F9 dictate daemon
```

- Tap (press and release quickly): toggles recording on and off.
- Hold (keep it pressed): records while held and transcribes on release
  (push-to-talk).

On Wayland, apps cannot register global hotkeys, so the daemon skips this and
prints a note; bind `dictate toggle` (or `start`/`stop`) in your compositor
instead (see above).

On macOS, grant Accessibility (for typing) and Microphone to the terminal or
app that runs `dictate daemon`.

## Model

Speech recognition uses NVIDIA Parakeet TDT 0.6B v2 (English), released under
CC-BY-4.0, using the int8 ONNX export by istupakov.

## CPUs without AVX2 / system ONNX Runtime

The default build bundles an ONNX Runtime that needs AVX2 (x86_64). On older
CPUs, install ONNX Runtime from your package manager and build against the
system library instead:

```sh
sudo pacman -S onnxruntime-cpu
cargo build --release --no-default-features --features system-ort
```

If the library is not on the default search path, point at it with
`ORT_DYLIB_PATH=/full/path/to/libonnxruntime.so`.

## License

Code: MIT (see LICENSE). Speech model: NVIDIA Parakeet TDT 0.6B v2,
CC-BY-4.0, downloaded at runtime by `dictate download` — not bundled.
