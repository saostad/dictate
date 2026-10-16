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
| Linux | x86_64 | `dictate-linux-x86_64.tar.gz` (works on any x86_64 CPU, no ONNX Runtime install needed) |
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

The x86_64 binary bundles ONNX Runtime and works on any x86_64 CPU, no
ONNX Runtime install needed.

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

If the daemon isn't running, `toggle`/`start`/`stop`/`cancel` start it
automatically (its output goes to the log below) and wait up to 30 s for it.

The daemon plays a short tone when recording starts and stops; set `DICTATE_SOUNDS=0` to silence it.

Failures (and empty recordings) also pop a desktop notification; set `DICTATE_NOTIFY=0` to disable them.

When the daemon's stderr is not a terminal (autostart), it also writes every
line to `daemon.log` in the platform data directory:
`~/.local/share/dictate/daemon.log` on Linux,
`~/Library/Application Support/dictate/daemon.log` on macOS,
`%APPDATA%\dictate\data\daemon.log` on Windows.

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

## Tray menu

`dictate daemon` shows a tray icon: grey when idle, red while recording,
amber while transcribing. Its menu shows the live state, starts/stops/cancels
dictation, toggles sound cues, notifications and type-while-speaking (saved to
the settings file), switches the global shortcut (on Wayland it explains the
compositor keybinds instead), opens the settings file, and quits the daemon.
Without a tray host the daemon logs `dictate: no tray icon` once and keeps
running headless.

## Modes

| Mode | Setting | What happens | Accuracy | Speed |
|---|---|---|---|---|
| Whole (default) | `mode = "whole"` | types once at stop | highest: hears everything at once | slowest after release on long dictations |
| Pauses | `mode = "pauses"` | types each pause segment while speaking | high: 1.5 s of left context per piece | most text is already typed at stop |
| Live | `mode = "live"` | types while speaking, chunk by chunk | good: true streaming model | fastest: text appears as you talk |

Set the mode in the settings file or with `DICTATE_MODE=live`. The tray
menu's "Type while speaking" checkbox toggles pauses mode for now (a Mode
submenu is coming).

Pauses mode finds pauses with the Silero voice detector, which
`dictate download` fetches (2.3 MB). Cancel can't un-type pieces already
typed. Trade-off: each piece is transcribed with only 1.5 s of the audio
before it, so a word right after a long pause is occasionally misheard.

Live mode needs its own model and a fast machine:

```sh
dictate download --live   # about 900 MB, NVIDIA Nemotron streaming 0.6B (English)
```

At startup the daemon benchmarks the model (3 chunks): it stays in Live only
when the median chunk takes under 392 ms per 560 ms of audio. Otherwise it
logs and notifies (`dictate: this computer is too slow for Live mode`) and
uses whole mode for that session. Without the model it does the same
(`dictate: Live model missing`).

## Settings file

Settings live in `config.toml` in the platform config directory:

| OS | Path |
|---|---|
| Linux | `~/.config/dictate/config.toml` |
| macOS | `~/Library/Application Support/dictate/config.toml` |
| Windows | `%APPDATA%\dictate\config\config.toml` |

All keys are optional; a missing file means defaults:

```toml
sounds = true                 # short tone when recording starts/stops
notifications = true          # desktop notifications for errors
mode = "whole"                # dictation mode: "whole", "pauses" or "live"
hotkey = "Ctrl+Shift+Space"   # global shortcut: tap to toggle, hold to talk
```

The environment wins: `DICTATE_SOUNDS=0`, `DICTATE_NOTIFY=0`,
`DICTATE_MODE=...` and `DICTATE_HOTKEY=...` override the file. The old
`streaming` key and `DICTATE_STREAMING` still work (they map to whole/pauses).
The tray menu edits the file for you; `Open settings file` creates it with
defaults when missing.

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
CC-BY-4.0, using the int8 ONNX export by istupakov. Pause segmentation (pauses
mode) uses Silero VAD v6, released under MIT; `dictate download` fetches it
too (2.3 MB, next to the speech model). Live mode uses NVIDIA Nemotron
streaming 0.6B (English), int8 ONNX, fetched by `dictate download --live`
(about 900 MB, next to the speech model).

The model lives in the platform data directory
(`~/.local/share/dictate/models/parakeet-tdt-0.6b-v2-int8` on Linux). Set
`DICTATE_MODEL_DIR` to keep it elsewhere; `download` and every command that
loads the model honor it.

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

## Troubleshooting

Streaming diagnosis: set `DICTATE_DEBUG_SEGMENTS=<dir>` to save every emitted segment (`<recording>-<N>.wav` plus its `<recording>-<N>-ctx.wav` context prefix) and the whole recording (`<recording>-full.wav`) as 16 kHz WAVs.

## License

Code: MIT (see LICENSE). Speech model: NVIDIA Parakeet TDT 0.6B v2,
CC-BY-4.0, downloaded at runtime by `dictate download` — not bundled. Live
model: NVIDIA Nemotron streaming 0.6B, downloaded at runtime by
`dictate download --live` — not bundled.
