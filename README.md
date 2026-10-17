# dictate

Superwhisper-like voice typing in pure Rust: press a hotkey, speak, release,
and the text is typed into the focused input. No Tauri, webview or Electron.

Round 1 covers the foundation: download the speech model and transcribe a WAV
file from the command line.

## Install

On Linux or macOS, the install script fetches the latest release, checks its
SHA256 and links `dictate` into `~/.local/bin`:

```sh
curl -fsSL https://raw.githubusercontent.com/saostad/dictate/main/install.sh | sh
```

Then run `dictate download` and `dictate daemon` (steps 2 and 3 below).

Or download the archive for your OS from
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

The binary is not notarized by Apple, so after a browser download macOS
refuses to run it (or kills it silently). The install script avoids this
(curl downloads aren't quarantined). If you downloaded the archive in a
browser, remove the quarantine flag from the unpacked folder:

```sh
xattr -dr com.apple.quarantine ./dictate-macos-aarch64
```

Grant Microphone and Accessibility to the terminal (or app) that runs
`dictate daemon` (System Settings → Privacy & Security).

To run dictate as a regular menu bar app instead (its own "Dictate" entries
in Privacy & Security, no terminal), build it from source:

```sh
scripts/macos-app.sh   # builds Dictate.app and installs it to /Applications
open /Applications/Dictate.app
```

The script signs the app with a self-signed certificate it creates in its own
keychain (`dictate-signing.keychain-db`), so no Apple Developer account is
needed and macOS keeps the permissions across rebuilds. Add Dictate under
System Settings → General → Login Items to start it at login.

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
dictate mics
```

## Use it

Start the daemon once (it keeps the model loaded), then trigger recordings
with the control commands:

```sh
dictate daemon
dictate toggle   # start/stop recording
dictate status   # idle, recording or transcribing
dictate settings # open the Settings window
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
amber while transcribing. Its menu shows the live state (with the microphone
and mode in the tooltip), starts/stops/cancels dictation, opens the Settings
window, and quits the daemon.
Without a tray host the daemon logs `dictate: no tray icon` once and keeps
running headless.

## Settings window

Pick `Settings…` in the tray menu, or run `dictate settings`. One page with
everything, applied immediately:

- **Microphone**: pick the microphone (or the system default) and see which
  one the last recording used.
- **Dictation mode**: Whole recording, Type at pauses, or Live — with the
  loading, download and speed-check progress inline.
- **Shortcut**: pick the global hotkey (on Wayland it explains the
  compositor keybinds instead).
- **General**: start at login, sound cues, notifications.
- **About**: version, open the log, check for updates.

## Start at login

```sh
dictate autostart on      # start dictate when you log in
dictate autostart status  # show whether it is on
dictate autostart off     # turn it off
```

If you move the dictate binary, the daemon updates the login entry itself
on its next start. The Settings window has a checkbox for this too
(General → Start at login).

## Modes

| Mode | Setting | What happens | Accuracy | Speed |
|---|---|---|---|---|
| Whole (default) | `mode = "whole"` | types once at stop | highest: hears everything at once | slowest after release on long dictations |
| Pauses | `mode = "pauses"` | types each pause segment while speaking | high: 1.5 s of left context per piece | most text is already typed at stop |
| Live | `mode = "live"` | types while speaking, chunk by chunk | good: true streaming model | fastest: text appears as you talk |

Set the mode in the Settings window (it switches live, loading
the needed model on a background thread; picking Live downloads its model
first when missing), or with `DICTATE_MODE=live`.

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

## Advanced: where settings are stored

The Settings window manages settings for you (sound cues, notifications,
mode, microphone, shortcut). They live in `config.toml` in the platform config
directory (`~/.config/dictate/config.toml` on Linux,
`~/Library/Application Support/dictate/config.toml` on macOS,
`%APPDATA%\dictate\config\config.toml` on Windows).

The environment wins: `DICTATE_SOUNDS=0`, `DICTATE_NOTIFY=0`,
`DICTATE_MODE=...`, `DICTATE_HOTKEY=...` and `DICTATE_MICROPHONE=...` override
the file. The old `DICTATE_STREAMING` variable still works (it maps to
whole/pauses).

## Global hotkey

Outside a Wayland session, `dictate daemon` also registers a global hotkey:
**Ctrl+Shift+Space** by default, or whatever `DICTATE_HOTKEY` holds:

```sh
DICTATE_HOTKEY=Alt+F9 dictate daemon
```

On macOS the modifiers are named `Cmd`, `Option`, `Ctrl` and `Shift`:

```sh
DICTATE_HOTKEY=Cmd+Option+Space dictate daemon
```

You can also pick a preset or capture your own with the `Record shortcut…`
button in the Settings window (on macOS you can also press just modifier
keys, like Command+Option).

Shortcuts with only modifier keys (at least two modifiers, e.g. `Cmd+Option`)
work on macOS: pressing another key while holding them cancels that recording.
They need Input Monitoring permission for Dictate (or the terminal that runs
it) in System Settings → Privacy & Security. They are not available on
Windows/Linux; add a key there, for example Ctrl+Alt+Space.

- Tap (press and release quickly): toggles recording on and off.
- Hold (keep it pressed): records while held and transcribes on release
  (push-to-talk).

On Wayland, apps cannot register global hotkeys, so the daemon skips this and
prints a note; bind `dictate toggle` (or `start`/`stop`) in your compositor
instead (see above).

On macOS, grant Accessibility (for typing) and Microphone to the terminal or
app that runs `dictate daemon`. Modifier-only shortcuts (like Command+Option)
also need Input Monitoring for the same terminal or app.

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
