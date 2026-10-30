# dictate

Superwhisper-like voice typing in Rust: press a hotkey, speak, release,
and the text is typed into the focused input. No Tauri, webview or Electron.

Choose from five local speech models: **Parakeet TDT 0.6B v2**, **Fluid
Speech Mini**, **Whisper Tiny English**, **Nemotron** for Live mode, and
**Moonshine Tiny Streaming** for phrase-by-phrase dictation (Linux only).
Settings explains which to choose for speed, accuracy, or a weaker computer.
After the models are downloaded, transcription runs entirely on your machine;
audio and text are not sent to a remote inference service.

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
| Linux | x86_64 | `dictate-linux-x86_64.tar.gz` (includes ONNX Runtime; Parakeet/Fluid/Nemotron need AVX2, see [CPUs without AVX2](#cpus-without-avx2--system-onnx-runtime)) |
| Linux | aarch64 | `dictate-linux-aarch64.tar.gz` |
| macOS | Apple Silicon | `dictate-macos-aarch64.tar.gz` |
| Windows | x86_64 | `dictate-windows-x86_64-setup.exe` (installer) |

The macOS release targets Apple Silicon; there is no Intel macOS archive in
the release build matrix.

The Linux binaries need glibc 2.39+ (Ubuntu 24.04+, Fedora 40+, Arch).

On Windows, run the installer. It downloads the speech model (~660 MB),
adds Dictate to the Start menu, optionally starts it when you sign in, and
starts it in the system tray. Uninstall it from Settings → Apps.

On Linux and macOS:

1. Unpack the archive:

   ```sh
   tar -xzf dictate-linux-x86_64.tar.gz
   cd dictate-linux-x86_64
   ```

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

   To start with a small engine instead, skip the default download and run
   one of these pairs of commands:

   ```sh
   ./dictate download --whisper     # about 42 MB, Whole/Pauses
   DICTATE_SPEECH_MODEL=whisper ./dictate daemon
   ```

   ```sh
   ./dictate download --moonshine   # about 62 MB, Linux x86_64/arm64 only
   DICTATE_MODE=live DICTATE_SPEECH_MODEL=moonshine ./dictate daemon
   ```

   Run only one daemon at a time. The environment variable selects the small
   engine for that launch; see [stored settings](#advanced-where-settings-are-stored)
   to save your selection for future launches.

The Windows installer downloads only the default speech model. Run
`dictate download --whisper` later if you want the English Whisper model.

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

The installer is not code-signed, so SmartScreen may warn about it: click
"More info → Run anyway". Socket control commands (`dictate toggle`,
`start`/`stop`) are not available on Windows yet — dictate with the
global hotkey (Ctrl+Shift+Space by default).

### Linux notes

On Wayland, Dictate uses the local Global Shortcuts desktop portal. On
Hyprland, open Settings → Shortcut → **Enable shortcut / apply again** to
bind the default **Ctrl+Shift+Space**, or choose a preset. Other desktops
confirm the binding in their shortcut dialog. Manual compositor bindings
remain available; see [Use it](#use-it).

The Linux x86_64 release includes ONNX Runtime, so a separate install is not
needed. Its Parakeet, Fluid, and Nemotron paths require an AVX2-capable CPU.
Whisper has a separate CPU baseline; see
[CPUs without AVX2](#cpus-without-avx2--system-onnx-runtime).

## Build

The build compiles whisper.cpp through `whisper-rs`, so it needs **CMake** and
a C++ compiler (C++17) in addition to the Rust toolchain. By default
`whisper-rs-sys` generates its bindings with bindgen, which needs libclang
(install clang, or LLVM on Windows). On 64-bit Linux and macOS you can skip
libclang by exporting `WHISPER_DONT_GENERATE_BINDINGS=1` to use the crate's
checked-in bindings; do not do this on Windows, where those bindings do not
match the MSVC ABI. `.cargo/config.toml` builds ggml for the x86_64 baseline (AVX/AVX2/FMA/F16C off, no
`-march=native`); override the `GGML_*`/`WHISPER_*` variables in your shell
when targeting a different CPU class.

On Windows, Whisper runs on the GPU through Vulkan, so the build also needs
the [Vulkan SDK](https://vulkan.lunarg.com/) (`VULKAN_SDK` set by its
installer). From PowerShell:

```powershell
winget install -e --id LLVM.LLVM
winget install -e --id Kitware.CMake
winget install -e --id KhronosGroup.VulkanSDK
[Environment]::SetEnvironmentVariable('LIBCLANG_PATH', 'C:\Program Files\LLVM\bin', 'User')
```

Then open a new terminal so the new variables apply.

```sh
cargo build --release
```

Linux builds also need ALSA and libxkbcommon development packages plus
`pkg-config`. On Ubuntu, install the build dependencies with:

```sh
sudo apt-get install build-essential cmake pkg-config libasound2-dev libxkbcommon-dev
```

## Usage

```sh
dictate download                  # default NVIDIA Parakeet model
dictate download --fluid          # Parakeet + optional Fluid Speech Mini
dictate download --live           # Parakeet + optional Nemotron streaming model
dictate download --whisper        # optional Whisper Tiny English
dictate download --moonshine      # optional Moonshine streaming runtime/model
dictate transcribe x.wav
dictate transcribe --fluid x.wav
dictate transcribe --whisper x.wav
dictate transcribe --live x.wav
dictate transcribe --moonshine x.wav
dictate transcribe --segmented x.wav  # configured Whole/Pauses model, with pause segmentation
dictate listen
dictate mics
```

`download --whisper` fetches Whisper and the pause detector without Parakeet;
`download --moonshine` fetches its own model and runtime without Parakeet.
The `--fluid` and `--live` download commands also fetch the default Parakeet
model and pause detector if missing. Downloads need a network connection;
transcription uses the local files.

## Use it

Start the daemon once (it keeps the model loaded). On Linux and macOS, use
the local control commands to trigger recordings; on Windows, use the tray
or global hotkey:

```sh
dictate daemon
dictate toggle   # start/stop recording
dictate status   # idle, recording or transcribing
dictate settings # open the Settings window
```

On Linux and macOS, control commands other than `status` start the daemon
automatically if needed (its output goes to the log below) and wait up to
30 s for it.

The daemon plays a short tone when recording starts and stops; set `DICTATE_SOUNDS=0` to silence it.

Failures (and empty recordings) also pop a desktop notification; set `DICTATE_NOTIFY=0` to disable them.

When the daemon's stderr is not a terminal (autostart), it also writes every
line to `daemon.log` in the platform data directory:
`~/.local/share/dictate/daemon.log` on Linux,
`~/Library/Application Support/dictate/daemon.log` on macOS,
`%APPDATA%\dictate\data\daemon.log` on Windows.

Settings can configure a portal shortcut directly. On Hyprland, it checks
for conflicts, saves a backup, updates only its marked config block, reloads
the compositor, and verifies the binding. Failed changes restore the previous
config. Omarchy Lua configs (`hyprland.lua` loading `hypr.bindings`) and
legacy `hyprland.conf` configs are supported. The action receives both press
and release events, so tap-to-toggle and hold-to-talk work without spawning
a process on each key event.

For manual setup, Hyprland Lua uses:

```lua
hl.bind("CTRL + SHIFT + space", hl.dsp.global("io.github.saostad.Dictate:dictation"))
```

Keep the daemon running for the portal action to exist. Settings reports
the desktop connection separately from the actual key binding.

Alternatively, bind the control commands yourself. Legacy Hyprland
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
and mode in the tooltip), starts/stops/cancels dictation, switches dictation
mode, speech model and inference threads, opens the Settings window, and
quits the daemon.
Without a tray host the daemon logs `dictate: no tray icon` once and keeps
running headless.

## Settings window

Pick `Settings…` in the tray menu, or run `dictate settings` on Linux/macOS.
Most changes apply immediately; inference-thread changes take effect after
restarting the daemon:

- **Microphone**: pick the microphone (or the system default) and see which
  one the last recording used.
- **Speech model**: Parakeet TDT 0.6B v2, Fluid Speech Mini, Whisper Tiny
  English, Nemotron Streaming, or Moonshine Tiny Streaming. Each choice has a
  short explanation of its speed, accuracy, and memory trade-offs.
- **Dictation mode**: Whole recording, Type at pauses, or Live. Unsupported
  modes are disabled for the selected model. Choosing a different model
  selects a compatible mode automatically when needed.
- **Inference threads**: Auto, 1, or 2 threads for Whisper and Moonshine.
  Auto uses up to two threads for Whisper and single-thread for Moonshine.
  Moonshine's explicit 2 means its default threading, not exactly two.
  Restart the daemon after changing this setting.
- **Shortcut**: choose a preset or record your own keys, including on Wayland.
  Hyprland setup checks conflicts and backs up the user config; other
  Wayland desktops confirm the actual binding through their portal.
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

| Mode | Setting | Compatible models | What happens |
|---|---|---|---|
| Whole recording (default) | `mode = "whole"` | Parakeet, Fluid, Whisper | types once at stop; useful for accuracy and slower computers |
| Type at pauses | `mode = "pauses"` | Parakeet, Fluid, Whisper | types completed pause segments; Whisper's extra context decoding adds delay |
| Live | `mode = "live"` | Nemotron, Moonshine | types while speaking; Nemotron commits text sooner in our test, Moonshine commits completed phrases with less memory |

Choose the **speech model** first in Settings or the tray, then choose a
supported dictation mode. All five models are in the same model list; model
names do not appear as dictation modes. Switching between compatible Whole
and Pauses modes reuses the loaded model. Switching engines loads on a
background thread, downloading missing files first. Changes are refused
while recording, transcribing, or loading another model.

Selecting Nemotron or Moonshine switches to Live. Selecting Parakeet, Fluid,
or Whisper from Live switches to Whole recording; switching between those
three batch models preserves Whole or Pauses. A failed download, model load,
or Nemotron speed check keeps the previous complete selection.

Pauses mode finds pauses with the Silero voice detector, which
`dictate download` fetches (2.3 MB). Cancel can't un-type pieces already
typed. Parakeet, Fluid and Whisper decode each piece after the first with up
to 1.5 s of the audio before it, so a word at a cut is occasionally misheard.
Whisper removes the repeated context with whisper.cpp's token timestamps.
Decoding a natural-pause piece alone is cheaper but changed real words on the
reference clip, so that shortcut is not used. The first piece has no context
and decodes alone.

Live with Nemotron needs a fast machine and its own model:

```sh
dictate download --live   # about 900 MB, NVIDIA Nemotron streaming 0.6B (English)
```

At startup the daemon benchmarks Nemotron (3 chunks): it stays in Live only
when the median chunk takes at most 392 ms per 560 ms of audio. Otherwise it
logs and notifies (`dictate: this computer is too slow for Live mode`) and
tries Whole mode for that session. Without the model it does the same
(`dictate: Live model missing`). This fallback needs the configured
Whole/Pauses model to be downloaded too.

### Moonshine for Live dictation

```sh
dictate download --moonshine   # about 62 MB of model + native runtime
```

Moonshine Tiny Streaming is a 34M-parameter English model with a true
stateful stream: audio is fed while you speak, the model stays resident, and
only lines the model marks complete are typed. A still-changing line is
logged as `moonshine partial:` and is never sent to the focused application.
Stopping drains the last audio and types the final phrase; cancel does not
type it. Moonshine and Nemotron are separate model choices within Live
mode: use `mode = "live"` and `model = "moonshine"` for Moonshine.

The pinned native runtime is downloaded as a Linux x86_64 or arm64 release
archive, checked by size and SHA-256, and stored in
`moonshine-runtime-0.1.5` next to the models. `libmoonshine.so` loads the
`libonnxruntime.so.1` beside it. On macOS and Windows the upstream release has
no usable dynamic runtime yet, so `download --moonshine` reports that clearly
and the other models remain available. Set `DICTATE_MOONSHINE_RUNTIME_DIR` to point
at an existing runtime directory.

Moonshine's **Auto** setting (the default) and explicit **1** use its faster
single-thread mode. Explicit **2** uses Moonshine's own default threading,
which is not exactly two threads and is logged as such. Whisper honors the
exact 1/2 setting. Changing the setting takes effect after the daemon
restarts.

## Speech model

Pick one of five speech models in Settings or the tray under **Speech
model**. The compatible modes and trade-offs are:

| Speech model | Modes | Download | When to choose it |
|---|---|---|---|
| Parakeet TDT 0.6B v2 (default) | Whole / Pauses | about 660 MB | fastest transcription and best measured accuracy; higher memory use and an AVX2 CPU required with the bundled runtime |
| Fluid Speech Mini | Whole / Pauses | about 300 MB extra | good measured accuracy with less RAM than Parakeet; slower in our test |
| Whisper Tiny English | Whole / Pauses | about 42 MB | older CPUs or low RAM; fastest loading, with lower accuracy; Whole avoids its Pauses delay |
| Nemotron Streaming | Live | about 900 MB extra | accurate streaming text on a fast computer with more RAM |
| Moonshine Tiny Streaming | Live | about 62 MB including runtime | low-memory streaming and short finalization delay, with lower measured accuracy; Linux only |

Settings gives a quick choice guide based on our [local CPU benchmark](docs/benchmarks/2026-10-22/README.md):

- **Parakeet TDT 0.6B v2**: fastest transcription and best measured accuracy;
  choose it when you have enough RAM and an AVX2-capable CPU.
- **Fluid Speech Mini**: a balance of good accuracy and lower memory use than
  Parakeet, with slower transcription in our test.
- **Whisper Tiny English**: for older CPUs without AVX2 or a small memory
  budget; loads fastest, but makes more recognition errors. Prefer Whole
  recording because its Pauses context decoding adds substantial delay.
- **Nemotron Streaming**: accurate text as you speak on a fast computer with
  more available RAM; committed words appeared earlier than with Moonshine.
- **Moonshine Tiny Streaming**: streaming dictation with low memory use and less delay
  after stopping, at the cost of lower measured accuracy. Linux only.

These rankings come from a small read-speech sample on an i7-6700HQ. Actual
speed and accuracy depend on your computer and voice; they are not universal
accuracy or minimum-memory guarantees.

`dictate transcribe --fluid x.wav` (or `--whisper`) transcribes one file with
that model without changing the setting. Whisper uses the Q8_0 Tiny English
weights through whisper.cpp 1.8.3 and `whisper-rs` 0.16.0, with explicit
English, greedy decoding, no previous-text conditioning, and bounded
segments. In Pauses mode the first piece decodes alone; every later piece
decodes up to 1.5 s of left context plus the new audio, and whisper.cpp's
real token timestamps remove the repeated context. No word timestamps are
invented. The context decode is the main cost of Whisper Pauses and is not
skipped after pauses; a segment-only shortcut was measured and rejected
because it changed words on the reference clip. Measured timings and limits
are in
[docs/stt-implementation-2026-10-22.md](docs/stt-implementation-2026-10-22.md).

To compare all five engines on the same local audio, see the reproducible
[model benchmark guide](docs/model-benchmarks.md). It measures transcription
errors, speed, memory, and streaming latency, and saves the raw transcripts.
The [2026-10-22 comparison on an i7-6700HQ](docs/benchmarks/2026-10-22/README.md)
includes all five engines and 14 thread configurations.

## Advanced: where settings are stored

The Settings window manages settings for you (sound cues, notifications,
mode, speech model, inference threads, free model memory, microphone,
shortcut). They live in
`config.toml` in the platform config directory:

- Linux: `~/.config/dictate/config.toml`
- macOS: `~/Library/Application Support/dictate/config.toml`
- Windows: `%APPDATA%\dictate\config\config.toml`

Saved `mode` values describe behavior: `"whole"`, `"pauses"`, or `"live"`.
Saved `model` values select the engine: `"standard"` (Parakeet; retained for
compatibility), `"fluid"`, `"whisper"`, `"nemotron"`, or `"moonshine"`.
For example:

```toml
mode = "whole"
model = "whisper"
inference_threads = "auto"
```

For low-memory streaming:

```toml
mode = "live"
model = "moonshine"
inference_threads = "auto"
```

The speech model loads when the daemon starts and is freed after 20 idle
minutes (about 700 MB for Parakeet), so dictate uses almost no memory while
you are not dictating. Change this under **Free model memory** in the tray
icon's menu or the Settings window, or set `unload_after` to `"never"`,
`"5"`, `"20"`, `"30"` or `"60"` idle minutes (default `"20"`). The next
dictation reloads the model in a few seconds; your speech is recorded
meanwhile and typed once it has loaded.

Nemotron uses `mode = "live"` and `model = "nemotron"`. Streaming selections
also retain `batch_model` (standard, fluid, or whisper), the last Whole/Pauses
model used for startup fallback and the batch-only `listen` / unflagged
`transcribe` commands. Use `transcribe --live` or `--moonshine` to explicitly
transcribe a file with a streaming model.

Existing `mode = "moonshine"` files are read as Live + Moonshine. Legacy
`mode = "live"` files that stored a batch model are read as Live + Nemotron.
The daemon saves the new format with a timestamped backup beside
`config.toml`; the previous batch-model preference is preserved. Existing
Whole/Pauses selections already have a compatible format.

The environment wins: `DICTATE_SOUNDS=0`, `DICTATE_NOTIFY=0`,
`DICTATE_MODE=...`, `DICTATE_SPEECH_MODEL=...`,
`DICTATE_INFERENCE_THREADS=...`, `DICTATE_HOTKEY=...` and
`DICTATE_MICROPHONE=...` override the file. The old `DICTATE_STREAMING`
variable still works (it maps to whole/pauses). Model overrides also select
a compatible mode; the legacy `DICTATE_MODE=moonshine` spelling remains
accepted. Canonical streaming examples use `DICTATE_MODE=live` together with
`DICTATE_SPEECH_MODEL=nemotron` or `moonshine`.

## Global hotkey

`dictate daemon` registers a global hotkey, using the local desktop portal
on Wayland: **Ctrl+Shift+Space** by default, or whatever `DICTATE_HOTKEY` holds:

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

Wayland requires a desktop that implements the Global Shortcuts portal.
On Hyprland, enable the compositor binding once in Settings; choosing another
preset applies it immediately. The app never replaces a conflicting shortcut.
An unavailable portal is reported in Settings, and the daemon retries after
portal restarts. On other Wayland desktops the chosen preset is a preference:
the desktop controls the actual binding shown in Settings. If the portal is
unavailable, bind `dictate toggle` or `dictate start`/`dictate stop` manually.

On Linux, `dictate shortcut Ctrl+Shift+Space` requests the same setup as the
Settings picker. The command acknowledges the request; Settings and the
daemon log report whether applying it succeeded. If the daemon is not running,
the command starts it in the same way as the other socket control commands.

On macOS, grant Accessibility (for typing) and Microphone to the terminal or
app that runs `dictate daemon`. Modifier-only shortcuts (like Command+Option)
also need Input Monitoring for the same terminal or app.

## Model

Speech recognition uses NVIDIA Parakeet TDT 0.6B v2 (English), released under
CC-BY-4.0, using the int8 ONNX export by istupakov. The optional Fluid Speech
Mini model is by FluidVoice, released under CC BY 4.0, built on NVIDIA
Parakeet TDT 0.6B v3 and Moondream Parakeet Ultra; `dictate download --fluid`
fetches its ONNX export (about 300 MB). Pause segmentation (pauses mode) uses
Silero VAD v6, released under MIT; `dictate download` fetches it too (2.3 MB,
next to the speech model). Live can use NVIDIA Nemotron streaming 0.6B
(English), int8 ONNX, fetched by `dictate download --live` (about 900 MB, next
to the speech model).

The optional Whisper model is `ggml-tiny.en-q8_0.bin` (Q8_0, about 42 MB)
from the official whisper.cpp model repository, pinned to revision
`5359861c739e955e79d9a303bcbc70fb988958b1` and verified by exact size and
SHA-256. Whisper.cpp is MIT; the OpenAI Whisper weights are MIT. The
Moonshine model is the quantized `tiny-streaming-en` set from
`download.moonshine.ai`, and the runtime is the pinned Moonshine v0.1.5 Linux
release; both are MIT. Exact URLs, sizes and hashes are in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

The models live in the platform data directory
(`~/.local/share/dictate/models/parakeet-tdt-0.6b-v2-int8` and its siblings
such as `whisper-tiny-en`, `moonshine-tiny-streaming-en` and
`moonshine-runtime-0.1.5` on Linux). Set `DICTATE_MODEL_DIR` to keep them
elsewhere; `download` and every command that loads a model honor it. The
override names the Parakeet directory itself; the other engines are siblings
next to it. `DICTATE_MOONSHINE_RUNTIME_DIR` overrides only the runtime
directory.

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

Whisper Tiny English is built independently of that ONNX Runtime for the
generic x86_64 baseline: the build disables AVX, AVX2, AVX-512, FMA, F16C and
SSE4.2, so Whisper itself does not require those instruction sets. That makes
it the intended small engine on older x86_64 CPUs. The baseline build was
exercised on an AVX2-capable i7-6700HQ, not on a no-AVX2 machine. Moonshine
uses its own prebuilt ONNX Runtime; its CPU requirements come from that
upstream runtime and Dictate has not validated it on Sandy Bridge-era
hardware. The prebuilt Moonshine runtime is currently Linux-only; Whisper
builds on Linux, macOS and Windows. Whisper Whole mode is the simpler choice
on a slow CPU; Pauses mode adds a left-context decode per segment and
therefore costs more. Exact speed depends on the CPU, threads and clip
length, so Dictate makes no model-wide real-time guarantee. Use
`dictate transcribe --whisper x.wav` to measure a specific machine.

## Troubleshooting

If a model is missing, use the matching download command, such as
`dictate download --whisper` or `dictate download --moonshine`. A failed model
switch in a running daemon keeps its previous working selection. At startup,
Moonshine failures try the configured Whole/Pauses model instead; if that
fallback is also missing, startup fails until a usable model is downloaded.
A Whisper model with the wrong size is deleted and fetched again on the next
download.

If a Wayland shortcut is unavailable, check its status in Settings. Hyprland
requires both a running portal session and a compositor binding; choose a
preset or **Enable shortcut / apply again** to set it up. Conflicting bindings
are kept, and the previous config is backed up next to the user bindings file.
The portal lifecycle and desktop verification are documented in
[the Wayland shortcut notes](docs/wayland-shortcuts-2026-10-22.md).

Open Settings after unlocking the desktop. On the tested Hyprland system,
opening its renderer while locked could stall the daemon event loop; the tray
daemon runs normally with Settings closed.

Streaming diagnosis for `dictate daemon`: set `DICTATE_DEBUG_SEGMENTS=<dir>` to save every emitted segment (`<recording>-<N>.wav` plus its `<recording>-<N>-ctx.wav` context prefix) and the whole recording (`<recording>-full.wav`) as 16 kHz WAVs.

## License

Code: MIT (see LICENSE). Speech model: NVIDIA Parakeet TDT 0.6B v2,
CC-BY-4.0, downloaded at runtime by `dictate download` — not bundled. Fluid
Speech Mini by FluidVoice, CC BY 4.0, built on NVIDIA Parakeet TDT 0.6B v3
and Moondream Parakeet Ultra, downloaded at runtime by `dictate download
--fluid` — not bundled. Nemotron Streaming: NVIDIA Nemotron streaming 0.6B,
downloaded at runtime by `dictate download --live` — not bundled. Whisper
Tiny English (`ggml-tiny.en-q8_0.bin`) and whisper.cpp are MIT,
downloaded/built at runtime or build time — the model is not bundled. Moonshine Tiny Streaming
weights and the Moonshine v0.1.5 runtime are MIT, downloaded at runtime by
`dictate download --moonshine` — not bundled. See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for exact pins and notices.
