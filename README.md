# dictate

Superwhisper-like voice typing in pure Rust: press a hotkey, speak, release,
and the text is typed into the focused input. No Tauri, webview or Electron.

Round 1 covers the foundation: download the speech model and transcribe a WAV
file from the command line.

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
