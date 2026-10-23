# Small local engines: Whisper Tiny English and Moonshine Tiny Streaming

Date: 2026-10-22. Scope: the native, local-only CPU engines added to Dictate's
Whole/Pauses path (Whisper) and streaming mode (Moonshine). Nothing in this
work sends audio or text to a remote service. This note records what was
built, how it was verified, and what remains unverified.

The engine verification used the `main` checkout in
`/home/saeid/projects/dictate`. The verification did not change user
configuration or system services.

## Test machine

- CPU: Intel Core i7-6700HQ, 4 physical cores / 8 logical CPUs, AVX2 capable.
  This is **not** the Sandy Bridge-class machine from the earlier research.
- OS: Omarchy 4.0.6, kernel 7.2.8-9-omarchy, x86_64.
- Toolchain: Rust 1.99.1, Cargo 1.99.1.
- Verification models: `/home/saeid/.cache/dictate-verification-2026-10-22/models`
  (`DICTATE_MODEL_DIR` points at its `parakeet-tdt-0.6b-v2-int8` child; the
  other engines are siblings).
- Fixture: the repository's public-domain 11 s JFK clip
  (`tests/fixtures/jfk.wav`), plus a 48 kHz copy, a 3 s excerpt, a 1.5 s
  silence clip, and a clip with a deliberate repeated phrase and a 1.2 s gap.

Whisper is compiled for the generic x86_64 baseline: `GGML_NATIVE=OFF`,
`GGML_AVX=OFF`, `GGML_AVX2=OFF`, `GGML_AVX512=OFF`, `GGML_FMA=OFF`,
`GGML_F16C=OFF`, `GGML_SSE42=OFF`. Moonshine uses the pinned upstream runtime
and its ONNX Runtime. No runtime instruction-set claim is made beyond the
machine above.

## Whisper Tiny English

- Model: `ggml-tiny.en-q8_0.bin`, Q8_0, 43,550,795 bytes, pinned to
  the `ggerganov/whisper.cpp` model repository revision
  `5359861c739e955e79d9a303bcbc70fb988958b1`. Exact URL, size, SHA-256 and
  license are in [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).
- Runtime: whisper.cpp 1.8.3 through `whisper-rs` 0.16.0 /
  `whisper-rs-sys` 0.15.0, statically linked.
- Default model for Whole and Pauses when `model = "whisper"`; the old f16
  file is not a fallback. Missing or wrong-size files tell the user to run
  `dictate download --whisper`.
- Whole mode decodes the completed recording. Pauses mode decodes each VAD
  segment. **The first segment has no context and decodes alone. Every later
  segment decodes up to 1.5 s of preceding audio plus the new segment.** With
  token timestamps enabled, whole BPE words are kept only when their midpoint
  is at/after the context boundary, so repeated context is not typed.
- The context path is unconditional when context exists, including after a
  natural pause. A segment-only decode is used only by the first segment or
  by the timing fallback; the fallback discards the context decode's words
  and is reported as disjoint so the joiner does not drop an intentional
  repeated word.
- If a segment's timing is unusable, it is re-decoded alone rather than
  trusting overlapping text. Invalid timestamps are rejected, never clamped
  or invented.
- BPE fragments are grouped into words before trimming and decoded once per
  word as raw UTF-8 bytes, so a multi-byte character split across tokens
  survives. Punctuation attached to a word is preserved.
- Segment-level `no_speech_probability > 0.6` is discarded. Exact bracketed
  silence/music/noise markers are dropped; ordinary bracketed words are not.
  ggml INFO chatter is filtered, warnings/errors pass.
- Intentional repeated speech is preserved. Only the existing narrow 1–2 word
  overlap dedupe remains; there is no global immediate-repeat collapse.
- Q8_0 is the only shipped Whisper weight. It is independent of the bundled
  ONNX Runtime, so the build works on x86_64 CPUs without AVX2. The **runtime**
  was exercised only on the AVX2-capable i7-6700HQ above.

### Rejected speedup

Decoding natural-pause pieces alone and shortening the encoder window was
tested as a possible Pauses speedup. It changed words on the reference clip:

| Piece decode | Result on the JFK clip |
|---|---|
| 512 encoder positions | `... Americans. S not! ...` (`Ask not` lost) |
| 768 positions | `And so am I fellow Americans. ...` (`my` changed) |
| 1024 positions | `And so am I fellow Americans. ...` |
| full 1500 positions | `And so am I fellow Americans. ...` |

The accepted context path instead produces the full sentence, but each later
piece costs a full-window context decode. On this CPU that is about **6.5 s
per piece**, so Whisper Pauses took **27.4–27.6 s** for the 11 s clip. This is
the known cost of keeping the words; no real-time or percentile claim is made.

## Moonshine Tiny Streaming

- Model: `tiny-streaming-en`, `quantized_26_08_21` export from
  `download.moonshine.ai`; every file pinned by size and SHA-256.
- Runtime: Moonshine v0.1.5 Linux x86_64/arm64 release, including ONNX Runtime
  1.23.2. Pins, sizes, hashes and licenses are in
  [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).
- Linux only in this build: the upstream macOS/Windows assets are static
  libraries, so `download --moonshine` reports that clearly and other modes
  keep working. Not validated on Sandy Bridge-era hardware.
- One shared `Session` lifecycle is used by the daemon worker, the CLI test
  hook and the fake-backend tests: lazy stream open, feed, one final
  stop/finish, cancel, close. A new recording discards stale stream/cursor/
  failure state before its first chunk.
- Only lines the model marks complete are typed. A changing line is logged as
  `moonshine partial:` and never sent to the focused app. `finish` drains the
  last phrase; cancel discards it. A null-text native line stays an empty
  placeholder so later line indices are preserved.
- A failure is reported once per recording; later chunks are ignored until
  the next start, including the stop-time tail.
- Threads: Auto and explicit 1 use the runtime's documented single-thread
  mode. Explicit 2 uses ONNX Runtime's default threading and is logged as “not
  exactly 2”; the pinned API has no per-session thread count.
- The daemon does not load the Parakeet model when starting in or switching
  to Moonshine.

## CLI, config, tray and daemon

- `dictate download --whisper`, `--moonshine`, `--fluid`, `--live`, or no flag
  for the default model. The new Tiny assets verify pinned size and SHA-256;
  existing engines retain their original download validation.
- `dictate transcribe --whisper file.wav`, `--moonshine file.wav`,
  `--fluid`, `--live`, and `dictate transcribe --segmented file.wav`.
- `model = "whisper"` in config, `DICTATE_SPEECH_MODEL=whisper`, and
  `DICTATE_INFERENCE_THREADS` for Whisper/Moonshine. The Settings window and
  tray expose mode, model and thread choices; choices are disabled while
  recording/transcribing/loading.
- `DICTATE_DEBUG_SEGMENTS=<dir>` remains a **daemon** streaming diagnostic
  (per-segment WAVs, context prefix, full recording). The temporary CLI dump
  and the temporary `DICTATE_AUDIO_CTX` override used during round 3 were
  removed.

## Build and test results

Run from the repository root with the release binary in `target/release`:

```sh
cargo fmt --all -- --check
cargo test
cargo test --no-default-features --features system-ort
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --no-default-features --features system-ort -- -D warnings
cargo build --release
```

Final results:

- `cargo test`: **259 passed, 0 failed, 1 ignored**.
- `cargo test --no-default-features --features system-ort`:
  **266 passed, 0 failed, 1 ignored**.
- `cargo fmt --check`: clean.
- `cargo build --release`: success.
- Both Clippy configurations pass with `-D warnings`.
- The Python benchmark tooling suite passes **26 tests**, including word-error
  alignment, scoring, corpus pins, and report generation.

The new regression tests cover Whisper BPE/UTF-8 word grouping, timing
validation and fallback, context-boundary trimming, intentional repetition,
noise markers, and the Moonshine session lifecycle (start/finish/cancel/
failure/reuse) with a fake backend.

## Real-model results

Commands used `DICTATE_MODEL_DIR` pointing at the verification tree above and
`DICTATE_SPEECH_MODEL=whisper` for the segmented runs. Times are wall clock
for the process unless noted; engine-reported decode times are in the logs.

| Check | Output | Time |
|---|---|---:|
| Whisper whole, 11 s JFK (2 runs) | `And so my fellow Americans ask not what your country can do for you ask what you can do for your country.` | 1.708–1.735 s wall; 1.635–1.669 s decode, 45–50 ms load |
| Whisper whole, 3 s excerpt | `And so my fellow Americans.` | 1.248 s wall; 1.186 s decode |
| Whisper whole, 1.5 s silence | no stdout; `No speech recognized in 1.5 s of audio.` | 1.226 s wall; 1.157 s decode |
| Whisper whole, deliberate repeat | `And so my fellow Americans asked and so my fellow Americans asked` | 2.502 s wall; 2.434 s decode |
| Whisper Pauses, 48 kHz JFK (5 runs) | `And so my fellow Americans Ask not what your country can do for you ask what you can do for your country.` (identical every run) | 27.404–27.594 s wall; segments ~1.18–1.22 s then ~6.47–6.62 s each |
| Whisper Pauses, 1.5 s silence | no stdout; `No speech recognized in 1.5 s of audio.` | 0.131 s wall; no segment decoded |
| Moonshine, 11 s JFK | `And so my fellow America, Ask not. What your country can do for you? Ask what you can do for your country.` | 1.825–1.833 s wall; 93–99 ms load; 110 chunks, 15–16 ms/chunk |
| Moonshine, 3 s excerpt | `And so my fellow America,` | 0.527 s wall; 93 ms load |
| Moonshine, deliberate repeat | `And so my fellow America, Ask. And so my fellow America, Ask.` | 1.238 s wall; 92 chunks, 12 ms/chunk |
| Moonshine, 1.5 s silence | no stdout; `No speech recognized in 1.5 s of audio.` | 0.134 s wall; 91 ms load |
| Moonshine explicit 2, 3 s | same short text; log says default threads, not exactly 2 | 0.769 s wall; 30 chunks, 21 ms/chunk |

Earlier accepted runs measured 1.697 s decode / 43 ms load for Whisper whole
11 s and 1.790 s process / 89 ms load for Moonshine 11 s. Run-to-run variation
of a few percent is expected on this shared desktop; these are not percentile
measurements.

A separate paced run of the shared Rust `Session` (warmed model, audio fed at
real-time rate) produced the first non-empty partial at **1.07 s**, the first
completed phrase at **3.07 s**, and final committed text **108–127 ms after
the 11 s audio ended** in two runs. The fixture ends with trailing silence,
which this finalization includes; it is not a microphone or desktop-typing
latency. One paced run was executed with kernel-level network blocking
(seccomp, no network) and passed. Fresh downloads of all five model sets were
also completed successfully. The Tiny download paths check pinned sizes and
SHA-256; the benchmark inventory hashes the installed files for all engines.

## Known limits

- Absolute timings and the Whisper Pauses cost are for this i7-6700HQ only.
  No no-AVX2 hardware was available for a runtime check. The baseline build
  flags are verified, not a performance or compatibility guarantee.
- Moonshine is Linux-only here and was not run on Sandy Bridge-era hardware.
- No real microphone, tray host, or full daemon recording was exercised in
  this verification. The shared `Session` lifecycle has fake-backend tests;
  the CLI exercises the same code with the real Moonshine model.
- Whisper Pauses is slow because every later piece pays a full-window context
  decode. Whole mode is the cheaper choice on a slow CPU.
- One clip cannot establish accuracy across accents, names, noise, distant
  speech, or other speakers. There is no real-time or latency-percentile
  guarantee.
- The [all-five-model comparison](benchmarks/2026-10-22/README.md) contains
  the measured speed, memory, word errors, streaming latency and raw transcripts.
