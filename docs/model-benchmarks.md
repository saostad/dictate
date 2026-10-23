# Model benchmarks

`dictate benchmark` measures all five ASR engines on one machine:

| Engine flag | Model | Production decode path |
| --- | --- | --- |
| `standard` | Parakeet TDT 0.6B v2 int8 | `Asr::transcribe` / segment decode |
| `fluid` | Fluid Speech Mini int8 | `Asr::transcribe` / segment decode |
| `whisper` | Whisper Tiny English Q8_0 (whisper.cpp) | `Asr::transcribe` / segment decode |
| `live` | Nemotron streaming en 0.6B int8 | `Live::reset` + `transcribe_chunk` + 3 zero flushes |
| `moonshine` | Moonshine Tiny Streaming (native C) | `Moonshine::Session` feed/finish/cancel |

The harness is offline and local. It never downloads a model, never changes
user configuration, never types into a focused window, and never starts the
daemon. The Python orchestrator starts one process per configuration, so a
model is loaded once, stays resident for all measurements, and cannot share
threads or memory with another engine.

The benchmark is intentionally a small, reproducible comparison, not a product
claim. Read [Timing and scope](#timing-and-scope) before quoting numbers.

See the [measured i7-6700HQ comparison from 2026-10-22](benchmarks/2026-10-22/README.md)
for results, recommendations, raw transcripts and statistics for all five engines.

## Prerequisites

- Linux x86_64/arm64 for Moonshine's pinned native runtime. The other engines
  work wherever Dictate supports them, but RSS reporting is Linux-specific.
- The five model folders and `silero-vad-v6` already downloaded under one
  models root:
  - `parakeet-tdt-0.6b-v2-int8`
  - `fluid-speech-mini-onnx`
  - `whisper-tiny-en`
  - `nemotron-streaming-en-0.6b-int8`
  - `moonshine-tiny-streaming-en`
  - `moonshine-runtime-0.1.5`
  - `silero-vad-v6`
- `flac` and/or `ffmpeg` for the one-time corpus preparation.
- A release binary built with `cargo build --release`.

`--model-dir` accepts either the models root or the Parakeet child folder. The
latter is normalized to its parent.

## Prepare the corpus once

The repository contains a portable manifest,
`tests/fixtures/benchmark-corpus.json`, but not the audio. Build the 12 WAVs
and verify every pinned hash:

```bash
python3 scripts/prepare-benchmark-corpus.py \
  --output tests/fixtures \
  --cache ~/.cache/dictate-benchmarks/corpus
```

This is the corpus download step. It fetches the 12 fixed dataset rows
from the Hugging Face datasets-server `rows` endpoint, downloads the fresh
signed FLAC URL, verifies the pinned source FLAC SHA-256, decodes to canonical
16 kHz mono PCM16 WAV, verifies the pinned WAV SHA-256, and writes
`tests/fixtures/audio/`. Signed URLs are never written to the manifest.

If the WAVs and FLAC cache already exist, prepare again with `--offline` to
avoid all network access:

```bash
python3 scripts/prepare-benchmark-corpus.py --offline --output tests/fixtures
```

`flac -d` reproduces the pinned WAV bytes exactly. If `flac` is unavailable,
the script falls back to `ffmpeg` and rewrites the output as a canonical
44-byte-header WAV; a hash mismatch stops the script rather than silently
scoring different audio.

The corpus is 12 LibriSpeech `test-clean` clips, 81.345 seconds total. It is
two clips from each of six speakers (three male, three female) at fixed rows
`0, 2, 250, 254, 501, 502, 750, 751, 1000, 1001, 1502, 1503`, dataset revision
`71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`. See
[Corpus attribution and limits](#corpus-attribution-and-limits).

## Build

`src/main.rs` declares the module and dispatches the command:

```rust
mod benchmark;
```

```rust
"benchmark" => benchmark::run(args),
```

The `args` value is the remaining argument iterator after the command name.
`benchmark::run` takes `impl Iterator<Item = String>`. `dictate help` lists:

```text
dictate benchmark --engine <standard|fluid|whisper|live|moonshine> [options]
```

Then build:

```bash
cargo build --release
```

## Run the matrix

Full run (14 configurations, 5 warmed JFK runs, 3 paced JFK runs, all 12
corpus clips):

```bash
python3 scripts/benchmark-models.py \
  --binary target/release/dictate \
  --corpus tests/fixtures/benchmark-corpus.json \
  --model-dir ~/.local/share/dictate/models \
  --output benchmark-results/2026-10-22
```

The output directory must be empty unless `--force` is given. The default
output is `benchmark-results/<UTC timestamp>`.

Quick smoke run (1 JFK run, 1 paced run, first 2 corpus clips; still exercises
every engine):

```bash
python3 scripts/benchmark-models.py \
  --binary target/release/dictate \
  --corpus tests/fixtures/benchmark-corpus.json \
  --model-dir ~/.local/share/dictate/models \
  --quick \
  --output benchmark-results/quick
```

Useful filters:

```bash
--engines standard,moonshine
--threads 1
--timeout 3600
--no-model-hash
--strict
```

`--strict` exits nonzero if any configuration failed. By default failures are
kept in the report and the command still exits 0 after writing it. A failing
configuration does not stop the remaining ones.

### Optional network sandbox

On Linux with `libseccomp.so.2`, the included wrapper blocks socket creation,
connections, and send syscalls for each inference process and its children:

```bash
python3 scripts/benchmark-models.py \
  ... \
  --sandbox scripts/no-network.py
```

A `.py` wrapper is run with the current Python interpreter, with the benchmark
binary and its arguments appended. Any other executable is invoked directly as
`<wrapper> <binary> <args>`. The wrapper is optional; the harness itself never
contacts the network.

## Configuration matrix

The default matrix is 14 processes:

| Engine | `default` | `1` | `2` | `runtime-default` |
| --- | --- | --- | --- | --- |
| standard | app default | exact 1 | exact 2 | not available |
| fluid | app default | exact 1 | exact 2 | not available |
| whisper | app default | exact 1 | exact 2 | not available |
| live | app default | exact 1 | exact 2 | not available |
| moonshine | maps to single-thread | single-thread | **not available** | ONNX Runtime default |

Thread semantics:

- `standard`, `fluid`, `live`: native ONNX Runtime intra-op threads passed
  through the real constructor. `default` is `asr::intra_threads()` (available
  parallelism clamped to 1..=16).
- `whisper`: `whisper.cpp` `n_threads`. `default` is
  `config::small_model_threads()` (half the logical CPUs, capped at 2). The
  benchmark does not call `init_config`, so a user settings file cannot change
  the measured default.
- `moonshine`: the pinned v0.1.5 C API has no per-session thread-count
  argument. `single-thread` sets `MOONSHINE_ORT_SINGLE_THREAD=1` once before
  the runtime creates its ONNX Runtime sessions. `runtime-default` leaves ONNX
  Runtime's physical-core default in place. That default is usually more than
  two threads; the harness never labels it as an exact two-thread run.

The Python orchestrator runs standard/fluid/whisper/live with
`default`, `1`, `2`, and Moonshine with `1`, `runtime-default`. It does not run
a duplicate Moonshine `default` because that is the same policy as `1`.

## What each process measures

The Rust command writes one JSON document to stdout and exits. The Python
script captures that document plus stderr, and adds WER/CER and report tables.

1. **Preload** — read and validate the portable manifest, check WAV format,
   verify `wav_sha256`, and load every corpus clip and `tests/fixtures/jfk.wav`
   into memory before model load or timers. `preload.ms` is recorded but is not
   part of RTF.
2. **Load** — construct the engine once with the requested native thread
   setting. `model_load_ms` covers only this construction. `first_decode_ms`
   is the compute for the entire warmup JFK recording for every engine, so the
   streaming engines are comparable with Whole/Pauses. Streaming also records
   `first_chunk_ms` (the first 100 ms Moonshine chunk or first 560 ms Live
   chunk), which is a startup diagnostic, not the first recording decode. The
   warmup transcript and total are recorded and excluded from the measured JFK
   statistics.
3. **JFK** — five warmed, unpaced runs by default. Per run: decode time, RTF,
   transcript. The first decode is not counted as a measured run.
4. **Whole vs paused JFK** — for standard/fluid/whisper only, one paired warm
   whole decode vs the production `segment::Feeder` VAD, segment decode and
   `join_piece` path. This is a single trial and a reference comparison, not a
   statistical distribution or a third quality corpus. The paused run loads a
   fresh VAD; `vad_load_ms` is reported separately and is excluded from
   `paused_total_ms`. A failed paused segment marks the engine document
   `partial`; the per-segment errors stay in `whole_vs_paused_jfk`.
5. **Corpus quality** — each of the 12 clips is decoded once, unpaced, after
   warmup. Every raw reference, hypothesis, duration, decode time, RTF and
   failure is retained.
6. **Stress** — 5 s digital silence; 5 s deterministic low-level noise
   (uniform ±0.01); the first 3 s of JFK; and JFK repeated four times with
   0.25 s gaps. All are labeled synthetic. False words on silence/noise are
   reported separately and are **not** included in corpus WER. The duplicated
   JFK case is never included in corpus WER.
7. **Paced JFK** — Live and Moonshine only: three warmed runs by default, audio
   fed at real-time arrival, then the production finalization path. Recorded
   values:
   - `first_chunk_ms` (first chunk decode; startup diagnostic)
   - `first_provisional_ms` (Moonshine only; Live has no provisional text)
   - `first_committed_ms` and `first_committed_audio_ms`
   - `audio_arrival_end_ms`
   - `stop_to_final_ms` from audio end through flush/finalization
   - `finalization_compute_ms`
   - `max_backlog_ms` and `max_lag_ms`

   Slower-than-real-time Live is a result, not a gate. The harness does not
   disable or skip a slow engine.

### Live lifecycle

The Live path calls `Live::reset()`, feeds the model's native chunk size
(8960 samples / 560 ms), pads the final partial chunk with zeros, feeds
`live::FLUSH_CHUNKS` (3) zero chunks, then resets again. The transcript uses the
production rule: the first output has its leading space trimmed; later outputs
keep the model's spacing.

### Moonshine lifecycle

The Moonshine path uses `moonshine::configure_threads` once before model load,
then a reusable `Session<Moonshine>`: `feed` 100 ms chunks, `finish` to drain
the stream, and `cancel` before the next run. Only `Update::completed` text is
joined; provisional text is recorded for timing but never treated as final.

## Scoring

WER is computed in `scripts/benchmark-models.py`, not in Rust, so the raw
transcripts remain available for rescoring:

1. Unicode NFKC, then case-fold.
2. Remove only known bracketed non-speech annotations: `BLANK_AUDIO`, silence,
   music, noise, applause, laughter, inaudible, no speech, background noise,
   and unintelligible. Ordinary bracketed words remain in the score.
3. Delete apostrophes (`Eva's` → `evas`); replace hyphen variants with spaces.
4. Keep alphanumeric characters and whitespace; delete other punctuation.
5. Collapse whitespace. No number expansion or inverse text normalization.

Corpus WER is `total edits / total reference words`, not an average of
per-clip WERs. S/D/I come from a word-level Levenshtein alignment with an
explicit tie-break (substitution, then deletion, then insertion). CER uses the
same normalization without spaces and is `total character edits / total
reference characters`. Per-clip references, hypotheses, edit counts and CER
are preserved in `benchmark-clips.csv` and `benchmark.json`.

A failed corpus clip contributes an empty hypothesis (its reference words are
therefore deletions) and is also counted in `corpus_failed_clips`. This keeps
the aggregate honest about missing output; inspect the per-clip rows for the
error text.

JFK WER is computed against the public-domain reference
“And so my fellow Americans, ask not what your country can do for you, ask what
you can do for your country.” It is diagnostic only; the 12-clip corpus is the
quality score.

Silence/noise “false words” use the same normalization, so `[BLANK_AUDIO]`
does not count as a false word. The raw transcript is still shown.
Speech stress cases have a recognized-word count; their false-word field is
empty because those recordings contain speech.

## Memory and CPU

- `memory.after_load.rss_bytes` / `peak_rss_bytes` and
  `memory.at_end.rss_bytes` / `peak_rss_bytes` are read from Linux
  `/proc/self/status` `VmRSS` and `VmHWM` inside the benchmark process.
  They are whole-process RSS, including ONNX Runtime, model weights, scratch
  buffers and the loaded corpus. They are not “weights RAM”.
- `runtime.model_bytes` is the sum of the engine's model files (plus the
  Moonshine runtime), from the Rust-side file inventory. The Python report also
  records sizes and optional SHA-256 for every file under the models root.
- CPU per configuration is the parent-side `RUSAGE_CHILDREN` delta for
  user + system time while one child runs. The cumulative child `ru_maxrss`
  high-water mark is deliberately not reported as a per-configuration value;
  use the child's own `VmHWM` for peak RSS.
- The orchestrator removes `DICTATE_*` overrides and common BLAS/OMP thread
  environment variables, sets isolated `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
  `XDG_CACHE_HOME` and `TMPDIR`, and passes the model root explicitly. No user
  config or model directory is modified.

## Outputs

Each output directory contains:

| File | Contents |
| --- | --- |
| `benchmark.json` | metadata, full corpus manifest, every raw engine document, summaries, aggregate bests |
| `benchmark.csv` | one summary row per engine/thread configuration |
| `benchmark-clips.csv` | per-clip references, hypotheses, WER, CER, S/D/I, durations, RTF |
| `benchmark-stress.csv` | stress transcripts and false-word counts |
| `benchmark.md` | concise human-readable tables |
| `benchmark.html` | self-contained offline report with engine/thread filters, bars and transcript details |
| `commands.txt` | exact command line for each configuration |
| `raw/` | `<n>-<config>.stdout.txt` (engine JSON) and `.stderr.log` |
| `BENCHMARK-CORPUS-LICENSE.txt` | only when the corpus script writes into that directory |

The HTML uses inline CSS and JavaScript only: no CDNs, remote fonts, images or
scripts. The JSON is embedded as `application/json`; all dynamic text is set
with `textContent`.

The report metadata records:

- UTC time, hostname, OS, kernel, CPU model, logical/physical cores, RAM,
  governor, load average and Python version.
- Git `HEAD`, branch, dirty flag, porcelain status, diff stat, diff SHA-256 and
  a combined dirty-tree SHA-256 that includes untracked source files.
- Binary path, size and SHA-256.
- Corpus path, manifest SHA-256, source/revision/license and selected rows.
- Model root, every model file size and optional SHA-256.
- Flags, effective runs, timeout, sandbox wrapper and isolation details.

The `schema_version` of the Rust engine document and the combined report is
both 1.

## Corpus attribution and limits

The corpus is a fixed subset of **LibriSpeech `test-clean`**:

- Source: OpenSLR, <https://www.openslr.org/12/>
- Hugging Face dataset: `openslr/librispeech_asr`, config `clean`, split
  `test`, revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`
- License: **CC-BY-4.0**, <https://creativecommons.org/licenses/by/4.0/>
- Reference: V. Panayotov, G. Chen, D. Povey and S. Khudanpur, “LibriSpeech:
  An ASR corpus based on public domain audio books,” ICASSP 2015.

The selection is deliberately small and fixed: two 3–12 second clips from each
of six speakers, three male and three female, at rows
`0, 2, 250, 254, 501, 502, 750, 751, 1000, 1001, 1502, 1503`. It measures
read English from audiobooks. It does **not** represent conversational speech,
telephone audio, accents, noisy rooms, children, overlapping speakers, real
microphones, or the long tail of production dictation. Do not generalize the
WER ranking beyond this sample.

The manifest pins both the source FLAC SHA-256 and the canonical WAV SHA-256.
`prepare-benchmark-corpus.py` also checks the fixed row id, speaker id and
transcript text before decoding. A changed dataset revision or signed asset is
therefore a hard failure, not a silent corpus drift.

## Timing and scope

These are one-machine measurements, not a controlled benchmark laboratory:

- The file cache is warm; there is no cold-disk or first-download claim.
- The harness measures model compute, not microphone capture, VAD-only user
  latency, GUI paint time, keystroke injection, or end-to-end dictation.
- Live/Moonshine paced runs use `sleep_until` at real-time arrival and wall
  clock deltas. OS scheduling, CPU frequency, thermal throttling, background
  load and the governor can move results between runs.
- Five JFK runs and three paced runs are descriptive. They are not enough for
  confidence intervals or significance claims; the report includes median,
  mean, min, max, p90 and population standard deviation for transparency.
- RTF is processing time divided by audio duration. For Live/Moonshine it is
  compute time only; sleep time is excluded. `stop_to_final_ms` includes the
  flush/finalization compute and is not a typing-latency claim.
- `model_load_ms` covers construction, not download or OS page-cache warmup.
- Moonshine's `runtime-default` is not an exact thread count. Do not compare it
  as if it were a two-thread configuration.
- The standard/fluid/whisper whole-vs-paused JFK comparison uses the real VAD
  and join path, but JFK is short and read; it is one paired trial, a pipeline
  sanity check, not a segmentation quality benchmark or a statistical
  distribution. The VAD is loaded fresh for that trial, and its load time is
  reported separately rather than folded into the paused decode total.

## Troubleshooting

- **`model directory is missing`** — pass `--model-dir` to the folder that
  contains all six model folders and `silero-vad-v6`, not to the repository.
- **`corpus validation failed` / missing `tests/fixtures/audio`** — run
  `scripts/prepare-benchmark-corpus.py` first, or pass `--corpus` to the
  prepared cache manifest.
- **WAV SHA-256 mismatch** — re-run the prepare script. Do not edit the pinned
  hashes or score unverified audio.
- **`neither flac nor ffmpeg is installed`** — install `flac` (preferred) or
  `ffmpeg`, or use `--decoder`.
- **Output directory is not empty** — choose a new `--output` or pass
  `--force`.
- **A configuration times out** — raise `--timeout`; the raw stdout/stderr and
  timeout status are kept in the report.
- **Live is slower than real time** — that is recorded as a result. Do not
  lower the audio rate or skip runs to make it pass.
