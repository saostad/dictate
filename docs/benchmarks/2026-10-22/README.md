# Model comparison on this machine — 2026-10-22

All **14 configurations passed**, with no partial runs or engine failures. Every inference process ran under the included Linux seccomp wrapper, which blocks socket creation, connect, sendto and sendmsg. A control socket-creation probe failed with EPERM. Setup downloads were completed before measurement.

Machine: Intel i7-6700HQ, 4 cores / 8 logical CPUs, 15.5 GiB RAM, Omarchy 4.0.6. Governor: `powersave`; normal desktop services stayed running. These are CPU measurements from the portable build, with a warm file cache.

## App-default results

Models stay resident after loading. The quality set contains 12 LibriSpeech test-clean clips: 81.345 seconds, 200 reference words, six speakers. Corpus RTF is total processing time / total audio time; lower is faster. The 11-second JFK decode is the median of five warmed runs.

| Model | Threads | Load ms | JFK median ms | Corpus RTF | Strict WER | Peak MiB | First committed / after stop ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard / Parakeet | 8 | 2864 | 671 | 0.065 | 1.5% (3/200) | 1750 | — |
| Fluid Mini | 8 | 390 | 1309 | 0.163 | 2.0% (4/200) | 521 | — |
| Whisper Tiny Q8_0 | 2 | 40 | 1626 | 0.196 | 5.0% (10/200) | 246 | — |
| Nemotron Live | 8 | 3174 | 2490 | 0.247 | 1.5% (3/200) | 1491 | 1230 / 526 |
| Moonshine Tiny Streaming | 1 | 97 | 1583 | 0.186 | 6.0% (12/200) | 218 | 2628 / 127 |

Peak memory is the whole process over the full workload, including the 44.75-second synthetic speech stress case. Streaming first-text time starts at audio start; stop latency starts at audio end. These are medians of three paced trials; Moonshine provisional text is excluded because the application types only completed lines. The JFK clip has trailing silence.

## What to use

- **Resident Whole mode on this machine: keep Standard.** It had the fastest compute and tied Live for the lowest strict WER in this sample, with the largest memory footprint. Model loading took about 2.9 seconds and is excluded from warm timings.
- **Streaming with earlier committed words: Live.** Its first output arrived sooner than Moonshine, with higher memory use and a longer finalization tail.
- **Streaming with a small memory budget: Moonshine, Auto/1 thread.** It peaked around 218 MiB. Runtime-default threads were slower here (JFK 1,955 ms vs 1,583 ms) and gave the same strict WER. Its first completed phrase arrived later than Live.
- **Small Whole-mode fallback: Whisper Tiny Q8_0, Auto/2 threads.** It uses about 246 MiB peak and loads quickly. The build disables AVX/AVX2/FMA/F16C/SSE4.2 for Whisper, but actual no-AVX2 hardware was not available for validation.
- **Whisper Pauses has a measured cost.** The production 16 kHz VAD/context path took 20.459 seconds for 11 seconds of audio at Auto/2 threads, compared with a 1.626-second Whole median. This is one paired Pauses trial, not a distribution. The separate 48 kHz verification took about 27.4 seconds.
- **Fluid is a middle memory option.** It used around 521 MiB and made four strict reference differences. This clean read-speech corpus does not test its foreground-speaker behavior with nearby speech.

Auto is 8 inference threads for Standard/Fluid/Live, 2 for Whisper, and single-thread mode for Moonshine on this host. The exact 1/2-thread overrides for Standard/Fluid/Live are benchmark controls; their production loaders retain the app default. Moonshine runtime-default is not exactly two threads.

## Reading the accuracy scores

Strict WER includes lexical formatting differences: `Neoplatonists` versus `neo Platonists`, `defense` versus `defence`, and `15th` versus `fifteenth` occur in these outputs. Some differences are substantive recognition errors. Raw references and hypotheses are preserved so both can be inspected. The normalizer was fixed before the final matrix; ordinary bracketed words remain scoreable. There is no number expansion or spelling equivalence.

Every configuration emitted zero words on the five-second silence and deterministic low-level-noise cases. These cases are separate from corpus WER. They do not establish resistance to real room noise or competing speakers.

This sample is too small for broad accuracy, accent, noise, significance, or tail-latency claims. No microphone, tray host, desktop typing, Windows/macOS, or actual Sandy Bridge/no-AVX2 runtime test was performed. Moonshine support in this build is Linux-only.

## Statistics and raw data

- [Interactive compact comparison](comparison.html): app defaults or all 14 configurations, with transcript details.
- [Full interactive report](benchmark.html) and [Markdown tables](benchmark.md).
- [Summary CSV](benchmark.csv): load, warm timing distributions, RTF, WER/CER, memory and streaming metrics.
- [Per-clip CSV](benchmark-clips.csv): all references, hypotheses, S/D/I and timings.
- [Stress CSV](benchmark-stress.csv): silence, noise, short speech and repeated speech. False-word counts apply to silence/noise only.
- [Full JSON](benchmark.json): host, Git snapshot, binary/model/corpus hashes, all raw native documents and derived summaries.
- [Exact commands](commands.txt) and [raw stdout/stderr](raw/).

The five JFK trials and three paced trials have mean, median, min, max, p90 and population standard deviation in the full data. A p90 with three or five samples is descriptive. Quality clips are decoded once per configuration after warmup. Presentation formatting was refined after measurement; the measured timings, transcripts, memory and quality scores were retained.

## Validation

- `cargo fmt --all -- --check`: passed.
- `cargo test`: 259 passed, 0 failed, 1 ignored.
- `cargo test --no-default-features --features system-ort`: 266 passed, 0 failed, 1 ignored.
- Both `cargo clippy --all-targets` configurations with `-D warnings`: passed.
- `cargo build --release`: passed.
- `PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_benchmark_tools.py`: 26 passed.
- The ignored real Moonshine Session reuse/finish/cancel test was run separately and passed, including one network-blocked paced run.
- Interactive report checks cover five default rows, all 14 configurations, engine filtering and table-cell alignment. Desktop dark and mobile light previews passed.

The same release binary was used for all measured configurations, SHA-256 `1c75c466d4a34ae565aa9cbc0c3f60b264da21e8266fa320b835476ba5e1f1a4`. Its hash was checked again after the matrix.

## Reproduce

From the repository root, with the model and corpus caches on this machine:

```sh
cargo build --release
python3 scripts/prepare-benchmark-corpus.py --offline \
  --cache /home/saeid/.cache/dictate-benchmarks/2026-10-22/audio \
  --output /home/saeid/.cache/dictate-benchmarks/2026-10-22
python3 scripts/benchmark-models.py \
  --binary target/release/dictate \
  --corpus /home/saeid/.cache/dictate-benchmarks/2026-10-22/benchmark-corpus.json \
  --model-dir /home/saeid/.cache/dictate-verification-2026-10-22/models \
  --output benchmark-results/repeat \
  --sandbox scripts/no-network.py \
  --runs 5 --paced-runs 3 --timeout 2400 --strict
```

The output folder must be empty. For fresh setup and corpus attribution, see the [benchmark guide](../../model-benchmarks.md) and [corpus license](CORPUS-LICENSE.txt). Audio and weights remain outside Git. The [engine implementation note](../../stt-implementation-2026-10-22.md) documents backend behavior and compatibility limits.

Measurements were captured from the `main` checkout before committing the implementation. The benchmark run did not change user configuration or services.
