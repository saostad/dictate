# English dictation on a Sandy Bridge laptop

Research date: 22 October 2026. Scope: English dictation, as confirmed by the user. Repository inspected: Dictate 0.8.0, commit a64a4eb. Application source and settings were not changed.

A responsive offline experience is feasible on this machine with a smaller model and inference during recording. The strongest candidates from these tests are **Moonshine Tiny Streaming for live interaction** and **Whisper Tiny English with an optimized runtime for pause-based dictation**. The fastest tested engine was a small Zipformer, but its output needs more work before it is a sensible default. No single engine is the fastest across all CPUs, recordings, and accuracy requirements.

## Machine and current bottlenecks

The machine has an Intel i5-2410M at 2.30 GHz, two physical cores / four logical CPUs, 7.7 GiB usable RAM, AVX and SSE4.1, and no AVX2, FMA, or F16C. It is a much older target than Apple Silicon. Background Python services were actively consuming CPU during the research, and swap was already in use before model comparisons. Available memory and workload varied; no background service was stopped.

Plan around CPU inference. The GeForce GT 520M is a CUDA compute capability 2.1 device, below CTranslate2's documented GPU minimum. Investing in this GPU is a poor first approach. Sources: [NVIDIA legacy GPU table](https://developer.nvidia.com/cuda/gpus/legacy), [CTranslate2 hardware support](https://opennmt.net/CTranslate2/hardware_support.html).

The application already uses a resident daemon, INT8 Parakeet TDT 0.6B v2, background inference, and Silero VAD. The main problem is model compute and when it runs, rather than startup alone or text insertion:

- Whole mode starts ASR after recording stops. Even an engine faster than real time can leave a noticeable wait after a long recording.
- Pauses mode waits for 416 ms of silence, then decodes the completed piece plus up to 1.5 seconds of repeated context. Without a qualifying pause, it force-cuts at 15 seconds. Shortening silence alone cannot remove the decode cost, and making pieces too short increases repeated context and boundary errors.
- Live uses a separate 0.6B Nemotron model, 560 ms chunks, and three zero chunks on stop. Existing logs show 859–905 ms per chunk in its speed check, versus the required 392 ms. It cannot keep up on this machine in those measurements. The fallback is appropriate.
- Fluid Speech Mini is another quality option built on the Parakeet family. Its smaller file size is not evidence that it requires proportionally less computation.
- ASR currently selects the logical CPU count, up to 16, for intra-op threading. On this CPU that normally means four threads. A configurable or measured one/two-thread choice deserves testing under ordinary desktop load.

Code: [ASR and threading](../src/asr.rs), [pause segmentation](../src/segment.rs), [live engine and flush](../src/live.rs), [worker and timing logs](../src/daemon/mod.rs), [Fluid inference](../src/fluid.rs).

Existing daemon logs include 1.2–4.9 second segment decodes and longer 5.8–6.7 second decodes for approximately 12–13 seconds of audio. Typing those results generally took tens to low hundreds of milliseconds. Later logs also show slower decodes, illustrating the importance of desktop load. A fresh CLI process took about 10 seconds to decode the 11-second fixture with two physical cores available, plus 8.6–9.9 seconds for model loading. That CLI result includes first-inference effects; it is **not** a warm daemon benchmark or a controlled speedup comparison against the table below.

## Measurements on this machine

The test input was the repository's public-domain, 11-second, 16 kHz mono JFK fixture. Each candidate was loaded once, given one warmup, then measured three times. Models were tested sequentially. Streaming engines received audio in small blocks without real-time sleeps; finalization was included in total processing time. The final measurement pass kept downloads and dependencies on disk rather than RAM-backed /tmp. Earlier busy-desktop results are retained separately.

RTF is elapsed processing time divided by audio duration. Below 1 can keep up on average; more headroom is better. **Processing time is not the same as the wait after stopping speech:** a streaming engine performs much of this work while speech is arriving.

| Engine and configuration | Median processing / 11 s audio | Warm run range | RTF | Median stop flush compute | Peak Python process RSS |
|---|---:|---:|---:|---:|---:|
| Zipformer 20M / sherpa-onnx, fp32, 1 thread | 1.34 s | 1.30–1.34 s | 0.12 | 92 ms | 199 MiB |
| Zipformer 20M / sherpa-onnx, int8, 1 thread | 1.65 s | 1.58–2.06 s | 0.15 | 120 ms | 121 MiB |
| Whisper Tiny English / faster-whisper, int8, 2 threads, beam 1 | 2.55 s | 2.51–2.64 s | 0.23 | — | 441 MiB |
| Vosk small-en-us-0.15, library defaults | 3.06 s | 3.04–3.26 s | 0.28 | 19 ms | 220 MiB |
| Moonshine Tiny Streaming, native library defaults, 0.5 s updates | 5.32 s | 5.28–5.47 s | 0.48 | 269 ms | 234 MiB |
| Moonshine Small Streaming, native library defaults, 0.5 s updates | 12.65 s | 11.14–13.59 s | 1.15 | 869 ms | 451 MiB |

RSS includes the Python process, native runtime, and allocator behavior. It is not the expected memory footprint of a Rust integration. Stop flush excludes natural endpoint waiting and is not a full desktop latency measurement. Whisper was run on finished audio, so its processing time would be paid after stop in Whole mode.

A separate **microphone-speed simulation** fed Moonshine Tiny at the actual audio rate, with a warmed model: first nonempty partial at **1.43 seconds**, stop flush **383 ms**, final result **509 ms after the 11-second input ended**. This shows subsecond finalization is possible for this clip. It does not establish a latency percentile, include desktop text insertion, or guarantee the same result for uninterrupted or noisy speech. The clip also contains natural pauses and trailing silence.

The native libraries tested successfully executed on this non-AVX2 CPU. Versions: moonshine-voice 0.1.5, sherpa-onnx 1.13.8, vosk 0.3.45, faster-whisper 1.2.1, CTranslate2 4.8.3. Moonshine used its current quantized_26_08_21 model files.

## Recognition quality and candidate choice

**Moonshine Tiny Streaming:** The 34M model is the strongest live prototype candidate. The downloaded model files total about 43 MiB. It returned usable text and punctuation, but sometimes changed “Americans” to “America” and split the sentence awkwardly. Partial transcripts revise earlier words, so do not pass every update directly to Dictate's append-only typing path. Display provisional text in a small overlay and commit completed phrases, or design and test a stable-prefix policy. Rewriting text in arbitrary focused applications is a separate interaction problem. The native [C API](https://github.com/moonshine-ai/moonshine/blob/main/core/moonshine-c-api.h) can be called from Rust. English streaming weights and library code use MIT licensing. [Model catalog](https://github.com/moonshine-ai/moonshine/blob/main/docs/models/available-models.md), [license](https://github.com/moonshine-ai/moonshine/blob/main/LICENSE).

**Whisper Tiny English:** The fastest tested option that preserved every spoken word of this particular fixture, with reasonable punctuation. It used INT8, explicit English, greedy decoding (beam 1), no VAD, and no previous-text conditioning. It is a strong candidate for a Fast speech-model option in the existing Whole/Pauses paths. This is not native stateful streaming; repeatedly decoding an ever-growing recording would waste compute. Use bounded completed phrases with careful boundaries. Current Pauses mode also needs word timestamps to discard repeated context; the benchmark did not request Whisper word alignment, so that extra cost and the joining behavior still need measurement. The measured runtime was faster-whisper/CTranslate2. For production, either integrate its native backend or benchmark a native whisper.cpp implementation; the measured CTranslate2 timing must not be attributed to whisper.cpp. whisper.cpp is a relevant Rust-friendly C API candidate, but was not benchmarked here. [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

**Zipformer 20M through sherpa-onnx:** Best throughput and low memory use in this comparison. FP32 was faster than INT8 on this CPU in the final pass, so quantization should be measured rather than assumed to win. Both variants lost the opening words when the fixture started immediately. Adding one second of leading silence recovered most of that opening, although “so” became “saw”; this exposed boundary sensitivity, not a clean quality victory. Output had no punctuation. The model is trained on LibriSpeech, so accent, noise, and conversational testing are especially important. A larger streaming Zipformer could improve the tradeoff, but was not tested. The tested weights are Apache-2.0 and sherpa-onnx offers a Rust API. [Model documentation](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-20m-2023-02-17-english), [model card](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17), [Rust API](https://k2-fsa.github.io/sherpa/onnx/rust-api/index.html).

**Vosk Small:** A viable low resource streaming fallback. It produced early partials, finalized cheaply, and made one word substitution (“as” for “ask”), without punctuation or capitalization. It was not the fastest engine tested here. The small US English model is Apache-2.0. [Vosk model catalog](https://alphacephei.com/vosk/models).

**Moonshine Small Streaming:** More accurate model family, but no reliable real-time headroom with the tested implementation on this laptop. Keep it for faster hardware or investigate optimization separately. Published benchmark quality numbers must be interpreted carefully: the Moonshine English leaderboard figures use floating-point models, while the shipped runtime is quantized; even its clean-speech evaluation is not a test of live dictation. [Accuracy methodology](https://github.com/moonshine-ai/moonshine/blob/main/docs/models/accuracy.md).

Extremely small command-recognition models are not substitutes for unrestricted dictation. For example, Moonshine Micro targets command recognition and spelling on microcontrollers. [Micro documentation](https://github.com/moonshine-ai/moonshine/blob/main/micro/README.md).

## Recommended development approach

1. **Prototype Moonshine Tiny Streaming for responsive local English dictation.** Reuse microphone capture and the background worker, keep its model resident, use its stateful streaming API, and surface revisable partials separately from committed text. Keep the existing large-model mode available for quality.
2. **Prototype Whisper Tiny English as a Fast model for Whole/Pauses.** The existing segmentation path makes this a smaller architectural change. Run the same recordings through CTranslate2 and a native whisper.cpp build before choosing production packaging. Benchmark 2–5 second phrases as well as long clips because small models can have fixed per-call overhead.
3. **Make the choice depend on measured hardware capability.** Benchmark real speech, queue growth, first usable text, stop-to-final latency, and desktop responsiveness. Test one and two threads, avoid oversubscribing both physical cores unnecessarily, and include ordinary background load. Suggested acceptance targets: median stop-to-final below 500 ms, p95 below one second, and sustained RTF comfortably below 0.7, subject to acceptable recognition quality. These are development targets, not measured guarantees.
4. **Preserve the larger quality modes.** Smaller models can be worse on accents, names, code terms, distant speech, and background speakers. Evaluate the user's English recordings before setting defaults. A single famous public speech cannot establish a quality ranking.
5. **Offer optional inference on another computer** if the user wants the current larger-model quality with little computation on the laptop. A self-hosted service on the M1 over a trusted local network is a valid separate approach; it adds network and availability dependencies and was not implemented or measured.

Local technology is sufficient to justify these prototypes. Matching the present larger models' quality, under arbitrary noise and competing workloads, with consistently instant offline results on this CPU remains a much harder requirement.

## Reproduce and inspect

The isolated environment, benchmark scripts, model files, JSON measurements, earlier busy-desktop measurements, and paced result are in:

`/home/saeid/.cache/dictate-stt-research/2026-10-22/`

The final measurement pass used a /tmp symlink to this disk directory; reproduction commands use its persistent location directly. No microphone or user recording was captured; the daemon was not restarted and no configuration was changed.

```sh
cd /home/saeid/.cache/dictate-stt-research/2026-10-22
venv/bin/python bench.py zipformer-fp32 1
venv/bin/python bench.py faster-whisper-tiny.en 2
venv/bin/python bench.py moonshine-tiny
venv/bin/python paced.py
```

The same native engines and graph configuration were used through Python for practical measurement; the tests do not measure the full Rust application. No service/process termination, GPU setup, or system configuration changes were performed. Model load time is excluded from the warm table. The cold CLI baseline is recorded separately. Background activity was not controlled, so absolute and relative timings are provisional. Do not read this one-clip feasibility check as a production benchmark suite.
