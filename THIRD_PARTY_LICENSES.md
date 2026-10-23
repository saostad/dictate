# Third-party models, runtimes and licenses

Dictate's own code is MIT (see `LICENSE`). The following are separate works,
downloaded at runtime or compiled in; they are not relicensed by Dictate.

## Bundled in the binary

### whisper.cpp / whisper-rs

- Project: <https://github.com/ggml-org/whisper.cpp>
- Rust bindings: `whisper-rs` 0.16.0 and `whisper-rs-sys` 0.15.0
- Bundled whisper.cpp version: 1.8.3 (as recorded by
  `whisper-rs-sys` 0.15.0's `whisper.cpp/CMakeLists.txt`)
- License: MIT
- Copyright notice: `Copyright (c) 2023-2024 The ggml authors`
- Built with `GGML_NATIVE=OFF` and AVX/AVX2/FMA/F16C/AVX512/SSE4.2 disabled,
  so the binary runs on the x86_64 baseline. It is statically linked into
  Dictate.

### Rust dependencies

Cargo dependencies and their licenses are listed by
`cargo metadata` / `cargo deny` tooling. The direct dependencies introduced
for the small engines are `flate2` (MIT/Apache-2.0), `libloading` (ISC),
`sha2` (MIT/Apache-2.0), `tar` (MIT/Apache-2.0), and `whisper-rs` /
`whisper-rs-sys` (Unlicense, <https://unlicense.org>; Rust bindings to the
MIT-licensed whisper.cpp above).

## Downloaded models and runtimes

### Whisper Tiny English

- File: `ggml-tiny.en-q8_0.bin` (Q8_0 quantization)
- Source repository: <https://huggingface.co/ggerganov/whisper.cpp>
- Pinned revision: `5359861c739e955e79d9a303bcbc70fb988958b1`
- URL used by Dictate:
  <https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.en-q8_0.bin>
- Size: 43,550,795 bytes
- SHA-256:
  `5bc2b3860aa151a4c6e7bb095e1fcce7cf12c7b020ca08dcec0c6d018bb7dd94`
- License: MIT. The OpenAI Whisper weights are `Copyright (c) 2022 OpenAI`;
  the GGML conversion/quantization is distributed by the whisper.cpp project
  (`Copyright (c) 2023-2024 The ggml authors`).

### Moonshine Tiny Streaming English

- Model: `tiny-streaming-en`, quantized `quantized_26_08_21`
- Base URL:
  <https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21/>
- Files, exact sizes and SHA-256:

  | File | Bytes | SHA-256 |
  |---|---:|---|
  | `adapter.ort` | 1,319,664 | `22ecc949e146c49667fda28d102d4e30749a107dc88a396292aa8f277ef1347c` |
  | `cross_kv.ort` | 1,287,544 | `143a36667b8d05fd9d04e8c337b7ee121f37ef299aea6b3d82bdb3d3401950b4` |
  | `decoder_kv.ort` | 32,583,720 | `8852553f312adb6c9aa4d17418015049b30f412209ee569d336548c0044627de` |
  | `encoder.ort` | 7,675,440 | `a8414e1a5dedf9f2093d7680601dd8a9b0433e7020260eafe0e370ead91134ca` |
  | `frontend.model.ort` | 23,344 | `5121b561417b638afce0c6c31b760e37c93cf97f80d9b0031aad1fe7b6f25d61` |
  | `frontend.weights.ort` | 2,093,464 | `217da24ac6f522ebf02da8ef288e77d1ac68d50d4a6821433182e4fbf4204bbd` |
  | `streaming_config.json` | 509 | `74fe5ddebd63b17caf59e8a3b18c17547ff7bce1642050edbb1c3962674f8950` |
  | `tokenizer.bin` | 249,974 | `6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d` |

- Model license: MIT (`Copyright (c) 2025 Useful Sensors, Inc. (dba Moonshine AI)`)
- Project: <https://github.com/moonshine-ai/moonshine>
- Model catalog:
  <https://github.com/moonshine-ai/moonshine/blob/main/docs/models/available-models.md>

### Moonshine native runtime

- Version: Moonshine v0.1.5
- Release:
  <https://github.com/moonshine-ai/moonshine/releases/tag/v0.1.5>
- Linux x86_64 asset: `moonshine-voice-linux-x86_64.tar.gz`, 14,576,999
  bytes, SHA-256
  `9c3a87fea93ff2ad957938868f95a0a366dce9ff8ad86bde6cdcf5a4cadb51df`
- Linux arm64 asset: `moonshine-voice-linux-arm64.tar.gz`, 13,316,234 bytes,
  SHA-256 `1600c80a0806b7a2582307c98e7a56f4072e4b060498b08a0b75eb20af42def2`
- License: MIT
- The archive also contains ONNX Runtime 1.23.2 (`libonnxruntime.so.1`,
  Microsoft, MIT, `Copyright (c) Microsoft Corporation`). Dictate extracts
  both libraries under `moonshine-runtime-0.1.5/lib/` and loads them with
  `libloading`. The bundled Moonshine runtime code is MIT
  (`Copyright (c) 2025 Useful Sensors, Inc. (dba Moonshine AI)`).
- macOS/Windows upstream assets are static libraries; Dictate does not
  download or claim support for them in this build.

### Silero VAD v6

- Source:
  <https://github.com/snakers4/silero-vad/raw/1e261b036686cd0017d500ee96acd1c4ba572a9d/src/silero_vad/data/silero_vad.onnx>
- Size: 2,327,524 bytes
- SHA-256:
  `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`
- License: MIT (`Copyright (c) 2020-present Silero Team`)

### NVIDIA Parakeet TDT 0.6B v2 (int8 ONNX export)

- Project: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2>
- ONNX export: <https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx>
- License: CC-BY-4.0
- Downloaded by `dictate download`; not bundled.

### Fluid Speech Mini

- Project: <https://huggingface.co/FluidInference/FluidSpeechMini>
- Upstream models: NVIDIA Parakeet TDT 0.6B v3 and Moondream Parakeet Ultra
- License: CC BY 4.0
- Downloaded by `dictate download --fluid`; not bundled.

### NVIDIA Nemotron streaming 0.6B

- Model: `nemotron-streaming-en-0.6b-int8`
- Source: <https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int8>
- Pinned revision: `95df6c82aa796a3fc793f87633dcdb017ac12c07`
- License: PolyForm Shield 1.0.0
  (<https://polyformproject.org/licenses/shield/1.0.0/>), as stated by the
  model card
- Downloaded by `dictate download --live`; not bundled.

## Upstream copyright notices

These notices are reproduced from the upstream license files listed above.

- whisper.cpp: `Copyright (c) 2023-2024 The ggml authors`
- Moonshine code, runtime and models: `Copyright (c) 2025 Useful Sensors,
  Inc. (dba Moonshine AI)`
- ONNX Runtime (shipped inside the Moonshine runtime archive):
  `Copyright (c) Microsoft Corporation`
- OpenAI Whisper weights: `Copyright (c) 2022 OpenAI`
- Silero VAD: `Copyright (c) 2020-present Silero Team` (MIT), see
  <https://github.com/snakers4/silero-vad/blob/master/LICENSE>
- `whisper-rs` and `whisper-rs-sys`: Unlicense, see
  <https://unlicense.org> and each crate's `LICENSE` file

## MIT license text (applies to the MIT-licensed works above)

```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
