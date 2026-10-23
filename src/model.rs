use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use sha2::{Digest, Sha256};

const HF_REVISION: &str = "0bbb45a3365852604aef28b538a8f066f4ccaa85";
const HF_BASE_URL: &str = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve";
const MODEL_FILES: &[&str] = &[
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "vocab.txt",
];
const LIVE_REVISION: &str = "95df6c82aa796a3fc793f87633dcdb017ac12c07";
const LIVE_BASE_URL: &str =
    "https://huggingface.co/lokkju/nemotron-speech-streaming-en-0.6b-int8/resolve";
const LIVE_DIR_NAME: &str = "nemotron-streaming-en-0.6b-int8";
/// Live model files with their exact pinned sizes in bytes.
const LIVE_FILES: &[(&str, u64)] = &[
    ("encoder.onnx", 880_555_453),
    ("decoder_joint.onnx", 10_962_697),
    ("tokenizer.model", 251_056),
];
const FLUID_REVISION: &str = "fc592620325027419883cf272e70588d8a788c0a";
const FLUID_BASE_URL: &str =
    "https://huggingface.co/FluidVoice-Labs/fluid-speech-mini-onnx/resolve";
const FLUID_DIR_NAME: &str = "fluid-speech-mini-onnx";
/// Fluid Speech Mini files with their exact pinned sizes in bytes.
const FLUID_FILES: &[(&str, u64)] = &[
    ("config.json", 508),
    ("decoder.onnx", 24_461_296),
    ("encoder.onnx", 426_102),
    ("encoder.onnx.data", 253_429_760),
    ("frontend.onnx", 8_697_130),
    ("joint.onnx", 11_840_320),
    ("mel_filters.npy", 131_712),
    ("tokenizer.json", 1_159_959),
];
const VAD_URL: &str = "https://github.com/snakers4/silero-vad/raw/1e261b036686cd0017d500ee96acd1c4ba572a9d/src/silero_vad/data/silero_vad.onnx";
const VAD_DIR_NAME: &str = "silero-vad-v6";
const VAD_FILE_NAME: &str = "silero_vad.onnx";
/// Exact size of the pinned `silero_vad.onnx`. This file predates the SHA-256
/// verification used by the newer engines, so it is checked by exact size;
/// the pinned sha256 is
/// `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.
const VAD_FILE_SIZE: u64 = 2_327_524;
const PROGRESS_STEP_PERCENT: u64 = 5;
const MB: u64 = 1_000_000;

/// Whisper Tiny English (whisper.cpp GGML Q8_0) from the official whisper.cpp
/// model repository, pinned to one repository revision and verified by size
/// and SHA-256.
const WHISPER_REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";
const WHISPER_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve";
const WHISPER_DIR_NAME: &str = "whisper-tiny-en";
const WHISPER_FILE_NAME: &str = "ggml-tiny.en-q8_0.bin";
const WHISPER_FILE_SIZE: u64 = 43_550_795;
const WHISPER_FILE_SHA256: &str =
    "5bc2b3860aa151a4c6e7bb095e1fcce7cf12c7b020ca08dcec0c6d018bb7dd94";

/// Moonshine Tiny Streaming English, quantized `quantized_26_08_21` export
/// from the official model CDN. Every file is pinned by size and SHA-256.
const MOONSHINE_DIR_NAME: &str = "moonshine-tiny-streaming-en";
const MOONSHINE_MODEL_BASE_URL: &str =
    "https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21";
/// (file name, exact bytes, sha256). `decoder_kv_with_attention.ort` is only
/// needed for optional word timestamps and is deliberately not required.
const MOONSHINE_FILES: &[(&str, u64, &str)] = &[
    (
        "adapter.ort",
        1_319_664,
        "22ecc949e146c49667fda28d102d4e30749a107dc88a396292aa8f277ef1347c",
    ),
    (
        "cross_kv.ort",
        1_287_544,
        "143a36667b8d05fd9d04e8c337b7ee121f37ef299aea6b3d82bdb3d3401950b4",
    ),
    (
        "decoder_kv.ort",
        32_583_720,
        "8852553f312adb6c9aa4d17418015049b30f412209ee569d336548c0044627de",
    ),
    (
        "encoder.ort",
        7_675_440,
        "a8414e1a5dedf9f2093d7680601dd8a9b0433e7020260eafe0e370ead91134ca",
    ),
    (
        "frontend.model.ort",
        23_344,
        "5121b561417b638afce0c6c31b760e37c93cf97f80d9b0031aad1fe7b6f25d61",
    ),
    (
        "frontend.weights.ort",
        2_093_464,
        "217da24ac6f522ebf02da8ef288e77d1ac68d50d4a6821433182e4fbf4204bbd",
    ),
    (
        "streaming_config.json",
        509,
        "74fe5ddebd63b17caf59e8a3b18c17547ff7bce1642050edbb1c3962674f8950",
    ),
    (
        "tokenizer.bin",
        249_974,
        "6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d",
    ),
];
const MOONSHINE_RUNTIME_DIR_NAME: &str = "moonshine-runtime-0.1.5";
const MOONSHINE_RUNTIME_FILE: &str = "libmoonshine.so";
/// ONNX Runtime shared library shipped beside `libmoonshine.so`; the runtime
/// locates it with `$ORIGIN` RUNPATH, so both must be installed together.
const MOONSHINE_ORT_LIBRARY: &str = "libonnxruntime.so.1";
const MOONSHINE_RELEASE_BASE_URL: &str =
    "https://github.com/moonshine-ai/moonshine/releases/download/v0.1.5";

/// The model directory, honoring `DICTATE_MODEL_DIR` when it is set and
/// non-empty. Creates nothing.
pub fn model_dir() -> anyhow::Result<PathBuf> {
    model_dir_with(std::env::var_os("DICTATE_MODEL_DIR"))
}

fn model_dir_with(env: Option<std::ffi::OsString>) -> anyhow::Result<PathBuf> {
    if let Some(dir) = model_dir_override(env) {
        return Ok(dir);
    }
    let dirs = directories::ProjectDirs::from("", "", "dictate")
        .ok_or_else(|| anyhow::anyhow!("Could not find a home/data directory for this user"))?;
    Ok(dirs
        .data_dir()
        .join("models")
        .join("parakeet-tdt-0.6b-v2-int8"))
}

/// `DICTATE_MODEL_DIR` value as a path: `Some` when set and non-empty.
fn model_dir_override(env: Option<std::ffi::OsString>) -> Option<PathBuf> {
    match env {
        Some(dir) if !dir.is_empty() => Some(PathBuf::from(dir)),
        _ => None,
    }
}

/// Error unless every model file exists in `dir`.
pub fn check_model_dir(dir: &Path) -> anyhow::Result<()> {
    for name in MODEL_FILES {
        if !dir.join(name).is_file() {
            anyhow::bail!(
                "Model file {name} is missing in {}. Run \"dictate download\" first (about 660 MB).",
                dir.display()
            );
        }
    }
    Ok(())
}

/// The model directory, checked for completeness.
pub fn model_dir_checked() -> anyhow::Result<PathBuf> {
    let dir = model_dir()?;
    check_model_dir(&dir)?;
    Ok(dir)
}

/// Voice-detector dir for a parakeet model dir: the sibling
/// `silero-vad-v6` under the same models dir.
fn vad_dir_for(model_dir: &Path) -> PathBuf {
    sibling_dir_for(model_dir, VAD_DIR_NAME)
}

/// Fluid Speech Mini dir for a parakeet model dir: the sibling
/// `fluid-speech-mini-onnx` under the same models dir.
fn fluid_dir_for(model_dir: &Path) -> PathBuf {
    sibling_dir_for(model_dir, FLUID_DIR_NAME)
}

/// Live-model dir for a parakeet model dir: the sibling
/// `nemotron-streaming-en-0.6b-int8` under the same models dir.
fn live_dir_for(model_dir: &Path) -> PathBuf {
    sibling_dir_for(model_dir, LIVE_DIR_NAME)
}

/// Whisper Tiny English dir: the sibling `whisper-tiny-en`.
fn whisper_dir_for(model_dir: &Path) -> PathBuf {
    sibling_dir_for(model_dir, WHISPER_DIR_NAME)
}

/// Moonshine Tiny Streaming dir: the sibling
/// `moonshine-tiny-streaming-en`.
fn moonshine_dir_for(model_dir: &Path) -> PathBuf {
    sibling_dir_for(model_dir, MOONSHINE_DIR_NAME)
}

/// Moonshine native runtime dir: the sibling `moonshine-runtime-0.1.5`.
fn moonshine_runtime_dir_for(model_dir: &Path) -> PathBuf {
    sibling_dir_for(model_dir, MOONSHINE_RUNTIME_DIR_NAME)
}

/// A sibling of the parakeet model dir under the same models dir.
fn sibling_dir_for(model_dir: &Path, name: &str) -> PathBuf {
    match model_dir.parent() {
        Some(models) => models.join(name),
        None => model_dir.join(name),
    }
}

/// The Live model directory. Creates nothing.
pub fn live_model_dir() -> anyhow::Result<PathBuf> {
    Ok(live_dir_for(&model_dir()?))
}

/// Error unless every Live model file exists in `dir`.
pub fn check_live_model_dir(dir: &Path) -> anyhow::Result<()> {
    for (name, _) in LIVE_FILES {
        if !dir.join(name).is_file() {
            anyhow::bail!(
                "Model file {name} is missing in {}. Run \"dictate download --live\" first (about 900 MB).",
                dir.display()
            );
        }
    }
    Ok(())
}

/// The Live model directory, checked for completeness.
pub fn live_model_dir_checked() -> anyhow::Result<PathBuf> {
    let dir = live_model_dir()?;
    check_live_model_dir(&dir)?;
    Ok(dir)
}

/// The Fluid Speech Mini model directory. Creates nothing.
pub fn fluid_model_dir() -> anyhow::Result<PathBuf> {
    Ok(fluid_dir_for(&model_dir()?))
}

/// Error unless every Fluid Speech Mini file exists in `dir`.
pub fn check_fluid_model_dir(dir: &Path) -> anyhow::Result<()> {
    for (name, _) in FLUID_FILES {
        if !dir.join(name).is_file() {
            anyhow::bail!(
                "Model file {name} is missing in {}. Run \"dictate download --fluid\" first (about 300 MB).",
                dir.display()
            );
        }
    }
    Ok(())
}

/// The Fluid Speech Mini directory, checked for completeness.
pub fn fluid_model_dir_checked() -> anyhow::Result<PathBuf> {
    let dir = fluid_model_dir()?;
    check_fluid_model_dir(&dir)?;
    Ok(dir)
}

/// The Whisper Tiny English model directory. Creates nothing.
pub fn whisper_model_dir() -> anyhow::Result<PathBuf> {
    Ok(whisper_dir_for(&model_dir()?))
}

/// The Whisper model file inside its directory.
pub(crate) fn whisper_model_path_in(dir: &Path) -> PathBuf {
    dir.join(WHISPER_FILE_NAME)
}

/// Error unless the Whisper model file exists in `dir` with the pinned size.
pub fn check_whisper_model_dir(dir: &Path) -> anyhow::Result<()> {
    let path = whisper_model_path_in(dir);
    let len = path.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    if path.is_file() && len == WHISPER_FILE_SIZE {
        return Ok(());
    }
    let problem = if path.is_file() {
        format!("has wrong size ({len} bytes, expected {WHISPER_FILE_SIZE})")
    } else {
        "is missing".to_string()
    };
    anyhow::bail!(
        "Whisper model file {WHISPER_FILE_NAME} {problem} in {}. Run \"dictate download --whisper\" first (about 42 MB).",
        dir.display()
    )
}

/// The Whisper model directory, checked for completeness.
pub fn whisper_model_dir_checked() -> anyhow::Result<PathBuf> {
    let dir = whisper_model_dir()?;
    check_whisper_model_dir(&dir)?;
    Ok(dir)
}

/// The Moonshine Tiny Streaming model directory. Creates nothing.
pub fn moonshine_model_dir() -> anyhow::Result<PathBuf> {
    Ok(moonshine_dir_for(&model_dir()?))
}

/// Error unless every pinned Moonshine model file exists in `dir` at its
/// exact size. A startup check deliberately avoids hashing 47 MB; the
/// download path verifies SHA-256 instead.
pub fn check_moonshine_model_dir(dir: &Path) -> anyhow::Result<()> {
    for (name, size, _) in MOONSHINE_FILES {
        let path = dir.join(name);
        let len = path.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        if path.is_file() && len == *size {
            continue;
        }
        let problem = if path.is_file() {
            format!("has wrong size ({len} bytes, expected {size})")
        } else {
            "is missing".to_string()
        };
        anyhow::bail!(
            "Moonshine model file {name} {problem} in {}. Run \"dictate download --moonshine\" first (about 47 MB).",
            dir.display()
        );
    }
    Ok(())
}

/// The Moonshine model directory, checked for completeness.
pub fn moonshine_model_dir_checked() -> anyhow::Result<PathBuf> {
    let dir = moonshine_model_dir()?;
    check_moonshine_model_dir(&dir)?;
    Ok(dir)
}

/// The Moonshine native runtime directory, honoring
/// `DICTATE_MOONSHINE_RUNTIME_DIR` when it is set and non-empty.
pub fn moonshine_runtime_dir() -> anyhow::Result<PathBuf> {
    moonshine_runtime_dir_with(
        std::env::var_os("DICTATE_MOONSHINE_RUNTIME_DIR"),
        &model_dir()?,
    )
}

fn moonshine_runtime_dir_with(
    env: Option<std::ffi::OsString>,
    model_dir: &Path,
) -> anyhow::Result<PathBuf> {
    match env {
        Some(dir) if !dir.is_empty() => Ok(PathBuf::from(dir)),
        _ => Ok(moonshine_runtime_dir_for(model_dir)),
    }
}

/// Name of the loaded Moonshine shared library inside the runtime dir.
fn moonshine_library_name() -> &'static str {
    #[cfg(target_os = "linux")]
    {
        "libmoonshine.so"
    }
    #[cfg(target_os = "macos")]
    {
        "libmoonshine.dylib"
    }
    #[cfg(target_os = "windows")]
    {
        "moonshine.dll"
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        "libmoonshine.so"
    }
}

/// The Moonshine shared library path.
pub fn moonshine_library_path() -> anyhow::Result<PathBuf> {
    Ok(moonshine_runtime_dir()?
        .join("lib")
        .join(moonshine_library_name()))
}

/// Error unless the Moonshine runtime is installed and supported here.
pub fn check_moonshine_runtime() -> anyhow::Result<()> {
    if moonshine_runtime_asset().is_none() {
        anyhow::bail!(
            "Moonshine's prebuilt runtime is not available for this platform yet (Linux x86_64/arm64 only). All other modes still work; see README."
        );
    }
    let dir = moonshine_runtime_dir()?;
    let library = dir.join("lib").join(moonshine_library_name());
    let ort = dir.join("lib").join(MOONSHINE_ORT_LIBRARY);
    let library_ok = library
        .metadata()
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false);
    let ort_ok = ort
        .metadata()
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false);
    if library_ok && ort_ok {
        return Ok(());
    }
    let missing = if !library_ok {
        MOONSHINE_RUNTIME_FILE
    } else {
        MOONSHINE_ORT_LIBRARY
    };
    anyhow::bail!(
        "Moonshine runtime {missing} is missing in {}. Run \"dictate download --moonshine\" first (about 62 MB total), or set DICTATE_MOONSHINE_RUNTIME_DIR.",
        dir.display()
    )
}

/// Moonshine is ready: runtime library plus every model file.
pub fn check_moonshine_ready() -> anyhow::Result<()> {
    check_moonshine_runtime()?;
    check_moonshine_model_dir(&moonshine_model_dir()?)
}

/// Path of the voice-detector model. Creates nothing.
pub fn vad_model_path() -> anyhow::Result<PathBuf> {
    Ok(vad_dir_for(&model_dir()?).join(VAD_FILE_NAME))
}

/// The voice-detector path, checked for presence.
pub fn vad_model_path_checked() -> anyhow::Result<PathBuf> {
    let path = vad_model_path()?;
    check_vad_file(&path)?;
    Ok(path)
}

/// Error unless the voice-detector file exists at `path`.
fn check_vad_file(path: &Path) -> anyhow::Result<()> {
    if path.is_file() {
        return Ok(());
    }
    let dir = path
        .parent()
        .map(|parent| parent.display().to_string())
        .unwrap_or_else(|| ".".to_string());
    anyhow::bail!(
        "Voice detector {VAD_FILE_NAME} is missing in {dir}. Run \"dictate download\" first (about 2.3 MB)."
    );
}

/// Download any missing model files into the model directory.
pub fn download() -> anyhow::Result<()> {
    download_inner(false)
}

/// Download the speech model plus the Live streaming model.
pub fn download_live() -> anyhow::Result<()> {
    download_inner(true)
}

/// Download the speech model plus Fluid Speech Mini (the voice detector
/// Pauses mode needs comes with the speech model, as for `--live`).
pub fn download_fluid() -> anyhow::Result<()> {
    download()?;
    download_fluid_files_with_progress(|_, _| {})?;
    let dir = fluid_model_dir()?;
    println!("Fluid Speech Mini ready in {}", dir.display());
    Ok(())
}

/// Download Whisper Tiny English plus the voice detector Pauses needs, without
/// downloading the 660 MB Parakeet model.
pub fn download_whisper() -> anyhow::Result<()> {
    download_whisper_files_with_progress(|_, _| {})?;
    let dir = whisper_model_dir()?;
    println!("Whisper Tiny English ready in {}", dir.display());
    Ok(())
}

/// Download the Moonshine model and its pinned native runtime. Linux
/// x86_64/arm64 only; other platforms get an actionable error while every
/// other dictation mode keeps working.
pub fn download_moonshine() -> anyhow::Result<()> {
    if moonshine_runtime_asset().is_none() {
        anyhow::bail!(
            "Moonshine's prebuilt runtime is not available for this platform yet (Linux x86_64/arm64 only). All other modes still work; see README."
        );
    }
    download_moonshine_files_with_progress(|_, _| {})?;
    let dir = moonshine_model_dir()?;
    let runtime = moonshine_runtime_dir()?;
    println!(
        "Moonshine Tiny Streaming ready in {} (runtime in {})",
        dir.display(),
        runtime.display()
    );
    Ok(())
}

/// Pinned prebuilt Moonshine native release archive for this platform.
struct MoonshineRuntimeAsset {
    archive: &'static str,
    size: u64,
    sha256: &'static str,
    /// Exact size of the extracted `lib/libmoonshine.so`.
    library_size: u64,
    /// Exact size of the extracted `lib/libonnxruntime.so.1`.
    ort_size: u64,
}

fn moonshine_runtime_asset() -> Option<MoonshineRuntimeAsset> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        Some(MoonshineRuntimeAsset {
            archive: "moonshine-voice-linux-x86_64.tar.gz",
            size: 14_576_999,
            sha256: "9c3a87fea93ff2ad957938868f95a0a366dce9ff8ad86bde6cdcf5a4cadb51df",
            library_size: 12_157_928,
            ort_size: 22_326_072,
        })
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        Some(MoonshineRuntimeAsset {
            archive: "moonshine-voice-linux-arm64.tar.gz",
            size: 13_316_234,
            sha256: "1600c80a0806b7a2582307c98e7a56f4072e4b060498b08a0b75eb20af42def2",
            library_size: 11_490_352,
            ort_size: 18_693_384,
        })
    }
    #[cfg(not(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )))]
    {
        None
    }
}

/// Download the Whisper model, calling `progress(downloaded, total)` as bytes
/// land. The file is verified by exact size and SHA-256; a wrong file is
/// deleted and re-downloaded. Pauses mode also needs the Silero voice
/// detector, so this downloads it too (the CLI and the Settings/tray switch
/// use the same path).
pub fn download_whisper_files_with_progress<F>(mut progress: F) -> anyhow::Result<()>
where
    F: FnMut(u64, u64),
{
    let dir = whisper_model_dir()?;
    fs::create_dir_all(&dir)
        .with_context(|| format!("Could not create model directory {}", dir.display()))?;
    let dest = dir.join(WHISPER_FILE_NAME);
    let total = WHISPER_FILE_SIZE;
    if file_matches(&dest, WHISPER_FILE_SIZE, Some(WHISPER_FILE_SHA256))? {
        println!("{WHISPER_FILE_NAME}: already present");
        download_vad()?;
        progress(total, total);
        return Ok(());
    }
    let url = format!("{WHISPER_URL}/{WHISPER_REVISION}/{WHISPER_FILE_NAME}");
    eprintln!(
        "Downloading Whisper Tiny English ({})…",
        mb(WHISPER_FILE_SIZE)
    );
    let mut report = |file_bytes: u64| progress(file_bytes, total);
    download_file_report(&url, WHISPER_FILE_NAME, &dest, Some(&mut report))
        .with_context(|| format!("Downloading {WHISPER_FILE_NAME} from {url} failed"))?;
    verify_file(
        &dest,
        WHISPER_FILE_NAME,
        WHISPER_FILE_SIZE,
        WHISPER_FILE_SHA256,
    )
    .with_context(|| format!("Downloading {WHISPER_FILE_NAME} from {url} failed"))?;
    download_vad()?;
    progress(total, total);
    Ok(())
}

/// Download the Moonshine model files plus the native runtime, calling
/// `progress(downloaded, total)` across both. Present files count up front.
/// Every downloaded file is verified by size and SHA-256.
pub fn download_moonshine_files_with_progress<F>(mut progress: F) -> anyhow::Result<()>
where
    F: FnMut(u64, u64),
{
    let Some(asset) = moonshine_runtime_asset() else {
        anyhow::bail!(
            "Moonshine's prebuilt runtime is not available for this platform yet (Linux x86_64/arm64 only). All other modes still work; see README."
        );
    };
    let model_dir = moonshine_model_dir()?;
    let runtime_dir = moonshine_runtime_dir()?;
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("Could not create model directory {}", model_dir.display()))?;
    fs::create_dir_all(&runtime_dir).with_context(|| {
        format!(
            "Could not create Moonshine runtime directory {}",
            runtime_dir.display()
        )
    })?;

    let runtime_library = runtime_dir.join("lib").join(MOONSHINE_RUNTIME_FILE);
    let runtime_ort = runtime_dir.join("lib").join(MOONSHINE_ORT_LIBRARY);
    let runtime_present = runtime_library.is_file()
        && runtime_library.metadata().map(|m| m.len()).unwrap_or(0) == asset.library_size
        && runtime_ort.is_file()
        && runtime_ort.metadata().map(|m| m.len()).unwrap_or(0) == asset.ort_size;
    let model_total: u64 = MOONSHINE_FILES.iter().map(|(_, size, _)| size).sum();
    let total = model_total + asset.size;
    let mut completed: u64 = 0;

    if runtime_present {
        println!("{MOONSHINE_RUNTIME_FILE}: already present");
        completed += asset.size;
        progress(completed, total);
    } else {
        let url = format!("{MOONSHINE_RELEASE_BASE_URL}/{}", asset.archive);
        let archive_path = runtime_dir.join(format!("{}.download", asset.archive));
        eprintln!(
            "Downloading Moonshine runtime {} ({})…",
            asset.archive,
            mb(asset.size)
        );
        let mut report = |file_bytes: u64| progress(completed + file_bytes, total);
        download_file_report(&url, asset.archive, &archive_path, Some(&mut report))
            .with_context(|| format!("Downloading Moonshine runtime from {url} failed"))?;
        verify_file(&archive_path, asset.archive, asset.size, asset.sha256)
            .with_context(|| format!("Downloading Moonshine runtime from {url} failed"))?;
        extract_moonshine_runtime(&archive_path, &runtime_dir)
            .with_context(|| format!("Extracting Moonshine runtime from {url} failed"))?;
        if let Err(err) = fs::remove_file(&archive_path) {
            eprintln!(
                "warning: could not delete {}: {err}",
                archive_path.display()
            );
        }
        completed += asset.size;
        progress(completed, total);
    }

    for (name, size, sha256) in MOONSHINE_FILES {
        let dest = model_dir.join(name);
        if file_matches(&dest, *size, Some(sha256))? {
            println!("{name}: already present");
            completed += size;
            progress(completed, total);
            continue;
        }
        let url = format!("{MOONSHINE_MODEL_BASE_URL}/{name}");
        let mut report = |file_bytes: u64| progress(completed + file_bytes, total);
        download_file_report(&url, name, &dest, Some(&mut report))
            .with_context(|| format!("Downloading {name} from {url} failed"))?;
        verify_file(&dest, name, *size, sha256)
            .with_context(|| format!("Downloading {name} from {url} failed"))?;
        completed += size;
        progress(completed, total);
    }
    Ok(())
}

fn mb(bytes: u64) -> String {
    format!("{:.0} MB", bytes as f64 / MB as f64)
}

/// True when `path` exists with the expected size and (when given) SHA-256.
/// A wrong file is deleted so a fresh download replaces it.
fn file_matches(path: &Path, size: u64, sha256: Option<&str>) -> anyhow::Result<bool> {
    let len = path.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    if !path.is_file() || len != size {
        if path.is_file() {
            eprintln!(
                "{}: wrong size ({len} bytes, expected {size}); re-downloading",
                path.display()
            );
            let _ = fs::remove_file(path);
        }
        return Ok(false);
    }
    if let Some(expected) = sha256 {
        let actual = hash_file(path)?;
        if !actual.eq_ignore_ascii_case(expected) {
            eprintln!("{}: wrong SHA-256; re-downloading", path.display());
            let _ = fs::remove_file(path);
            return Ok(false);
        }
    }
    Ok(true)
}

fn hash_file(path: &Path) -> anyhow::Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("Could not read {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("Could not read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify a downloaded file's exact size and SHA-256, deleting it on any
/// mismatch so a later run cannot trust it.
fn verify_file(path: &Path, name: &str, expected: u64, sha256: &str) -> anyhow::Result<()> {
    verify_file_size(path, name, expected)?;
    let actual = hash_file(path)?;
    if actual.eq_ignore_ascii_case(sha256) {
        return Ok(());
    }
    if let Err(err) = fs::remove_file(path) {
        eprintln!(
            "warning: could not delete corrupt download {}: {err}",
            path.display()
        );
    }
    anyhow::bail!("{name} has wrong SHA-256 (the file was deleted)");
}

/// Extract the `lib/` and `include/` members of the pinned Moonshine release
/// archive directly into `dir`, stripping the archive's top-level folder.
/// Path traversal is rejected; only regular files and directories are
/// extracted.
fn extract_moonshine_runtime(archive_path: &Path, dir: &Path) -> anyhow::Result<()> {
    let file = fs::File::open(archive_path)
        .with_context(|| format!("Could not read {}", archive_path.display()))?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let mut components = path.components();
        // Drop the archive's top-level directory.
        let Some(_top) = components.next() else {
            continue;
        };
        let relative: PathBuf = components.collect();
        if relative.as_os_str().is_empty() {
            continue;
        }
        if relative
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            anyhow::bail!(
                "archive member {} escapes the runtime directory",
                path.display()
            );
        }
        if !(relative.starts_with("lib") || relative.starts_with("include")) {
            continue;
        }
        let kind = entry.header().entry_type();
        if !kind.is_file() && !kind.is_dir() {
            continue;
        }
        let dest = dir.join(&relative);
        entry
            .unpack(&dest)
            .with_context(|| format!("Could not write {}", dest.display()))?;
    }
    let library = dir.join("lib").join(MOONSHINE_RUNTIME_FILE);
    if !library.is_file() {
        anyhow::bail!(
            "the archive did not contain lib/{MOONSHINE_RUNTIME_FILE}; the pinned runtime layout changed"
        );
    }
    let ort = dir.join("lib").join(MOONSHINE_ORT_LIBRARY);
    if !ort.is_file() {
        anyhow::bail!(
            "the archive did not contain lib/{MOONSHINE_ORT_LIBRARY}; the pinned runtime layout changed"
        );
    }
    Ok(())
}

fn download_inner(with_live: bool) -> anyhow::Result<()> {
    let dir = model_dir()?;
    fs::create_dir_all(&dir)
        .with_context(|| format!("Could not create model directory {}", dir.display()))?;
    for name in MODEL_FILES {
        let dest = dir.join(name);
        let present = dest.is_file()
            && dest
                .metadata()
                .with_context(|| format!("Could not read metadata of {}", dest.display()))?
                .len()
                > 0;
        if present {
            println!("{name}: already present");
            continue;
        }
        let url = format!("{HF_BASE_URL}/{HF_REVISION}/{name}");
        download_file(&url, name, &dest)
            .with_context(|| format!("Downloading {name} from {url} failed"))?;
    }
    download_vad()?;
    if with_live {
        download_live_files()?;
    }
    println!("Model ready in {}", dir.display());
    Ok(())
}

/// Download any missing Live model files, verifying their exact pinned sizes.
/// A wrong-sized file is deleted and re-downloaded, never trusted.
fn download_live_files() -> anyhow::Result<()> {
    download_live_files_with_progress(|_, _| {})
}

/// Download any missing Live model files, calling
/// `progress(downloaded, total)` as bytes land across all files (`total` is
/// the pinned size of every Live file; already-present files count up front).
/// Used by the tray-menu download for progress notifications.
pub fn download_live_files_with_progress<F>(progress: F) -> anyhow::Result<()>
where
    F: FnMut(u64, u64),
{
    let dir = live_model_dir()?;
    download_files_with_progress(&dir, LIVE_BASE_URL, LIVE_REVISION, LIVE_FILES, progress)
}

/// Download any missing Fluid Speech Mini files, calling
/// `progress(downloaded, total)` as bytes land across all files (`total` is
/// the pinned size of every Fluid file; already-present files count up front).
/// Used by the Settings page's background model switch.
pub fn download_fluid_files_with_progress<F>(progress: F) -> anyhow::Result<()>
where
    F: FnMut(u64, u64),
{
    let dir = fluid_model_dir()?;
    download_files_with_progress(&dir, FLUID_BASE_URL, FLUID_REVISION, FLUID_FILES, progress)
}

/// Download the files of one pinned model export into `dir`, verifying each
/// file's exact size. A wrong-sized file is deleted and re-downloaded, never
/// trusted. `progress` gets `(downloaded, total)` bytes across all files.
fn download_files_with_progress<F>(
    dir: &Path,
    base_url: &str,
    revision: &str,
    files: &[(&str, u64)],
    mut progress: F,
) -> anyhow::Result<()>
where
    F: FnMut(u64, u64),
{
    fs::create_dir_all(dir)
        .with_context(|| format!("Could not create model directory {}", dir.display()))?;
    let total: u64 = files.iter().map(|(_, size)| size).sum();
    let mut completed: u64 = 0;
    for (name, size) in files {
        let dest = dir.join(name);
        let present_len = dest.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        if dest.is_file() && present_len == *size {
            println!("{name}: already present");
            completed += size;
            continue;
        }
        if dest.is_file() {
            eprintln!("{name}: wrong size ({present_len} bytes, expected {size}); re-downloading");
            // Windows cannot rename over an existing destination. Remove the
            // bad file before downloading the replacement.
            fs::remove_file(&dest)
                .with_context(|| format!("Could not remove {}", dest.display()))?;
        }
        let url = format!("{base_url}/{revision}/{name}");
        let mut report = |file_bytes: u64| progress(completed + file_bytes, total);
        download_file_report(&url, name, &dest, Some(&mut report))
            .with_context(|| format!("Downloading {name} from {url} failed"))?;
        verify_file_size(&dest, name, *size)
            .with_context(|| format!("Downloading {name} from {url} failed"))?;
        completed += size;
        progress(completed, total);
    }
    Ok(())
}

/// Download the voice-detector model next to the speech model, unless it is
/// already there with the exact pinned size. A wrong-sized file is deleted
/// and reported clearly, never trusted.
pub(crate) fn download_vad() -> anyhow::Result<()> {
    let dest = vad_model_path()?;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Could not create model directory {}", parent.display()))?;
    }
    let present_len = dest.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    if dest.is_file() && present_len == VAD_FILE_SIZE {
        println!("{VAD_FILE_NAME}: already present");
        return Ok(());
    }
    if dest.is_file() {
        eprintln!(
            "{VAD_FILE_NAME}: wrong size ({present_len} bytes, expected {VAD_FILE_SIZE}); re-downloading"
        );
        // Windows cannot rename over an existing destination.
        fs::remove_file(&dest).with_context(|| format!("Could not remove {}", dest.display()))?;
    }
    download_file(VAD_URL, VAD_FILE_NAME, &dest)
        .with_context(|| format!("Downloading {VAD_FILE_NAME} from {VAD_URL} failed"))?;
    verify_file_size(&dest, VAD_FILE_NAME, VAD_FILE_SIZE)
        .with_context(|| format!("Downloading {VAD_FILE_NAME} from {VAD_URL} failed"))?;
    Ok(())
}

/// The downloaded file must have exactly `expected` bytes (the integrity
/// check: exact size stands in for the sha256). A mismatch deletes the file
/// and errors clearly.
fn verify_file_size(path: &Path, file_name: &str, expected: u64) -> anyhow::Result<()> {
    let len = fs::metadata(path)
        .with_context(|| format!("Could not read metadata of {}", path.display()))?
        .len();
    if len == expected {
        return Ok(());
    }
    if let Err(err) = fs::remove_file(path) {
        eprintln!(
            "warning: could not delete wrong-sized download {}: {err}",
            path.display()
        );
    }
    anyhow::bail!(
        "{file_name} has wrong size: {len} bytes, expected {expected} bytes (the file was deleted)"
    );
}

fn download_file(url: &str, name: &str, dest: &Path) -> anyhow::Result<()> {
    download_file_report(url, name, dest, None)
}

/// Download one file like [`download_file`], calling `report` with the file's
/// downloaded byte count as chunks land (for overall progress).
fn download_file_report(
    url: &str,
    name: &str,
    dest: &Path,
    report: Option<&mut dyn FnMut(u64)>,
) -> anyhow::Result<()> {
    let part = dest.with_file_name(format!("{name}.part"));
    let result = download_file_inner(url, name, &part, dest, report);
    if result.is_err() {
        remove_part(&part);
    }
    result
}

fn remove_part(part: &Path) {
    match fs::remove_file(part) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => eprintln!(
            "warning: could not delete partial download {}: {err}",
            part.display()
        ),
    }
}

fn download_file_inner(
    url: &str,
    name: &str,
    part: &Path,
    dest: &Path,
    mut report: Option<&mut dyn FnMut(u64)>,
) -> anyhow::Result<()> {
    // Fail instead of hanging forever when the server never answers (a stuck
    // background download would keep the daemon in Loading).
    let mut response = ureq::get(url)
        .config()
        .timeout_connect(Some(std::time::Duration::from_secs(30)))
        .timeout_recv_response(Some(std::time::Duration::from_secs(60)))
        .build()
        .call()?;
    let total = response.body().content_length();
    let mut reader = response.body_mut().as_reader();
    let mut file = fs::File::create(part)
        .with_context(|| format!("Could not write partial download {}", part.display()))?;
    let mut downloaded: u64 = 0;
    let mut next_report_percent = PROGRESS_STEP_PERCENT;
    let mut next_report_mb = 10;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader
            .read(&mut buf)
            .context("Could not read response body")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .with_context(|| format!("Could not write partial download {}", part.display()))?;
        downloaded += n as u64;
        if let Some(report) = report.as_mut() {
            report(downloaded);
        }
        match total {
            Some(total) => {
                let percent = downloaded * 100 / total;
                if percent >= next_report_percent {
                    eprintln!(
                        "{name}: {percent}% ({} / {} MB)",
                        downloaded / MB,
                        total / MB
                    );
                    next_report_percent = percent + PROGRESS_STEP_PERCENT;
                }
            }
            None => {
                let mb = downloaded / MB;
                if mb >= next_report_mb {
                    eprintln!("{name}: {mb} MB");
                    next_report_mb = mb + 10;
                }
            }
        }
    }
    if let Some(total) = total
        && downloaded != total
    {
        anyhow::bail!("downloaded {downloaded} bytes but Content-Length was {total}");
    }
    drop(file);
    // Windows refuses to rename over an existing file. Every caller wants the
    // freshly downloaded file to replace it.
    #[cfg(target_os = "windows")]
    if dest.exists() {
        fs::remove_file(dest).with_context(|| format!("Could not remove {}", dest.display()))?;
    }
    fs::rename(part, dest)
        .with_context(|| format!("Could not move {} to {}", part.display(), dest.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn fresh_dir(tag: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "dictate-test-model-{}-{tag}-{n}",
            std::process::id()
        ))
    }

    fn cleanup(dir: &Path) {
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn missing_file_error_names_file_and_suggests_download() {
        let dir = fresh_dir("missing");
        fs::create_dir_all(&dir).unwrap();
        let err = check_model_dir(&dir).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("encoder-model.int8.onnx"), "unexpected: {msg}");
        assert!(msg.contains("dictate download"), "unexpected: {msg}");
        assert!(
            msg.contains(&dir.display().to_string()),
            "unexpected: {msg}"
        );
        cleanup(&dir);
    }

    #[test]
    fn complete_dir_passes_check() {
        let dir = fresh_dir("complete");
        fs::create_dir_all(&dir).unwrap();
        for name in MODEL_FILES {
            fs::write(dir.join(name), b"stub").unwrap();
        }
        check_model_dir(&dir).unwrap();
        cleanup(&dir);
    }

    #[test]
    fn vad_dir_is_a_sibling_of_the_speech_model_dir() {
        let model_dir = Path::new("/data/models/parakeet-tdt-0.6b-v2-int8");
        assert_eq!(
            vad_dir_for(model_dir),
            PathBuf::from("/data/models/silero-vad-v6")
        );
        assert_eq!(
            vad_dir_for(model_dir).join(VAD_FILE_NAME),
            PathBuf::from("/data/models/silero-vad-v6/silero_vad.onnx")
        );
    }

    #[test]
    fn missing_vad_error_names_file_and_suggests_download() {
        let dir = fresh_dir("vad-missing");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(VAD_FILE_NAME);
        let err = check_vad_file(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains(VAD_FILE_NAME), "unexpected: {msg}");
        assert!(msg.contains("dictate download"), "unexpected: {msg}");
        assert!(
            msg.contains(&dir.display().to_string()),
            "unexpected: {msg}"
        );
        cleanup(&dir);
    }

    #[test]
    fn present_vad_file_passes_check() {
        let dir = fresh_dir("vad-present");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(VAD_FILE_NAME);
        fs::write(&path, b"stub").unwrap();
        check_vad_file(&path).unwrap();
        cleanup(&dir);
    }

    #[test]
    fn verify_file_size_accepts_exact_size() {
        let dir = fresh_dir("vad-size-ok");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(VAD_FILE_NAME);
        fs::write(&path, b"12345").unwrap();
        verify_file_size(&path, VAD_FILE_NAME, 5).unwrap();
        assert!(path.is_file(), "a good file must be kept");
        cleanup(&dir);
    }

    #[test]
    fn verify_file_size_deletes_wrong_size() {
        let dir = fresh_dir("vad-size-bad");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(VAD_FILE_NAME);
        fs::write(&path, b"1234").unwrap();
        let err = verify_file_size(&path, VAD_FILE_NAME, 5).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains(VAD_FILE_NAME), "unexpected: {msg}");
        assert!(msg.contains('4'), "unexpected: {msg}");
        assert!(msg.contains('5'), "unexpected: {msg}");
        assert!(!path.exists(), "a wrong-sized file must be deleted");
        cleanup(&dir);
    }

    #[test]
    fn live_dir_is_a_sibling_of_the_speech_model_dir() {
        let model_dir = Path::new("/data/models/parakeet-tdt-0.6b-v2-int8");
        assert_eq!(
            live_dir_for(model_dir),
            PathBuf::from("/data/models/nemotron-streaming-en-0.6b-int8")
        );
    }

    #[test]
    fn missing_live_file_error_suggests_download_live() {
        let dir = fresh_dir("live-missing");
        fs::create_dir_all(&dir).unwrap();
        let err = check_live_model_dir(&dir).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("encoder.onnx"), "unexpected: {msg}");
        assert!(msg.contains("dictate download --live"), "unexpected: {msg}");
        assert!(
            msg.contains(&dir.display().to_string()),
            "unexpected: {msg}"
        );
        cleanup(&dir);
    }

    #[test]
    fn complete_live_dir_passes_check() {
        let dir = fresh_dir("live-complete");
        fs::create_dir_all(&dir).unwrap();
        for (name, _) in LIVE_FILES {
            fs::write(dir.join(name), b"stub").unwrap();
        }
        check_live_model_dir(&dir).unwrap();
        cleanup(&dir);
    }

    #[test]
    fn fluid_dir_is_a_sibling_of_the_speech_model_dir() {
        let model_dir = Path::new("/data/models/parakeet-tdt-0.6b-v2-int8");
        assert_eq!(
            fluid_dir_for(model_dir),
            PathBuf::from("/data/models/fluid-speech-mini-onnx")
        );
    }

    #[test]
    fn missing_fluid_file_error_suggests_download_fluid() {
        let dir = fresh_dir("fluid-missing");
        fs::create_dir_all(&dir).unwrap();
        let err = check_fluid_model_dir(&dir).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("config.json"), "unexpected: {msg}");
        assert!(
            msg.contains("dictate download --fluid"),
            "unexpected: {msg}"
        );
        assert!(
            msg.contains(&dir.display().to_string()),
            "unexpected: {msg}"
        );
        cleanup(&dir);
    }

    #[test]
    fn complete_fluid_dir_passes_check() {
        let dir = fresh_dir("fluid-complete");
        fs::create_dir_all(&dir).unwrap();
        for (name, _) in FLUID_FILES {
            fs::write(dir.join(name), b"stub").unwrap();
        }
        check_fluid_model_dir(&dir).unwrap();
        cleanup(&dir);
    }

    #[test]
    fn whisper_default_is_the_q8_0_file() {
        assert_eq!(WHISPER_FILE_NAME, "ggml-tiny.en-q8_0.bin");
        assert_eq!(WHISPER_FILE_SIZE, 43_550_795);
        assert_eq!(
            whisper_model_path_in(Path::new("/data/models/whisper-tiny-en")),
            PathBuf::from("/data/models/whisper-tiny-en/ggml-tiny.en-q8_0.bin")
        );
    }

    #[test]
    fn whisper_dir_is_a_sibling_of_the_speech_model_dir() {
        let model_dir = Path::new("/data/models/parakeet-tdt-0.6b-v2-int8");
        assert_eq!(
            whisper_dir_for(model_dir),
            PathBuf::from("/data/models/whisper-tiny-en")
        );
    }

    #[test]
    fn missing_whisper_file_error_suggests_download_whisper() {
        let dir = fresh_dir("whisper-missing");
        fs::create_dir_all(&dir).unwrap();
        let err = check_whisper_model_dir(&dir).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains(WHISPER_FILE_NAME), "unexpected: {msg}");
        assert!(
            msg.contains("dictate download --whisper"),
            "unexpected: {msg}"
        );
        assert!(
            msg.contains(&dir.display().to_string()),
            "unexpected: {msg}"
        );
        cleanup(&dir);
    }

    #[test]
    fn moonshine_dir_is_a_sibling_of_the_speech_model_dir() {
        let model_dir = Path::new("/data/models/parakeet-tdt-0.6b-v2-int8");
        assert_eq!(
            moonshine_dir_for(model_dir),
            PathBuf::from("/data/models/moonshine-tiny-streaming-en")
        );
        assert_eq!(
            moonshine_runtime_dir_for(model_dir),
            PathBuf::from("/data/models/moonshine-runtime-0.1.5")
        );
    }

    #[test]
    fn missing_moonshine_file_error_suggests_download_moonshine() {
        let dir = fresh_dir("moonshine-missing");
        fs::create_dir_all(&dir).unwrap();
        let err = check_moonshine_model_dir(&dir).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("adapter.ort"), "unexpected: {msg}");
        assert!(
            msg.contains("dictate download --moonshine"),
            "unexpected: {msg}"
        );
        assert!(
            msg.contains(&dir.display().to_string()),
            "unexpected: {msg}"
        );
        cleanup(&dir);
    }

    #[test]
    fn complete_moonshine_dir_passes_check() {
        let dir = fresh_dir("moonshine-complete");
        fs::create_dir_all(&dir).unwrap();
        for (name, size, _) in MOONSHINE_FILES {
            let file = fs::File::create(dir.join(name)).unwrap();
            file.set_len(*size).unwrap();
        }
        check_moonshine_model_dir(&dir).unwrap();
        cleanup(&dir);
    }

    #[test]
    fn moonshine_runtime_env_overrides_the_sibling_directory() {
        let model = Path::new("/data/models/parakeet-tdt-0.6b-v2-int8");
        assert_eq!(
            moonshine_runtime_dir_with(Some("/opt/moonshine".into()), model).unwrap(),
            PathBuf::from("/opt/moonshine")
        );
        assert_eq!(
            moonshine_runtime_dir_with(Some("".into()), model).unwrap(),
            PathBuf::from("/data/models/moonshine-runtime-0.1.5")
        );
    }

    #[test]
    fn model_dir_override_uses_env_when_set() {
        assert_eq!(
            model_dir_override(Some("/tmp/dictate-models".into())),
            Some(PathBuf::from("/tmp/dictate-models"))
        );
    }

    #[test]
    fn model_dir_override_ignores_empty_env() {
        assert_eq!(model_dir_override(Some("".into())), None);
    }

    #[test]
    fn model_dir_override_is_none_when_unset() {
        assert_eq!(model_dir_override(None), None);
    }

    #[test]
    fn model_dir_with_env_uses_override() {
        let dir = model_dir_with(Some("/tmp/dictate-models".into())).unwrap();
        assert_eq!(dir, PathBuf::from("/tmp/dictate-models"));
    }

    #[test]
    fn model_dir_with_unset_env_uses_project_data_dir() {
        let dir = model_dir_with(None).unwrap();
        assert!(
            dir.join("encoder-model.int8.onnx")
                .ends_with("models/parakeet-tdt-0.6b-v2-int8/encoder-model.int8.onnx"),
            "unexpected: {}",
            dir.display()
        );
    }
}
