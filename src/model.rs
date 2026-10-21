use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;

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
/// Exact size of the pinned `silero_vad.onnx`. `sha2` is not in Cargo.lock,
/// so the download is verified by exact size instead of sha256 (pinned
/// sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`,
/// for reference if hashing ever becomes available).
const VAD_FILE_SIZE: u64 = 2_327_524;
const PROGRESS_STEP_PERCENT: u64 = 5;
const MB: u64 = 1_000_000;

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
fn download_vad() -> anyhow::Result<()> {
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
