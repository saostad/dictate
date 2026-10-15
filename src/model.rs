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

/// Download any missing model files into the model directory.
pub fn download() -> anyhow::Result<()> {
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
    println!("Model ready in {}", dir.display());
    Ok(())
}

fn download_file(url: &str, name: &str, dest: &Path) -> anyhow::Result<()> {
    let part = dest.with_file_name(format!("{name}.part"));
    let result = download_file_inner(url, name, &part, dest);
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

fn download_file_inner(url: &str, name: &str, part: &Path, dest: &Path) -> anyhow::Result<()> {
    let mut response = ureq::get(url).call()?;
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
