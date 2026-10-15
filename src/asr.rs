use std::path::Path;
use std::sync::OnceLock;

use anyhow::Context;
use parakeet_rs::ort::environment::Environment;
use parakeet_rs::{ExecutionConfig, ParakeetTDT, Transcriber};

pub struct Asr {
    model: ParakeetTDT,
}

fn shared_environment() -> anyhow::Result<Environment> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env.clone());
    }
    let env = build_environment()?;
    if ENV.set(env.clone()).is_err() {
        // Another thread initialized first; prefer its environment.
        if let Some(first) = ENV.get() {
            return Ok(first.clone());
        }
    }
    Ok(env)
}

#[cfg(feature = "bundled-ort")]
fn build_environment() -> anyhow::Result<Environment> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !std::arch::is_x86_feature_detected!("avx2") {
        anyhow::bail!(
            "This CPU has no AVX2, but this build bundles an ONNX Runtime that needs AVX2. Download the prebuilt Linux release from https://github.com/saostad/dictate/releases/latest (it works on any x86_64 CPU), or rebuild with: cargo build --release --no-default-features --features system-ort"
        );
    }
    Ok(parakeet_rs::ort::init().build()?)
}

#[cfg(feature = "system-ort")]
fn build_environment() -> anyhow::Result<Environment> {
    let exe = std::env::current_exe().ok();
    let exe_dir = exe.as_ref().and_then(|p| p.parent());
    let path = ort_dylib_path(std::env::var_os("ORT_DYLIB_PATH"), exe_dir, |p| p.exists());
    let builder = parakeet_rs::ort::init_from(&path).map_err(|err| {
        anyhow::anyhow!(
            "Could not load ONNX Runtime from {}: {err:#}. Put the onnxruntime library next to the dictate binary, install it (Arch: sudo pacman -S onnxruntime-cpu), or set ORT_DYLIB_PATH to its full path.",
            path.display()
        )
    })?;
    Ok(builder.build()?)
}

/// Path of the ONNX Runtime library for system-ort builds: `ORT_DYLIB_PATH`
/// when it is set and non-empty, otherwise the library file next to the
/// running binary when it exists, otherwise the platform's default library
/// file name (resolved through the system search path).
#[cfg(feature = "system-ort")]
fn ort_dylib_path(
    env: Option<std::ffi::OsString>,
    exe_dir: Option<&Path>,
    mut exists: impl FnMut(&Path) -> bool,
) -> std::path::PathBuf {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
    if let Some(path) = env
        && !path.is_empty()
    {
        return std::path::PathBuf::from(path);
    }
    let file_name = format!("{DLL_PREFIX}onnxruntime{DLL_SUFFIX}");
    if let Some(dir) = exe_dir {
        let next_to_binary = dir.join(&file_name);
        if exists(&next_to_binary) {
            return next_to_binary;
        }
    }
    std::path::PathBuf::from(file_name)
}

/// True when the transcript holds no recognized speech.
pub(crate) fn is_empty_transcript(text: &str) -> bool {
    text.trim().is_empty()
}

/// Message for `stderr` when no speech was recognized in `seconds` of audio.
pub(crate) fn no_speech_message(seconds: f32) -> String {
    format!("No speech recognized in {seconds:.1} s of audio.")
}

impl Asr {
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        load_inner(dir)
            .with_context(|| format!("Loading speech model from {} failed", dir.display()))
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> anyhow::Result<String> {
        let result = self
            .model
            .transcribe_samples(samples.to_vec(), 16_000, 1, None)?;
        Ok(result.text.trim().to_string())
    }
}

fn load_inner(dir: &Path) -> anyhow::Result<Asr> {
    let env = shared_environment()?;
    let config = ExecutionConfig::new().with_environment(env);
    let model = ParakeetTDT::from_pretrained(dir, Some(config))?;
    Ok(Asr { model })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_transcript_check_ignores_whitespace() {
        assert!(is_empty_transcript(""));
        assert!(is_empty_transcript("   "));
        assert!(!is_empty_transcript("hello"));
    }

    #[test]
    fn no_speech_message_names_duration() {
        assert_eq!(
            no_speech_message(3.9),
            "No speech recognized in 3.9 s of audio."
        );
    }
}

#[cfg(all(test, feature = "system-ort"))]
mod system_ort_tests {
    use super::*;
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
    use std::path::PathBuf;

    fn default_lib_name() -> PathBuf {
        PathBuf::from(format!("{DLL_PREFIX}onnxruntime{DLL_SUFFIX}"))
    }

    #[test]
    fn ort_dylib_path_uses_env_when_set() {
        let path = ort_dylib_path(Some("/opt/ort/libonnxruntime.so".into()), None, |_| true);
        assert_eq!(path, PathBuf::from("/opt/ort/libonnxruntime.so"));
    }

    #[test]
    fn ort_dylib_path_env_wins_over_bundled_lib() {
        let path = ort_dylib_path(
            Some("/opt/ort/libonnxruntime.so".into()),
            Some(Path::new("/opt/dictate")),
            |_| true,
        );
        assert_eq!(path, PathBuf::from("/opt/ort/libonnxruntime.so"));
    }

    #[test]
    fn ort_dylib_path_uses_lib_next_to_binary() {
        let dir = Path::new("/opt/dictate");
        let path = ort_dylib_path(None, Some(dir), |_| true);
        assert_eq!(path, dir.join(default_lib_name()));
    }

    #[test]
    fn ort_dylib_path_empty_env_falls_through_to_bundled_lib() {
        let dir = Path::new("/opt/dictate");
        let path = ort_dylib_path(Some("".into()), Some(dir), |_| true);
        assert_eq!(path, dir.join(default_lib_name()));
    }

    #[test]
    fn ort_dylib_path_probes_exe_dir() {
        let dir = Path::new("/opt/dictate");
        let mut probed = Vec::new();
        let path = ort_dylib_path(None, Some(dir), |p| {
            probed.push(p.to_path_buf());
            false
        });
        assert_eq!(probed, vec![dir.join(default_lib_name())]);
        assert_eq!(path, default_lib_name());
    }

    #[test]
    fn ort_dylib_path_falls_back_when_unset() {
        assert_eq!(ort_dylib_path(None, None, |_| true), default_lib_name());
    }

    #[test]
    fn ort_dylib_path_falls_back_on_empty_env() {
        let path = ort_dylib_path(Some("".into()), None, |_| true);
        assert_eq!(path, default_lib_name());
    }
}
