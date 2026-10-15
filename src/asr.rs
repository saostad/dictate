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
            "This CPU has no AVX2, but this build bundles an ONNX Runtime that needs AVX2. Install ONNX Runtime from your package manager (Arch: sudo pacman -S onnxruntime-cpu) and rebuild with: cargo build --release --no-default-features --features system-ort"
        );
    }
    Ok(parakeet_rs::ort::init().build()?)
}

#[cfg(feature = "system-ort")]
fn build_environment() -> anyhow::Result<Environment> {
    let path = ort_dylib_path(std::env::var_os("ORT_DYLIB_PATH"));
    let builder = parakeet_rs::ort::init_from(&path).map_err(|err| {
        anyhow::anyhow!(
            "Could not load ONNX Runtime from {}: {err:#}. Install it (Arch: sudo pacman -S onnxruntime-cpu) or set ORT_DYLIB_PATH to the full path of the onnxruntime library file.",
            path.display()
        )
    })?;
    Ok(builder.build()?)
}

/// Path of the system ONNX Runtime library: `ORT_DYLIB_PATH` when it is set
/// and non-empty, otherwise the platform's default library file name.
#[cfg(feature = "system-ort")]
fn ort_dylib_path(env: Option<std::ffi::OsString>) -> std::path::PathBuf {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
    match env {
        Some(path) if !path.is_empty() => std::path::PathBuf::from(path),
        _ => std::path::PathBuf::from(format!("{DLL_PREFIX}onnxruntime{DLL_SUFFIX}")),
    }
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
        let path = ort_dylib_path(Some("/opt/ort/libonnxruntime.so".into()));
        assert_eq!(path, PathBuf::from("/opt/ort/libonnxruntime.so"));
    }

    #[test]
    fn ort_dylib_path_falls_back_on_empty_env() {
        let path = ort_dylib_path(Some("".into()));
        assert_eq!(path, default_lib_name());
    }

    #[test]
    fn ort_dylib_path_falls_back_when_unset() {
        assert_eq!(ort_dylib_path(None), default_lib_name());
    }
}
