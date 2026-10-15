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
    let env = parakeet_rs::ort::init().build()?;
    if ENV.set(env.clone()).is_err() {
        // Another thread initialized first; prefer its environment.
        if let Some(first) = ENV.get() {
            return Ok(first.clone());
        }
    }
    Ok(env)
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
