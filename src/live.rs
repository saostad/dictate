//! Live streaming mode: NVIDIA Nemotron speech-to-text while speaking.
//!
//! Unlike whole/pauses mode (Parakeet TDT on finished audio), the Live engine
//! feeds 8960-sample (560 ms) chunks to the streaming model as they arrive and
//! types each chunk's new text immediately. Chunk outputs are final and never
//! revised; the model's text carries its own spaces, so only the first output
//! of a recording has its leading space trimmed.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Context;
use parakeet_rs::ort::environment::Environment;
use parakeet_rs::{ExecutionConfig, Nemotron};

/// Audio samples per streaming chunk: 8960 = 560 ms at 16 kHz.
pub(crate) const CHUNK_SAMPLES: usize = 8960;
/// Audio duration of one chunk, in milliseconds.
pub(crate) const CHUNK_MS: u64 = 560;
/// Zero chunks fed after the padded remainder to flush the final words.
pub(crate) const FLUSH_CHUNKS: usize = 3;
/// Speed gate: the median benchmark chunk must take at most 70% of real time.
pub(crate) const SPEED_LIMIT_MS: u64 = 392;

/// True when the benchmark median is fast enough for Live mode.
pub(crate) fn is_fast_enough(median: Duration) -> bool {
    median <= Duration::from_millis(SPEED_LIMIT_MS)
}

/// Real-time factor of an average chunk time: processing time / audio time.
pub(crate) fn rtf(avg_ms: f64) -> f64 {
    avg_ms / CHUNK_MS as f64
}

/// Deterministic low-level noise for the speed benchmark: uniform pseudo-noise
/// at ±0.01, so the benchmark exercises real inference without silence
/// shortcuts.
fn benchmark_noise() -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    (0..CHUNK_SAMPLES)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let unit = ((state >> 16) & 0x7FFF) as f32 / 32768.0;
            unit * 0.02 - 0.01
        })
        .collect()
}

/// The Live streaming recognizer.
pub struct Live {
    model: Nemotron,
}

impl Live {
    /// Load the Live model from `dir` with the shared ORT environment and
    /// `threads` intra-op threads.
    pub fn load(dir: &Path, env: &Environment, threads: usize) -> anyhow::Result<Self> {
        let config = ExecutionConfig::new()
            .with_environment(env.clone())
            .with_intra_threads(threads);
        let model = Nemotron::from_pretrained(dir, Some(config))
            .with_context(|| format!("Loading Live model from {} failed", dir.display()))?;
        eprintln!("using {threads} threads");
        Ok(Self { model })
    }

    /// Audio samples per streaming chunk (8960 for this export).
    pub fn chunk_samples(&self) -> usize {
        self.model.chunk_samples()
    }

    /// Transcribe one chunk: returns only the new text for this chunk.
    pub fn transcribe_chunk(&mut self, chunk: &[f32]) -> anyhow::Result<String> {
        Ok(self.model.transcribe_chunk(chunk)?)
    }

    /// Back to a fresh recording: clears the model's streaming state.
    pub fn reset(&mut self) {
        self.model.reset();
    }

    /// Speed gate: process 3 chunks of low-level noise and return the median
    /// per-chunk time, then reset. A failing chunk counts as infinitely slow
    /// (Live falls back to whole-recording mode).
    pub fn benchmark(&mut self) -> Duration {
        let chunk = benchmark_noise();
        let mut times = Vec::with_capacity(3);
        for _ in 0..3 {
            let start = Instant::now();
            if self.transcribe_chunk(&chunk).is_err() {
                self.reset();
                return Duration::MAX;
            }
            times.push(start.elapsed());
        }
        times.sort();
        self.reset();
        times[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_gate_is_70_percent_of_real_time() {
        assert!(is_fast_enough(Duration::from_millis(0)));
        assert!(is_fast_enough(Duration::from_millis(391)));
        assert!(is_fast_enough(Duration::from_millis(SPEED_LIMIT_MS)));
        assert!(!is_fast_enough(Duration::from_millis(SPEED_LIMIT_MS + 1)));
        assert_eq!(SPEED_LIMIT_MS, CHUNK_MS * 7 / 10);
    }

    #[test]
    fn chunk_constant_is_560_ms_at_16_khz() {
        assert_eq!(CHUNK_SAMPLES, 8960);
        assert_eq!(CHUNK_SAMPLES as u64 * 1000 / 16_000, CHUNK_MS);
    }

    #[test]
    fn rtf_divides_by_chunk_audio_time() {
        assert!((rtf(560.0) - 1.0).abs() < 1e-9);
        assert!((rtf(1015.0) - 1.8125).abs() < 1e-9);
    }

    #[test]
    fn benchmark_noise_is_low_level_and_deterministic() {
        let noise = benchmark_noise();
        assert_eq!(noise.len(), CHUNK_SAMPLES);
        assert!(
            noise.iter().all(|sample| sample.abs() <= 0.01),
            "benchmark noise must stay low-level"
        );
        assert_ne!(noise[0], noise[1], "benchmark noise must vary");
        assert_eq!(noise, benchmark_noise());
    }
}
