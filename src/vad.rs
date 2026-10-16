//! Silero VAD v6 speech probabilities for pause segmentation.
//!
//! The detector scores 512-sample frames at 16 kHz (32 ms each). Every call
//! feeds `input` f32 `[1, 576]` (64 samples of carried context + 512 new),
//! `state` f32 `[2, 1, 128]` (zeros at the start, then the previous `stateN`)
//! and `sr` i64 scalar 16000, and reads `output` `[1, 1]` (the speech
//! probability) plus the new `stateN`. State and context live in the `Vad`,
//! so one `Vad` scores exactly one recording; [`Vad::reset`] starts another.

use std::path::Path;

use anyhow::Context;
use parakeet_rs::ort::environment::Environment;
use parakeet_rs::ort::session::Session;
use parakeet_rs::ort::value::Tensor;

/// New samples per detector frame (32 ms at 16 kHz).
pub(crate) const FRAME_LEN: usize = 512;
/// Samples of carried context prepended to every frame.
const CONTEXT_LEN: usize = 64;
/// RNN state elements ([2, 1, 128] = 256).
const STATE_LEN: usize = 256;
/// Sample rate the detector was trained on.
const SAMPLE_RATE: i64 = 16_000;

/// Silero VAD session plus its running state and carried context.
pub(crate) struct Vad {
    session: Session,
    state: Vec<f32>,
    context: [f32; CONTEXT_LEN],
}

impl Vad {
    /// Load the voice detector from the standard path with the shared ORT
    /// environment.
    pub fn load_default() -> anyhow::Result<Self> {
        let path = crate::model::vad_model_path_checked()?;
        let env = crate::asr::shared_environment()?;
        Self::load(&path, &env)
    }

    /// Load the voice detector from `path` (one intra-op thread: a frame
    /// costs a fraction of a millisecond). The session keeps the environment
    /// alive, so the borrow ends here.
    pub fn load(path: &Path, env: &Environment) -> anyhow::Result<Self> {
        let builder = Session::builder(env)
            .with_context(|| format!("Could not set up the voice detector ({})", path.display()))?;
        // `with_intra_threads` returns a builder-carrying error (not a
        // std error), so it maps through Display instead of context.
        let mut builder = builder.with_intra_threads(1).map_err(|err| {
            anyhow::anyhow!(
                "Could not set up the voice detector ({}): {err}",
                path.display()
            )
        })?;
        let session = builder.commit_from_file(path).with_context(|| {
            format!("Could not load the voice detector from {}", path.display())
        })?;
        Ok(Self {
            session,
            state: vec![0.0; STATE_LEN],
            context: [0.0; CONTEXT_LEN],
        })
    }

    /// Speech probability for one 512-sample frame at 16 kHz. Updates the
    /// RNN state and the carried context for the next frame.
    pub fn prob(&mut self, frame512: &[f32]) -> anyhow::Result<f32> {
        if frame512.len() != FRAME_LEN {
            anyhow::bail!(
                "voice detector frame must hold {FRAME_LEN} samples, got {}",
                frame512.len()
            );
        }
        let mut input = Vec::with_capacity(CONTEXT_LEN + FRAME_LEN);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(frame512);
        self.context
            .copy_from_slice(&frame512[FRAME_LEN - CONTEXT_LEN..]);
        let input = Tensor::from_array(([1usize, CONTEXT_LEN + FRAME_LEN], input))
            .context("Could not build the voice detector input")?;
        let state = Tensor::from_array(([2usize, 1, 128], self.state.clone()))
            .context("Could not build the voice detector state")?;
        let sr = Tensor::from_array(((), vec![SAMPLE_RATE]))
            .context("Could not build the voice detector sample rate")?;
        let outputs = self
            .session
            .run(parakeet_rs::ort::inputs![
                "input" => input,
                "state" => state,
                "sr" => sr,
            ])
            .with_context(|| "Could not run the voice detector on an audio frame")?;
        let prob = output_first_f32(&outputs, "output")?;
        let state = output_all_f32(&outputs, "stateN")?;
        if state.len() != STATE_LEN {
            anyhow::bail!(
                "voice detector state has {} elements, expected {STATE_LEN}",
                state.len()
            );
        }
        self.state = state;
        Ok(prob)
    }

    /// Back to a fresh recording: zero state and context.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.context = [0.0; CONTEXT_LEN];
    }
}

/// First element of a float output tensor, with a clear error.
fn output_first_f32(
    outputs: &parakeet_rs::ort::session::SessionOutputs,
    name: &str,
) -> anyhow::Result<f32> {
    let value = outputs
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("voice detector output is missing \"{name}\""))?;
    let (_, data) = value
        .try_extract_tensor::<f32>()
        .with_context(|| format!("voice detector output \"{name}\" is not float"))?;
    data.first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("voice detector output \"{name}\" holds no data"))
}

/// All elements of a float output tensor, with a clear error.
fn output_all_f32(
    outputs: &parakeet_rs::ort::session::SessionOutputs,
    name: &str,
) -> anyhow::Result<Vec<f32>> {
    let value = outputs
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("voice detector output is missing \"{name}\""))?;
    let (_, data) = value
        .try_extract_tensor::<f32>()
        .with_context(|| format!("voice detector output \"{name}\" is not float"))?;
    Ok(data.to_vec())
}
