//! Fluid Speech Mini: the alternative Whole/Pauses speech model.
//!
//! This is a direct port of the model's Python reference (`transcribe.py`):
//! Parakeet-style log-mel features, the split conv front end run in time
//! chunks, the FastConformer encoder, and greedy token-and-duration (TDT)
//! decoding over the decoder/joint graphs. Tokens are joined with the model's
//! BPE tokenizer; word timestamps come from the encoder frame each token was
//! emitted at (one frame = 80 ms).

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::path::Path;

use anyhow::{Context, bail};
use parakeet_rs::ort::AsPointer;
use parakeet_rs::ort::environment::Environment;
use parakeet_rs::ort::ortsys;
use parakeet_rs::ort::session::Session;
use parakeet_rs::ort::session::builder::GraphOptimizationLevel;
use parakeet_rs::ort::value::Tensor;
use realfft::RealFftPlanner;

use crate::asr::Word;

/// `log(x + 2^-24)` guard for the mel power spectrum.
/// 2^-24.
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;
/// Normalization epsilon added to the mel standard deviation.
const NORM_EPS: f32 = 1e-5;
/// Seconds of audio represented by one encoder frame (10 ms hop × 8).
const FRAME_SECONDS: f32 = 0.08;
/// Long clips are cut at quiet points into pieces of about this many seconds.
const SPLIT_TARGET_SECONDS: f64 = 30.0;
/// The split point is searched within ±`SPLIT_SEARCH_SECONDS / 2` of the target.
const SPLIT_SEARCH_SECONDS: f64 = 10.0;
/// Audio energy is measured in windows of 1/50 s for the quiet-point search.
const ENERGY_WINDOWS_PER_SECOND: usize = 50;
/// Frames per mel/FFT block (bounds the temporary allocation, like Python).
const MEL_BLOCK_FRAMES: usize = 256;

/// The parts of `config.json` the inference path uses.
#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) blank_token_id: u32,
    pub(crate) pad_token_id: u32,
    pub(crate) eos_token_id: Option<u32>,
    pub(crate) durations: Vec<usize>,
    pub(crate) max_symbols_per_step: usize,
    pub(crate) decoder_layers: usize,
    pub(crate) decoder_hidden_size: usize,
    pub(crate) encoder_hidden_size: usize,
    pub(crate) sampling_rate: usize,
    pub(crate) n_fft: usize,
    pub(crate) hop_length: usize,
    pub(crate) win_length: usize,
    pub(crate) preemphasis: Option<f32>,
    pub(crate) num_mel_bins: usize,
    pub(crate) max_single_pass_seconds: f64,
    pub(crate) subsampling_kernel: usize,
    pub(crate) subsampling_stride: usize,
    pub(crate) subsampling_layers: usize,
    pub(crate) frontend_chunk_frames: usize,
}

impl Config {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Could not read {}", path.display()))?;
        Self::from_json(&text)
            .with_context(|| format!("Could not parse model config {}", path.display()))
    }

    /// Parse `config.json`. The reference reads the same keys.
    pub(crate) fn from_json(text: &str) -> anyhow::Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(text).context("invalid JSON in config.json")?;
        let number = |key: &str| -> anyhow::Result<&serde_json::Value> {
            value
                .get(key)
                .ok_or_else(|| anyhow::anyhow!("config.json is missing \"{key}\""))
        };
        let usize_field = |key: &str| -> anyhow::Result<usize> {
            number(key)?
                .as_u64()
                .map(|number| number as usize)
                .ok_or_else(|| {
                    anyhow::anyhow!("config.json \"{key}\" must be a non-negative integer")
                })
        };
        let durations = number("durations")?
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("config.json \"durations\" must be a list"))?
            .iter()
            .map(|duration| {
                duration
                    .as_u64()
                    .map(|duration| duration as usize)
                    .ok_or_else(|| anyhow::anyhow!("config.json \"durations\" must hold integers"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let eos_token_id = value
            .get("eos_token_id")
            .and_then(serde_json::Value::as_u64)
            .map(|id| id as u32);
        let config = Self {
            vocab_size: usize_field("vocab_size")?,
            blank_token_id: usize_field("blank_token_id")? as u32,
            pad_token_id: usize_field("pad_token_id")? as u32,
            eos_token_id,
            durations,
            max_symbols_per_step: usize_field("max_symbols_per_step")?,
            decoder_layers: usize_field("decoder_layers")?,
            decoder_hidden_size: usize_field("decoder_hidden_size")?,
            encoder_hidden_size: usize_field("encoder_hidden_size")?,
            sampling_rate: usize_field("sampling_rate")?,
            n_fft: usize_field("n_fft")?,
            hop_length: usize_field("hop_length")?,
            win_length: usize_field("win_length")?,
            preemphasis: number("preemphasis")?.as_f64().map(|value| value as f32),
            num_mel_bins: usize_field("num_mel_bins")?,
            max_single_pass_seconds: number("max_single_pass_seconds")?.as_f64().ok_or_else(
                || anyhow::anyhow!("config.json \"max_single_pass_seconds\" must be a number"),
            )?,
            subsampling_kernel: usize_field("subsampling_kernel")?,
            subsampling_stride: usize_field("subsampling_stride")?,
            subsampling_layers: usize_field("subsampling_layers")?,
            frontend_chunk_frames: usize_field("frontend_chunk_frames")?,
        };
        for (name, value) in [
            ("max_symbols_per_step", config.max_symbols_per_step),
            ("hop_length", config.hop_length),
            ("n_fft", config.n_fft),
            ("win_length", config.win_length),
            ("num_mel_bins", config.num_mel_bins),
            ("subsampling_stride", config.subsampling_stride),
            ("frontend_chunk_frames", config.frontend_chunk_frames),
        ] {
            if value == 0 {
                bail!("config.json \"{name}\" must be greater than 0");
            }
        }
        Ok(config)
    }
}

/// The tokenizer embedded in `tokenizer.json`: a BPE vocab plus Metaspace
/// decoding (the replacement character `▁` stands for a space).
pub(crate) struct Tokenizer {
    pieces: HashMap<u32, String>,
    skip: HashSet<u32>,
    space: String,
}

impl Tokenizer {
    fn load(path: &Path, config: &Config) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Could not read {}", path.display()))?;
        Self::from_json(&text, config.blank_token_id, config.pad_token_id)
            .with_context(|| format!("Could not parse tokenizer {}", path.display()))
    }

    /// Parse a BPE `tokenizer.json`. Blank and pad are always skipped; added
    /// tokens marked special (including them) are skipped too.
    pub(crate) fn from_json(text: &str, blank: u32, pad: u32) -> anyhow::Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(text).context("invalid JSON in tokenizer.json")?;
        let vocab = value
            .get("model")
            .and_then(|model| model.get("vocab"))
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("tokenizer.json has no model.vocab"))?;
        let mut pieces = HashMap::with_capacity(vocab.len());
        for (piece, id) in vocab {
            let Some(id) = id.as_u64() else {
                continue;
            };
            pieces.insert(id as u32, piece.clone());
        }
        let mut skip = HashSet::from([blank, pad]);
        if let Some(added) = value
            .get("added_tokens")
            .and_then(serde_json::Value::as_array)
        {
            for token in added {
                let (Some(id), Some(content)) = (
                    token.get("id").and_then(serde_json::Value::as_u64),
                    token.get("content").and_then(serde_json::Value::as_str),
                ) else {
                    continue;
                };
                let id = id as u32;
                pieces.insert(id, content.to_string());
                if token
                    .get("special")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    skip.insert(id);
                }
            }
        }
        let space = value
            .get("decoder")
            .and_then(|decoder| decoder.get("replacement"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("▁")
            .to_string();
        Ok(Self {
            pieces,
            skip,
            space,
        })
    }

    /// Decode token ids to text: skip special/blank/pad, turn `▁` into a
    /// space, and drop the one leading space Metaspace adds.
    pub(crate) fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        for id in ids {
            if self.skip.contains(id) {
                continue;
            }
            if let Some(piece) = self.pieces.get(id) {
                text.push_str(piece);
            }
        }
        let text = text.replace(&self.space, " ");
        text.strip_prefix(' ').unwrap_or(&text).to_string()
    }

    /// The piece text for one token, with `▁` already turned into a space.
    fn piece_text(&self, id: u32) -> Option<String> {
        if self.skip.contains(&id) {
            return None;
        }
        self.pieces
            .get(&id)
            .map(|piece| piece.replace(&self.space, " "))
    }

    /// Group emitted tokens into words at spaces and attach timestamps.
    pub(crate) fn words(&self, tokens: &[(u32, usize)]) -> Vec<Word> {
        let pieces = tokens
            .iter()
            .filter_map(|(id, frame)| self.piece_text(*id).map(|text| (text, *frame)))
            .collect::<Vec<_>>();
        group_words(&pieces)
    }
}

/// Group `(text, encoder frame)` pieces into words. A space starts a new word
/// (and is not kept); other characters, including punctuation, stay attached
/// to the word they follow. Word start is its first token's frame time; word
/// end is its last token's frame time plus one frame.
pub(crate) fn group_words(pieces: &[(String, usize)]) -> Vec<Word> {
    let mut words = Vec::new();
    let mut current: Option<(String, usize, usize)> = None;
    for (text, frame) in pieces {
        for character in text.chars() {
            if character == ' ' {
                if let Some((text, start, end)) = current.take() {
                    words.push(word(text, start, end));
                }
            } else {
                let word = current.get_or_insert_with(|| (String::new(), *frame, *frame));
                word.0.push(character);
                word.2 = *frame;
            }
        }
    }
    if let Some((text, start, end)) = current {
        words.push(word(text, start, end));
    }
    words
}

fn word(text: String, start_frame: usize, end_frame: usize) -> Word {
    Word {
        text,
        start: start_frame as f32 * FRAME_SECONDS,
        end: end_frame as f32 * FRAME_SECONDS + FRAME_SECONDS,
    }
}

/// One mel filter bank: `[num_mel_bins, n_fft / 2 + 1]` little-endian f32
/// values from an `.npy` file.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NpyF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NpyF32 {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("Could not read {}", path.display()))?;
        parse_npy_f32(&bytes).with_context(|| format!("Could not parse {}", path.display()))
    }
}

/// Parse a `.npy` v1/v2/v3 file holding little-endian f32 (or f64) data.
/// Only the simple header fields the mel filters use are supported:
/// descriptor, fortran-order flag and shape.
pub(crate) fn parse_npy_f32(bytes: &[u8]) -> anyhow::Result<NpyF32> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..MAGIC.len()] != MAGIC {
        bail!("not a .npy file (bad magic)");
    }
    let major = bytes[6];
    let (header_start, header_len): (usize, usize) = match major {
        1 => (10, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        2 | 3 => {
            if bytes.len() < 12 {
                bail!("truncated .npy header");
            }
            (
                12,
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            )
        }
        other => bail!("unsupported .npy version {other}"),
    };
    let header_end = header_start
        .checked_add(header_len)
        .ok_or_else(|| anyhow::anyhow!("invalid .npy header length"))?;
    let header_bytes = bytes
        .get(header_start..header_end)
        .ok_or_else(|| anyhow::anyhow!("truncated .npy header"))?;
    let header = std::str::from_utf8(header_bytes).context(".npy header is not UTF-8")?;
    let descr = quoted_field(header, "descr")?;
    if header.contains("'fortran_order': True") {
        bail!("Fortran-order .npy arrays are not supported");
    }
    let shape = shape_field(header)?;
    let count = shape.iter().try_fold(1usize, |count, size| {
        count
            .checked_mul(*size)
            .ok_or_else(|| anyhow::anyhow!("invalid .npy shape"))
    })?;
    let data_bytes = bytes
        .get(header_end..)
        .ok_or_else(|| anyhow::anyhow!("truncated .npy data"))?;
    let data = match descr.as_str() {
        "<f4" | "|f4" | "=f4" => {
            if count
                .checked_mul(4)
                .is_none_or(|bytes| data_bytes.len() < bytes)
            {
                bail!("truncated .npy data");
            }
            data_bytes[..count * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()
        }
        "<f8" | "|f8" | "=f8" => {
            if count
                .checked_mul(8)
                .is_none_or(|bytes| data_bytes.len() < bytes)
            {
                bail!("truncated .npy data");
            }
            data_bytes[..count * 8]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|chunk| f64::from_le_bytes(*chunk) as f32)
                .collect()
        }
        other => bail!("unsupported .npy descriptor {other:?}"),
    };
    Ok(NpyF32 { shape, data })
}

/// The string value of a `'key': 'value'` field in an `.npy` header.
fn quoted_field(header: &str, key: &str) -> anyhow::Result<String> {
    let marker = format!("'{key}':");
    let rest = header
        .split_once(&marker)
        .map(|(_, rest)| rest)
        .ok_or_else(|| anyhow::anyhow!(".npy header has no {key}"))?;
    let rest = rest.trim_start();
    let rest = rest
        .strip_prefix('\'')
        .ok_or_else(|| anyhow::anyhow!(".npy {key} is not a string"))?;
    let (value, _) = rest
        .split_once('\'')
        .ok_or_else(|| anyhow::anyhow!(".npy {key} is unterminated"))?;
    Ok(value.to_string())
}

/// The integer shape tuple of an `.npy` header.
fn shape_field(header: &str) -> anyhow::Result<Vec<usize>> {
    let rest = header
        .split_once("'shape':")
        .map(|(_, rest)| rest)
        .ok_or_else(|| anyhow::anyhow!(".npy header has no shape"))?;
    let (_, rest) = rest
        .split_once('(')
        .ok_or_else(|| anyhow::anyhow!(".npy shape is not a tuple"))?;
    let (inner, _) = rest
        .split_once(')')
        .ok_or_else(|| anyhow::anyhow!(".npy shape is unterminated"))?;
    inner
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!(".npy shape has a non-integer dimension {part:?}"))
        })
        .collect()
}

/// Symmetric Hann window of `win` samples, cast to f32 then zero-padded
/// (centered) to `n_fft` samples as f64, exactly like the Python reference.
pub(crate) fn hann_window(n_fft: usize, win: usize) -> Vec<f64> {
    let mut window = vec![0.0; n_fft];
    if win == 0 || n_fft < win {
        return window;
    }
    let left = (n_fft - win) / 2;
    for n in 0..win {
        let value = (0.5
            - 0.5 * (2.0 * PI * n as f64 / (win.saturating_sub(1).max(1)) as f64).cos())
            as f32;
        window[left + n] = f64::from(value);
    }
    window
}

/// Log-mel features: `([1, T, mels] f32, [1, T] i64 mask, T)`.
pub(crate) fn log_mel(
    samples: &[f32],
    config: &Config,
    filters: &NpyF32,
) -> anyhow::Result<(Vec<f32>, Vec<i64>, usize)> {
    let bins = config.n_fft / 2 + 1;
    if filters.shape != [config.num_mel_bins, bins] {
        bail!(
            "mel filters have shape {:?}, expected [{}, {}]",
            filters.shape,
            config.num_mel_bins,
            bins
        );
    }
    if samples.is_empty() {
        bail!("cannot compute mel features of empty audio");
    }
    let mut audio = samples.to_vec();
    if let Some(preemphasis) = config.preemphasis {
        let mut emphasized = Vec::with_capacity(audio.len());
        emphasized.push(audio[0]);
        for index in 1..audio.len() {
            emphasized.push(audio[index] - preemphasis * audio[index - 1]);
        }
        audio = emphasized;
    }
    let window = hann_window(config.n_fft, config.win_length);
    let padding = config.n_fft / 2;
    let mut padded = vec![0.0f32; padding];
    padded.extend_from_slice(&audio);
    padded.extend(std::iter::repeat_n(0.0f32, padding));
    let frames = 1 + audio.len() / config.hop_length;
    let mut mel = vec![0.0f32; frames * config.num_mel_bins];

    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(config.n_fft);
    let mut input = vec![0.0f64; config.n_fft];
    let mut output = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();
    let mut power = vec![0.0f32; bins];
    for block_start in (0..frames).step_by(MEL_BLOCK_FRAMES) {
        let block_end = (block_start + MEL_BLOCK_FRAMES).min(frames);
        for frame in block_start..block_end {
            let offset = frame * config.hop_length;
            for (index, sample) in padded[offset..offset + config.n_fft].iter().enumerate() {
                input[index] = f64::from(*sample) * window[index];
            }
            r2c.process_with_scratch(&mut input, &mut output, &mut scratch)?;
            for (bin, value) in output.iter().enumerate() {
                power[bin] = (value.re * value.re + value.im * value.im) as f32;
            }
            let row = &mut mel[frame * config.num_mel_bins..(frame + 1) * config.num_mel_bins];
            for (mel_index, out) in row.iter_mut().enumerate() {
                let filter = &filters.data[mel_index * bins..(mel_index + 1) * bins];
                let mut sum = 0.0f32;
                for (weight, value) in filter.iter().zip(&power) {
                    sum += weight * value;
                }
                *out = (sum + LOG_ZERO_GUARD).ln();
            }
        }
    }

    let valid = (audio.len() / config.hop_length).min(frames);
    let mut mean = vec![0.0f32; config.num_mel_bins];
    for frame in 0..valid {
        for mel_index in 0..config.num_mel_bins {
            mean[mel_index] += mel[frame * config.num_mel_bins + mel_index];
        }
    }
    let valid_f32 = valid as f32;
    for value in &mut mean {
        *value /= valid_f32;
    }
    let mut variance = vec![0.0f32; config.num_mel_bins];
    for frame in 0..valid {
        for mel_index in 0..config.num_mel_bins {
            let difference = mel[frame * config.num_mel_bins + mel_index] - mean[mel_index];
            variance[mel_index] += difference * difference;
        }
    }
    let divisor = valid.saturating_sub(1).max(1) as f32;
    for value in &mut variance {
        *value /= divisor;
    }
    for frame in 0..frames {
        let row = &mut mel[frame * config.num_mel_bins..(frame + 1) * config.num_mel_bins];
        if frame < valid {
            for (mel_index, value) in row.iter_mut().enumerate() {
                *value = (*value - mean[mel_index]) / (variance[mel_index].sqrt() + NORM_EPS);
            }
        } else {
            row.fill(0.0);
        }
    }
    let mask = (0..frames).map(|frame| i64::from(frame < valid)).collect();
    Ok((mel, mask, frames))
}

/// The Fluid Speech Mini recognizer: four ONNX graphs plus the tokenizer.
pub struct FluidModel {
    config: Config,
    filters: NpyF32,
    tokenizer: Tokenizer,
    frontend: Session,
    encoder: Session,
    decoder: Session,
    joint: Session,
}

impl FluidModel {
    /// Load the model with an explicit intra-op thread count. Production
    /// reaches this through [`Asr::load_fluid`](crate::asr::Asr::load_fluid),
    /// which supplies the app default; the benchmark needs the same four
    /// sessions with a measured 1/2-thread setting.
    pub(crate) fn load_with_threads(dir: &Path, threads: usize) -> anyhow::Result<Self> {
        let config = Config::load(&dir.join("config.json"))?;
        let filters = NpyF32::load(&dir.join("mel_filters.npy"))?;
        let tokenizer = Tokenizer::load(&dir.join("tokenizer.json"), &config)?;
        let env = crate::asr::shared_environment()?;
        let frontend = open_session(&env, &dir.join("frontend.onnx"), threads)?;
        let encoder = open_session(&env, &dir.join("encoder.onnx"), threads)?;
        let decoder = open_session(&env, &dir.join("decoder.onnx"), 1)?;
        let joint = open_session(&env, &dir.join("joint.onnx"), 1)?;
        Ok(Self {
            config,
            filters,
            tokenizer,
            frontend,
            encoder,
            decoder,
            joint,
        })
    }

    /// Transcribe 16 kHz mono audio. Empty or very short audio yields `""`,
    /// and clips longer than one model pass are split at quiet points.
    pub fn transcribe(&mut self, samples: &[f32]) -> anyhow::Result<String> {
        if samples.len() < self.config.hop_length * 2 {
            return Ok(String::new());
        }
        let max_samples =
            (self.config.max_single_pass_seconds * self.config.sampling_rate as f64) as usize;
        if samples.len() <= max_samples {
            return self.transcribe_pass(samples);
        }
        let mut pieces = Vec::new();
        for piece in self.split_long(samples) {
            let text = self.transcribe_pass(&piece)?.trim().to_string();
            if !text.is_empty() {
                pieces.push(text);
            }
        }
        Ok(pieces.join(" "))
    }

    /// Transcribe with word-level timestamps (seconds).
    pub fn transcribe_words(&mut self, samples: &[f32]) -> anyhow::Result<Vec<Word>> {
        if samples.len() < self.config.hop_length * 2 {
            return Ok(Vec::new());
        }
        let max_samples =
            (self.config.max_single_pass_seconds * self.config.sampling_rate as f64) as usize;
        if samples.len() <= max_samples {
            return self.transcribe_words_pass(samples);
        }
        let mut words = Vec::new();
        let mut shift = 0.0f32;
        for piece in self.split_long(samples) {
            let piece_words = self.transcribe_words_pass(&piece)?;
            let end = shift + piece.len() as f32 / self.config.sampling_rate as f32;
            words.extend(piece_words.into_iter().map(|word| Word {
                text: word.text,
                start: word.start + shift,
                end: word.end + shift,
            }));
            shift = end;
        }
        Ok(words)
    }

    fn transcribe_pass(&mut self, samples: &[f32]) -> anyhow::Result<String> {
        let (features, mask, frames) = log_mel(samples, &self.config, &self.filters)?;
        let (encoded, encoded_frames, valid) = self.encode(&features, mask, frames)?;
        let tokens = self.greedy_tdt(&encoded, encoded_frames, valid)?;
        let ids = tokens.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        Ok(self.tokenizer.decode(&ids))
    }

    fn transcribe_words_pass(&mut self, samples: &[f32]) -> anyhow::Result<Vec<Word>> {
        let (features, mask, frames) = log_mel(samples, &self.config, &self.filters)?;
        let (encoded, encoded_frames, valid) = self.encode(&features, mask, frames)?;
        let tokens = self.greedy_tdt(&encoded, encoded_frames, valid)?;
        Ok(self.tokenizer.words(&tokens))
    }

    /// Encoder frames after the three subsampling layers.
    fn subsampled(&self, frames: usize) -> usize {
        subsampled_frames(
            frames,
            self.config.subsampling_kernel,
            self.config.subsampling_stride,
            self.config.subsampling_layers,
        )
    }

    /// Run the split front end in time chunks (one frame of context on each
    /// side), then the encoder. Returns the encoder frames and the valid
    /// frame count.
    fn encode(
        &mut self,
        features: &[f32],
        mask: Vec<i64>,
        frames: usize,
    ) -> anyhow::Result<(Vec<f32>, usize, usize)> {
        let hidden_size = self.config.encoder_hidden_size;
        let mels = self.config.num_mel_bins;
        let total = self.subsampled(frames);
        let factor = self
            .config
            .subsampling_stride
            .pow(self.config.subsampling_layers as u32);
        let step = self.config.frontend_chunk_frames;
        let mut hidden: Vec<f32> = Vec::with_capacity(total * hidden_size);
        for start_frame in (0..total).step_by(step.max(1)) {
            let end_frame = (start_frame + step).min(total);
            let start_sample = if start_frame == 0 {
                0
            } else {
                factor * (start_frame - 1)
            };
            let end_sample = (factor * (end_frame + 1)).min(frames);
            let feature_slice = &features[start_sample * mels..end_sample * mels];
            let mask_slice = mask[start_sample..end_sample].to_vec();
            let length = end_sample - start_sample;
            let outputs = self.frontend.run(parakeet_rs::ort::inputs![
                "input_features" => Tensor::from_array((
                    [1usize, length, mels],
                    feature_slice.to_vec(),
                ))?,
                "attention_mask" => Tensor::from_array((
                    [1usize, length],
                    mask_slice,
                ))?,
            ])?;
            let (shape, data) = outputs["hidden_states"].try_extract_tensor::<f32>()?;
            let frames_out = shape.as_ref()[1] as usize;
            let skip = usize::from(start_frame > 0);
            let take = end_frame - start_frame;
            if skip + take > frames_out {
                bail!(
                    "front end returned {frames_out} frames, expected at least {}",
                    skip + take
                );
            }
            let begin = skip * hidden_size;
            hidden.extend_from_slice(&data[begin..begin + take * hidden_size]);
        }
        if hidden.len() != total * hidden_size {
            bail!(
                "front end produced {} hidden values, expected {}",
                hidden.len(),
                total * hidden_size
            );
        }
        let valid = self.subsampled(mask.iter().filter(|value| **value != 0).count());
        let outputs = self.encoder.run(parakeet_rs::ort::inputs![
            "hidden_states" => Tensor::from_array((
                [1usize, total, hidden_size],
                hidden,
            ))?,
            "length" => Tensor::from_array(([1usize], vec![valid as i64]))?,
        ])?;
        let (shape, data) = outputs["encoder_out"].try_extract_tensor::<f32>()?;
        let encoded_frames = shape.as_ref()[1] as usize;
        Ok((data.to_vec(), encoded_frames, valid))
    }

    /// Greedy token-and-duration decoding: returns each emitted token with the
    /// encoder frame it was read from.
    fn greedy_tdt(
        &mut self,
        encoded: &[f32],
        frames: usize,
        valid: usize,
    ) -> anyhow::Result<Vec<(u32, usize)>> {
        if frames == 0 || valid == 0 {
            return Ok(Vec::new());
        }
        let config = &self.config;
        let hidden = config.decoder_hidden_size;
        let layers = config.decoder_layers;
        let state = vec![0.0f32; layers * hidden];
        let blank = config.blank_token_id;
        let outputs = self.decoder.run(parakeet_rs::ort::inputs![
            "token" => Tensor::from_array(([1usize, 1usize], vec![i64::from(blank)]))?,
            "h_in" => Tensor::from_array(([layers, 1usize, hidden], state.clone()))?,
            "c_in" => Tensor::from_array(([layers, 1usize, hidden], state))?,
        ])?;
        let (_, decoder_out) = outputs["decoder_out"].try_extract_tensor::<f32>()?;
        let (_, h_out) = outputs["h_out"].try_extract_tensor::<f32>()?;
        let (_, c_out) = outputs["c_out"].try_extract_tensor::<f32>()?;
        let mut decoder_out = decoder_out.to_vec();
        let mut h = h_out.to_vec();
        let mut c = c_out.to_vec();
        drop(outputs);

        let budget = config.max_symbols_per_step.saturating_mul(frames) - 1;
        let mut tokens: Vec<(u32, usize)> = Vec::new();
        let mut position = 0usize;
        loop {
            let frame = position.min(frames - 1);
            let frame_values = &encoded
                [frame * config.encoder_hidden_size..(frame + 1) * config.encoder_hidden_size];
            let outputs = self.joint.run(parakeet_rs::ort::inputs![
                "encoder_frame" => Tensor::from_array((
                    [1usize, config.encoder_hidden_size],
                    frame_values.to_vec(),
                ))?,
                "decoder_out" => Tensor::from_array((
                    [1usize, hidden],
                    decoder_out.clone(),
                ))?,
            ])?;
            let (_, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
            let token = argmax(&logits[..config.vocab_size]) as u32;
            let duration_index = argmax(&logits[config.vocab_size..]);
            let step = config.durations.get(duration_index).copied().unwrap_or(0);
            let step = if token == blank && step == 0 { 1 } else { step };
            tokens.push((token, frame));
            position += step;
            if position >= valid || tokens.len() >= budget || config.eos_token_id == Some(token) {
                return Ok(tokens);
            }
            if token != blank {
                let outputs = self.decoder.run(parakeet_rs::ort::inputs![
                    "token" => Tensor::from_array(([1usize, 1usize], vec![i64::from(token)]))?,
                    "h_in" => Tensor::from_array(([layers, 1usize, hidden], h.clone()))?,
                    "c_in" => Tensor::from_array(([layers, 1usize, hidden], c.clone()))?,
                ])?;
                let (_, new_out) = outputs["decoder_out"].try_extract_tensor::<f32>()?;
                let (_, new_h) = outputs["h_out"].try_extract_tensor::<f32>()?;
                let (_, new_c) = outputs["c_out"].try_extract_tensor::<f32>()?;
                decoder_out = new_out.to_vec();
                h = new_h.to_vec();
                c = new_c.to_vec();
            }
        }
    }

    /// Cut audio longer than one pass at quiet points: about 30 s per piece,
    /// searching ±5 s for the lowest energy.
    fn split_long(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let rate = self.config.sampling_rate;
        let hop = (rate / ENERGY_WINDOWS_PER_SECOND).max(1);
        let mut energy = Vec::with_capacity(samples.len().div_ceil(hop));
        let mut start = 0;
        while start < samples.len() {
            let end = (start + hop).min(samples.len());
            let sum: f64 = samples[start..end]
                .iter()
                .map(|sample| (*sample as f64) * (*sample as f64))
                .sum();
            energy.push((sum / hop as f64).sqrt());
            start += hop;
        }
        let smooth = smooth_energy(&energy);
        let max_samples = (self.config.max_single_pass_seconds * rate as f64) as usize;
        let mut pieces = Vec::new();
        let mut start = 0;
        while samples.len() - start > max_samples {
            let lo = (start
                + ((SPLIT_TARGET_SECONDS - SPLIT_SEARCH_SECONDS / 2.0) * rate as f64) as usize)
                / hop;
            let hi = ((start
                + ((SPLIT_TARGET_SECONDS + SPLIT_SEARCH_SECONDS / 2.0) * rate as f64) as usize)
                / hop)
                .min(smooth.len());
            let quietest = smooth
                .get(lo..hi)
                .and_then(|window| {
                    window
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(index, _)| index)
                })
                .unwrap_or(0);
            let cut = (lo + quietest) * hop;
            if cut <= start || cut >= samples.len() {
                break;
            }
            pieces.push(samples[start..cut].to_vec());
            start = cut;
        }
        pieces.push(samples[start..].to_vec());
        pieces
    }
}

/// The front end's subsampling formula, repeated once per conv layer
/// (`floor((frames + pad) / stride + 1)` with `pad = kernel/2*2 - kernel`).
pub(crate) fn subsampled_frames(
    mut frames: usize,
    kernel: usize,
    stride: usize,
    layers: usize,
) -> usize {
    let pad = (kernel as i64 - 1) / 2 * 2 - kernel as i64;
    for _ in 0..layers {
        frames = (((frames as i64 + pad) / stride as i64) + 1) as usize;
    }
    frames
}

/// The moving average NumPy computes for `convolve(energy, ones(10)/10,
/// same)`: zero padding outside the signal, centered on each sample.
fn smooth_energy(energy: &[f64]) -> Vec<f64> {
    const KERNEL: usize = 10;
    // `same` starts at index (KERNEL - 1) / 2 of the full convolution, so the
    // window around output i covers input i-5..=i+4.
    const BEFORE: usize = KERNEL - 1 - (KERNEL - 1) / 2;
    const AFTER: usize = (KERNEL - 1) / 2;
    let weight = 1.0 / KERNEL as f64;
    (0..energy.len())
        .map(|index| {
            let first = index.saturating_sub(BEFORE);
            let last = (index + AFTER).min(energy.len() - 1);
            energy[first..=last].iter().sum::<f64>() * weight
        })
        .collect()
}

/// Index of the largest value (first on ties), like NumPy's `argmax`.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best = index;
            best_value = *value;
        }
    }
    best
}

/// Open one ONNX Runtime session: CPU, `threads` intra-op threads, and the
/// CPU memory arena and memory pattern disabled (the conv front end's
/// activations are freed between runs instead of staying in an arena).
fn open_session(env: &Environment, path: &Path, threads: usize) -> anyhow::Result<Session> {
    let mut builder = Session::builder(env)
        .with_context(|| {
            format!(
                "Could not create an ONNX Runtime session for {}",
                path.display()
            )
        })?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| anyhow::anyhow!("Could not configure {}: {err:?}", path.display()))?
        .with_intra_threads(threads)
        .map_err(|err| anyhow::anyhow!("Could not configure {}: {err:?}", path.display()))?;
    ortsys![unsafe DisableCpuMemArena(builder.ptr_mut())?];
    ortsys![unsafe DisableMemPattern(builder.ptr_mut())?];
    builder
        .commit_from_file(path)
        .with_context(|| format!("Could not load {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Config {
        Config::from_json(
            r#"{
                "vocab_size": 8193,
                "blank_token_id": 8192,
                "pad_token_id": 2,
                "eos_token_id": 3,
                "durations": [0, 1, 2, 3, 4],
                "max_symbols_per_step": 10,
                "decoder_layers": 2,
                "decoder_hidden_size": 640,
                "encoder_hidden_size": 1024,
                "sampling_rate": 16000,
                "n_fft": 512,
                "hop_length": 160,
                "win_length": 400,
                "preemphasis": 0.97,
                "num_mel_bins": 4,
                "max_single_pass_seconds": 40.0,
                "subsampling_kernel": 3,
                "subsampling_stride": 2,
                "subsampling_layers": 3,
                "frontend_chunk_frames": 50
            }"#,
        )
        .unwrap()
    }

    /// Build an `.npy` v1 file header plus f32 payload.
    fn npy_bytes(shape: &[usize], data: &[f32]) -> Vec<u8> {
        let shape_text = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape_text},), }}");
        let mut header_bytes = header.into_bytes();
        // Real .npy pads the header so the data starts 64-byte aligned.
        while (10 + header_bytes.len() + 1) % 64 != 0 {
            header_bytes.push(b' ');
        }
        header_bytes.push(b'\n');
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1);
        bytes.push(0);
        bytes.extend_from_slice(&(header_bytes.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn parses_npy_header_and_f32_payload() {
        let bytes = npy_bytes(&[2, 3], &[1.0, 2.5, -3.0, 4.0, 5.0, 6.0]);
        let parsed = parse_npy_f32(&bytes).unwrap();
        assert_eq!(parsed.shape, vec![2, 3]);
        assert_eq!(parsed.data, vec![1.0, 2.5, -3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_non_npy_and_truncated_input() {
        assert!(parse_npy_f32(b"not a numpy file").is_err());
        let bytes = npy_bytes(&[1], &[1.0]);
        assert!(parse_npy_f32(&bytes[..20]).is_err());
    }

    #[test]
    fn rejects_npy_shape_whose_size_overflows() {
        // `count * 4` would overflow: the header lies about the payload size.
        let bytes = npy_bytes(&[usize::MAX / 2], &[1.0]);
        assert!(parse_npy_f32(&bytes).is_err());
    }

    #[test]
    fn rejects_zero_config_values() {
        let json = r#"{"vocab_size": 8193, "blank_token_id": 8192, "pad_token_id": 2,
            "durations": [0, 1, 2, 3, 4], "max_symbols_per_step": 0, "decoder_layers": 2,
            "decoder_hidden_size": 640, "encoder_hidden_size": 1024, "sampling_rate": 16000,
            "n_fft": 512, "hop_length": 160, "win_length": 400, "preemphasis": 0.97, "num_mel_bins": 128,
            "max_single_pass_seconds": 40.0, "subsampling_kernel": 3, "subsampling_stride": 2,
            "subsampling_layers": 3, "frontend_chunk_frames": 50}"#;
        let err = Config::from_json(json).unwrap_err();
        assert!(
            format!("{err:#}").contains("max_symbols_per_step"),
            "unexpected: {err:#}"
        );
        assert!(
            Config::from_json(&json.replace(
                "\"max_symbols_per_step\": 0",
                "\"max_symbols_per_step\": 10"
            ))
            .is_ok()
        );
    }

    #[test]
    fn hann_window_is_symmetric_and_centered() {
        let window = hann_window(512, 400);
        assert_eq!(window.len(), 512);
        // Zero padding on both sides: 56 samples each.
        assert!(window[..56].iter().all(|value| *value == 0.0));
        assert!(window[456..].iter().all(|value| *value == 0.0));
        assert!(window[56].abs() < 1e-9);
        assert!(window[455].abs() < 1e-9);
        assert!((window[255] - 1.0).abs() < 1e-4);
        assert!((window[56 + 200] - window[455 - 200]).abs() < 1e-9);
    }

    #[test]
    fn mel_features_have_expected_shape_and_are_finite() {
        let config = tiny_config();
        let filters = NpyF32 {
            shape: vec![config.num_mel_bins, config.n_fft / 2 + 1],
            data: vec![1.0; config.num_mel_bins * (config.n_fft / 2 + 1)],
        };
        let samples = (0..1000)
            .map(|index| (index as f32 * 0.1).sin() * 0.5)
            .collect::<Vec<_>>();
        let (mel, mask, frames) = log_mel(&samples, &config, &filters).unwrap();
        assert_eq!(frames, 1 + samples.len() / config.hop_length);
        assert_eq!(mel.len(), frames * config.num_mel_bins);
        assert_eq!(mask.len(), frames);
        assert!(mel.iter().all(|value| value.is_finite()));
        assert_eq!(mask.iter().filter(|value| **value != 0).count(), 6);
        // Frames past the valid count are zeroed.
        assert!(
            mel[(frames - 1) * config.num_mel_bins..]
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    #[test]
    fn tokenizer_skips_special_tokens_and_decodes_metaspace() {
        let json = r#"{
            "model": {"vocab": {"▁hello": 0, "▁world": 1, "!": 4, "<blank>": 8192}},
            "added_tokens": [
                {"id": 8192, "content": "<blank>", "special": true},
                {"id": 2, "content": "<pad>", "special": true},
                {"id": 3, "content": "<eos>", "special": true}
            ],
            "decoder": {"replacement": "▁"}
        }"#;
        let tokenizer = Tokenizer::from_json(json, 8192, 2).unwrap();
        assert_eq!(tokenizer.decode(&[0, 1, 4]), "hello world!");
        assert_eq!(tokenizer.decode(&[8192, 0, 2, 3, 1]), "hello world");
        assert_eq!(tokenizer.decode(&[]), "");
    }

    #[test]
    fn words_group_at_spaces_with_frame_timestamps() {
        let pieces = vec![
            (" He".to_string(), 10),
            ("llo".to_string(), 10),
            (",".to_string(), 11),
            (" world".to_string(), 20),
        ];
        let words = group_words(&pieces);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Hello,");
        assert!((words[0].start - 0.8).abs() < 1e-6);
        assert!((words[0].end - 0.96).abs() < 1e-6);
        assert_eq!(words[1].text, "world");
        assert!((words[1].start - 1.6).abs() < 1e-6);
        assert!((words[1].end - 1.68).abs() < 1e-6);
    }

    #[test]
    fn group_words_ignores_empty_and_leading_space_pieces() {
        let pieces = vec![
            (" ".to_string(), 0),
            ("you".to_string(), 2),
            (",".to_string(), 2),
            (" ".to_string(), 3),
            ("ask".to_string(), 4),
        ];
        let words = group_words(&pieces);
        assert_eq!(
            words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            vec!["you,", "ask"]
        );
    }

    #[test]
    fn subsampling_matches_the_front_end_formula() {
        assert_eq!(subsampled_frames(1, 3, 2, 3), 1);
        assert_eq!(subsampled_frames(100, 3, 2, 3), 13);
        assert_eq!(subsampled_frames(101, 3, 2, 3), 13);
        assert_eq!(subsampled_frames(108, 3, 2, 3), 14);
    }

    #[test]
    fn smooth_energy_matches_numpy_same_mode() {
        let energy = vec![1.0, 2.0, 3.0, 4.0];
        let smooth = smooth_energy(&energy);
        // NumPy: convolve([1,2,3,4], [0.1;10], 'same') is [1,1,1,1].
        for value in smooth {
            assert!((value - 1.0).abs() < 1e-9);
        }
    }
}
