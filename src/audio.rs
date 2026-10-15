use std::path::Path;

use anyhow::Context;
use hound::SampleFormat;
use rubato::audioadapter::Adapter;
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Resampler};

pub(crate) const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Input chunk size (frames) for the FFT resampler, one-shot and streaming.
const RESAMPLE_CHUNK: usize = 1024;

/// Load a WAV file as mono 16 kHz f32 samples in [-1, 1].
pub fn load_wav_16k_mono(path: &Path) -> anyhow::Result<Vec<f32>> {
    load_inner(path).with_context(|| format!("Could not read WAV {}", path.display()))
}

/// Load a WAV file as mono f32 samples in [-1, 1] at the file's own sample
/// rate, for callers (like the streaming feeder) that resample themselves.
pub(crate) fn load_wav_mono(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    load_native(path).with_context(|| format!("Could not read WAV {}", path.display()))
}

fn load_inner(path: &Path) -> anyhow::Result<Vec<f32>> {
    let (mono, sample_rate) = load_native(path)?;
    if sample_rate == TARGET_SAMPLE_RATE || mono.is_empty() {
        return Ok(mono);
    }
    resample(&mono, sample_rate)
        .with_context(|| format!("Could not resample audio from {sample_rate} Hz to 16000 Hz"))
}

fn load_native(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels == 0 {
        anyhow::bail!("unsupported WAV format: 0 channels");
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<_, _>>()?,
        (SampleFormat::Int, bits @ (8 | 16 | 24 | 32)) => {
            let divisor = 2f32.powi(i32::from(bits) - 1);
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|v| v as f32 / divisor))
                .collect::<Result<_, _>>()?
        }
        (format, bits) => {
            anyhow::bail!("unsupported WAV sample format: {format:?} {bits}-bit");
        }
    };
    let channels = usize::from(spec.channels);
    Ok((crate::mic::downmix(&samples, channels), spec.sample_rate))
}

pub(crate) fn resample(mono: &[f32], sample_rate: u32) -> anyhow::Result<Vec<f32>> {
    let mut resampler = Fft::<f32>::new(
        sample_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        RESAMPLE_CHUNK,
        1,
        FixedSync::Input,
    )?;
    let channel = mono.to_vec();
    let input = SequentialSliceOfVecs::new(std::slice::from_ref(&channel), 1, mono.len())?;
    let output = resampler.process_all(&input, mono.len(), None)?;
    let frames = output.frames();
    let mut data = output.take_data();
    data.truncate(frames);
    Ok(data)
}

/// Stateful resampler to 16 kHz for the streaming feeder: `push` device-rate
/// chunks as they arrive, `finish` flushes the tail. Output matches the
/// one-shot [`resample`] exactly (same resampler, startup delay trimmed,
/// exact end length), so segmenting the live stream equals segmenting a file.
pub(crate) struct StreamResampler {
    inner: Fft<f32>,
    from_rate: u32,
    input_chunk: usize,
    holdback: usize,
    delay_left: usize,
    input_buf: Vec<f32>,
    output_buf: Vec<f32>,
    total_in: u64,
    emitted: u64,
    finished: bool,
}

impl StreamResampler {
    /// New streaming resampler from `from_rate` Hz to 16 kHz.
    pub(crate) fn new(from_rate: u32) -> anyhow::Result<Self> {
        if from_rate == 0 {
            anyhow::bail!("Invalid sample rate 0 Hz");
        }
        let inner = Fft::<f32>::new(
            from_rate as usize,
            TARGET_SAMPLE_RATE as usize,
            RESAMPLE_CHUNK,
            1,
            FixedSync::Input,
        )
        .map_err(|err| {
            anyhow::anyhow!("Could not create a resampler from {from_rate} Hz to 16000 Hz: {err}")
        })?;
        let input_chunk = inner.input_frames_next();
        let delay = inner.output_delay();
        // Hold back the delay plus one worst-case chunk so `finish` can trim
        // the end to the exact length.
        let holdback = delay + inner.output_frames_max();
        Ok(Self {
            inner,
            from_rate,
            input_chunk,
            holdback,
            delay_left: delay,
            input_buf: Vec::new(),
            output_buf: Vec::new(),
            total_in: 0,
            emitted: 0,
            finished: false,
        })
    }

    /// Push device-rate samples; returns the 16 kHz output ready so far.
    /// Output lags the input by a small hold-back window (see `finish`).
    pub(crate) fn push(&mut self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        if self.finished {
            anyhow::bail!("audio pushed to the resampler after finish");
        }
        debug_assert_eq!(self.inner.input_frames_next(), self.input_chunk);
        self.total_in += input.len() as u64;
        self.input_buf.extend_from_slice(input);
        while self.input_buf.len() >= self.input_chunk {
            let chunk: Vec<f32> = self.input_buf.drain(..self.input_chunk).collect();
            self.process_chunk(chunk)?;
        }
        Ok(self.take_ready())
    }

    /// Flush the tail: returns the remaining 16 kHz output, so the total
    /// output length equals the one-shot result exactly. Idempotent.
    pub(crate) fn finish(&mut self) -> anyhow::Result<Vec<f32>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        if self.total_in == 0 {
            return Ok(Vec::new());
        }
        let expected = (self.inner.resample_ratio() * self.total_in as f64).ceil() as u64;
        // Pad the input with zeros to full chunks and process it all.
        let remainder = self.input_buf.len() % self.input_chunk;
        if remainder > 0 {
            self.input_buf
                .resize(self.input_buf.len() + (self.input_chunk - remainder), 0.0);
        }
        while self.input_buf.len() >= self.input_chunk {
            let chunk: Vec<f32> = self.input_buf.drain(..self.input_chunk).collect();
            self.process_chunk(chunk)?;
        }
        // Pump zero chunks until the delay plus the exact remainder is
        // buffered (the resampler emits indefinitely, so this terminates).
        let want = self.delay_left as u64 + (expected - self.emitted);
        while (self.output_buf.len() as u64) < want {
            self.process_chunk(vec![0.0; self.input_chunk])?;
        }
        let drop = self.delay_left.min(self.output_buf.len());
        self.output_buf.drain(..drop);
        self.delay_left = 0;
        let take = (expected - self.emitted) as usize;
        let out: Vec<f32> = self.output_buf.drain(..take).collect();
        self.emitted += out.len() as u64;
        self.output_buf.clear();
        Ok(out)
    }

    /// Resample one full input chunk into the output buffer.
    fn process_chunk(&mut self, chunk: Vec<f32>) -> anyhow::Result<()> {
        let input = SequentialSliceOfVecs::new(std::slice::from_ref(&chunk), 1, chunk.len())
            .map_err(|err| {
                anyhow::anyhow!(
                    "Could not resample audio from {} Hz to 16000 Hz: {err}",
                    self.from_rate
                )
            })?;
        let output = self.inner.process(&input, None).map_err(|err| {
            anyhow::anyhow!(
                "Could not resample audio from {} Hz to 16000 Hz: {err}",
                self.from_rate
            )
        })?;
        let frames = output.frames();
        let mut data = output.take_data();
        data.truncate(frames);
        self.output_buf.extend(data);
        Ok(())
    }

    /// Drop the startup delay from the front, then emit everything except
    /// the hold-back tail (kept so `finish` can trim the end exactly).
    fn take_ready(&mut self) -> Vec<f32> {
        let drop = self.delay_left.min(self.output_buf.len());
        self.output_buf.drain(..drop);
        self.delay_left -= drop;
        let emit = self.output_buf.len().saturating_sub(self.holdback);
        let out: Vec<f32> = self.output_buf.drain(..emit).collect();
        self.emitted += out.len() as u64;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn fresh_path(tag: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "dictate-test-audio-{}-{tag}-{n}.wav",
            std::process::id()
        ))
    }

    fn cleanup(path: &Path) {
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn stereo_44k_resamples_to_16k_mono() {
        let path = fresh_path("stereo44");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..44_100 {
            let v = (i * 7 % 2000) as i16 - 1000;
            writer.write_sample(v).unwrap();
            writer.write_sample(-v).unwrap();
        }
        writer.finalize().unwrap();

        let mono = load_wav_16k_mono(&path).unwrap();
        cleanup(&path);
        assert!(
            (15_840..=16_160).contains(&mono.len()),
            "unexpected length: {}",
            mono.len()
        );
        // Left and right cancel out, so correct channel averaging yields ~silence.
        let peak = mono.iter().fold(0f32, |max, s| max.max(s.abs()));
        assert!(peak < 1e-4, "unexpected peak: {peak}");
    }

    #[test]
    fn mono_16k_f32_passes_through_unchanged() {
        let path = fresh_path("mono16");
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32) / 1600.0 * 2.0 - 1.0).collect();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for s in &samples {
            writer.write_sample(*s).unwrap();
        }
        writer.finalize().unwrap();

        let mono = load_wav_16k_mono(&path).unwrap();
        cleanup(&path);
        assert_eq!(mono, samples);
    }

    /// 48 kHz test signal with an abrupt onset and a sweep (stresses
    /// chunk edges): 0.5 s silence, then 200→3000 Hz at amplitude 0.5.
    fn sweep_48k(secs: f32) -> Vec<f32> {
        let len = (secs * 48_000.0) as usize;
        let mut phase = 0.0f32;
        (0..len)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                if t < 0.5 {
                    return 0.0;
                }
                let freq = 200.0 + (t - 0.5) * 1500.0;
                phase += freq * 2.0 * std::f32::consts::PI / 48_000.0;
                0.5 * phase.sin()
            })
            .collect()
    }

    fn stream_all(from_rate: u32, input: &[f32], push_len: usize) -> Vec<f32> {
        let mut resampler = StreamResampler::new(from_rate).unwrap();
        let mut out = Vec::new();
        for piece in input.chunks(push_len) {
            out.extend(resampler.push(piece).unwrap());
        }
        out.extend(resampler.finish().unwrap());
        out
    }

    #[test]
    fn streaming_resampler_matches_one_shot() {
        // Lengths around the chunk size, the feeder cadence and a long clip.
        for len in [0, 1, 100, 1023, 1024, 1025, 4800, 110_400] {
            let input = &sweep_48k(2.5)[..len.min(120_000)];
            let expected = resample(input, 48_000).unwrap();
            for push_len in [4800, 997] {
                let got = stream_all(48_000, input, push_len);
                assert_eq!(
                    got.len(),
                    expected.len(),
                    "length must match one-shot (len {len}, push {push_len})"
                );
                let max_diff = got
                    .iter()
                    .zip(expected.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_diff < 1e-6,
                    "streaming output must match one-shot (len {len}, push {push_len}): max diff {max_diff}"
                );
            }
        }
    }

    #[test]
    fn streaming_resampler_rejects_rate_zero() {
        let Err(err) = StreamResampler::new(0) else {
            panic!("sample rate 0 must fail");
        };
        assert!(format!("{err:#}").contains('0'), "unexpected: {err:#}");
    }

    #[test]
    fn streaming_resampler_finish_is_idempotent_but_push_after_finish_fails() {
        let mut resampler = StreamResampler::new(48_000).unwrap();
        let input = sweep_48k(0.2);
        let mut out = Vec::new();
        for piece in input.chunks(1000) {
            out.extend(resampler.push(piece).unwrap());
        }
        out.extend(resampler.finish().unwrap());
        assert_eq!(resampler.finish().unwrap(), Vec::<f32>::new());
        assert!(resampler.push(&[0.0; 100]).is_err());
        assert_eq!(out.len(), resample(&input, 48_000).unwrap().len());
    }

    #[test]
    fn non_wav_error_names_the_file() {
        let path = fresh_path("notwav");
        std::fs::write(&path, b"this is not a wav file").unwrap();
        let err = load_wav_16k_mono(&path).unwrap_err();
        cleanup(&path);
        let msg = format!("{err:#}");
        assert!(msg.contains("Could not read WAV"), "unexpected: {msg}");
        assert!(
            msg.contains(&path.display().to_string()),
            "unexpected: {msg}"
        );
    }
}
