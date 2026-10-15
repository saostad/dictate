use std::path::Path;

use anyhow::Context;
use hound::SampleFormat;
use rubato::audioadapter::Adapter;
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Resampler};

pub(crate) const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Load a WAV file as mono 16 kHz f32 samples in [-1, 1].
pub fn load_wav_16k_mono(path: &Path) -> anyhow::Result<Vec<f32>> {
    load_inner(path).with_context(|| format!("Could not read WAV {}", path.display()))
}

fn load_inner(path: &Path) -> anyhow::Result<Vec<f32>> {
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
    let mono = crate::mic::downmix(&samples, channels);
    if spec.sample_rate == TARGET_SAMPLE_RATE || mono.is_empty() {
        return Ok(mono);
    }
    resample(&mono, spec.sample_rate).with_context(|| {
        format!(
            "Could not resample audio from {} Hz to 16000 Hz",
            spec.sample_rate
        )
    })
}

pub(crate) fn resample(mono: &[f32], sample_rate: u32) -> anyhow::Result<Vec<f32>> {
    let mut resampler = Fft::<f32>::new(
        sample_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        1024,
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
