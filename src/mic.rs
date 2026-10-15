use std::sync::{Arc, Mutex};

use anyhow::Context;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample};

/// 16 kHz mono samples below this count are too short to transcribe (0.3 s).
pub(crate) const MIN_RECORDING_SAMPLES: usize = 4800;

/// True when `samples_16k` holds less than 0.3 s of 16 kHz mono audio.
pub(crate) fn recording_too_short(samples_16k: &[f32]) -> bool {
    samples_16k.len() < MIN_RECORDING_SAMPLES
}

/// Average interleaved frames to mono. Returns an empty vec for
/// `channels == 0` instead of panicking.
pub(crate) fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels == 0 {
        return Vec::new();
    }
    interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

pub(crate) struct Recorder {
    stream: cpal::Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    stream_error: Arc<Mutex<Option<String>>>,
    sample_rate: u32,
}

impl Recorder {
    /// Start recording from the default input device.
    pub(crate) fn start() -> anyhow::Result<Self> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or_else(|| {
            anyhow::anyhow!(
                "No microphone found: the system has no default input device. Check your sound settings."
            )
        })?;
        let supported = device
            .default_input_config()
            .context("Could not get the microphone's default input config")?;
        let channels = usize::from(supported.channels());
        let sample_rate = supported.sample_rate();
        let config = supported.config();
        let samples = Arc::new(Mutex::new(Vec::with_capacity((sample_rate as usize) * 60)));
        let stream_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let stream = match supported.sample_format() {
            SampleFormat::F32 => {
                build_stream::<f32>(&device, &config, channels, &samples, &stream_error)
            }
            SampleFormat::I16 => {
                build_stream::<i16>(&device, &config, channels, &samples, &stream_error)
            }
            SampleFormat::U16 => {
                build_stream::<u16>(&device, &config, channels, &samples, &stream_error)
            }
            format => {
                anyhow::bail!(
                    "Unsupported microphone sample format: {format:?} (only F32, I16 and U16 are supported)"
                )
            }
        }
        .context("Could not open the microphone input stream")?;
        stream
            .play()
            .context("Could not start the microphone stream")?;
        Ok(Self {
            stream,
            samples,
            stream_error,
            sample_rate,
        })
    }

    /// Device sample rate of this recording.
    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Shared capture buffer for the streaming feeder thread.
    /// The `Recorder` (with its non-`Send` stream) stays on the event-loop
    /// thread; only this `Arc` crosses threads.
    pub(crate) fn buffer_handle(&self) -> Arc<Mutex<Vec<f32>>> {
        self.samples.clone()
    }

    /// First stream error text, if any. Never panics on a poisoned mutex.
    pub(crate) fn stream_error(&self) -> Option<String> {
        stream_error_text(&self.stream_error)
    }

    /// Drain the mono device-rate samples captured since the last call.
    /// Empties the buffer so streaming never grows memory. Never panics.
    pub(crate) fn take_new(&self) -> Vec<f32> {
        take_buffer(&self.samples)
    }

    /// Stop recording and return the audio as 16 kHz mono samples.
    pub(crate) fn stop(self) -> anyhow::Result<Vec<f32>> {
        // Drain most samples before stopping, then the trailing few after
        // the stream is dropped, so no sample is lost.
        let mut mono = self.take_new();
        drop(self.stream);
        if let Some(text) = stream_error_text(&self.stream_error) {
            anyhow::bail!("Microphone stream error: {text}");
        }
        mono.extend(take_buffer(&self.samples));
        if mono.is_empty() || self.sample_rate == crate::audio::TARGET_SAMPLE_RATE {
            return Ok(mono);
        }
        crate::audio::resample(&mono, self.sample_rate).with_context(|| {
            format!(
                "Could not resample microphone audio from {} Hz to 16000 Hz",
                self.sample_rate
            )
        })
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    samples: &Arc<Mutex<Vec<f32>>>,
    stream_error: &Arc<Mutex<Option<String>>>,
) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let data_samples = samples.clone();
    let data_errors = stream_error.clone();
    let callback_errors = stream_error.clone();
    device.build_input_stream(
        *config,
        move |data: &[T], _: &_| push_samples(data, channels, &data_samples, &data_errors),
        move |err| store_stream_error(&callback_errors, err.to_string()),
        None,
    )
}

/// Convert one input callback's worth of interleaved samples to f32, downmix
/// to mono and append to the shared buffer. Never panics: a poisoned buffer
/// lock is reported through the shared error slot instead.
fn push_samples<T>(
    data: &[T],
    channels: usize,
    samples: &Mutex<Vec<f32>>,
    stream_error: &Mutex<Option<String>>,
) where
    T: cpal::Sample,
    f32: FromSample<T>,
{
    let interleaved: Vec<f32> = data.iter().map(|&s| f32::from_sample(s)).collect();
    let mono = downmix(&interleaved, channels);
    match samples.lock() {
        Ok(mut buffer) => buffer.extend_from_slice(&mono),
        Err(_) => store_stream_error(
            stream_error,
            "microphone sample buffer lock was poisoned".to_string(),
        ),
    }
}

/// Record the first stream error. Never panics, even on a poisoned mutex.
fn store_stream_error(slot: &Mutex<Option<String>>, text: String) {
    let mut guard = match slot.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if guard.is_none() {
        *guard = Some(text);
    }
}

fn stream_error_text(slot: &Mutex<Option<String>>) -> Option<String> {
    match slot.lock() {
        Ok(guard) => guard.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    }
}

/// Drain a shared capture buffer, leaving it empty. Never panics, even on a
/// poisoned mutex.
pub(crate) fn take_buffer(buffer: &Mutex<Vec<f32>>) -> Vec<f32> {
    match buffer.lock() {
        Ok(mut guard) => std::mem::take(&mut *guard),
        Err(poisoned) => std::mem::take(&mut *poisoned.into_inner()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_stereo_averages_channels() {
        let mono = downmix(&[1.0, -1.0, 0.5, 0.5, 0.0, 1.0], 2);
        assert_eq!(mono, vec![0.0, 0.5, 0.5]);
    }

    #[test]
    fn downmix_mono_passes_through() {
        let samples = vec![0.25, -0.5, 1.0];
        assert_eq!(downmix(&samples, 1), samples);
    }

    #[test]
    fn too_short_threshold_is_300_ms() {
        assert!(recording_too_short(&[]));
        assert!(recording_too_short(&vec![0.0; MIN_RECORDING_SAMPLES - 1]));
        assert!(!recording_too_short(&vec![0.0; MIN_RECORDING_SAMPLES]));
    }
}
