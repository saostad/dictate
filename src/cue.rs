//! Short sound cues when recording starts and stops.
//!
//! The daemon plays a tone on a successful start (880 Hz) and stop (660 Hz),
//! each 70 ms with a 5 ms fade in/out so there is no click. `DICTATE_SOUNDS=0`
//! disables the cues. Playback runs on a short-lived thread (following
//! `cpal`'s `beep` example) so the event loop never blocks, and failures only
//! warn once per run: a broken speaker must never stop dictation.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

/// Start cue: 880 Hz.
pub(crate) const START_FREQ_HZ: f32 = 880.0;
/// Stop cue: 660 Hz.
pub(crate) const STOP_FREQ_HZ: f32 = 660.0;
/// Length of each cue.
pub(crate) const CUE_DURATION_MS: u32 = 70;
/// Linear fade in/out at both ends of the tone, so there is no click.
const FADE_MS: u32 = 5;
/// Peak amplitude of the tone (well below clipping).
const PEAK: f32 = 0.3;

/// Set once the first cue failure has been reported.
static WARNED: AtomicBool = AtomicBool::new(false);

/// True unless `DICTATE_SOUNDS` is exactly `"0"`.
pub(crate) fn sounds_enabled(env: Option<&str>) -> bool {
    !matches!(env, Some("0"))
}

/// Generate a sine tone of `freq_hz` lasting `duration_ms` at `sample_rate`,
/// with a short linear fade in/out.
pub(crate) fn tone(freq_hz: f32, duration_ms: u32, sample_rate: u32) -> Vec<f32> {
    let len = duration_ms as usize * sample_rate as usize / 1000;
    let fade = (FADE_MS as usize * sample_rate as usize / 1000).min(len / 2);
    (0..len)
        .map(|i| {
            let envelope = if fade == 0 {
                1.0
            } else if i < fade {
                i as f32 / fade as f32
            } else if i >= len - fade {
                (len - 1 - i) as f32 / fade as f32
            } else {
                1.0
            };
            let phase = i as f32 * freq_hz * 2.0 * std::f32::consts::PI / sample_rate as f32;
            PEAK * envelope * phase.sin()
        })
        .collect()
}

/// Play the start cue (880 Hz) unless disabled. Returns immediately; failures
/// warn once per run and never reach the caller.
pub(crate) fn play_start() {
    play_cue(START_FREQ_HZ);
}

/// Play the stop cue (660 Hz) unless disabled. Returns immediately; failures
/// warn once per run and never reach the caller.
pub(crate) fn play_stop() {
    play_cue(STOP_FREQ_HZ);
}

fn play_cue(freq_hz: f32) {
    if !sounds_enabled(std::env::var("DICTATE_SOUNDS").ok().as_deref()) {
        return;
    }
    std::thread::spawn(move || {
        if let Err(err) = play_blocking(freq_hz)
            && !WARNED.swap(true, Ordering::SeqCst)
        {
            eprintln!("warning: could not play sound cue: {err:#}");
        }
    });
}

fn play_blocking(freq_hz: f32) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("the system has no default output device"))?;
    let supported = device
        .default_output_config()
        .context("could not get the default output config")?;
    let config = supported.config();
    match supported.sample_format() {
        SampleFormat::F32 => run::<f32>(&device, &config, freq_hz),
        SampleFormat::I16 => run::<i16>(&device, &config, freq_hz),
        SampleFormat::U16 => run::<u16>(&device, &config, freq_hz),
        format => anyhow::bail!(
            "unsupported output sample format: {format:?} (only F32, I16 and U16 are supported)"
        ),
    }
}

/// Play one cue on `device` and return when it has finished. The stream lives
/// only for this call.
fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig, freq_hz: f32) -> anyhow::Result<()>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = usize::from(config.channels);
    if channels == 0 {
        anyhow::bail!("the default output config has 0 channels");
    }
    let samples = tone(freq_hz, CUE_DURATION_MS, config.sample_rate);
    let cursor = std::sync::Mutex::new(0usize);
    let stream_error = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
    let callback_errors = stream_error.clone();
    let stream = device
        .build_output_stream(
            *config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                let mut pos = match cursor.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                for frame in data.chunks_mut(channels) {
                    let value = if *pos < samples.len() {
                        samples[*pos]
                    } else {
                        0.0
                    };
                    *pos += 1;
                    let sample = T::from_sample(value);
                    for out in frame {
                        *out = sample;
                    }
                }
            },
            move |err| {
                let mut guard = match callback_errors.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                if guard.is_none() {
                    *guard = Some(err.to_string());
                }
            },
            None,
        )
        .context("could not open the audio output stream")?;
    stream
        .play()
        .context("could not start the audio output stream")?;
    // The tone plus a margin for device latency, then the stream is dropped.
    std::thread::sleep(std::time::Duration::from_millis(
        u64::from(CUE_DURATION_MS) + 100,
    ));
    let text = match stream_error.lock() {
        Ok(guard) => guard.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    };
    if let Some(text) = text {
        anyhow::bail!("audio output stream error: {text}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_length_matches_duration_and_rate() {
        assert_eq!(tone(880.0, 70, 48_000).len(), 3360);
        assert_eq!(tone(660.0, 70, 44_100).len(), 3087);
        assert_eq!(tone(880.0, 70, 16_000).len(), 1120);
    }

    #[test]
    fn tone_fades_in_and_out() {
        let samples = tone(880.0, 70, 48_000);
        assert!(samples[0].abs() < 1e-9);
        assert!(samples[samples.len() - 1].abs() < 1e-9);
        // The first 60 samples sit inside the 5 ms fade-in (240 samples),
        // so they stay quiet while the middle of the tone is loud.
        for (i, sample) in samples.iter().enumerate().take(60) {
            assert!(
                sample.abs() < 0.08,
                "sample {i} escapes the fade-in: {sample}"
            );
        }
        let peak: f32 = samples
            .iter()
            .skip(1000)
            .take(1000)
            .map(|sample| sample.abs())
            .fold(0.0, f32::max);
        assert!(peak > 0.25, "middle of the tone is too quiet: {peak}");
    }

    #[test]
    fn tone_peak_never_exceeds_point_three() {
        for (freq, rate) in [(880.0, 48_000), (660.0, 44_100), (880.0, 16_000)] {
            let peak: f32 = tone(freq, 70, rate)
                .iter()
                .map(|sample| sample.abs())
                .fold(0.0, f32::max);
            assert!(peak <= 0.3 + 1e-6, "peak {peak} at {freq} Hz/{rate} Hz");
        }
    }

    #[test]
    fn sounds_disable_only_on_zero() {
        assert!(!sounds_enabled(Some("0")));
        assert!(sounds_enabled(None));
        assert!(sounds_enabled(Some("")));
        assert!(sounds_enabled(Some("1")));
    }
}
