use std::str::FromStr;
#[cfg(target_os = "linux")]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, I24, Sample, SampleFormat, SizedSample, U24};

/// 16 kHz mono samples below this count are too short to transcribe (0.3 s).
pub(crate) const MIN_RECORDING_SAMPLES: usize = 4800;

/// Log one microphone line: stderr, plus the daemon log file when active
/// (the same two sinks `dlog!` writes to).
fn mlog(message: &str) {
    eprintln!("{message}");
    crate::daemon::mirror_to_log(message);
}

/// The audio host to record from and list microphones on. On Linux this is
/// PulseAudio when it connects (PipeWire serves the pulse protocol, with
/// user-meaningful device names); when the connect fails, ALSA with a
/// one-time log + notification. Other OSes use the default host.
#[cfg(target_os = "linux")]
pub(crate) fn audio_host() -> cpal::Host {
    if cpal::available_hosts().contains(&cpal::HostId::PulseAudio) {
        match cpal::host_from_id(cpal::HostId::PulseAudio) {
            Ok(host) => return host,
            Err(err) => warn_alsa_fallback(&err),
        }
    }
    // PulseAudio is absent or unreachable: ALSA directly (`default_host()`
    // would retry the failed PulseAudio connect first).
    cpal::host_from_id(cpal::HostId::Alsa).unwrap_or_else(|_| cpal::default_host())
}

/// The audio host to record from and list microphones on (the default host
/// outside Linux; only Linux has the PulseAudio/ALSA choice).
#[cfg(not(target_os = "linux"))]
pub(crate) fn audio_host() -> cpal::Host {
    cpal::default_host()
}

/// Set once the ALSA fallback has been reported.
#[cfg(target_os = "linux")]
static ALSA_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

/// Log + notify the PulseAudio→ALSA fallback, once per process.
#[cfg(target_os = "linux")]
fn warn_alsa_fallback(err: &cpal::Error) {
    if ALSA_FALLBACK_WARNED.swap(true, Ordering::SeqCst) {
        return;
    }
    let body = format!(
        "Could not connect to PipeWire/PulseAudio ({err}). Microphone names may be technical."
    );
    mlog(&format!("error: dictate: using ALSA audio: {body}"));
    crate::notify::show(
        crate::notify::Kind::Error,
        "dictate: using ALSA audio",
        &body,
    );
}

/// One microphone: its stable device id (`"host:device"`, as shown by
/// `dictate mics`), its display name, and whether it is the system default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Mic {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

/// List the microphones on [`audio_host`]: input devices only, monitors and
/// `alsa:null` hidden, duplicate names numbered. See [`microphone_list`].
pub(crate) fn list_microphones() -> anyhow::Result<Vec<Mic>> {
    list_microphones_on(&audio_host())
}

/// List the microphones on one already-opened host.
fn list_microphones_on(host: &cpal::Host) -> anyhow::Result<Vec<Mic>> {
    let default_id = host
        .default_input_device()
        .and_then(|device| device.id().ok())
        .map(|id| id.to_string());
    let devices = host
        .input_devices()
        .context("Could not list the microphones")?;
    let mut entries = Vec::new();
    for device in devices {
        let (Ok(id), Ok(description)) = (device.id(), device.description()) else {
            continue;
        };
        entries.push((id.to_string(), description.name().to_string()));
    }
    Ok(microphone_list(entries, default_id.as_deref()))
}

/// Build the microphone list from `(id, name)` entries: hide monitors (a
/// pulse id ending in `.monitor` or a name starting with `Monitor of `) and
/// `alsa:null`, mark the default by id, and number duplicate names (`Name`,
/// `Name (2)`, `Name (3)`). Pure, so the rules are unit-testable.
pub(crate) fn microphone_list(
    entries: Vec<(String, String)>,
    default_id: Option<&str>,
) -> Vec<Mic> {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut mics = Vec::with_capacity(entries.len());
    for (id, name) in entries {
        if id == "alsa:null" || id.ends_with(".monitor") || name.starts_with("Monitor of ") {
            continue;
        }
        let seen = counts.entry(name.clone()).or_insert(0);
        *seen += 1;
        let display = if *seen == 1 {
            name
        } else {
            format!("{name} ({seen})")
        };
        let is_default = default_id == Some(id.as_str());
        mics.push(Mic {
            id,
            name: display,
            is_default,
        });
    }
    mics
}

/// One supported input config range as plain data, so
/// [`prefer_f32_config`] is unit-testable without audio hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConfigRange {
    channels: u16,
    min_rate: u32,
    max_rate: u32,
    format: SampleFormat,
}

/// Prefer a float input config: the first F32 range, at `default_rate` when
/// that lies inside the range, else the closest supported rate (the range
/// edge). `None` when no range offers F32 (the caller then uses the default
/// config). Ranges with `min_rate > max_rate` are skipped rather than
/// panicking in `clamp`.
fn prefer_f32_config(ranges: &[ConfigRange], default_rate: u32) -> Option<cpal::StreamConfig> {
    let range = ranges
        .iter()
        .find(|range| range.format == SampleFormat::F32 && range.min_rate <= range.max_rate)?;
    Some(cpal::StreamConfig {
        channels: range.channels,
        sample_rate: default_rate.clamp(range.min_rate, range.max_rate),
        buffer_size: cpal::BufferSize::Default,
    })
}

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
    mic_name: String,
}

/// The default input device, or the no-microphone error.
fn default_input_device(host: &cpal::Host) -> anyhow::Result<cpal::Device> {
    host.default_input_device().ok_or_else(|| {
        anyhow::anyhow!(
            "No microphone found: the system has no default input device. Check your sound settings."
        )
    })
}

/// Find a device by its `"host:device"` id string. `None` when the id is
/// unparsable or no such device exists on this host.
fn device_by_id_str(host: &cpal::Host, selected: &str) -> Option<cpal::Device> {
    let id = cpal::DeviceId::from_str(selected).ok()?;
    host.device_by_id(&id)
}

/// Display text for a saved microphone id: the current friendly name when the
/// device is listed, else the raw id. Pure, for tests.
fn saved_mic_display(mics: &[Mic], saved: &str) -> String {
    mics.iter()
        .find(|mic| mic.id == saved)
        .map(|mic| mic.name.clone())
        .unwrap_or_else(|| saved.to_string())
}

/// Record from the default because the saved microphone `selected` is gone:
/// log + notify the fallback (the saved choice is kept, so it is used again
/// when it comes back), and return the default device.
fn fallback_to_default(host: &cpal::Host, selected: &str) -> anyhow::Result<cpal::Device> {
    let device = default_input_device(host)?;
    let default_name = device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| "(unknown microphone)".to_string());
    let saved = list_microphones_on(host)
        .ok()
        .as_deref()
        .map(|mics| saved_mic_display(mics, selected))
        .unwrap_or_else(|| selected.to_string());
    let body = format!("\"{saved}\" isn't available. Using the system default: {default_name}.");
    mlog(&format!("error: dictate: microphone not found: {body}"));
    crate::notify::show(
        crate::notify::Kind::Error,
        "dictate: microphone not found",
        &body,
    );
    Ok(device)
}

/// The `(id, name)` label of a device for the `recording from:` line.
/// Identity reads only fail on a device that vanished mid-call; use
/// placeholders rather than failing a recording that may still work.
fn device_label(device: &cpal::Device) -> (String, String) {
    let id = device
        .id()
        .map(|id| id.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    let name = device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| id.clone());
    (id, name)
}

impl Recorder {
    /// Start recording from `mic` (a `"host:device"` id, `None` or empty for
    /// the system default). A missing or unparsable id falls back to the
    /// default with a log + notification; the saved choice is kept.
    pub(crate) fn start(mic: Option<&str>) -> anyhow::Result<Self> {
        let host = audio_host();
        let device = match mic {
            Some(selected) if !selected.is_empty() => match device_by_id_str(&host, selected) {
                Some(device) => device,
                None => fallback_to_default(&host, selected)?,
            },
            _ => default_input_device(&host)?,
        };
        let (mic_id, mic_name) = device_label(&device);
        mlog(&format!("recording from: {mic_name} ({mic_id})"));
        let default_supported = device
            .default_input_config()
            .context("Could not get the microphone's default input config")?;
        // Prefer float samples when the device offers them; otherwise use the
        // default config (any PCM format it reports is handled below).
        let (config, format) = match device.supported_input_configs() {
            Ok(ranges) => {
                let plain: Vec<ConfigRange> = ranges
                    .map(|range| ConfigRange {
                        channels: range.channels(),
                        min_rate: range.min_sample_rate(),
                        max_rate: range.max_sample_rate(),
                        format: range.sample_format(),
                    })
                    .collect();
                match prefer_f32_config(&plain, default_supported.sample_rate()) {
                    Some(config) => (config, SampleFormat::F32),
                    None => (
                        default_supported.config(),
                        default_supported.sample_format(),
                    ),
                }
            }
            Err(err) => {
                mlog(&format!(
                    "warning: could not list the microphone's supported configs ({err}); using the default config"
                ));
                (
                    default_supported.config(),
                    default_supported.sample_format(),
                )
            }
        };
        let channels = usize::from(config.channels);
        let sample_rate = config.sample_rate;
        mlog(&format!(
            "mic format: {format:?} {channels}ch {sample_rate} Hz"
        ));
        let samples = Arc::new(Mutex::new(Vec::with_capacity((sample_rate as usize) * 60)));
        let stream_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let stream = match format {
            SampleFormat::I8 => build_stream::<i8>(&device, &config, channels, &samples, &stream_error),
            SampleFormat::I16 => {
                build_stream::<i16>(&device, &config, channels, &samples, &stream_error)
            }
            SampleFormat::I24 => {
                build_stream::<I24>(&device, &config, channels, &samples, &stream_error)
            }
            SampleFormat::I32 => {
                build_stream::<i32>(&device, &config, channels, &samples, &stream_error)
            }
            SampleFormat::I64 => {
                build_stream::<i64>(&device, &config, channels, &samples, &stream_error)
            }
            SampleFormat::U8 => build_stream::<u8>(&device, &config, channels, &samples, &stream_error),
            SampleFormat::U16 => {
                build_stream::<u16>(&device, &config, channels, &samples, &stream_error)
            }
            SampleFormat::U24 => {
                build_stream::<U24>(&device, &config, channels, &samples, &stream_error)
            }
            SampleFormat::U32 => {
                build_stream::<u32>(&device, &config, channels, &samples, &stream_error)
            }
            SampleFormat::U64 => {
                build_stream::<u64>(&device, &config, channels, &samples, &stream_error)
            }
            SampleFormat::F32 => {
                build_stream::<f32>(&device, &config, channels, &samples, &stream_error)
            }
            SampleFormat::F64 => {
                build_stream::<f64>(&device, &config, channels, &samples, &stream_error)
            }
            format => {
                anyhow::bail!(
                    "Unsupported microphone sample format: {format:?} (only PCM sample formats are supported)"
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
            mic_name,
        })
    }

    /// Device sample rate of this recording.
    pub(crate) fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Display name of the microphone this recording uses (for the tray).
    pub(crate) fn mic_name(&self) -> &str {
        &self.mic_name
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

    fn entry(id: &str, name: &str) -> (String, String) {
        (id.to_string(), name.to_string())
    }

    #[test]
    fn microphone_list_marks_default_and_keeps_order() {
        let mics = microphone_list(
            vec![
                entry("pulseaudio:alsa_input.usb", "USB Microphone"),
                entry("pulseaudio:alsa_input.pci", "Built-in Audio Analog Stereo"),
            ],
            Some("pulseaudio:alsa_input.pci"),
        );
        assert_eq!(
            mics,
            vec![
                Mic {
                    id: "pulseaudio:alsa_input.usb".to_string(),
                    name: "USB Microphone".to_string(),
                    is_default: false,
                },
                Mic {
                    id: "pulseaudio:alsa_input.pci".to_string(),
                    name: "Built-in Audio Analog Stereo".to_string(),
                    is_default: true,
                },
            ]
        );
    }

    #[test]
    fn microphone_list_hides_monitors_and_null() {
        let mics = microphone_list(
            vec![
                entry(
                    "pulseaudio:alsa_output.pci.monitor",
                    "Monitor of Built-in Audio",
                ),
                entry("pulseaudio:other.monitor", "Something else"),
                entry("pulseaudio:alsa_input.pci", "Monitor of Weird Name"),
                entry("alsa:null", "null"),
                entry("pulseaudio:alsa_input.pci", "Built-in Audio Analog Stereo"),
            ],
            None,
        );
        assert_eq!(mics.len(), 1);
        assert_eq!(mics[0].name, "Built-in Audio Analog Stereo");
    }

    #[test]
    fn saved_mic_display_prefers_friendly_name() {
        let mics = vec![Mic {
            id: "pulseaudio:alsa_input.pci".to_string(),
            name: "Built-in Audio Analog Stereo".to_string(),
            is_default: true,
        }];
        assert_eq!(
            saved_mic_display(&mics, "pulseaudio:alsa_input.pci"),
            "Built-in Audio Analog Stereo"
        );
        assert_eq!(
            saved_mic_display(&mics, "pulseaudio:gone"),
            "pulseaudio:gone"
        );
        assert_eq!(saved_mic_display(&[], "not an id"), "not an id");
    }

    fn range(channels: u16, min_rate: u32, max_rate: u32, format: SampleFormat) -> ConfigRange {
        ConfigRange {
            channels,
            min_rate,
            max_rate,
            format,
        }
    }

    #[test]
    fn prefers_f32_at_default_rate_when_in_range() {
        let ranges = vec![
            range(2, 8000, 48_000, SampleFormat::I16),
            range(2, 8000, 96_000, SampleFormat::F32),
        ];
        let config = prefer_f32_config(&ranges, 48_000).unwrap();
        assert_eq!(config.channels, 2);
        assert_eq!(config.sample_rate, 48_000);
    }

    #[test]
    fn prefers_f32_at_closest_rate_when_default_is_out_of_range() {
        let ranges = vec![range(1, 8000, 48_000, SampleFormat::F32)];
        assert_eq!(
            prefer_f32_config(&ranges, 96_000).unwrap().sample_rate,
            48_000
        );
        assert_eq!(prefer_f32_config(&ranges, 4000).unwrap().sample_rate, 8000);
    }

    #[test]
    fn no_f32_range_means_default_config() {
        let ranges = vec![
            range(2, 8000, 48_000, SampleFormat::I16),
            range(2, 8000, 48_000, SampleFormat::I32),
        ];
        assert!(prefer_f32_config(&ranges, 48_000).is_none());
        assert!(prefer_f32_config(&[], 48_000).is_none());
    }

    #[test]
    fn first_f32_range_wins_and_degenerate_ranges_are_skipped() {
        let ranges = vec![
            range(2, 96_000, 8000, SampleFormat::F32),
            range(1, 8000, 48_000, SampleFormat::F32),
            range(2, 8000, 48_000, SampleFormat::F32),
        ];
        let config = prefer_f32_config(&ranges, 48_000).unwrap();
        assert_eq!(config.channels, 1);
        assert_eq!(config.sample_rate, 48_000);
        // Only a degenerate F32 range: no pick (and no panic).
        assert!(prefer_f32_config(&ranges[..1], 48_000).is_none());
    }

    #[test]
    fn microphone_list_numbers_duplicate_names() {
        let mics = microphone_list(
            vec![
                entry("pulseaudio:one", "USB Microphone"),
                entry("pulseaudio:two", "USB Microphone"),
                entry("pulseaudio:three", "USB Microphone"),
            ],
            None,
        );
        let names: Vec<&str> = mics.iter().map(|mic| mic.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["USB Microphone", "USB Microphone (2)", "USB Microphone (3)"]
        );
    }
}
