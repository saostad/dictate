use std::str::FromStr;
#[cfg(target_os = "linux")]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    BufferSize, FromSample, I24, Sample, SampleFormat, SizedSample, SupportedBufferSize, U24,
};

/// 16 kHz mono samples below this count are too short to transcribe (0.3 s).
pub(crate) const MIN_RECORDING_SAMPLES: usize = 4800;

/// Log one microphone line: stderr, plus the daemon log file when active
/// (the same two sinks `dlog!` writes to).
fn mlog(message: &str) {
    eprintln!("{message}");
    crate::daemon::mirror_to_log(message);
}

/// Debug timing for mic starts is off unless `DICTATE_DEBUG_MIC` is set to a
/// non-empty value other than `0`.
fn mic_debug() -> bool {
    matches!(std::env::var("DICTATE_DEBUG_MIC").as_deref(), Ok(value) if !value.is_empty() && value != "0")
}

/// One timed step of a mic start, for `DICTATE_DEBUG_MIC` profiling.
struct StartStep {
    name: &'static str,
    ms: u128,
}

/// Log each step's ms at debug level (`DICTATE_DEBUG_MIC=1`); silent otherwise.
fn debug_mic_start(steps: &[StartStep], cached: bool) {
    if !mic_debug() {
        return;
    }
    let total: u128 = steps.iter().map(|step| step.ms).sum();
    let parts: Vec<String> = steps
        .iter()
        .map(|step| format!("{}={}ms", step.name, step.ms))
        .collect();
    mlog(&format!(
        "mic start ({}): {} (total {total}ms)",
        if cached { "cached" } else { "fresh" },
        parts.join(" ")
    ));
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
    buffer_size: SupportedBufferSize,
}

/// Prefer a float input config: the first F32 range, at `default_rate` when
/// that lies inside the range, else the closest supported rate (the range
/// edge). `None` when no range offers F32 (the caller then uses the default
/// config). Ranges with `min_rate > max_rate` are skipped rather than
/// panicking in `clamp`. The buffer size is ~20 ms, clamped to the range.
fn prefer_f32_config(ranges: &[ConfigRange], default_rate: u32) -> Option<cpal::StreamConfig> {
    let range = ranges
        .iter()
        .find(|range| range.format == SampleFormat::F32 && range.min_rate <= range.max_rate)?;
    let sample_rate = default_rate.clamp(range.min_rate, range.max_rate);
    Some(cpal::StreamConfig {
        channels: range.channels,
        sample_rate,
        buffer_size: choose_buffer_size(range.buffer_size, sample_rate),
    })
}

/// About 20 ms of audio frames at `sample_rate` (48 kHz → 960).
fn wanted_buffer_frames(sample_rate: u32) -> u32 {
    (sample_rate / 50).max(1)
}

/// Pick the stream buffer size: ~20 ms of frames clamped into the device's
/// supported range, or `Default` when the device reports no range (or a
/// degenerate one). Pure, for tests.
fn choose_buffer_size(supported: SupportedBufferSize, sample_rate: u32) -> BufferSize {
    let SupportedBufferSize::Range { min, max } = supported else {
        return BufferSize::Default;
    };
    if min > max {
        return BufferSize::Default;
    }
    BufferSize::Fixed(wanted_buffer_frames(sample_rate).clamp(min, max))
}

/// The default config with a small fixed buffer when the device reports a
/// supported range (`Default` otherwise).
fn default_config_with_buffer(supported: &cpal::SupportedStreamConfig) -> cpal::StreamConfig {
    let mut config = supported.config();
    config.buffer_size = choose_buffer_size(*supported.buffer_size(), config.sample_rate);
    config
}

/// One-line description of the chosen buffer size for the `mic buffer:` log.
fn describe_buffer_size(buffer_size: BufferSize) -> String {
    match buffer_size {
        BufferSize::Fixed(frames) => format!("fixed {frames} frames (~20 ms)"),
        BufferSize::Default => "default (the device reports no buffer range)".to_string(),
    }
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
    /// Input data callbacks delivered (for the stop drain).
    callbacks: Arc<AtomicU64>,
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

/// A microphone resolved to a device and stream config, ready to open.
/// The daemon caches one per selected microphone in [`MicSession`].
struct ResolvedMic {
    device: cpal::Device,
    config: cpal::StreamConfig,
    format: SampleFormat,
    mic_id: String,
    mic_name: String,
}

/// Resolve `mic` (a `"host:device"` id, `None` or empty for the system default)
/// to a device and stream config on an already-opened host. A missing or
/// unparsable id falls back to the default with a log + notification; the
/// saved choice is kept.
fn resolve_mic(
    host: &cpal::Host,
    mic: Option<&str>,
    steps: &mut Vec<StartStep>,
) -> anyhow::Result<ResolvedMic> {
    let lookup_start = Instant::now();
    let device = match mic {
        Some(selected) if !selected.is_empty() => match device_by_id_str(host, selected) {
            Some(device) => device,
            None => fallback_to_default(host, selected)?,
        },
        _ => default_input_device(host)?,
    };
    steps.push(StartStep {
        name: "lookup",
        ms: lookup_start.elapsed().as_millis(),
    });
    let (mic_id, mic_name) = device_label(&device);
    let config_start = Instant::now();
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
                    buffer_size: *range.buffer_size(),
                })
                .collect();
            match prefer_f32_config(&plain, default_supported.sample_rate()) {
                Some(config) => (config, SampleFormat::F32),
                None => (
                    default_config_with_buffer(&default_supported),
                    default_supported.sample_format(),
                ),
            }
        }
        Err(err) => {
            mlog(&format!(
                "warning: could not list the microphone's supported configs ({err}); using the default config"
            ));
            (
                default_config_with_buffer(&default_supported),
                default_supported.sample_format(),
            )
        }
    };
    steps.push(StartStep {
        name: "config",
        ms: config_start.elapsed().as_millis(),
    });
    let resolved = ResolvedMic {
        device,
        config,
        format,
        mic_id,
        mic_name,
    };
    log_recording_from(&resolved);
    Ok(resolved)
}

/// Log which microphone a recording uses. Every recording logs these lines,
/// whether its device was freshly resolved or came from the cache.
fn log_recording_from(resolved: &ResolvedMic) {
    mlog(&format!(
        "recording from: {} ({})",
        resolved.mic_name, resolved.mic_id
    ));
    let channels = usize::from(resolved.config.channels);
    let sample_rate = resolved.config.sample_rate;
    mlog(&format!(
        "mic format: {:?} {channels}ch {sample_rate} Hz",
        resolved.format
    ));
    mlog(&format!(
        "mic buffer: {}",
        describe_buffer_size(resolved.config.buffer_size)
    ));
}

/// Open the stream for a resolved microphone and start recording.
fn open_recorder(resolved: &ResolvedMic, steps: &mut Vec<StartStep>) -> anyhow::Result<Recorder> {
    let open_start = Instant::now();
    let device = &resolved.device;
    let config = &resolved.config;
    let format = resolved.format;
    let channels = usize::from(config.channels);
    let sample_rate = config.sample_rate;
    let samples = Arc::new(Mutex::new(Vec::with_capacity((sample_rate as usize) * 60)));
    let stream_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let callbacks = Arc::new(AtomicU64::new(0));
    let stream = match format {
            SampleFormat::I8 => build_stream::<i8>(device, config, channels, &samples, &stream_error, &callbacks),
            SampleFormat::I16 => {
                build_stream::<i16>(device, config, channels, &samples, &stream_error, &callbacks)
            }
            SampleFormat::I24 => {
                build_stream::<I24>(device, config, channels, &samples, &stream_error, &callbacks)
            }
            SampleFormat::I32 => {
                build_stream::<i32>(device, config, channels, &samples, &stream_error, &callbacks)
            }
            SampleFormat::I64 => {
                build_stream::<i64>(device, config, channels, &samples, &stream_error, &callbacks)
            }
            SampleFormat::U8 => build_stream::<u8>(device, config, channels, &samples, &stream_error, &callbacks),
            SampleFormat::U16 => {
                build_stream::<u16>(device, config, channels, &samples, &stream_error, &callbacks)
            }
            SampleFormat::U24 => {
                build_stream::<U24>(device, config, channels, &samples, &stream_error, &callbacks)
            }
            SampleFormat::U32 => {
                build_stream::<u32>(device, config, channels, &samples, &stream_error, &callbacks)
            }
            SampleFormat::U64 => {
                build_stream::<u64>(device, config, channels, &samples, &stream_error, &callbacks)
            }
            SampleFormat::F32 => {
                build_stream::<f32>(device, config, channels, &samples, &stream_error, &callbacks)
            }
            SampleFormat::F64 => {
                build_stream::<f64>(device, config, channels, &samples, &stream_error, &callbacks)
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
    steps.push(StartStep {
        name: "open",
        ms: open_start.elapsed().as_millis(),
    });
    Ok(Recorder {
        stream,
        samples,
        stream_error,
        callbacks,
        sample_rate,
        mic_name: resolved.mic_name.clone(),
    })
}

/// Long-lived audio session for the daemon: one cpal host plus the resolved
/// device and stream config for the selected microphone. Creating a host
/// re-connects to the audio server, so keeping one (and the resolved device)
/// is what makes `start` fast.
pub(crate) struct MicSession {
    host: cpal::Host,
    cached: Option<CachedMic>,
}

/// A resolved microphone plus the selection it was resolved for (`""` is the
/// system default).
struct CachedMic {
    selection: String,
    resolved: ResolvedMic,
}

impl MicSession {
    /// Open the audio host once; the daemon keeps the session for its whole run.
    pub(crate) fn new() -> Self {
        let host_start = Instant::now();
        let host = audio_host();
        if mic_debug() {
            mlog(&format!("mic host: {}ms", host_start.elapsed().as_millis()));
        }
        Self { host, cached: None }
    }

    /// List the microphones on the kept host (no reconnect). Enumeration
    /// always queries the server, so this sees plug/unplug.
    pub(crate) fn list_microphones(&self) -> anyhow::Result<Vec<Mic>> {
        list_microphones_on(&self.host)
    }

    /// Drop the cached device and stream config (a mic change, or a listing
    /// change on refresh). The host is kept.
    pub(crate) fn invalidate(&mut self) {
        self.cached = None;
    }

    /// Start recording from `mic` (a `"host:device"` id, `None` or empty for
    /// the system default), reusing the cached device and config when the
    /// selection matches. When opening fails, the cache is dropped and the
    /// start is retried once with a fresh host and lookup.
    pub(crate) fn start(&mut self, mic: Option<&str>) -> anyhow::Result<Recorder> {
        let selection = mic.unwrap_or("").to_string();
        if let Some(cached) = &self.cached
            && cached.selection == selection
        {
            let mut steps = Vec::new();
            log_recording_from(&cached.resolved);
            match open_recorder(&cached.resolved, &mut steps) {
                Ok(recorder) => {
                    debug_mic_start(&steps, true);
                    return Ok(recorder);
                }
                Err(err) => {
                    if mic_debug() {
                        mlog(&format!(
                            "mic start: cached open failed ({err:#}); retrying with a fresh lookup"
                        ));
                    }
                    self.cached = None;
                }
            }
        } else {
            // A different selection than the cached one: re-resolve.
            self.cached = None;
        }
        let mut steps = Vec::new();
        let resolved = resolve_mic(&self.host, mic, &mut steps)?;
        match open_recorder(&resolved, &mut steps) {
            Ok(recorder) => {
                debug_mic_start(&steps, false);
                self.cached = Some(CachedMic {
                    selection,
                    resolved,
                });
                Ok(recorder)
            }
            Err(err) => {
                if mic_debug() {
                    mlog(&format!(
                        "mic start: open failed ({err:#}); retrying once with a fresh host"
                    ));
                }
                // Retry once: the server may have restarted, so reconnect.
                self.host = audio_host();
                self.cached = None;
                let mut steps = Vec::new();
                let resolved = resolve_mic(&self.host, mic, &mut steps)?;
                let recorder = open_recorder(&resolved, &mut steps)?;
                debug_mic_start(&steps, false);
                self.cached = Some(CachedMic {
                    selection,
                    resolved,
                });
                Ok(recorder)
            }
        }
    }
}

impl Recorder {
    /// Start recording from `mic` (a `"host:device"` id, `None` or empty for
    /// the system default). One-shot host for short-lived callers (`listen`);
    /// the daemon uses [`MicSession`] instead. A missing or unparsable id
    /// falls back to the default with a log + notification; the choice is kept.
    pub(crate) fn start(mic: Option<&str>) -> anyhow::Result<Self> {
        let mut steps = Vec::new();
        let host_start = Instant::now();
        let host = audio_host();
        steps.push(StartStep {
            name: "host",
            ms: host_start.elapsed().as_millis(),
        });
        let resolved = resolve_mic(&host, mic, &mut steps)?;
        let recorder = open_recorder(&resolved, &mut steps)?;
        debug_mic_start(&steps, false);
        Ok(recorder)
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

    /// Wait for one more input callback after the stop request, so audio held
    /// in the device's buffer is delivered before the stream is dropped.
    /// Bounded at 200 ms: a stalled mic just waits that out, never hangs.
    pub(crate) fn drain(&self) {
        let drain_start = Instant::now();
        let seen = self.callbacks.load(Ordering::Relaxed);
        let deadline = drain_start + Duration::from_millis(200);
        while self.callbacks.load(Ordering::Relaxed) == seen && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        if mic_debug() {
            mlog(&format!(
                "mic drain: {}ms",
                drain_start.elapsed().as_millis()
            ));
        }
    }

    /// Stop recording and return the audio as 16 kHz mono samples.
    pub(crate) fn stop(self) -> anyhow::Result<Vec<f32>> {
        // Let the device deliver its buffered tail first, then drain most
        // samples before stopping and the trailing few after the stream is
        // dropped, so no sample is lost.
        self.drain();
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
    callbacks: &Arc<AtomicU64>,
) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let data_samples = samples.clone();
    let data_errors = stream_error.clone();
    let data_callbacks = callbacks.clone();
    let callback_errors = stream_error.clone();
    device.build_input_stream(
        *config,
        move |data: &[T], _: &_| {
            push_samples(data, channels, &data_samples, &data_errors, &data_callbacks)
        },
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
    callbacks: &AtomicU64,
) where
    T: cpal::Sample,
    f32: FromSample<T>,
{
    if !data.is_empty() {
        callbacks.fetch_add(1, Ordering::Relaxed);
    }
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
            buffer_size: SupportedBufferSize::Unknown,
        }
    }

    fn ranged_buffer(
        channels: u16,
        min_rate: u32,
        max_rate: u32,
        format: SampleFormat,
        min: u32,
        max: u32,
    ) -> ConfigRange {
        ConfigRange {
            channels,
            min_rate,
            max_rate,
            format,
            buffer_size: SupportedBufferSize::Range { min, max },
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
    fn buffer_size_is_20_ms_clamped_to_the_range() {
        let range = |min: u32, max: u32| SupportedBufferSize::Range { min, max };
        // 48 kHz wants 960 frames; inside the range it is used as-is.
        assert_eq!(
            choose_buffer_size(range(1, 4096), 48_000),
            BufferSize::Fixed(960)
        );
        assert_eq!(wanted_buffer_frames(48_000), 960);
        assert_eq!(wanted_buffer_frames(16_000), 320);
        // Outside the range it clamps to the edge.
        assert_eq!(
            choose_buffer_size(range(1, 512), 48_000),
            BufferSize::Fixed(512)
        );
        assert_eq!(
            choose_buffer_size(range(2048, 8192), 48_000),
            BufferSize::Fixed(2048)
        );
        // No range (or a degenerate one): the default buffer size.
        assert_eq!(
            choose_buffer_size(SupportedBufferSize::Unknown, 48_000),
            BufferSize::Default
        );
        assert_eq!(
            choose_buffer_size(range(512, 128), 48_000),
            BufferSize::Default
        );
    }

    #[test]
    fn preferred_f32_config_carries_the_clamped_buffer() {
        let ranges = vec![ranged_buffer(2, 8000, 96_000, SampleFormat::F32, 1, 512)];
        let config = prefer_f32_config(&ranges, 48_000).unwrap();
        assert_eq!(config.buffer_size, BufferSize::Fixed(512));
        let ranges = vec![ranged_buffer(1, 8000, 48_000, SampleFormat::F32, 1, 8192)];
        let config = prefer_f32_config(&ranges, 48_000).unwrap();
        assert_eq!(config.buffer_size, BufferSize::Fixed(960));
        // Unknown range: default.
        let ranges = vec![range(2, 8000, 96_000, SampleFormat::F32)];
        let config = prefer_f32_config(&ranges, 48_000).unwrap();
        assert_eq!(config.buffer_size, BufferSize::Default);
    }

    #[test]
    fn buffer_descriptions_name_fixed_and_default() {
        assert_eq!(
            describe_buffer_size(BufferSize::Fixed(960)),
            "fixed 960 frames (~20 ms)"
        );
        assert_eq!(
            describe_buffer_size(BufferSize::Default),
            "default (the device reports no buffer range)"
        );
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
