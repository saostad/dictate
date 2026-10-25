//! Settings file: `config.toml` in the platform config directory.
//!
//! A tiny TOML subset parsed by hand (no new crates): `#` comments, blank
//! lines, `key = true|false` and `key = "string"`. Keys are `sounds` and
//! `notifications` (bool, default true), `mode`
//! (`"whole"`/`"pauses"`/`"live"`, default `"whole"`), `model`
//! (`"standard"`/`"fluid"`/`"whisper"`/`"nemotron"`/`"moonshine"`), `hotkey`
//! (string, default `Ctrl+Shift+Space`), `microphone` (string, default `""` =
//! the system default input) and `inference_threads` (`"auto"`, `"1"` or
//! `"2"`, default `"auto"`) and `unload_after` (`"never"`, `"5"`, `"20"`,
//! `"30"` or `"60"` idle minutes before the daemon frees the speech model,
//! default `"20"`). A missing
//! file means defaults, never an error. Legacy `mode = "moonshine"` and Live
//! configs are migrated to a mode/model pair; `batch_model` retains their
//! previous offline model for startup fallback. On macOS the hotkey modifiers are
//! spelled `Cmd`, `Option`, `Ctrl`, `Shift` (e.g. `Cmd+Option+Space`); the
//! Settings window's `Record shortcut…` button captures one. Shortcuts with
//! only modifier keys (at least two, e.g. `Cmd+Option`) work on macOS and
//! need Input Monitoring permission; pressing another key while holding them
//! cancels that recording. They are not available on Windows/Linux.
//!
//! The environment still wins: `DICTATE_SOUNDS`, `DICTATE_NOTIFY`, `DICTATE_MODE`,
//! `DICTATE_SPEECH_MODEL`, `DICTATE_INFERENCE_THREADS`, `DICTATE_HOTKEY` and
//! `DICTATE_MICROPHONE` override the
//! file (`DICTATE_STREAMING` still works when `DICTATE_MODE` is unset). The
//! daemon resolves file + env once
//! at startup into the effective switches below, which `cue`, `notify` and the
//! daemon read instead of touching the environment per call (the microphone is
//! resolved per recording instead: it is read rarely, and only the daemon's
//! settings hold the file value).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

pub(crate) use crate::selection::{DictationMode, ModelChoice, Selection};
use anyhow::Context;

/// Backend dispatch codes, including the legacy Moonshine code. User-facing
/// behavior is [`DictationMode`]; saved settings use a [`Selection`] pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Mode {
    /// Type once at stop (most accurate).
    #[default]
    Whole,
    /// Type each pause segment while speaking.
    Pauses,
    /// True streaming: type while speaking (needs a fast machine).
    Live,
    /// Offline Moonshine Tiny Streaming: complete phrases while speaking.
    Moonshine,
}

impl Mode {
    /// The `config.toml` / `DICTATE_MODE` spelling.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Mode::Whole => "whole",
            Mode::Pauses => "pauses",
            Mode::Live => "live",
            Mode::Moonshine => "moonshine",
        }
    }

    /// Short display name for the tray tooltip and the `Mode: <name>`
    /// notification.
    pub(crate) fn display_name(self) -> &'static str {
        match self {
            Mode::Whole => "Whole",
            Mode::Pauses => "Pauses",
            Mode::Live => "Live",
            Mode::Moonshine => "Live",
        }
    }

    /// Parse one mode value; `None` for anything else.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "whole" => Some(Mode::Whole),
            "pauses" => Some(Mode::Pauses),
            "live" => Some(Mode::Live),
            "moonshine" => Some(Mode::Moonshine),
            _ => None,
        }
    }

    fn from_discriminant(value: u8) -> Self {
        match value {
            1 => Mode::Pauses,
            2 => Mode::Live,
            3 => Mode::Moonshine,
            _ => Mode::Whole,
        }
    }

    /// True for the modes that type while the user is still speaking.
    pub(crate) fn is_streaming(self) -> bool {
        matches!(self, Mode::Live | Mode::Moonshine)
    }
}

/// Last Whole/Pauses model, also retained for startup fallback from Live.
/// The full user-facing engine choice is [`ModelChoice`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SpeechModel {
    /// NVIDIA Parakeet TDT 0.6B v2: the default and most accurate choice on
    /// CPUs that can run the bundled AVX2 ONNX Runtime.
    #[default]
    Standard,
    /// Fluid Speech Mini: better when other people talk nearby.
    Fluid,
    /// Whisper Tiny English (whisper.cpp, English-only, greedy).
    Whisper,
}

impl SpeechModel {
    /// The `config.toml` / `DICTATE_SPEECH_MODEL` spelling.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            SpeechModel::Standard => "standard",
            SpeechModel::Fluid => "fluid",
            SpeechModel::Whisper => "whisper",
        }
    }

    /// Display name in the Settings window and notifications.
    pub(crate) const fn display_name(self) -> &'static str {
        match self {
            SpeechModel::Standard => "Parakeet TDT 0.6B v2",
            SpeechModel::Fluid => "Fluid Speech Mini",
            SpeechModel::Whisper => "Whisper Tiny English",
        }
    }

    /// Parse one model value; `None` for anything else.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "standard" => Some(SpeechModel::Standard),
            "fluid" => Some(SpeechModel::Fluid),
            "whisper" => Some(SpeechModel::Whisper),
            _ => None,
        }
    }

    fn from_discriminant(value: u8) -> Self {
        match value {
            1 => SpeechModel::Fluid,
            2 => SpeechModel::Whisper,
            _ => SpeechModel::Standard,
        }
    }
}

/// Inference threads for the small local engines. Whisper's `Auto` stays
/// conservative: half the available logical CPUs, capped at two, so a
/// four-thread laptop uses two threads and a two-thread laptop uses one.
/// Moonshine's `Auto` selects its faster single-thread mode; explicit 2
/// leaves Moonshine's default threading in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum InferenceThreads {
    #[default]
    Auto,
    One,
    Two,
}

impl InferenceThreads {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            InferenceThreads::Auto => "auto",
            InferenceThreads::One => "1",
            InferenceThreads::Two => "2",
        }
    }

    pub(crate) fn display_name(self) -> &'static str {
        match self {
            InferenceThreads::Auto => "Auto (1–2 by engine)",
            InferenceThreads::One => "1 thread",
            InferenceThreads::Two => "2 threads (Moonshine: default)",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(InferenceThreads::Auto),
            "1" => Some(InferenceThreads::One),
            "2" => Some(InferenceThreads::Two),
            _ => None,
        }
    }

    /// Concrete Whisper thread count. `available` is logical parallelism (the
    /// same value `std::thread::available_parallelism` reports). Moonshine
    /// uses [`crate::moonshine::Threads::for_setting`] instead because its
    /// pinned API cannot request exactly two.
    pub(crate) fn count(self, available: Option<usize>) -> usize {
        match self {
            InferenceThreads::One => 1,
            InferenceThreads::Two => 2,
            InferenceThreads::Auto => match available {
                Some(n) if n >= 4 => 2,
                _ => 1,
            },
        }
    }

    fn from_discriminant(value: u8) -> Self {
        match value {
            1 => InferenceThreads::One,
            2 => InferenceThreads::Two,
            _ => InferenceThreads::Auto,
        }
    }
}

/// Idle minutes before the daemon frees the speech model's memory. The next
/// recording reloads it (a few seconds) while the audio is already captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum UnloadAfter {
    Never,
    Minutes5,
    #[default]
    Minutes20,
    Minutes30,
    Minutes60,
}

impl UnloadAfter {
    pub(crate) const ALL: [UnloadAfter; 5] = [
        UnloadAfter::Never,
        UnloadAfter::Minutes5,
        UnloadAfter::Minutes20,
        UnloadAfter::Minutes30,
        UnloadAfter::Minutes60,
    ];

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            UnloadAfter::Never => "never",
            UnloadAfter::Minutes5 => "5",
            UnloadAfter::Minutes20 => "20",
            UnloadAfter::Minutes30 => "30",
            UnloadAfter::Minutes60 => "60",
        }
    }

    pub(crate) fn display_name(self) -> &'static str {
        match self {
            UnloadAfter::Never => "Never (keep loaded)",
            UnloadAfter::Minutes5 => "After 5 minutes idle",
            UnloadAfter::Minutes20 => "After 20 minutes idle (default)",
            UnloadAfter::Minutes30 => "After 30 minutes idle",
            UnloadAfter::Minutes60 => "After 1 hour idle",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|choice| choice.as_str() == value)
    }

    /// The idle time before unloading, `None` for Never.
    pub(crate) fn duration(self) -> Option<std::time::Duration> {
        let minutes = match self {
            UnloadAfter::Never => return None,
            UnloadAfter::Minutes5 => 5,
            UnloadAfter::Minutes20 => 20,
            UnloadAfter::Minutes30 => 30,
            UnloadAfter::Minutes60 => 60,
        };
        Some(std::time::Duration::from_secs(minutes * 60))
    }
}

/// Settings from the file (before env overrides).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Settings {
    pub sounds: bool,
    pub notifications: bool,
    pub mode: Mode,
    pub model: SpeechModel,
    pub inference_threads: InferenceThreads,
    pub unload_after: UnloadAfter,
    pub hotkey: String,
    pub microphone: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sounds: true,
            notifications: true,
            mode: Mode::Whole,
            model: SpeechModel::Standard,
            inference_threads: InferenceThreads::Auto,
            unload_after: UnloadAfter::Minutes20,
            hotkey: crate::hotkey::DEFAULT_HOTKEY.to_string(),
            microphone: String::new(),
        }
    }
}

/// Effective "play sound cues" switch (file + env, resolved at startup).
static SOUNDS_ON: AtomicBool = AtomicBool::new(true);
/// Effective "show notifications" switch (file + env, resolved at startup).
static NOTIFY_ON: AtomicBool = AtomicBool::new(true);
/// Effective dictation mode (file + env, resolved at startup).
static MODE: AtomicU8 = AtomicU8::new(0);
/// Effective speech model for Whole/Pauses (file + env, resolved at startup).
static SPEECH_MODEL: AtomicU8 = AtomicU8::new(0);
/// Effective inference-thread setting for the small engines (file + env).
static INFERENCE_THREADS: AtomicU8 = AtomicU8::new(0);

/// Combine one bool setting with its env override: unset env keeps the file
/// value, `"0"` turns it off, anything else turns it on.
pub(crate) fn apply_env(file_value: bool, env: Option<&str>) -> bool {
    match env {
        None => file_value,
        Some("0") => false,
        Some(_) => true,
    }
}

/// Combine the mode setting with its env overrides: `DICTATE_MODE` wins when
/// set to a valid non-empty value, otherwise `DICTATE_STREAMING` is honored
/// (`"0"` means whole, anything else means pauses), otherwise the file value.
/// An invalid `DICTATE_MODE` falls through like an unset one.
pub(crate) fn resolve_mode(
    file: Mode,
    mode_env: Option<&str>,
    streaming_env: Option<&str>,
) -> Mode {
    if let Some(value) = mode_env
        && !value.is_empty()
        && let Some(mode) = Mode::parse(value)
    {
        return mode;
    }
    match streaming_env {
        None => file,
        Some("0") => Mode::Whole,
        Some(_) => Mode::Pauses,
    }
}

/// Combine the speech-model setting with its env override:
/// `DICTATE_SPEECH_MODEL` wins when set to a valid non-empty value, otherwise
/// the file value. An invalid or empty env falls through like an unset one.
pub(crate) fn resolve_speech_model(file: SpeechModel, env: Option<&str>) -> SpeechModel {
    match env {
        Some(value) if !value.is_empty() => SpeechModel::parse(value).unwrap_or(file),
        _ => file,
    }
}

/// Combine the inference-thread setting with its env override:
/// `DICTATE_INFERENCE_THREADS` wins when set to a valid non-empty value,
/// otherwise the file value. An invalid or empty env falls through.
pub(crate) fn resolve_inference_threads(
    file: InferenceThreads,
    env: Option<&str>,
) -> InferenceThreads {
    match env {
        Some(value) if !value.is_empty() => InferenceThreads::parse(value).unwrap_or(file),
        _ => file,
    }
}

/// Effective microphone selection: `DICTATE_MICROPHONE` wins when set and
/// non-empty, else the file value when non-empty, else the system default
/// (`None`). An empty value counts as unset on both sides.
pub(crate) fn resolve_microphone(file: &str, env: Option<&str>) -> Option<String> {
    match env {
        Some(selected) if !selected.is_empty() => Some(selected.to_string()),
        _ => {
            if file.is_empty() {
                None
            } else {
                Some(file.to_string())
            }
        }
    }
}

/// Current env values for the mode resolution, plus a warning when
/// `DICTATE_MODE` is set to an invalid non-empty value.
fn mode_env_now() -> (Option<String>, Option<String>, Option<String>) {
    let mode = std::env::var("DICTATE_MODE").ok();
    let streaming = std::env::var("DICTATE_STREAMING").ok();
    let warning = match mode.as_deref() {
        Some(value) if !value.is_empty() && Mode::parse(value).is_none() => Some(format!(
            "warning: invalid DICTATE_MODE \"{value}\": expected \"whole\", \"pauses\", \"live\" or \"moonshine\""
        )),
        _ => None,
    };
    (mode, streaming, warning)
}

/// Resolve the effective switches from the file settings plus the current
/// environment. The daemon calls this once with the loaded file; short-lived
/// processes call it with defaults so `DICTATE_*=0` still applies.
pub(crate) fn init_effective(settings: &Settings) {
    refresh_from(
        settings.sounds,
        settings.notifications,
        settings.mode,
        settings.model,
        settings.inference_threads,
    );
}

fn refresh_from(
    sounds: bool,
    notifications: bool,
    mode: Mode,
    model: SpeechModel,
    inference_threads: InferenceThreads,
) {
    SOUNDS_ON.store(
        apply_env(sounds, std::env::var("DICTATE_SOUNDS").ok().as_deref()),
        Ordering::SeqCst,
    );
    NOTIFY_ON.store(
        apply_env(
            notifications,
            std::env::var("DICTATE_NOTIFY").ok().as_deref(),
        ),
        Ordering::SeqCst,
    );
    refresh_selection(mode, model);
    refresh_inference_threads(inference_threads);
}

/// Effective sound-cue switch, read by `cue` on every start/stop.
pub(crate) fn sounds_on() -> bool {
    SOUNDS_ON.load(Ordering::SeqCst)
}

/// Effective notification switch, read by `notify` on every event.
pub(crate) fn notifications_on() -> bool {
    NOTIFY_ON.load(Ordering::SeqCst)
}

/// Effective dictation mode, read by the daemon on every start.
pub(crate) fn mode() -> Mode {
    Mode::from_discriminant(MODE.load(Ordering::SeqCst))
}

/// Effective speech model for Whole/Pauses, read when a model is loaded.
pub(crate) fn speech_model() -> SpeechModel {
    SpeechModel::from_discriminant(SPEECH_MODEL.load(Ordering::SeqCst))
}

/// Effective inference-thread setting for Whisper/Moonshine.
pub(crate) fn inference_threads() -> InferenceThreads {
    InferenceThreads::from_discriminant(INFERENCE_THREADS.load(Ordering::SeqCst))
}

/// Concrete Whisper thread count, based on the effective setting and the
/// machine's logical parallelism. Moonshine uses its own policy.
pub(crate) fn small_model_threads() -> usize {
    inference_threads().count(
        std::thread::available_parallelism()
            .ok()
            .map(|threads| threads.get()),
    )
}

/// Update the effective sound switch after the tray changed the file value
/// (env still wins).
pub(crate) fn refresh_sounds(file_value: bool) {
    SOUNDS_ON.store(
        apply_env(file_value, std::env::var("DICTATE_SOUNDS").ok().as_deref()),
        Ordering::SeqCst,
    );
}

/// Update the effective notification switch after the tray changed the file
/// value (env still wins).
pub(crate) fn refresh_notifications(file_value: bool) {
    NOTIFY_ON.store(
        apply_env(file_value, std::env::var("DICTATE_NOTIFY").ok().as_deref()),
        Ordering::SeqCst,
    );
}

/// Update the effective selection after a settings change (env still wins).
/// Invalid environment overrides warn on stderr and fall through.
pub(crate) fn refresh_selection(file_mode: Mode, file_model: SpeechModel) {
    let (mode_env, streaming_env, warning) = mode_env_now();
    if let Some(warning) = warning {
        eprintln!("{warning}");
    }
    let model_env = std::env::var("DICTATE_SPEECH_MODEL").ok();
    if let Some(value) = model_env.as_deref()
        && !value.is_empty()
        && ModelChoice::parse(value).is_none()
    {
        eprintln!(
            "warning: invalid DICTATE_SPEECH_MODEL {value:?}: expected standard, fluid, whisper, nemotron or moonshine"
        );
    }
    let (mode, model) = resolve_selection(
        file_mode,
        file_model,
        mode_env.as_deref(),
        streaming_env.as_deref(),
        model_env.as_deref(),
    );
    MODE.store(mode as u8, Ordering::SeqCst);
    SPEECH_MODEL.store(model as u8, Ordering::SeqCst);
}

/// Resolve environment overrides as a compatible selection. A model override
/// chooses its compatible mode; a Live mode override keeps a chosen Moonshine.
pub(crate) fn resolve_selection(
    file_mode: Mode,
    file_model: SpeechModel,
    mode_env: Option<&str>,
    streaming_env: Option<&str>,
    model_env: Option<&str>,
) -> (Mode, SpeechModel) {
    let mut mode = resolve_mode(file_mode, mode_env, streaming_env);
    if mode == Mode::Live && file_mode == Mode::Moonshine {
        mode = Mode::Moonshine;
    }
    let mut batch = resolve_speech_model(file_model, model_env);
    if let Some(model) = model_env.and_then(ModelChoice::parse) {
        let selected = Selection::from_runtime(mode, batch).with_model(model);
        mode = selected.runtime_mode();
        if let Some(model) = selected.model.batch_model() {
            batch = model;
        }
    }
    (mode, batch)
}

pub(crate) fn effective_selection(settings: &Settings) -> Selection {
    let (mode, model) = resolve_selection(
        settings.mode,
        settings.model,
        std::env::var("DICTATE_MODE").ok().as_deref(),
        std::env::var("DICTATE_STREAMING").ok().as_deref(),
        std::env::var("DICTATE_SPEECH_MODEL").ok().as_deref(),
    );
    Selection::from_runtime(mode, model)
}

/// Update the effective inference-thread setting after the settings file or
/// page changed it (env still wins). An invalid
/// `DICTATE_INFERENCE_THREADS` warns on stderr and falls through.
pub(crate) fn refresh_inference_threads(file_value: InferenceThreads) {
    let env = std::env::var("DICTATE_INFERENCE_THREADS").ok();
    if let Some(value) = env.as_deref()
        && !value.is_empty()
        && InferenceThreads::parse(value).is_none()
    {
        eprintln!(
            "warning: invalid DICTATE_INFERENCE_THREADS \"{value}\": expected \"auto\", \"1\" or \"2\""
        );
    }
    INFERENCE_THREADS.store(
        resolve_inference_threads(file_value, env.as_deref()) as u8,
        Ordering::SeqCst,
    );
}

/// Path of the settings file, if a home directory exists.
pub(crate) fn config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "dictate")
        .map(|dirs| dirs.config_dir().join("config.toml"))
}

/// Rewrite only legacy streaming selections, preserving a complete backup.
/// Ordinary loads and CLI commands stay read-only; the daemon calls this once.
pub(crate) fn migrate_selection(settings: &Settings) -> anyhow::Result<Option<PathBuf>> {
    let Some(path) = config_path() else {
        return Ok(None);
    };
    migrate_selection_at(&path, settings)
}

fn migrate_selection_at(path: &Path, settings: &Settings) -> anyhow::Result<Option<PathBuf>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let mut raw_mode = None;
    let mut raw_model = None;
    for line in text.lines() {
        let Some((key, value)) = strip_comment(line).split_once('=') else {
            continue;
        };
        if let Ok(value) = parse_string(value.trim()) {
            match key.trim() {
                "mode" => raw_mode = Mode::parse(&value),
                "model" => raw_model = ModelChoice::parse(&value),
                _ => {}
            }
        }
    }
    let legacy = raw_mode == Some(Mode::Moonshine)
        || (raw_mode == Some(Mode::Live)
            && raw_model.is_none_or(|model| model.batch_model().is_some()));
    if !legacy {
        return Ok(None);
    }
    anyhow::ensure!(
        parse(&text).map_err(anyhow::Error::msg)? == *settings,
        "Settings changed during migration; leaving the file untouched"
    );
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let backup = path.with_extension(format!("toml.before-mode-model-split-{stamp}.bak"));
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&backup)?;
    file.set_permissions(std::fs::metadata(path)?.permissions())?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    anyhow::ensure!(
        std::fs::read_to_string(path)? == text,
        "Settings changed during migration; backup retained at {}",
        backup.display()
    );
    save_to(path, settings).with_context(|| {
        format!(
            "Migrating legacy {} selection (backup: {})",
            settings.mode.as_str(),
            backup.display()
        )
    })?;
    Ok(Some(backup))
}

/// Load the settings file. A missing file (or no home directory) yields
/// defaults with no problem; a read or parse failure yields defaults plus a
/// human-readable problem the daemon reports while keeping them.
pub(crate) fn load() -> (Settings, Option<String>) {
    let Some(path) = config_path() else {
        return (Settings::default(), None);
    };
    match std::fs::read_to_string(&path) {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => (Settings::default(), None),
        Err(err) => (
            Settings::default(),
            Some(format!("Could not read {}: {err}", path.display())),
        ),
        Ok(text) => match parse(&text) {
            Ok(settings) => (settings, None),
            Err(problem) => (Settings::default(), Some(problem)),
        },
    }
}

/// Parse settings text. Unknown keys, malformed lines and bad values are
/// errors naming the line; a repeated key keeps its last value.
pub(crate) fn parse(text: &str) -> Result<Settings, String> {
    let mut settings = Settings::default();
    let mut chosen_model = None;
    let mut batch_model = None;
    for (index, raw_line) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!(
                "config.toml line {line_number}: expected key = value"
            ));
        };
        let key = key.trim();
        let value = value.trim();
        match key {
            "sounds" | "notifications" => {
                let parsed = match value {
                    "true" => true,
                    "false" => false,
                    _ => {
                        return Err(format!(
                            "config.toml line {line_number}: invalid value for \"{key}\": expected true or false"
                        ));
                    }
                };
                if key == "sounds" {
                    settings.sounds = parsed;
                } else {
                    settings.notifications = parsed;
                }
            }
            // Legacy key: `streaming = true` means pauses mode, `false`
            // means whole. The next save writes `mode` instead.
            "streaming" => {
                let parsed = match value {
                    "true" => Mode::Pauses,
                    "false" => Mode::Whole,
                    _ => {
                        return Err(format!(
                            "config.toml line {line_number}: invalid value for \"streaming\": expected true or false"
                        ));
                    }
                };
                settings.mode = parsed;
            }
            "mode" => {
                let parsed = parse_string(value).map_err(|problem| {
                    format!("config.toml line {line_number}: invalid value for \"mode\": {problem}")
                })?;
                settings.mode = Mode::parse(&parsed).ok_or_else(|| {
                    format!(
                        "config.toml line {line_number}: invalid value for \"mode\": expected \"whole\", \"pauses\", \"live\" or \"moonshine\""
                    )
                })?;
            }
            "model" => {
                let parsed = parse_string(value).map_err(|problem| {
                    format!(
                        "config.toml line {line_number}: invalid value for \"model\": {problem}"
                    )
                })?;
                chosen_model = Some(ModelChoice::parse(&parsed).ok_or_else(|| {
                    format!(
                        "config.toml line {line_number}: invalid value for \"model\": expected \"standard\", \"fluid\" or \"whisper\" (Whole/Pauses), or \"nemotron\" / \"moonshine\" (Live)"
                    )
                })?);
            }
            "batch_model" => {
                let parsed = parse_string(value).map_err(|problem| {
                    format!("config.toml line {line_number}: invalid batch_model: {problem}")
                })?;
                batch_model = Some(SpeechModel::parse(&parsed).ok_or_else(|| format!("config.toml line {line_number}: batch_model must be standard, fluid or whisper"))?);
            }
            "inference_threads" => {
                let parsed = parse_string(value).map_err(|problem| {
                    format!(
                        "config.toml line {line_number}: invalid value for \"inference_threads\": {problem}"
                    )
                })?;
                settings.inference_threads = InferenceThreads::parse(&parsed).ok_or_else(|| {
                    format!(
                        "config.toml line {line_number}: invalid value for \"inference_threads\": expected \"auto\", \"1\" or \"2\""
                    )
                })?;
            }
            "unload_after" => {
                let parsed = parse_string(value).map_err(|problem| {
                    format!(
                        "config.toml line {line_number}: invalid value for \"unload_after\": {problem}"
                    )
                })?;
                settings.unload_after = UnloadAfter::parse(&parsed).ok_or_else(|| {
                    format!(
                        "config.toml line {line_number}: invalid value for \"unload_after\": expected \"never\", \"5\", \"20\", \"30\" or \"60\""
                    )
                })?;
            }
            "hotkey" => {
                settings.hotkey = parse_string(value).map_err(|problem| {
                    format!(
                        "config.toml line {line_number}: invalid value for \"hotkey\": {problem}"
                    )
                })?;
            }
            "microphone" => {
                settings.microphone = parse_string(value).map_err(|problem| {
                    format!(
                        "config.toml line {line_number}: invalid value for \"microphone\": {problem}"
                    )
                })?;
            }
            _ => {
                return Err(format!(
                    "config.toml line {line_number}: unknown key \"{key}\""
                ));
            }
        }
    }
    if let Some(model) = chosen_model {
        if let Some(batch) = model.batch_model() {
            settings.model = batch;
        } else {
            if !settings.mode.is_streaming() {
                return Err(format!("{} requires mode = \"live\"", model.display_name()));
            }
            Selection::new(DictationMode::Live, model)?.apply(&mut settings);
        }
    }
    if settings.mode.is_streaming()
        && let Some(batch) = batch_model
    {
        settings.model = batch;
    }
    Ok(settings)
}

/// Cut a `#` comment starting outside a quoted string.
fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    let mut escaped = false;
    for (index, c) in line.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
        } else if c == '#' {
            return line[..index].trim_end();
        }
    }
    line
}

/// Parse one `"string"` value with `\"` and `\\` escapes.
fn parse_string(value: &str) -> Result<String, &'static str> {
    let rest = value.strip_prefix('"').ok_or("expected a quoted string")?;
    let mut out = String::new();
    let mut chars = rest.chars();
    loop {
        match chars.next() {
            None => return Err("unterminated string"),
            Some('\\') => match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some(_) => return Err("unsupported escape sequence"),
                None => return Err("unterminated string"),
            },
            Some('"') => break,
            Some(c) => out.push(c),
        }
    }
    if !chars.as_str().trim().is_empty() {
        return Err("unexpected text after value");
    }
    Ok(out)
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Render all keys, each with a one-line comment.
fn render(settings: &Settings) -> String {
    let selection = Selection::from_settings(settings);
    let mut text = format!(
        "# Play a short tone when recording starts and stops.\nsounds = {}\n# Show desktop notifications for errors and empty recordings.\nnotifications = {}\n# Dictation mode: \"whole\" (type once at stop), \"pauses\" (type at pauses), or \"live\" (stream while speaking).\nmode = \"{}\"\n# Speech model for whole/pauses: \"standard\" (Parakeet), \"fluid\" or \"whisper\". Live: \"nemotron\" or \"moonshine\".\nmodel = \"{}\"\n# Inference threads: \"auto\" (Whisper up to 2, Moonshine single-thread), \"1\" or \"2\" (Moonshine 2 = default threading, not exactly 2). Restart the daemon after changing it.\ninference_threads = \"{}\"\n# Free the speech model's memory after this many idle minutes: \"never\", \"5\", \"20\" (default), \"30\" or \"60\". The next recording reloads it.\nunload_after = \"{}\"\n# Global hotkey: tap to toggle, hold to talk. macOS names: Cmd, Option, Ctrl, Shift (e.g. Cmd+Option+Space); modifier-only shortcuts (at least two modifiers, macOS only, need Input Monitoring; pressing another key cancels).\nhotkey = \"{}\"\n# Microphone: \"\" means the system default input, or a device id (see \"dictate mics\").\nmicrophone = \"{}\"\n",
        settings.sounds,
        settings.notifications,
        selection.mode.as_str(),
        selection.model.as_str(),
        settings.inference_threads.as_str(),
        settings.unload_after.as_str(),
        escape(&settings.hotkey),
        escape(&settings.microphone)
    );
    if settings.mode.is_streaming() {
        text.push_str(&format!("# Last Whole/Pauses model, retained for returning from Live and startup fallback.\nbatch_model = \"{}\"\n", settings.model.as_str()));
    }
    text
}

/// Save the settings to the settings file, atomically (write `.tmp`, rename).
pub(crate) fn save(settings: &Settings) -> anyhow::Result<()> {
    let Some(path) = config_path() else {
        anyhow::bail!("Could not find a home directory for this user");
    };
    save_to(&path, settings)
}

/// Save the settings to `path`, atomically (write `.tmp`, rename).
pub(crate) fn save_to(path: &Path, settings: &Settings) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Could not create directory {}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .map(|name| format!("{}.tmp", name.to_string_lossy()))
        .unwrap_or_else(|| "config.toml.tmp".to_string());
    let tmp = path.with_file_name(file_name);
    std::fs::write(&tmp, render(settings))
        .with_context(|| format!("Could not write {}", tmp.display()))?;
    if let Err(err) = std::fs::rename(&tmp, path)
        .with_context(|| format!("Could not move {} to {}", tmp.display(), path.display()))
    {
        if let Err(cleanup) = std::fs::remove_file(&tmp) {
            eprintln!("warning: could not delete {}: {cleanup}", tmp.display());
        }
        return Err(err);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn fresh_path(tag: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "dictate-test-config-{}-{tag}-{n}/config.toml",
            std::process::id()
        ))
    }

    #[test]
    fn empty_file_means_defaults() {
        assert_eq!(parse("").unwrap(), Settings::default());
        assert_eq!(
            parse("# only a comment\n\n  \n").unwrap(),
            Settings::default()
        );
        assert_eq!(Settings::default().mode, Mode::Whole);
    }

    #[test]
    fn parses_all_keys() {
        let settings = parse(
            "sounds = false\nnotifications = false\nmode = \"live\"\nhotkey = \"F8\"\nmicrophone = \"pulseaudio:alsa_input.pci\"\n",
        )
        .unwrap();
        assert!(!settings.sounds);
        assert!(!settings.notifications);
        assert_eq!(settings.mode, Mode::Live);
        assert_eq!(settings.hotkey, "F8");
        assert_eq!(settings.microphone, "pulseaudio:alsa_input.pci");
        assert_eq!(parse("mode = \"whole\"\n").unwrap().mode, Mode::Whole);
        assert_eq!(parse("mode = \"pauses\"\n").unwrap().mode, Mode::Pauses);
        assert_eq!(
            parse("mode = \"moonshine\"\n").unwrap().mode,
            Mode::Moonshine
        );
    }

    #[test]
    fn legacy_streaming_key_migrates_to_mode() {
        assert_eq!(parse("streaming = true\n").unwrap().mode, Mode::Pauses);
        assert_eq!(parse("streaming = false\n").unwrap().mode, Mode::Whole);
        // The last occurrence wins, across both spellings.
        assert_eq!(
            parse("streaming = true\nmode = \"whole\"\n").unwrap().mode,
            Mode::Whole
        );
        assert_eq!(
            parse("mode = \"live\"\nstreaming = false\n").unwrap().mode,
            Mode::Whole
        );
        // A migrated save writes `mode` and drops `streaming`.
        let migrated = parse("streaming = true\n").unwrap();
        let text = render(&migrated);
        assert!(text.contains("mode = \"pauses\""), "unexpected:\n{text}");
        assert!(!text.contains("streaming"), "unexpected:\n{text}");
    }

    #[test]
    fn bad_mode_says_expected_values() {
        let err = parse("mode = \"fast\"\n").unwrap_err();
        assert!(err.contains("line 1"), "unexpected: {err}");
        assert!(
            err.contains("expected \"whole\", \"pauses\", \"live\" or \"moonshine\""),
            "unexpected: {err}"
        );
        let err = parse("mode = true\n").unwrap_err();
        assert!(
            err.contains("invalid value for \"mode\""),
            "unexpected: {err}"
        );
    }

    #[test]
    fn mode_env_precedence() {
        // No env: the file wins (including live).
        assert_eq!(resolve_mode(Mode::Live, None, None), Mode::Live);
        assert_eq!(resolve_mode(Mode::Whole, None, None), Mode::Whole);
        // DICTATE_MODE wins over everything.
        assert_eq!(
            resolve_mode(Mode::Whole, Some("live"), Some("0")),
            Mode::Live
        );
        assert_eq!(
            resolve_mode(Mode::Live, Some("whole"), Some("1")),
            Mode::Whole
        );
        // Empty DICTATE_MODE counts as unset.
        assert_eq!(resolve_mode(Mode::Whole, Some(""), None), Mode::Whole);
        // Invalid DICTATE_MODE falls through to DICTATE_STREAMING / file.
        assert_eq!(
            resolve_mode(Mode::Whole, Some("fast"), Some("1")),
            Mode::Pauses
        );
        assert_eq!(resolve_mode(Mode::Live, Some("fast"), None), Mode::Live);
        // Legacy DICTATE_STREAMING when DICTATE_MODE is unset.
        assert_eq!(resolve_mode(Mode::Whole, None, Some("1")), Mode::Pauses);
        assert_eq!(resolve_mode(Mode::Whole, None, Some("0")), Mode::Whole);
        assert_eq!(resolve_mode(Mode::Live, None, Some("0")), Mode::Whole);
        assert_eq!(resolve_mode(Mode::Live, None, Some("")), Mode::Pauses);
    }

    #[test]
    fn mode_display_names() {
        assert_eq!(Mode::Whole.display_name(), "Whole");
        assert_eq!(Mode::Pauses.display_name(), "Pauses");
        assert_eq!(Mode::Live.display_name(), "Live");
        assert_eq!(Mode::Moonshine.display_name(), "Live");
        assert_eq!(Mode::Moonshine.as_str(), "moonshine");
        assert!(Mode::Live.is_streaming());
        assert!(Mode::Moonshine.is_streaming());
        assert!(!Mode::Whole.is_streaming());
    }

    #[test]
    fn parses_speech_model_and_defaults_to_standard() {
        assert_eq!(Settings::default().model, SpeechModel::Standard);
        assert_eq!(
            parse("model = \"fluid\"\n").unwrap().model,
            SpeechModel::Fluid
        );
        assert_eq!(
            parse("model = \"standard\"\n").unwrap().model,
            SpeechModel::Standard
        );
        assert_eq!(
            parse("model = \"whisper\"\n").unwrap().model,
            SpeechModel::Whisper
        );
    }

    #[test]
    fn bad_speech_model_says_expected_values() {
        let err = parse("model = \"large\"\n").unwrap_err();
        assert!(err.contains("line 1"), "unexpected: {err}");
        assert!(
            err.contains("expected \"standard\", \"fluid\" or \"whisper\""),
            "unexpected: {err}"
        );
        let err = parse("model = true\n").unwrap_err();
        assert!(
            err.contains("invalid value for \"model\""),
            "unexpected: {err}"
        );
    }

    #[test]
    fn speech_model_env_precedence() {
        assert_eq!(
            resolve_speech_model(SpeechModel::Standard, Some("fluid")),
            SpeechModel::Fluid
        );
        assert_eq!(
            resolve_speech_model(SpeechModel::Fluid, Some("whisper")),
            SpeechModel::Whisper
        );
        assert_eq!(
            resolve_speech_model(SpeechModel::Whisper, Some("standard")),
            SpeechModel::Standard
        );
        // Empty and invalid values fall through to the file value.
        assert_eq!(
            resolve_speech_model(SpeechModel::Fluid, Some("")),
            SpeechModel::Fluid
        );
        assert_eq!(
            resolve_speech_model(SpeechModel::Whisper, Some("large")),
            SpeechModel::Whisper
        );
        assert_eq!(
            resolve_speech_model(SpeechModel::Standard, None),
            SpeechModel::Standard
        );
    }

    #[test]
    fn speech_model_display_names() {
        assert_eq!(SpeechModel::Standard.display_name(), "Parakeet TDT 0.6B v2");
        assert_eq!(SpeechModel::Fluid.display_name(), "Fluid Speech Mini");
        assert_eq!(SpeechModel::Whisper.display_name(), "Whisper Tiny English");
        assert_eq!(SpeechModel::Standard.as_str(), "standard");
        assert_eq!(SpeechModel::Fluid.as_str(), "fluid");
        assert_eq!(SpeechModel::Whisper.as_str(), "whisper");
    }

    #[test]
    fn inference_threads_parse_precedence_and_auto_cap() {
        assert_eq!(
            Settings::default().inference_threads,
            InferenceThreads::Auto
        );
        assert_eq!(
            parse("inference_threads = \"1\"\n")
                .unwrap()
                .inference_threads,
            InferenceThreads::One
        );
        assert_eq!(
            parse("inference_threads = \"2\"\n")
                .unwrap()
                .inference_threads,
            InferenceThreads::Two
        );
        let err = parse("inference_threads = \"4\"\n").unwrap_err();
        assert!(
            err.contains("expected \"auto\", \"1\" or \"2\""),
            "unexpected: {err}"
        );
        assert_eq!(
            resolve_inference_threads(InferenceThreads::Auto, Some("2")),
            InferenceThreads::Two
        );
        assert_eq!(
            resolve_inference_threads(InferenceThreads::Two, Some("1")),
            InferenceThreads::One
        );
        assert_eq!(
            resolve_inference_threads(InferenceThreads::One, Some("bogus")),
            InferenceThreads::One
        );
        assert_eq!(
            resolve_inference_threads(InferenceThreads::Two, Some("")),
            InferenceThreads::Two
        );
        assert_eq!(InferenceThreads::Auto.count(Some(4)), 2);
        assert_eq!(InferenceThreads::Auto.count(Some(8)), 2);
        assert_eq!(InferenceThreads::Auto.count(Some(2)), 1);
        assert_eq!(InferenceThreads::Auto.count(None), 1);
        assert_eq!(InferenceThreads::One.count(Some(8)), 1);
        assert_eq!(InferenceThreads::Two.count(Some(1)), 2);
    }

    #[test]
    fn microphone_defaults_to_empty_and_parses() {
        assert_eq!(Settings::default().microphone, "");
        let settings = parse("microphone = \"pulseaudio:alsa_input.pci\"\n").unwrap();
        assert_eq!(settings.microphone, "pulseaudio:alsa_input.pci");
        let settings = parse("microphone = \"\"\n").unwrap();
        assert_eq!(settings.microphone, "");
        let err = parse("microphone = pulseaudio:x\n").unwrap_err();
        assert!(
            err.contains("invalid value for \"microphone\""),
            "unexpected: {err}"
        );
    }

    #[test]
    fn microphone_env_precedence() {
        // Env wins when set and non-empty, even over a file value.
        assert_eq!(
            resolve_microphone("pulseaudio:file", Some("pulseaudio:env")),
            Some("pulseaudio:env".to_string())
        );
        // Empty env counts as unset: the file wins.
        assert_eq!(
            resolve_microphone("pulseaudio:file", Some("")),
            Some("pulseaudio:file".to_string())
        );
        assert_eq!(
            resolve_microphone("pulseaudio:file", None),
            Some("pulseaudio:file".to_string())
        );
        // Empty on both sides means the system default.
        assert_eq!(resolve_microphone("", Some("")), None);
        assert_eq!(resolve_microphone("", None), None);
    }

    #[test]
    fn ignores_comments_and_whitespace() {
        let settings = parse(
            "  # leading comment\n  sounds=false  # trailing comment\n\nhotkey = \"a#b\" # keep the hash\n",
        )
        .unwrap();
        assert!(!settings.sounds);
        assert!(settings.notifications);
        assert_eq!(settings.hotkey, "a#b");
    }

    #[test]
    fn unload_after_parses_presets_only() {
        assert_eq!(Settings::default().unload_after, UnloadAfter::Minutes20);
        assert_eq!(
            parse("unload_after = \"30\"\n").unwrap().unload_after,
            UnloadAfter::Minutes30
        );
        assert!(parse("unload_after = \"7\"\n").is_err());
        assert_eq!(UnloadAfter::Never.duration(), None);
        assert_eq!(
            UnloadAfter::Minutes5.duration(),
            Some(std::time::Duration::from_secs(300))
        );
    }

    #[test]
    fn repeated_key_keeps_last_value() {
        let settings = parse("sounds = false\nsounds = true\n").unwrap();
        assert!(settings.sounds);
    }

    #[test]
    fn round_trips_through_render() {
        for settings in [
            Settings::default(),
            Settings {
                sounds: false,
                notifications: false,
                mode: Mode::Pauses,
                model: SpeechModel::Fluid,
                inference_threads: InferenceThreads::Two,
                unload_after: UnloadAfter::Minutes20,
                hotkey: "say \"hi\" \\ now".to_string(),
                microphone: "pulseaudio:alsa_input.pci".to_string(),
            },
            Settings {
                mode: Mode::Live,
                model: SpeechModel::Whisper,
                inference_threads: InferenceThreads::One,
                ..Settings::default()
            },
        ] {
            let text = render(&settings);
            assert_eq!(parse(&text).unwrap(), settings, "round trip failed");
        }
        for mode in [Mode::Whole, Mode::Pauses, Mode::Live, Mode::Moonshine] {
            for model in [
                SpeechModel::Standard,
                SpeechModel::Fluid,
                SpeechModel::Whisper,
            ] {
                let settings = Settings {
                    mode,
                    model,
                    ..Settings::default()
                };
                let text = render(&settings);
                let restored = parse(&text).unwrap();
                assert_eq!(restored, settings, "lost selection or fallback: {text}");
                assert_eq!(
                    Selection::from_settings(&restored),
                    Selection::from_settings(&settings)
                );
            }
        }
    }

    #[test]
    fn unknown_key_names_line_and_key() {
        assert_eq!(
            parse("sounds = true\nfoo = 1\n").unwrap_err(),
            "config.toml line 2: unknown key \"foo\""
        );
    }

    #[test]
    fn bad_bool_says_expected_true_or_false() {
        let err = parse("sounds = yes\n").unwrap_err();
        assert!(err.contains("line 1"), "unexpected: {err}");
        assert!(err.contains("expected true or false"), "unexpected: {err}");
    }

    #[test]
    fn unquoted_string_is_rejected() {
        let err = parse("hotkey = F8\n").unwrap_err();
        assert!(
            err.contains("expected a quoted string"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn unterminated_string_is_rejected() {
        assert!(parse("hotkey = \"F8\n").is_err());
    }

    #[test]
    fn trailing_text_after_string_is_rejected() {
        assert!(parse("hotkey = \"F8\" extra\n").is_err());
    }

    #[test]
    fn line_without_equals_is_rejected() {
        let err = parse("sounds\n").unwrap_err();
        assert!(err.contains("expected key = value"), "unexpected: {err}");
    }

    #[test]
    fn env_wins_over_file_value() {
        assert!(apply_env(true, None));
        assert!(!apply_env(false, None));
        assert!(!apply_env(true, Some("0")));
        assert!(!apply_env(false, Some("0")));
        assert!(apply_env(false, Some("1")));
        assert!(apply_env(false, Some("yes")));
        assert!(apply_env(false, Some("")));
    }

    #[test]
    fn legacy_streaming_configs_migrate_without_losing_the_batch_model() {
        for (old, expected) in [
            ("moonshine", ModelChoice::Moonshine),
            ("live", ModelChoice::Nemotron),
        ] {
            for batch in [
                SpeechModel::Standard,
                SpeechModel::Fluid,
                SpeechModel::Whisper,
            ] {
                let legacy = format!(
                    "mode = \"{old}\"\nmodel = \"{}\"\nhotkey = \"F8\"\n",
                    batch.as_str()
                );
                let settings = parse(&legacy).unwrap();
                let selected = Selection::from_settings(&settings);
                assert_eq!(selected.mode, DictationMode::Live);
                assert_eq!(selected.model, expected);
                let migrated = render(&settings);
                assert!(migrated.contains("mode = \"live\""));
                assert!(migrated.contains(&format!("model = \"{}\"", expected.as_str())));
                assert!(migrated.contains(&format!("batch_model = \"{}\"", batch.as_str())));
                assert_eq!(parse(&migrated).unwrap(), settings);
            }
        }
    }

    #[test]
    fn canonical_streaming_model_keys_are_order_independent() {
        let first =
            parse("mode = \"live\"\nmodel = \"moonshine\"\nbatch_model = \"whisper\"\n").unwrap();
        let reordered =
            parse("batch_model = \"whisper\"\nmodel = \"moonshine\"\nmode = \"live\"\n").unwrap();
        assert_eq!(first, reordered);
        assert_eq!(first.mode, Mode::Moonshine);
        assert_eq!(first.model, SpeechModel::Whisper);
        assert_eq!(
            Selection::from_settings(&first).model,
            ModelChoice::Moonshine
        );
        assert!(
            parse("mode = \"pauses\"\nmodel = \"moonshine\"\n")
                .unwrap_err()
                .contains("requires mode")
        );
        assert!(
            parse("mode = \"whole\"\nmodel = \"nemotron\"\n")
                .unwrap_err()
                .contains("requires mode")
        );
    }

    #[test]
    fn environment_model_overrides_resolve_to_compatible_modes() {
        assert_eq!(
            resolve_selection(
                Mode::Whole,
                SpeechModel::Fluid,
                None,
                None,
                Some("moonshine")
            ),
            (Mode::Moonshine, SpeechModel::Fluid)
        );
        assert_eq!(
            resolve_selection(
                Mode::Moonshine,
                SpeechModel::Standard,
                None,
                None,
                Some("whisper")
            ),
            (Mode::Whole, SpeechModel::Whisper)
        );
        assert_eq!(
            resolve_selection(
                Mode::Pauses,
                SpeechModel::Fluid,
                None,
                None,
                Some("whisper")
            ),
            (Mode::Pauses, SpeechModel::Whisper)
        );
        assert_eq!(
            resolve_selection(
                Mode::Moonshine,
                SpeechModel::Whisper,
                Some("live"),
                None,
                None
            ),
            (Mode::Moonshine, SpeechModel::Whisper)
        );
        assert_eq!(
            resolve_selection(
                Mode::Whole,
                SpeechModel::Fluid,
                Some("moonshine"),
                None,
                None
            ),
            (Mode::Moonshine, SpeechModel::Fluid)
        );
    }

    #[test]
    fn migration_backs_up_and_only_runs_once() {
        let path = fresh_path("selection-migration");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let original = "# original comment\nmode = \"moonshine\"\nmodel = \"fluid\"\nsounds = false\nhotkey = \"F8\"\n";
        std::fs::write(&path, original).unwrap();
        let settings = parse(original).unwrap();
        let backup = migrate_selection_at(&path, &settings).unwrap().unwrap();
        assert_eq!(std::fs::read_to_string(backup).unwrap(), original);
        assert_eq!(
            parse(&std::fs::read_to_string(&path).unwrap()).unwrap(),
            settings
        );
        assert!(migrate_selection_at(&path, &settings).unwrap().is_none());
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn save_to_writes_and_reads_back() {
        let path = fresh_path("save");
        let settings = Settings {
            sounds: false,
            notifications: true,
            mode: Mode::Whole,
            model: SpeechModel::Standard,
            inference_threads: InferenceThreads::Auto,
            unload_after: UnloadAfter::Never,
            hotkey: "F8".to_string(),
            microphone: String::new(),
        };
        save_to(&path, &settings).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# Play a short tone"), "unexpected:\n{text}");
        assert!(
            text.contains("# Speech model for whole"),
            "unexpected:\n{text}"
        );
        assert_eq!(parse(&text).unwrap(), settings);
        assert!(!path.with_file_name("config.toml.tmp").exists());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
