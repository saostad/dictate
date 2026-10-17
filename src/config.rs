//! Settings file: `config.toml` in the platform config directory.
//!
//! A tiny TOML subset parsed by hand (no new crates): `#` comments, blank
//! lines, `key = true|false` and `key = "string"`. Keys are `sounds` and
//! `notifications` (bool, default true), `mode` (`"whole"`/`"pauses"`/`"live"`,
//! default `"whole"`), `hotkey` (string, default `Ctrl+Shift+Space`) and
//! `microphone` (string, default `""` = the system default input). A missing
//! file means defaults, never an error. On macOS the hotkey modifiers are
//! spelled `Cmd`, `Option`, `Ctrl`, `Shift` (e.g. `Cmd+Option+Space`); the
//! Settings window's `Record shortcut…` button captures one, and shortcuts
//! with only modifier keys aren't supported yet.
//!
//! The environment still wins: `DICTATE_SOUNDS`, `DICTATE_NOTIFY`, `DICTATE_MODE`,
//! `DICTATE_HOTKEY` and `DICTATE_MICROPHONE` override the file (`DICTATE_STREAMING`
//! still works when `DICTATE_MODE` is unset). The daemon resolves file + env once
//! at startup into the effective switches below, which `cue`, `notify` and the
//! daemon read instead of touching the environment per call (the microphone is
//! resolved per recording instead: it is read rarely, and only the daemon's
//! settings hold the file value).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use anyhow::Context;

/// Dictation mode: when the text is typed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Mode {
    /// Type once at stop (most accurate).
    #[default]
    Whole,
    /// Type each pause segment while speaking.
    Pauses,
    /// True streaming: type while speaking (needs a fast machine).
    Live,
}

impl Mode {
    /// The `config.toml` / `DICTATE_MODE` spelling.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Mode::Whole => "whole",
            Mode::Pauses => "pauses",
            Mode::Live => "live",
        }
    }

    /// Short display name for the tray tooltip and the `Mode: <name>`
    /// notification.
    pub(crate) fn display_name(self) -> &'static str {
        match self {
            Mode::Whole => "Whole",
            Mode::Pauses => "Pauses",
            Mode::Live => "Live",
        }
    }

    /// Parse one mode value; `None` for anything else.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "whole" => Some(Mode::Whole),
            "pauses" => Some(Mode::Pauses),
            "live" => Some(Mode::Live),
            _ => None,
        }
    }

    fn from_discriminant(value: u8) -> Self {
        match value {
            1 => Mode::Pauses,
            2 => Mode::Live,
            _ => Mode::Whole,
        }
    }
}

/// Settings from the file (before env overrides).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Settings {
    pub sounds: bool,
    pub notifications: bool,
    pub mode: Mode,
    pub hotkey: String,
    pub microphone: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sounds: true,
            notifications: true,
            mode: Mode::Whole,
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
            "warning: invalid DICTATE_MODE \"{value}\": expected \"whole\", \"pauses\" or \"live\""
        )),
        _ => None,
    };
    (mode, streaming, warning)
}

/// Resolve the effective switches from the file settings plus the current
/// environment. The daemon calls this once with the loaded file; short-lived
/// processes call it with defaults so `DICTATE_*=0` still applies.
pub(crate) fn init_effective(settings: &Settings) {
    refresh_from(settings.sounds, settings.notifications, settings.mode);
}

fn refresh_from(sounds: bool, notifications: bool, mode: Mode) {
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
    refresh_mode(mode);
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

/// Update the effective mode after the tray changed the file value (env still
/// wins). An invalid `DICTATE_MODE` warns on stderr and falls through.
pub(crate) fn refresh_mode(file_value: Mode) {
    let (mode_env, streaming_env, warning) = mode_env_now();
    if let Some(warning) = warning {
        eprintln!("{warning}");
    }
    MODE.store(
        resolve_mode(file_value, mode_env.as_deref(), streaming_env.as_deref()) as u8,
        Ordering::SeqCst,
    );
}

/// Path of the settings file, if a home directory exists.
pub(crate) fn config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "dictate")
        .map(|dirs| dirs.config_dir().join("config.toml"))
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
                        "config.toml line {line_number}: invalid value for \"mode\": expected \"whole\", \"pauses\" or \"live\""
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
    format!(
        "# Play a short tone when recording starts and stops.\nsounds = {}\n# Show desktop notifications for errors and empty recordings.\nnotifications = {}\n# Dictation mode: \"whole\" (type once at stop), \"pauses\" (type at pauses), \"live\" (type while speaking).\nmode = \"{}\"\n# Global hotkey: tap to toggle, hold to talk. macOS names: Cmd, Option, Ctrl, Shift (e.g. Cmd+Option+Space); modifier-only shortcuts aren't supported yet.\nhotkey = \"{}\"\n# Microphone: \"\" means the system default input, or a device id (see \"dictate mics\").\nmicrophone = \"{}\"\n",
        settings.sounds,
        settings.notifications,
        settings.mode.as_str(),
        escape(&settings.hotkey),
        escape(&settings.microphone)
    )
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
            err.contains("expected \"whole\", \"pauses\" or \"live\""),
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
                hotkey: "say \"hi\" \\ now".to_string(),
                microphone: "pulseaudio:alsa_input.pci".to_string(),
            },
            Settings {
                mode: Mode::Live,
                ..Settings::default()
            },
        ] {
            let text = render(&settings);
            assert_eq!(parse(&text).unwrap(), settings, "round trip failed");
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
    fn save_to_writes_and_reads_back() {
        let path = fresh_path("save");
        let settings = Settings {
            sounds: false,
            notifications: true,
            mode: Mode::Whole,
            hotkey: "F8".to_string(),
            microphone: String::new(),
        };
        save_to(&path, &settings).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# Play a short tone"), "unexpected:\n{text}");
        assert_eq!(parse(&text).unwrap(), settings);
        assert!(!path.with_file_name("config.toml.tmp").exists());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
