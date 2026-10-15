//! Settings file: `config.toml` in the platform config directory.
//!
//! A tiny TOML subset parsed by hand (no new crates): `#` comments, blank
//! lines, `key = true|false` and `key = "string"`. Keys are `sounds` and
//! `notifications` (bool, default true), `streaming` (bool, default false)
//! and `hotkey` (string, default `Ctrl+Shift+Space`). A missing file means
//! defaults, never an error.
//!
//! The environment still wins: `DICTATE_SOUNDS`, `DICTATE_NOTIFY`,
//! `DICTATE_STREAMING` and `DICTATE_HOTKEY` override the file. The daemon
//! resolves file + env once at startup into the effective switches below,
//! which `cue`, `notify` and the daemon read instead of touching the
//! environment per call.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;

/// Settings from the file (before env overrides).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Settings {
    pub sounds: bool,
    pub notifications: bool,
    pub streaming: bool,
    pub hotkey: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sounds: true,
            notifications: true,
            streaming: false,
            hotkey: crate::hotkey::DEFAULT_HOTKEY.to_string(),
        }
    }
}

/// Effective "play sound cues" switch (file + env, resolved at startup).
static SOUNDS_ON: AtomicBool = AtomicBool::new(true);
/// Effective "show notifications" switch (file + env, resolved at startup).
static NOTIFY_ON: AtomicBool = AtomicBool::new(true);
/// Effective "type while speaking" switch (file + env, resolved at startup).
static STREAMING_ON: AtomicBool = AtomicBool::new(false);

/// Combine one bool setting with its env override: unset env keeps the file
/// value, `"0"` turns it off, anything else turns it on.
pub(crate) fn apply_env(file_value: bool, env: Option<&str>) -> bool {
    match env {
        None => file_value,
        Some("0") => false,
        Some(_) => true,
    }
}

/// Resolve the effective switches from the file settings plus the current
/// environment. The daemon calls this once with the loaded file; short-lived
/// processes call it with defaults so `DICTATE_*=0` still applies.
pub(crate) fn init_effective(settings: &Settings) {
    refresh_from(settings.sounds, settings.notifications, settings.streaming);
}

fn refresh_from(sounds: bool, notifications: bool, streaming: bool) {
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
    STREAMING_ON.store(
        apply_env(
            streaming,
            std::env::var("DICTATE_STREAMING").ok().as_deref(),
        ),
        Ordering::SeqCst,
    );
}

/// Effective sound-cue switch, read by `cue` on every start/stop.
pub(crate) fn sounds_on() -> bool {
    SOUNDS_ON.load(Ordering::SeqCst)
}

/// Effective notification switch, read by `notify` on every event.
pub(crate) fn notifications_on() -> bool {
    NOTIFY_ON.load(Ordering::SeqCst)
}

/// Effective streaming switch, read by the daemon on every start.
pub(crate) fn streaming_on() -> bool {
    STREAMING_ON.load(Ordering::SeqCst)
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

/// Update the effective streaming switch after the tray changed the file
/// value (env still wins).
pub(crate) fn refresh_streaming(file_value: bool) {
    STREAMING_ON.store(
        apply_env(
            file_value,
            std::env::var("DICTATE_STREAMING").ok().as_deref(),
        ),
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
            "sounds" | "notifications" | "streaming" => {
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
                } else if key == "notifications" {
                    settings.notifications = parsed;
                } else {
                    settings.streaming = parsed;
                }
            }
            "hotkey" => {
                settings.hotkey = parse_string(value).map_err(|problem| {
                    format!(
                        "config.toml line {line_number}: invalid value for \"hotkey\": {problem}"
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
        "# Play a short tone when recording starts and stops.\nsounds = {}\n# Show desktop notifications for errors and empty recordings.\nnotifications = {}\n# Type while speaking (off: type once at stop).\nstreaming = {}\n# Global hotkey: tap to toggle, hold to talk.\nhotkey = \"{}\"\n",
        settings.sounds,
        settings.notifications,
        settings.streaming,
        escape(&settings.hotkey)
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
        assert!(!Settings::default().streaming);
    }

    #[test]
    fn parses_all_keys() {
        let settings =
            parse("sounds = false\nnotifications = false\nstreaming = true\nhotkey = \"F8\"\n")
                .unwrap();
        assert!(!settings.sounds);
        assert!(!settings.notifications);
        assert!(settings.streaming);
        assert_eq!(settings.hotkey, "F8");
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
                streaming: true,
                hotkey: "say \"hi\" \\ now".to_string(),
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
            streaming: false,
            hotkey: "F8".to_string(),
        };
        save_to(&path, &settings).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("# Play a short tone"), "unexpected:\n{text}");
        assert_eq!(parse(&text).unwrap(), settings);
        assert!(!path.with_file_name("config.toml.tmp").exists());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
