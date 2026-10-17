//! Global hotkey for the daemon: tap to toggle, hold to talk.
//!
//! The hotkey comes from `DICTATE_HOTKEY`, defaulting to Ctrl+Shift+Space.
//! [`hotkey_action`] maps each press/release event to a daemon command (or
//! none); the event loop in [`crate::daemon`] owns the timing and tap state.

use std::str::FromStr;

use global_hotkey::{HotKeyState, hotkey::HotKey};

use crate::daemon::{Command, State};

/// Hotkey used when `DICTATE_HOTKEY` is unset or empty.
pub(crate) const DEFAULT_HOTKEY: &str = "Ctrl+Shift+Space";

/// Shortcut presets on macOS: parser strings, labelled per OS at display time.
const MAC_SHORTCUT_PRESETS: [&str; 5] = [
    "Ctrl+Shift+Space",
    "Cmd+Shift+Space",
    "Option+Space",
    "Ctrl+Option+D",
    "F8",
];

/// Shortcut presets on Windows and Linux.
const OTHER_SHORTCUT_PRESETS: [&str; 4] =
    ["Ctrl+Shift+Space", "Ctrl+Alt+Space", "Ctrl+Alt+D", "F8"];

/// Shortcut choices offered in the Settings window. Callers pass
/// `cfg!(target_os = "macos")`. Values stay parser strings; only the shown
/// label changes per OS.
pub(crate) fn shortcut_presets(mac: bool) -> &'static [&'static str] {
    if mac {
        &MAC_SHORTCUT_PRESETS
    } else {
        &OTHER_SHORTCUT_PRESETS
    }
}

/// Display label for a stored shortcut value: modifiers renamed per OS, the
/// key part kept as written. Pure, so the combo, tray and logs share it.
pub(crate) fn shortcut_label(value: &str, mac: bool) -> String {
    value
        .split('+')
        .map(|part| {
            let part = part.trim();
            match part.to_ascii_lowercase().as_str() {
                "ctrl" | "control" => {
                    if mac {
                        "Control".to_string()
                    } else {
                        "Ctrl".to_string()
                    }
                }
                "alt" | "option" => if mac { "Option" } else { "Alt" }.to_string(),
                "cmd" | "command" | "super" | "meta" => {
                    if mac { "Command" } else { "Super" }.to_string()
                }
                "shift" => "Shift".to_string(),
                _ => part.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join("+")
}

/// A press held at least this long is push-to-talk; a shorter one is a tap.
pub(crate) const HOLD_THRESHOLD_MS: u64 = 400;

/// The parsed hotkey plus the string shown to the user.
#[derive(Debug)]
pub(crate) struct HotkeyConfig {
    pub hotkey: HotKey,
    pub display: String,
}

/// Resolve the daemon hotkey from the `DICTATE_HOTKEY` value (`None` when
/// unset). An empty value means the default.
pub(crate) fn resolve_hotkey(env: Option<&str>) -> Result<HotkeyConfig, String> {
    match env {
        Some(value) if !value.is_empty() => HotKey::from_str(value)
            .map(|hotkey| HotkeyConfig {
                hotkey,
                display: value.to_string(),
            })
            .map_err(|err| {
                format!("Invalid DICTATE_HOTKEY \"{value}\": {err}. Example: Ctrl+Shift+Space")
            }),
        _ => HotKey::from_str(DEFAULT_HOTKEY)
            .map(|hotkey| HotkeyConfig {
                hotkey,
                display: DEFAULT_HOTKEY.to_string(),
            })
            .map_err(|err| {
                format!(
                    "Internal error: default hotkey \"{DEFAULT_HOTKEY}\" failed to parse: {err}"
                )
            }),
    }
}

/// Resolve a hotkey from the settings file. An empty value means the default.
pub(crate) fn resolve_file_hotkey(value: &str) -> Result<HotkeyConfig, String> {
    if value.is_empty() {
        return resolve_hotkey(None);
    }
    HotKey::from_str(value)
        .map(|hotkey| HotkeyConfig {
            hotkey,
            display: value.to_string(),
        })
        .map_err(|err| {
            format!(
                "Invalid hotkey \"{value}\" in the settings file: {err}. Example: Ctrl+Shift+Space"
            )
        })
}

/// True on Linux in a Wayland session, where apps cannot register global
/// hotkeys. Never true on other systems.
pub(crate) fn wayland_session() -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    std::env::var_os("WAYLAND_DISPLAY").is_some_and(|value| !value.is_empty())
}

/// True on Hyprland, where the Settings page lists the user's compositor
/// binds from `hyprctl`. Never true on other systems.
pub(crate) fn hyprland_session() -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some_and(|value| !value.is_empty())
}

/// Pure tap-vs-hold decision: which daemon command (if any) a hotkey event
/// triggers. `held_ms` is the press length (only used for releases);
/// `tap_armed` is true when a short tap started the current recording and the
/// next press must stop it.
pub(crate) fn hotkey_action(
    state: State,
    event: HotKeyState,
    held_ms: u64,
    tap_armed: bool,
) -> Option<Command> {
    match event {
        HotKeyState::Pressed => match state {
            State::Idle => Some(Command::Start),
            State::Recording if tap_armed => Some(Command::Stop),
            State::Recording | State::Transcribing => None,
        },
        HotKeyState::Released => match state {
            State::Recording if !tap_armed && held_ms >= HOLD_THRESHOLD_MS => Some(Command::Stop),
            _ => None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use global_hotkey::hotkey::{Code, Modifiers};

    #[test]
    fn tap_tap_toggles() {
        // First tap starts recording and arms the toggle.
        assert_eq!(
            hotkey_action(State::Idle, HotKeyState::Pressed, 0, false),
            Some(Command::Start)
        );
        // Short release keeps recording (caller arms the tap).
        assert_eq!(
            hotkey_action(State::Recording, HotKeyState::Released, 120, false),
            None
        );
        // Second press stops; its release is ignored (now transcribing).
        assert_eq!(
            hotkey_action(State::Recording, HotKeyState::Pressed, 0, true),
            Some(Command::Stop)
        );
        assert_eq!(
            hotkey_action(State::Transcribing, HotKeyState::Released, 120, false),
            None
        );
    }

    #[test]
    fn hold_release_is_push_to_talk() {
        assert_eq!(
            hotkey_action(State::Idle, HotKeyState::Pressed, 0, false),
            Some(Command::Start)
        );
        assert_eq!(
            hotkey_action(State::Recording, HotKeyState::Released, 1000, false),
            Some(Command::Stop)
        );
    }

    #[test]
    fn hold_boundary_is_400_ms() {
        assert_eq!(
            hotkey_action(State::Recording, HotKeyState::Released, 400, false),
            Some(Command::Stop)
        );
        assert_eq!(
            hotkey_action(State::Recording, HotKeyState::Released, 399, false),
            None
        );
    }

    #[test]
    fn press_while_transcribing_is_ignored() {
        assert_eq!(
            hotkey_action(State::Transcribing, HotKeyState::Pressed, 0, false),
            None
        );
        assert_eq!(
            hotkey_action(State::Transcribing, HotKeyState::Pressed, 0, true),
            None
        );
        assert_eq!(
            hotkey_action(State::Transcribing, HotKeyState::Released, 500, false),
            None
        );
    }

    #[test]
    fn repeat_press_while_holding_does_nothing() {
        // A press that did not start the recording (hold in progress, or a
        // recording started over the socket) is not a toggle press.
        assert_eq!(
            hotkey_action(State::Recording, HotKeyState::Pressed, 0, false),
            None
        );
    }

    #[test]
    fn resolve_defaults_when_unset_or_empty() {
        for env in [None, Some("")] {
            let config = resolve_hotkey(env).unwrap();
            assert_eq!(config.display, "Ctrl+Shift+Space");
            assert_eq!(config.hotkey.key, Code::Space);
            assert!(config.hotkey.mods.contains(Modifiers::CONTROL));
            assert!(config.hotkey.mods.contains(Modifiers::SHIFT));
        }
    }

    #[test]
    fn resolve_accepts_custom_hotkey() {
        let config = resolve_hotkey(Some("Alt+F9")).unwrap();
        assert_eq!(config.display, "Alt+F9");
        assert_eq!(config.hotkey.key, Code::F9);
        assert!(config.hotkey.mods.contains(Modifiers::ALT));
    }

    #[test]
    fn file_hotkey_resolves_and_defaults_when_empty() {
        let config = resolve_file_hotkey("F8").unwrap();
        assert_eq!(config.display, "F8");
        assert_eq!(config.hotkey.key, Code::F8);
        let config = resolve_file_hotkey("").unwrap();
        assert_eq!(config.display, "Ctrl+Shift+Space");
    }

    #[test]
    fn file_hotkey_rejects_bad_value_with_example() {
        let err = resolve_file_hotkey("Shift+Foo").unwrap_err();
        assert!(
            err.starts_with("Invalid hotkey \"Shift+Foo\" in the settings file: "),
            "unexpected: {err}"
        );
        assert!(
            err.ends_with(". Example: Ctrl+Shift+Space"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn shortcut_labels_rename_modifiers_per_os() {
        let cases = [
            (
                "Ctrl+Shift+Space",
                "Control+Shift+Space",
                "Ctrl+Shift+Space",
            ),
            ("cmd+option+K", "Command+Option+K", "Super+Alt+K"),
            ("Super+Alt+D", "Command+Option+D", "Super+Alt+D"),
            ("F8", "F8", "F8"),
        ];
        for (value, mac_label, other_label) in cases {
            assert_eq!(shortcut_label(value, true), mac_label, "{value} on mac");
            assert_eq!(
                shortcut_label(value, false),
                other_label,
                "{value} elsewhere"
            );
        }
    }

    #[test]
    fn every_shortcut_preset_parses() {
        for mac in [true, false] {
            for preset in shortcut_presets(mac) {
                assert!(HotKey::from_str(preset).is_ok(), "{preset} does not parse");
            }
        }
    }

    #[test]
    fn resolve_rejects_bad_value_with_example() {
        let err = resolve_hotkey(Some("Shift+Foo")).unwrap_err();
        assert!(
            err.starts_with("Invalid DICTATE_HOTKEY \"Shift+Foo\": "),
            "unexpected: {err}"
        );
        assert!(
            err.ends_with(". Example: Ctrl+Shift+Space"),
            "unexpected: {err}"
        );
        assert!(err.contains("Foo"), "unexpected: {err}");
    }
}
