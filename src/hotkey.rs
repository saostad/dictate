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

/// Modifier keys of a modifier-only shortcut (the macOS event tap watches one
/// of these): Command, Control, Option, Shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModifierSet {
    pub(crate) cmd: bool,
    pub(crate) ctrl: bool,
    pub(crate) option: bool,
    pub(crate) shift: bool,
}

impl ModifierSet {
    /// How many distinct modifiers are set.
    fn distinct_count(self) -> usize {
        [self.cmd, self.ctrl, self.option, self.shift]
            .into_iter()
            .filter(|held| *held)
            .count()
    }
}

/// The modifier set when every `+` part of the value (trimmed,
/// case-insensitive) names a modifier — the same names [`shortcut_label`]
/// maps — else `None`.
fn all_modifiers(value: &str) -> Option<ModifierSet> {
    let mut set = ModifierSet {
        cmd: false,
        ctrl: false,
        option: false,
        shift: false,
    };
    for part in value.split('+') {
        match part.trim().to_ascii_lowercase().as_str() {
            "cmd" | "command" | "super" | "meta" => set.cmd = true,
            "ctrl" | "control" => set.ctrl = true,
            "alt" | "option" => set.option = true,
            "shift" => set.shift = true,
            _ => return None,
        }
    }
    Some(set)
}

/// Parse a modifier-only shortcut value (`"Cmd+Option"`): `Some` only when
/// every part names a modifier and at least two distinct modifiers are set.
#[allow(dead_code)] // wired up in the daemon next
pub(crate) fn parse_modifier_only(value: &str) -> Option<ModifierSet> {
    let set = all_modifiers(value)?;
    (set.distinct_count() >= 2).then_some(set)
}

/// User-facing error for an all-modifier value that can't be used: `None`
/// when the value isn't all modifiers, or when it works (two or more
/// modifiers on macOS).
#[allow(dead_code)] // wired up in the daemon next
pub(crate) fn modifier_only_error(value: &str, mac: bool) -> Option<String> {
    let set = all_modifiers(value)?;
    match set.distinct_count() {
        0 => None,
        1 => Some(
            if mac {
                "Use at least two modifier keys (for example Command+Option), or add a key."
            } else {
                "Use at least two modifier keys (for example Ctrl+Alt), or add a key."
            }
            .to_string(),
        ),
        _ if mac => None,
        _ => Some(
            "Shortcuts with only modifier keys work only on macOS. Add a key, for example Ctrl+Alt+Space."
                .to_string(),
        ),
    }
}

/// A watched modifier chord plus its press state: active once the exact held
/// set matched, until anything changes.
#[allow(dead_code)] // wired up in the daemon next
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChordTracker {
    pub(crate) chord: ModifierSet,
    pub(crate) active: bool,
}

/// What one tap event means for the watched chord.
#[allow(dead_code)] // wired up in the daemon next
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChordEvent {
    Pressed,
    Released,
    Interrupted,
}

#[allow(dead_code)] // wired up in the daemon next
impl ChordTracker {
    /// Feed one flags-changed event: `Pressed` when the held set becomes
    /// exactly the chord, `Released` when an active chord stops matching.
    /// Anything else (including repeats) is `None`.
    pub(crate) fn on_flags(&mut self, held: ModifierSet) -> Option<ChordEvent> {
        if self.active {
            if held == self.chord {
                None
            } else {
                self.active = false;
                Some(ChordEvent::Released)
            }
        } else if held == self.chord {
            self.active = true;
            Some(ChordEvent::Pressed)
        } else {
            None
        }
    }

    /// Feed one plain key-down: `Interrupted` while a chord is active (the
    /// chord stays armed, so its release still reports `Released`).
    pub(crate) fn on_key_down(&mut self) -> Option<ChordEvent> {
        if self.active {
            Some(ChordEvent::Interrupted)
        } else {
            None
        }
    }
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

    fn cmd_option() -> ModifierSet {
        ModifierSet {
            cmd: true,
            ctrl: false,
            option: true,
            shift: false,
        }
    }

    #[test]
    fn modifier_only_values_need_two_distinct_modifiers() {
        assert_eq!(parse_modifier_only("Cmd+Option"), Some(cmd_option()));
        assert_eq!(parse_modifier_only("option+command"), Some(cmd_option()));
        assert_eq!(parse_modifier_only("Cmd"), None);
        assert_eq!(parse_modifier_only("Cmd+Space"), None);
        assert_eq!(parse_modifier_only(""), None);
        // The same modifier twice is one distinct modifier.
        assert_eq!(parse_modifier_only("Cmd+Command"), None);
        // Trimmed and case-insensitive.
        assert_eq!(
            parse_modifier_only("  ctrl + ALT + shift "),
            Some(ModifierSet {
                cmd: false,
                ctrl: true,
                option: true,
                shift: true,
            })
        );
    }

    #[test]
    fn modifier_only_errors_name_the_fix() {
        assert_eq!(
            modifier_only_error("Cmd", true).as_deref(),
            Some("Use at least two modifier keys (for example Command+Option), or add a key.")
        );
        assert_eq!(
            modifier_only_error("Ctrl", false).as_deref(),
            Some("Use at least two modifier keys (for example Ctrl+Alt), or add a key.")
        );
        assert_eq!(modifier_only_error("Cmd+Option", true), None);
        assert_eq!(
            modifier_only_error("Ctrl+Alt", false).as_deref(),
            Some(
                "Shortcuts with only modifier keys work only on macOS. Add a key, for example Ctrl+Alt+Space."
            )
        );
        assert_eq!(modifier_only_error("Ctrl+Shift+Space", true), None);
        assert_eq!(modifier_only_error("not a shortcut", false), None);
    }

    #[test]
    fn chord_presses_only_on_the_exact_held_set() {
        let chord = cmd_option();
        let mut tracker = ChordTracker {
            chord,
            active: false,
        };
        // A superset does not press.
        let held = ModifierSet {
            shift: true,
            ..chord
        };
        assert_eq!(tracker.on_flags(held), None);
        assert!(!tracker.active);
        // Neither does a subset.
        let held = ModifierSet {
            option: false,
            ..chord
        };
        assert_eq!(tracker.on_flags(held), None);
        // The exact set presses; repeats don't.
        assert_eq!(tracker.on_flags(chord), Some(ChordEvent::Pressed));
        assert_eq!(tracker.on_flags(chord), None);
    }

    #[test]
    fn chord_press_releases_on_any_change() {
        let chord = cmd_option();
        let mut tracker = ChordTracker {
            chord,
            active: false,
        };
        // Holding Cmd, then adding Option, presses.
        let cmd_only = ModifierSet {
            option: false,
            ..chord
        };
        assert_eq!(tracker.on_flags(cmd_only), None);
        assert_eq!(tracker.on_flags(chord), Some(ChordEvent::Pressed));
        // Releasing only Option releases; a repeat of that is quiet.
        assert_eq!(tracker.on_flags(cmd_only), Some(ChordEvent::Released));
        assert_eq!(tracker.on_flags(cmd_only), None);
    }

    #[test]
    fn chord_key_down_interrupts_but_stays_armed() {
        let chord = cmd_option();
        let mut tracker = ChordTracker {
            chord,
            active: false,
        };
        assert_eq!(tracker.on_key_down(), None);
        assert_eq!(tracker.on_flags(chord), Some(ChordEvent::Pressed));
        assert_eq!(tracker.on_key_down(), Some(ChordEvent::Interrupted));
        // Still armed, so a later release still reports Released.
        let empty = ModifierSet {
            cmd: false,
            ctrl: false,
            option: false,
            shift: false,
        };
        assert_eq!(tracker.on_flags(empty), Some(ChordEvent::Released));
        assert_eq!(tracker.on_key_down(), None);
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
