//! Tray icon with a settings menu for the daemon.
//!
//! The icon is a filled circle, grey when idle, red while recording, amber
//! while transcribing. The menu shows the live state, starts/stops/cancels
//! dictation, toggles sound cues and notifications, chooses the microphone and
//! switches the dictation mode (all saved to the settings file), switches the
//! global shortcut, opens the settings file, and quits. Menu clicks arrive as
//! `UserEvent::Menu` via a winit proxy, like hotkeys.

use anyhow::Context;
use tray_icon::menu::{CheckMenuItem, Menu, MenuId, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use crate::config::{Mode, Settings};
use crate::daemon::State;
use crate::mic::Mic;

/// Icon edge length in pixels.
pub(crate) const ICON_SIZE: u32 = 32;
/// Idle icon color: grey.
const IDLE_GREY: [u8; 4] = [158, 158, 158, 255];
/// Recording icon color: red (#E53935).
const RECORDING_RED: [u8; 4] = [229, 57, 53, 255];
/// Transcribing icon color: amber (#FFB300).
const TRANSCRIBING_AMBER: [u8; 4] = [255, 179, 0, 255];

/// Shortcut choices offered in the tray menu.
pub(crate) const SHORTCUT_OPTIONS: [&str; 4] =
    ["Ctrl+Shift+Space", "Ctrl+Alt+Space", "Ctrl+Alt+D", "F8"];

/// Wayland note shown instead of shortcut choices (apps can't take shortcuts).
pub(crate) const WAYLAND_SHORTCUT_NOTE: &str =
    "Set in your compositor (Hyprland: F8 / Super+Alt+D)";

/// README section explaining the compositor keybinds.
pub(crate) const README_USE_IT_URL: &str = "https://github.com/saostad/dictate#use-it";

/// Stable tray identity (SNI id and title), so bars can pin the icon.
pub(crate) const TRAY_ID: &str = "dictate";

/// Modes offered in the Mode submenu, in order, with their radio labels.
pub(crate) const MODE_OPTIONS: [(Mode, &str); 3] = [
    (Mode::Whole, "Whole recording (most accurate)"),
    (Mode::Pauses, "Type at pauses"),
    (Mode::Live, "Live — text as you speak (fast computers)"),
];

/// Text of the disabled status label while a mode switch loads in the
/// background.
pub(crate) const LOADING_TEXT: &str = "dictate — Loading…";

/// Pixels, width and height of the tray icon for `state`: a filled circle in
/// the state color on a transparent background.
pub(crate) fn icon_rgba(state: State) -> (Vec<u8>, u32, u32) {
    let color = match state {
        State::Idle => IDLE_GREY,
        State::Recording => RECORDING_RED,
        State::Transcribing => TRANSCRIBING_AMBER,
    };
    let size = ICON_SIZE as i32;
    let center = size / 2;
    let radius = 12;
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let dx = x - center;
            let dy = y - center;
            if dx * dx + dy * dy <= radius * radius {
                rgba.extend_from_slice(&color);
            } else {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
            }
        }
    }
    (rgba, ICON_SIZE, ICON_SIZE)
}

/// Text of the disabled status label for `state`, with the microphone in
/// use when known.
pub(crate) fn status_text(state: State, mic: Option<&str>) -> String {
    let plain = match state {
        State::Idle => "dictate — Idle",
        State::Recording => "dictate — Recording…",
        State::Transcribing => "dictate — Transcribing…",
    };
    match (state, mic) {
        (State::Idle, Some(mic)) => format!("{plain} · {mic}"),
        (_, Some(mic)) => format!("{plain} ({mic})"),
        (_, None) => plain.to_string(),
    }
}

/// Text of the start/stop item for `state`.
pub(crate) fn toggle_text(state: State) -> &'static str {
    match state {
        State::Idle => "Start dictation",
        State::Recording | State::Transcribing => "Stop dictation",
    }
}

/// Tooltip text for `state`, `mode` and the microphone in use: the status
/// (with the mic) plus the mode name, so hovering the icon shows both.
pub(crate) fn tooltip_text(state: State, mode: Mode, mic: Option<&str>) -> String {
    format!("{} ({})", status_text(state, mic), mode.display_name())
}

/// Label of the `System default` microphone menu item: with the current
/// default microphone name when known.
pub(crate) fn system_default_label(default_name: Option<&str>) -> String {
    match default_name {
        Some(name) => format!("System default ({name})"),
        None => "System default".to_string(),
    }
}

/// A settings change made from the tray menu: a checkbox toggle, a mode or
/// microphone switch, or a shortcut switch. Pure, so the confirmation text
/// is unit-testable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SettingsChange {
    Sounds(bool),
    Notifications(bool),
    Mode(Mode),
    Shortcut(String),
    Microphone { id: String, name: String },
}

impl SettingsChange {
    /// Body of the Info notification confirming the change.
    pub(crate) fn notification_body(&self) -> String {
        match self {
            SettingsChange::Sounds(on) => format!("Sound cues: {}", on_off(*on)),
            SettingsChange::Notifications(on) => {
                format!("Notifications: {}", on_off(*on))
            }
            SettingsChange::Mode(mode) => format!("Mode: {}", mode.display_name()),
            SettingsChange::Shortcut(display) => format!("Shortcut: {display}"),
            SettingsChange::Microphone { name, .. } => format!("Microphone: {name}"),
        }
    }

    /// Daemon-log line for the change: the key and value.
    pub(crate) fn log_line(&self) -> String {
        match self {
            SettingsChange::Sounds(on) => format!("settings: sounds = {on}"),
            SettingsChange::Notifications(on) => format!("settings: notifications = {on}"),
            SettingsChange::Mode(mode) => format!("settings: mode = {}", mode.as_str()),
            SettingsChange::Shortcut(display) => format!("settings: hotkey = {display}"),
            SettingsChange::Microphone { id, .. } => {
                if id.is_empty() {
                    "settings: microphone = (system default)".to_string()
                } else {
                    format!("settings: microphone = {id}")
                }
            }
        }
    }
}

/// `on`/`off` word for a toggled switch.
fn on_off(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}

/// One menu click, already matched to its item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TrayAction {
    Toggle,
    Cancel,
    Sounds,
    Notifications,
    Microphone(String),
    RefreshMics,
    Mode(Mode),
    Shortcut(String),
    HowTo,
    OpenConfig,
    Quit,
}

/// The `Shortcut` submenu: a Wayland note, or radio-style shortcut choices.
enum ShortcutMenu {
    Wayland {
        how_to: MenuItem,
    },
    Hotkeys {
        options: Vec<(String, CheckMenuItem)>,
    },
}

/// The `Microphone` submenu and its current entries: `System default`, one
/// radio-style item per microphone, a separator, and `Refresh list`. Rebuilt
/// from a fresh listing whenever the daemon refreshes the microphones.
struct MicrophoneMenu {
    submenu: Submenu,
    default_item: CheckMenuItem,
    items: Vec<(String, CheckMenuItem)>,
    separator: PredefinedMenuItem,
    refresh: MenuItem,
}

/// Handles of the entries inside the `Microphone` submenu: `System
/// default`, one item per microphone, the separator, and `Refresh list`.
type MicrophoneEntries = (
    CheckMenuItem,
    Vec<(String, CheckMenuItem)>,
    PredefinedMenuItem,
    MenuItem,
);

/// Append the microphone entries to `submenu`: `System default` (with the
/// current default name when known), one item per microphone, a separator,
/// and `Refresh list`. `selected` is the saved microphone id (`""` = system
/// default). Returns the new handles. The caller adds the context.
fn append_microphone_items(
    submenu: &Submenu,
    mics: &[Mic],
    selected: &str,
    default_name: Option<&str>,
) -> anyhow::Result<MicrophoneEntries> {
    let default_item = CheckMenuItem::with_id(
        "dictate-mic-default",
        system_default_label(default_name),
        true,
        selected.is_empty(),
        None,
    );
    submenu.append(&default_item)?;
    let mut items = Vec::with_capacity(mics.len());
    for (index, mic) in mics.iter().enumerate() {
        let item = CheckMenuItem::with_id(
            format!("dictate-mic-{index}"),
            mic.name.clone(),
            true,
            mic.id == selected,
            None,
        );
        submenu.append(&item)?;
        items.push((mic.id.clone(), item));
    }
    let separator = PredefinedMenuItem::separator();
    submenu.append(&separator)?;
    let refresh = MenuItem::with_id("dictate-mic-refresh", "Refresh list", true, None);
    submenu.append(&refresh)?;
    Ok((default_item, items, separator, refresh))
}

/// The tray icon and every menu item the daemon updates or matches.
pub(crate) struct Tray {
    icon: TrayIcon,
    status: MenuItem,
    toggle: MenuItem,
    cancel: MenuItem,
    sounds: CheckMenuItem,
    notifications: CheckMenuItem,
    microphone: MicrophoneMenu,
    modes: Vec<(Mode, CheckMenuItem)>,
    shortcut: ShortcutMenu,
    open_config: MenuItem,
    quit: MenuItem,
}

/// Build the tray icon with its full menu, and forward menu clicks to the
/// event loop. Must run on the main thread after the loop started.
pub(crate) fn build_tray(
    settings: &Settings,
    state: State,
    mics: &[Mic],
    proxy: winit::event_loop::EventLoopProxy<crate::daemon::UserEvent>,
) -> anyhow::Result<Tray> {
    let menu = Menu::new();
    let status = MenuItem::with_id("dictate-status", status_text(state, None), false, None);
    let toggle = MenuItem::with_id(
        "dictate-toggle",
        toggle_text(state),
        state != State::Transcribing,
        None,
    );
    let cancel = MenuItem::with_id(
        "dictate-cancel",
        "Cancel recording",
        state == State::Recording,
        None,
    );
    let sounds =
        CheckMenuItem::with_id("dictate-sounds", "Sound cues", true, settings.sounds, None);
    let notifications = CheckMenuItem::with_id(
        "dictate-notifications",
        "Notifications",
        true,
        settings.notifications,
        None,
    );
    let microphone_submenu = Submenu::with_id("dictate-microphone", "Microphone", true);
    let default_name = mics
        .iter()
        .find(|mic| mic.is_default)
        .map(|mic| mic.name.as_str());
    let (default_item, mic_items, mic_separator, mic_refresh) = append_microphone_items(
        &microphone_submenu,
        mics,
        &settings.microphone,
        default_name,
    )
    .context("Could not build the tray menu")?;
    let microphone = MicrophoneMenu {
        submenu: microphone_submenu,
        default_item,
        items: mic_items,
        separator: mic_separator,
        refresh: mic_refresh,
    };
    let mode_submenu = Submenu::with_id("dictate-mode", "Mode", true);
    let mut modes = Vec::with_capacity(MODE_OPTIONS.len());
    for (mode, label) in MODE_OPTIONS {
        let item = CheckMenuItem::with_id(
            format!("dictate-mode-{}", mode.as_str()),
            label,
            true,
            settings.mode == mode,
            None,
        );
        mode_submenu
            .append(&item)
            .context("Could not build the tray menu")?;
        modes.push((mode, item));
    }
    let submenu = Submenu::with_id("dictate-shortcut", "Shortcut", true);
    let shortcut = if crate::hotkey::wayland_session() {
        let note = MenuItem::with_id("dictate-shortcut-note", WAYLAND_SHORTCUT_NOTE, false, None);
        let how_to = MenuItem::with_id("dictate-howto", "How to change it…", true, None);
        submenu
            .append_items(&[&note, &how_to])
            .context("Could not build the tray menu")?;
        ShortcutMenu::Wayland { how_to }
    } else {
        let mut options = Vec::with_capacity(SHORTCUT_OPTIONS.len());
        for name in SHORTCUT_OPTIONS {
            let item = CheckMenuItem::with_id(
                format!("dictate-shortcut-{name}"),
                name,
                true,
                name == settings.hotkey,
                None,
            );
            submenu
                .append(&item)
                .context("Could not build the tray menu")?;
            options.push((name.to_string(), item));
        }
        ShortcutMenu::Hotkeys { options }
    };
    let open_config = MenuItem::with_id("dictate-open-config", "Open settings file", true, None);
    let quit = MenuItem::with_id("dictate-quit", "Quit dictate", true, None);
    menu.append(&status)
        .context("Could not build the tray menu")?;
    menu.append(&toggle)
        .context("Could not build the tray menu")?;
    menu.append(&cancel)
        .context("Could not build the tray menu")?;
    menu.append(&PredefinedMenuItem::separator())
        .context("Could not build the tray menu")?;
    menu.append(&sounds)
        .context("Could not build the tray menu")?;
    menu.append(&notifications)
        .context("Could not build the tray menu")?;
    menu.append(&microphone.submenu)
        .context("Could not build the tray menu")?;
    menu.append(&mode_submenu)
        .context("Could not build the tray menu")?;
    menu.append(&submenu)
        .context("Could not build the tray menu")?;
    menu.append(&open_config)
        .context("Could not build the tray menu")?;
    menu.append(&PredefinedMenuItem::separator())
        .context("Could not build the tray menu")?;
    menu.append(&quit)
        .context("Could not build the tray menu")?;
    let (rgba, width, height) = icon_rgba(state);
    let icon = Icon::from_rgba(rgba, width, height)
        .map_err(|err| anyhow::anyhow!("Internal error: tray icon pixels rejected: {err}"))?;
    let icon = TrayIconBuilder::new()
        .with_id(TRAY_ID)
        .with_title(TRAY_ID)
        .with_menu(Box::new(menu))
        .with_tooltip(tooltip_text(state, settings.mode, None))
        .with_icon(icon)
        .build()
        .map_err(|err| anyhow::anyhow!("Could not create tray icon: {err}"))?;
    tray_icon::menu::MenuEvent::set_event_handler(Some(move |event| {
        if proxy
            .send_event(crate::daemon::UserEvent::Menu(event))
            .is_err()
        {
            // The event loop is closed; the daemon is shutting down.
        }
    }));
    Ok(Tray {
        icon,
        status,
        toggle,
        cancel,
        sounds,
        notifications,
        microphone,
        modes,
        shortcut,
        open_config,
        quit,
    })
}

impl Tray {
    /// Refresh the icon, tooltip, label and item states after a state or
    /// settings change. While a mode switch loads in the background, the label
    /// reads Loading instead of the state. `mic` is the microphone in use,
    /// shown in the label and tooltip when known.
    pub(crate) fn sync_state(
        &self,
        state: State,
        mode: Mode,
        loading: bool,
        mic: Option<&str>,
    ) -> anyhow::Result<()> {
        let (rgba, width, height) = icon_rgba(state);
        let icon = Icon::from_rgba(rgba, width, height)
            .map_err(|err| anyhow::anyhow!("Internal error: tray icon pixels rejected: {err}"))?;
        self.icon
            .set_icon(Some(icon))
            .map_err(|err| anyhow::anyhow!("Could not update the tray icon: {err}"))?;
        self.icon
            .set_tooltip(Some(tooltip_text(state, mode, mic)))
            .map_err(|err| anyhow::anyhow!("Could not update the tray tooltip: {err}"))?;
        if loading {
            self.status.set_text(LOADING_TEXT);
        } else {
            self.status.set_text(status_text(state, mic));
        }
        self.toggle.set_text(toggle_text(state));
        self.toggle.set_enabled(state != State::Transcribing);
        self.cancel.set_enabled(state == State::Recording);
        Ok(())
    }

    /// Rebuild the Microphone submenu from a fresh listing: remove the old
    /// entries, then append `System default`, one item per microphone, a
    /// separator, and `Refresh list`. `selected` is the saved microphone id
    /// (`""` = system default).
    pub(crate) fn rebuild_microphones(
        &mut self,
        mics: &[Mic],
        selected: &str,
    ) -> anyhow::Result<()> {
        let submenu = &self.microphone.submenu;
        submenu
            .remove(&self.microphone.default_item)
            .context("Could not rebuild the microphone menu")?;
        for (_, item) in &self.microphone.items {
            submenu
                .remove(item)
                .context("Could not rebuild the microphone menu")?;
        }
        submenu
            .remove(&self.microphone.separator)
            .context("Could not rebuild the microphone menu")?;
        submenu
            .remove(&self.microphone.refresh)
            .context("Could not rebuild the microphone menu")?;
        let default_name = mics
            .iter()
            .find(|mic| mic.is_default)
            .map(|mic| mic.name.as_str());
        let (default_item, items, separator, refresh) =
            append_microphone_items(submenu, mics, selected, default_name)
                .context("Could not rebuild the microphone menu")?;
        self.microphone.default_item = default_item;
        self.microphone.items = items;
        self.microphone.separator = separator;
        self.microphone.refresh = refresh;
        Ok(())
    }

    /// Map a menu click to its action, if it came from this menu.
    pub(crate) fn match_event(&self, id: &MenuId) -> Option<TrayAction> {
        if *id == *self.toggle.id() {
            return Some(TrayAction::Toggle);
        }
        if *id == *self.cancel.id() {
            return Some(TrayAction::Cancel);
        }
        if *id == *self.sounds.id() {
            return Some(TrayAction::Sounds);
        }
        if *id == *self.notifications.id() {
            return Some(TrayAction::Notifications);
        }
        if *id == *self.microphone.default_item.id() {
            return Some(TrayAction::Microphone(String::new()));
        }
        if let Some((mic_id, _)) = self
            .microphone
            .items
            .iter()
            .find(|(_, item)| *id == *item.id())
        {
            return Some(TrayAction::Microphone(mic_id.clone()));
        }
        if *id == *self.microphone.refresh.id() {
            return Some(TrayAction::RefreshMics);
        }
        if let Some((mode, _)) = self.modes.iter().find(|(_, item)| *id == *item.id()) {
            return Some(TrayAction::Mode(*mode));
        }
        if *id == *self.open_config.id() {
            return Some(TrayAction::OpenConfig);
        }
        if *id == *self.quit.id() {
            return Some(TrayAction::Quit);
        }
        match &self.shortcut {
            ShortcutMenu::Wayland { how_to } => {
                if *id == *how_to.id() {
                    Some(TrayAction::HowTo)
                } else {
                    None
                }
            }
            ShortcutMenu::Hotkeys { options } => options
                .iter()
                .find(|(_, item)| *id == *item.id())
                .map(|(name, _)| TrayAction::Shortcut(name.clone())),
        }
    }

    /// Set the `Sound cues` check explicitly (whatever the backend toggled).
    pub(crate) fn set_sounds_checked(&self, checked: bool) {
        self.sounds.set_checked(checked);
    }

    /// Set the `Notifications` check explicitly.
    pub(crate) fn set_notifications_checked(&self, checked: bool) {
        self.notifications.set_checked(checked);
    }

    /// Check exactly the current mode in the Mode submenu (radio behavior).
    pub(crate) fn sync_mode_checks(&self, current: Mode) {
        for (mode, item) in &self.modes {
            item.set_checked(*mode == current);
        }
    }

    /// Check exactly the selected microphone in the Microphone submenu (radio
    /// behavior; `""` checks System default).
    pub(crate) fn sync_microphone_checks(&self, selected: &str) {
        self.microphone
            .default_item
            .set_checked(selected.is_empty());
        for (id, item) in &self.microphone.items {
            item.set_checked(id == selected);
        }
    }

    /// Check exactly the current shortcut (radio behavior).
    pub(crate) fn sync_shortcut_checks(&self, current: &str) {
        if let ShortcutMenu::Hotkeys { options } = &self.shortcut {
            for (name, item) in options {
                item.set_checked(name.as_str() == current);
            }
        }
    }
}

/// Program and arguments to open a file or URL with the platform default.
pub(crate) fn open_argv(target: &str) -> (String, Vec<String>) {
    #[cfg(target_os = "windows")]
    return (
        "cmd".to_string(),
        vec![
            "/C".to_string(),
            "start".to_string(),
            String::new(),
            target.to_string(),
        ],
    );
    #[cfg(target_os = "macos")]
    return ("open".to_string(), vec![target.to_string()]);
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    return ("xdg-open".to_string(), vec![target.to_string()]);
}

/// Open a file or URL with the platform default handler (fire and forget).
pub(crate) fn open_target(target: &str) -> anyhow::Result<()> {
    let (program, args) = open_argv(target);
    std::process::Command::new(&program)
        .args(&args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("Could not open {target} with {program}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn center_pixel(state: State) -> [u8; 4] {
        let (rgba, width, height) = icon_rgba(state);
        assert_eq!((width, height), (32, 32));
        assert_eq!(rgba.len(), 32 * 32 * 4);
        let start = (16 * 32 + 16) * 4;
        [
            rgba[start],
            rgba[start + 1],
            rgba[start + 2],
            rgba[start + 3],
        ]
    }

    #[test]
    fn icon_is_32x32_with_state_colored_center() {
        assert_eq!(center_pixel(State::Idle), [158, 158, 158, 255]);
        assert_eq!(center_pixel(State::Recording), [229, 57, 53, 255]);
        assert_eq!(center_pixel(State::Transcribing), [255, 179, 0, 255]);
    }

    #[test]
    fn icon_corners_stay_transparent() {
        let (rgba, _, _) = icon_rgba(State::Recording);
        assert_eq!(&rgba[0..4], &[0, 0, 0, 0]);
        let end = rgba.len();
        assert_eq!(&rgba[end - 4..end], &[0, 0, 0, 0]);
    }

    #[test]
    fn tray_id_stays_stable() {
        assert_eq!(TRAY_ID, "dictate");
    }

    #[test]
    fn status_and_toggle_texts() {
        assert_eq!(status_text(State::Idle, None), "dictate — Idle");
        assert_eq!(status_text(State::Recording, None), "dictate — Recording…");
        assert_eq!(
            status_text(State::Transcribing, None),
            "dictate — Transcribing…"
        );
        assert_eq!(toggle_text(State::Idle), "Start dictation");
        assert_eq!(toggle_text(State::Recording), "Stop dictation");
        assert_eq!(toggle_text(State::Transcribing), "Stop dictation");
    }

    #[test]
    fn status_text_shows_microphone_in_use() {
        assert_eq!(
            status_text(State::Idle, Some("Built-in Audio Analog Stereo")),
            "dictate — Idle · Built-in Audio Analog Stereo"
        );
        assert_eq!(
            status_text(State::Recording, Some("Built-in Audio Analog Stereo")),
            "dictate — Recording… (Built-in Audio Analog Stereo)"
        );
        assert_eq!(
            status_text(State::Transcribing, Some("USB Microphone")),
            "dictate — Transcribing… (USB Microphone)"
        );
    }

    #[test]
    fn system_default_label_includes_default_name() {
        assert_eq!(
            system_default_label(Some("Built-in Audio Analog Stereo")),
            "System default (Built-in Audio Analog Stereo)"
        );
        assert_eq!(system_default_label(None), "System default");
    }

    #[test]
    fn every_shortcut_option_parses() {
        for name in SHORTCUT_OPTIONS {
            assert!(
                global_hotkey::hotkey::HotKey::from_str(name).is_ok(),
                "{name} does not parse"
            );
        }
    }

    #[test]
    fn tooltip_shows_state_and_mode_name() {
        assert_eq!(
            tooltip_text(State::Idle, Mode::Whole, None),
            "dictate — Idle (Whole)"
        );
        assert_eq!(
            tooltip_text(State::Idle, Mode::Pauses, None),
            "dictate — Idle (Pauses)"
        );
        assert_eq!(
            tooltip_text(State::Recording, Mode::Live, None),
            "dictate — Recording… (Live)"
        );
        assert_eq!(
            tooltip_text(State::Transcribing, Mode::Whole, None),
            "dictate — Transcribing… (Whole)"
        );
        assert_eq!(
            tooltip_text(
                State::Idle,
                Mode::Pauses,
                Some("Built-in Audio Analog Stereo")
            ),
            "dictate — Idle · Built-in Audio Analog Stereo (Pauses)"
        );
        assert_eq!(
            tooltip_text(State::Recording, Mode::Live, Some("USB Microphone")),
            "dictate — Recording… (USB Microphone) (Live)"
        );
    }

    #[test]
    fn mode_menu_lists_every_mode_once_with_its_label() {
        let modes: Vec<Mode> = MODE_OPTIONS.iter().map(|(mode, _)| *mode).collect();
        assert_eq!(modes, vec![Mode::Whole, Mode::Pauses, Mode::Live]);
        assert_eq!(MODE_OPTIONS[0].1, "Whole recording (most accurate)");
        assert_eq!(MODE_OPTIONS[1].1, "Type at pauses");
        assert_eq!(
            MODE_OPTIONS[2].1,
            "Live — text as you speak (fast computers)"
        );
    }

    #[test]
    fn loading_label_text() {
        assert_eq!(LOADING_TEXT, "dictate — Loading…");
    }

    #[test]
    fn settings_change_confirmation_texts() {
        let cases = [
            (
                SettingsChange::Sounds(true),
                "Sound cues: on",
                "settings: sounds = true",
            ),
            (
                SettingsChange::Sounds(false),
                "Sound cues: off",
                "settings: sounds = false",
            ),
            (
                SettingsChange::Notifications(true),
                "Notifications: on",
                "settings: notifications = true",
            ),
            (
                SettingsChange::Notifications(false),
                "Notifications: off",
                "settings: notifications = false",
            ),
            (
                SettingsChange::Mode(Mode::Whole),
                "Mode: Whole",
                "settings: mode = whole",
            ),
            (
                SettingsChange::Mode(Mode::Pauses),
                "Mode: Pauses",
                "settings: mode = pauses",
            ),
            (
                SettingsChange::Mode(Mode::Live),
                "Mode: Live",
                "settings: mode = live",
            ),
            (
                SettingsChange::Shortcut("Ctrl+Alt+D".to_string()),
                "Shortcut: Ctrl+Alt+D",
                "settings: hotkey = Ctrl+Alt+D",
            ),
            (
                SettingsChange::Microphone {
                    id: "pulseaudio:alsa_input.pci".to_string(),
                    name: "Built-in Audio Analog Stereo".to_string(),
                },
                "Microphone: Built-in Audio Analog Stereo",
                "settings: microphone = pulseaudio:alsa_input.pci",
            ),
            (
                SettingsChange::Microphone {
                    id: String::new(),
                    name: "System default (Built-in Audio Analog Stereo)".to_string(),
                },
                "Microphone: System default (Built-in Audio Analog Stereo)",
                "settings: microphone = (system default)",
            ),
        ];
        for (change, body, log) in cases {
            assert_eq!(change.notification_body(), body, "{change:?}");
            assert_eq!(change.log_line(), log, "{change:?}");
        }
    }

    #[test]
    fn open_argv_uses_platform_handler() {
        let (program, args) = open_argv("/tmp/settings.toml");
        #[cfg(target_os = "windows")]
        {
            assert_eq!(program, "cmd");
            assert_eq!(args, vec!["/C", "start", "", "/tmp/settings.toml"]);
        }
        #[cfg(target_os = "macos")]
        {
            assert_eq!(program, "open");
            assert_eq!(args, vec!["/tmp/settings.toml"]);
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        {
            assert_eq!(program, "xdg-open");
            assert_eq!(args, vec!["/tmp/settings.toml"]);
        }
    }
}
