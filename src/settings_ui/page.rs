//! Settings page: one scrolling page with headed sections.
//!
//! The page renders borrowed daemon state ([`PageView`]) and returns user
//! intents ([`PageAction`]); the daemon applies them through the same code
//! paths the tray menu uses, so there is no duplicate logic.

use crate::config::{Mode, Settings};
use crate::daemon::LiveSpeedCheck;
use crate::mic::Mic;

/// GitHub releases page (the About section's "Check for updates" button).
pub(crate) const RELEASES_URL: &str = "https://github.com/saostad/dictate/releases/latest";

/// Page margin in logical pixels: the CentralPanel frame's inner margin, so
/// content never touches the window edge.
pub(crate) const PAGE_MARGIN: f32 = 16.0;
/// Explanation text size in logical pixels. egui's `small()` is 9 px by
/// default, too small to read; the page renders it at 12 px instead.
pub(crate) const EXPLAIN_SIZE: f32 = 12.0;
/// Space above and below each section separator.
const SECTION_GAP: f32 = 8.0;

/// Dictation modes offered on the page, in order, with their radio label and
/// one-line explanation.
pub(crate) const MODE_OPTIONS: [(Mode, &str, &str); 3] = [
    (
        Mode::Whole,
        "Whole recording",
        "Most accurate. Text appears after you release the key.",
    ),
    (
        Mode::Pauses,
        "Type at pauses",
        "Types each phrase when you pause. Faster on slow computers; may occasionally mishear the word after a pause.",
    ),
    (
        Mode::Live,
        "Live",
        "Text appears as you speak. Needs a fast computer and a ~900 MB download.",
    ),
];

/// User intents from one page frame, applied by the daemon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PageAction {
    SelectMic(String),
    RefreshMics,
    SelectMode(Mode),
    SelectShortcut(String),
    ToggleSounds,
    ToggleNotifications,
    SetAutostart(bool),
    OpenLog,
    OpenReleases,
}

/// Everything the page renders, borrowed from daemon state.
pub(crate) struct PageView<'a> {
    pub settings: &'a Settings,
    pub mics: &'a [Mic],
    pub active_mic: Option<&'a str>,
    /// A mode switch is loading this mode in the background, if any.
    pub switch_target: Option<Mode>,
    /// The switch is downloading the Live model first.
    pub downloading: bool,
    /// Live-model download progress, 0–100.
    pub download_percent: u64,
    /// The Live speed check is running now.
    pub checking_speed: bool,
    /// The last Live speed-check outcome this session, if one ran.
    pub last_speed_check: Option<LiveSpeedCheck>,
    pub hotkey_display: &'a str,
    pub wayland: bool,
    /// Cached start-at-login state (`None` when the last query failed).
    pub autostart_on: Option<bool>,
    pub autostart_error: Option<&'a str>,
    pub shortcut_error: Option<&'a str>,
    pub version: &'static str,
}

/// `Recording from: <name>`: the device the last recording actually used, or
/// a placeholder when nothing recorded yet.
pub(crate) fn recording_from_line(active_mic: Option<&str>) -> String {
    match active_mic {
        Some(name) => format!("Recording from: {name}"),
        None => "Recording from: no recording yet".to_string(),
    }
}

/// Status line while a mode switch loads in the background.
pub(crate) fn mode_loading_line(target: Mode) -> String {
    format!("Loading {} model…", target.display_name())
}

/// Status line while the Live model downloads.
pub(crate) fn mode_download_line(percent: u64) -> String {
    format!("Downloading Live model… {percent}%")
}

/// Status line for a finished Live speed check.
pub(crate) fn speed_check_line(check: LiveSpeedCheck) -> String {
    let verdict = if check.fast {
        "fast enough"
    } else {
        "too slow"
    };
    format!(
        "Speed check: {} ms per 560 ms of audio ({verdict})",
        check.ms
    )
}

/// Wayland shortcut line: the configured shortcut the user binds in their
/// compositor, when one is known.
pub(crate) fn wayland_shortcut_line(configured: Option<&str>) -> Option<String> {
    configured
        .filter(|keys| !keys.is_empty())
        .map(|keys| format!("Your shortcut: {keys}"))
}

/// Render one page frame, collecting the user's intents.
pub(crate) fn show_page(ui: &mut egui::Ui, view: &PageView) -> Vec<PageAction> {
    ui.style_mut().text_styles.insert(
        egui::TextStyle::Small,
        egui::FontId::proportional(EXPLAIN_SIZE),
    );
    let mut actions = Vec::new();
    egui::CentralPanel::default()
        .frame(egui::Frame::central_panel(ui.style()).inner_margin(PAGE_MARGIN))
        .show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                microphone_section(ui, view, &mut actions);
                section_gap(ui);
                mode_section(ui, view, &mut actions);
                section_gap(ui);
                shortcut_section(ui, view, &mut actions);
                section_gap(ui);
                general_section(ui, view, &mut actions);
                section_gap(ui);
                about_section(ui, view, &mut actions);
            });
        });
    actions
}

/// Space between two sections: a gap, a separator, a gap.
fn section_gap(ui: &mut egui::Ui) {
    ui.add_space(SECTION_GAP);
    ui.separator();
    ui.add_space(SECTION_GAP);
}

/// A section heading: heading size, strong.
fn heading(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).heading().strong());
}

/// One-line grey explanation under a control.
fn explain(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).small().weak());
}

/// One-line inline error under a control, when there is one.
fn inline_error(ui: &mut egui::Ui, error: Option<&str>) {
    if let Some(error) = error {
        ui.label(
            egui::RichText::new(error)
                .small()
                .color(ui.visuals().error_fg_color),
        );
    }
}

fn microphone_section(ui: &mut egui::Ui, view: &PageView, actions: &mut Vec<PageAction>) {
    heading(ui, "Microphone");
    let default_name = view
        .mics
        .iter()
        .find(|mic| mic.is_default)
        .map(|mic| mic.name.as_str());
    if ui
        .radio(
            view.settings.microphone.is_empty(),
            crate::tray::system_default_label(default_name),
        )
        .clicked()
    {
        actions.push(PageAction::SelectMic(String::new()));
    }
    for mic in view.mics {
        if ui
            .radio(view.settings.microphone == mic.id, &mic.name)
            .clicked()
        {
            actions.push(PageAction::SelectMic(mic.id.clone()));
        }
    }
    explain(ui, "The microphone new recordings use.");
    if ui.button("Refresh").clicked() {
        actions.push(PageAction::RefreshMics);
    }
    explain(ui, "Re-list the available microphones.");
    ui.label(recording_from_line(view.active_mic));
}

fn mode_section(ui: &mut egui::Ui, view: &PageView, actions: &mut Vec<PageAction>) {
    heading(ui, "Dictation mode");
    for (mode, label, explanation) in MODE_OPTIONS {
        if ui.radio(view.settings.mode == mode, label).clicked() {
            actions.push(PageAction::SelectMode(mode));
        }
        explain(ui, explanation);
    }
    if view.downloading {
        ui.label(mode_download_line(view.download_percent));
    } else if let Some(target) = view.switch_target {
        ui.label(mode_loading_line(target));
    }
    if view.checking_speed {
        ui.label("Checking Live speed…");
    }
    if let Some(check) = view.last_speed_check {
        ui.label(speed_check_line(check));
    }
}

fn shortcut_section(ui: &mut egui::Ui, view: &PageView, actions: &mut Vec<PageAction>) {
    heading(ui, "Shortcut");
    if view.wayland {
        ui.label("Apps can't register global shortcuts on Wayland. Bind this shortcut in your compositor:");
        if let Some(line) = wayland_shortcut_line(Some(view.hotkey_display)) {
            ui.label(line);
        }
        ui.hyperlink_to("How to set it", crate::tray::README_USE_IT_URL);
        return;
    }
    egui::ComboBox::from_label("")
        .selected_text(view.settings.hotkey.as_str())
        .show_ui(ui, |ui| {
            for preset in crate::tray::SHORTCUT_OPTIONS {
                if ui
                    .selectable_label(view.settings.hotkey == preset, preset)
                    .clicked()
                {
                    actions.push(PageAction::SelectShortcut(preset.to_string()));
                }
            }
        });
    explain(ui, "Tap to toggle dictation, hold to talk.");
    inline_error(ui, view.shortcut_error);
}

fn general_section(ui: &mut egui::Ui, view: &PageView, actions: &mut Vec<PageAction>) {
    heading(ui, "General");
    let mut autostart = view.autostart_on.unwrap_or(false);
    if ui.checkbox(&mut autostart, "Start at login").changed() {
        actions.push(PageAction::SetAutostart(autostart));
    }
    explain(ui, "Start Dictate automatically when you log in.");
    inline_error(ui, view.autostart_error);
    let mut sounds = view.settings.sounds;
    if ui.checkbox(&mut sounds, "Sound cues").changed() {
        actions.push(PageAction::ToggleSounds);
    }
    explain(ui, "Play a short tone when recording starts and stops.");
    let mut notifications = view.settings.notifications;
    if ui.checkbox(&mut notifications, "Notifications").changed() {
        actions.push(PageAction::ToggleNotifications);
    }
    explain(
        ui,
        "Show desktop notifications for errors and empty recordings.",
    );
}

fn about_section(ui: &mut egui::Ui, view: &PageView, actions: &mut Vec<PageAction>) {
    heading(ui, "About");
    ui.label(format!("Dictate {}", view.version));
    ui.horizontal(|ui| {
        if ui.button("Open log").clicked() {
            actions.push(PageAction::OpenLog);
        }
        if ui.button("Check for updates").clicked() {
            actions.push(PageAction::OpenReleases);
        }
    });
    explain(
        ui,
        "Open the daemon log file, or the releases page in your browser.",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_options_cover_every_mode_with_its_text() {
        let modes: Vec<Mode> = MODE_OPTIONS.iter().map(|(mode, _, _)| *mode).collect();
        assert_eq!(modes, vec![Mode::Whole, Mode::Pauses, Mode::Live]);
        assert_eq!(MODE_OPTIONS[0].1, "Whole recording");
        assert_eq!(
            MODE_OPTIONS[0].2,
            "Most accurate. Text appears after you release the key."
        );
        assert_eq!(MODE_OPTIONS[1].1, "Type at pauses");
        assert_eq!(
            MODE_OPTIONS[1].2,
            "Types each phrase when you pause. Faster on slow computers; may occasionally mishear the word after a pause."
        );
        assert_eq!(MODE_OPTIONS[2].1, "Live");
        assert_eq!(
            MODE_OPTIONS[2].2,
            "Text appears as you speak. Needs a fast computer and a ~900 MB download."
        );
    }

    #[test]
    fn recording_from_line_names_the_last_mic() {
        assert_eq!(
            recording_from_line(Some("Built-in Audio Analog Stereo")),
            "Recording from: Built-in Audio Analog Stereo"
        );
        assert_eq!(
            recording_from_line(None),
            "Recording from: no recording yet"
        );
    }

    #[test]
    fn mode_status_lines() {
        assert_eq!(mode_loading_line(Mode::Live), "Loading Live model…");
        assert_eq!(mode_loading_line(Mode::Pauses), "Loading Pauses model…");
        assert_eq!(mode_download_line(25), "Downloading Live model… 25%");
        assert_eq!(mode_download_line(100), "Downloading Live model… 100%");
        assert_eq!(
            speed_check_line(LiveSpeedCheck {
                ms: 120,
                fast: true
            }),
            "Speed check: 120 ms per 560 ms of audio (fast enough)"
        );
        assert_eq!(
            speed_check_line(LiveSpeedCheck {
                ms: 905,
                fast: false
            }),
            "Speed check: 905 ms per 560 ms of audio (too slow)"
        );
    }

    #[test]
    fn wayland_shortcut_line_needs_known_keys() {
        assert_eq!(
            wayland_shortcut_line(Some("Ctrl+Shift+Space")),
            Some("Your shortcut: Ctrl+Shift+Space".to_string())
        );
        assert_eq!(wayland_shortcut_line(Some("")), None);
        assert_eq!(wayland_shortcut_line(None), None);
    }

    #[test]
    fn page_layout_numbers() {
        assert_eq!(PAGE_MARGIN, 16.0);
        assert!(EXPLAIN_SIZE >= 12.0);
        assert_eq!(SECTION_GAP, 8.0);
    }

    #[test]
    fn releases_url_points_at_latest() {
        assert_eq!(
            RELEASES_URL,
            "https://github.com/saostad/dictate/releases/latest"
        );
    }
}
