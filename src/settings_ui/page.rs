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
    RefreshBinds,
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
    pub wayland: bool,
    /// True on Hyprland, where the shortcut section lists compositor binds.
    pub hyprland: bool,
    /// Formatted `hyprctl binds` lines mentioning dictate (`None` when not on
    /// Hyprland, or when the query failed — the page shows generic text then).
    pub dictate_binds: Option<&'a [String]>,
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

/// One `hyprctl binds` entry mentioning dictate: the formatted keys
/// (`Super+Alt+D`), the display label (the bind description, or the dictate
/// action parsed from the arg when the description is empty), and whether
/// it fires on release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HyprBind {
    keys: String,
    label: String,
    release: bool,
}

/// Parse `hyprctl binds -j` output into the binds whose arg or description
/// mentions dictate (case-insensitive, so Lua-defined binds with opaque
/// numeric args still match via their description). The JSON is parsed by
/// hand (no new crates for one query); malformed entries are skipped.
pub(crate) fn parse_hyprland_binds(json: &str) -> Vec<HyprBind> {
    split_json_objects(json)
        .iter()
        .filter_map(|object| parse_bind_object(object))
        .collect()
}

/// Format parsed binds as `Super+Alt+D — Toggle dictate` lines: the keys,
/// an em dash, and the label, with ` (on release)` appended for release
/// binds.
pub(crate) fn format_hyprland_binds(binds: &[HyprBind]) -> Vec<String> {
    binds
        .iter()
        .map(|bind| {
            let suffix = if bind.release { " (on release)" } else { "" };
            format!("{} — {}{suffix}", bind.keys, bind.label)
        })
        .collect()
}

/// Split a top-level JSON array into its `{...}` object spans, ignoring
/// braces inside strings. Unbalanced input yields the spans found so far.
fn split_json_objects(array: &str) -> Vec<&str> {
    let bytes = array.as_bytes();
    let mut objects = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
        } else if b == b'"' {
            in_string = true;
        } else if b == b'{' {
            if depth == 0 {
                start = i;
            }
            depth += 1;
        } else if b == b'}' && depth > 0 {
            depth -= 1;
            if depth == 0 {
                objects.push(&array[start..=i]);
            }
        }
    }
    objects
}

/// Skip ASCII whitespace.
fn skip_ws(bytes: &[u8], i: &mut usize) {
    while bytes.get(*i).is_some_and(|b| b.is_ascii_whitespace()) {
        *i += 1;
    }
}

/// Read a JSON string starting at the opening quote in `bytes[*i]`,
/// advancing `*i` past the closing quote. Returns the unescaped text, or
/// `None` when unterminated or invalid.
fn read_json_string(bytes: &[u8], i: &mut usize) -> Option<String> {
    if bytes.get(*i) != Some(&b'"') {
        return None;
    }
    *i += 1;
    let mut out: Vec<u8> = Vec::new();
    loop {
        let b = *bytes.get(*i)?;
        *i += 1;
        match b {
            b'"' => return String::from_utf8(out).ok(),
            b'\\' => {
                let esc = *bytes.get(*i)?;
                *i += 1;
                match esc {
                    b'"' => out.push(b'"'),
                    b'\\' => out.push(b'\\'),
                    b'/' => out.push(b'/'),
                    b'b' => out.push(0x08),
                    b'f' => out.push(0x0C),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'u' => {
                        let hex = std::str::from_utf8(bytes.get(*i..*i + 4)?).ok()?;
                        *i += 4;
                        let code = u32::from_str_radix(hex, 16).ok()?;
                        let ch = char::from_u32(code).unwrap_or('\u{FFFD}');
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                    _ => return None,
                }
            }
            _ => out.push(b),
        }
    }
}

/// Read a JSON `true`/`false` literal at `bytes[*i]`.
fn read_bool(bytes: &[u8], i: &mut usize) -> Option<bool> {
    if bytes.get(*i..*i + 4) == Some(b"true") {
        *i += 4;
        Some(true)
    } else if bytes.get(*i..*i + 5) == Some(b"false") {
        *i += 5;
        Some(false)
    } else {
        None
    }
}

/// Read a JSON unsigned integer at `bytes[*i]`.
fn read_u32(bytes: &[u8], i: &mut usize) -> Option<u32> {
    let start = *i;
    while bytes.get(*i).is_some_and(|b| b.is_ascii_digit()) {
        *i += 1;
    }
    if *i == start {
        return None;
    }
    std::str::from_utf8(&bytes[start..*i]).ok()?.parse().ok()
}

/// Skip one JSON value (string, object, array or literal) at `bytes[*i]`.
fn skip_json_value(bytes: &[u8], i: &mut usize) -> bool {
    skip_ws(bytes, i);
    let Some(&b) = bytes.get(*i) else {
        return false;
    };
    match b {
        b'"' => read_json_string(bytes, i).is_some(),
        b'{' | b'[' => {
            let close = if b == b'{' { b'}' } else { b']' };
            *i += 1;
            let mut depth = 1;
            while depth > 0 {
                let Some(&c) = bytes.get(*i) else {
                    return false;
                };
                *i += 1;
                if c == b'"' {
                    *i -= 1;
                    if read_json_string(bytes, i).is_none() {
                        return false;
                    }
                } else if c == b {
                    depth += 1;
                } else if c == close {
                    depth -= 1;
                }
            }
            true
        }
        _ => {
            while let Some(&c) = bytes.get(*i) {
                if c == b',' || c == b'}' || c == b']' {
                    break;
                }
                *i += 1;
            }
            true
        }
    }
}

/// Parse one `hyprctl binds` entry into a [`HyprBind`]: `None` when a needed
/// field is missing or malformed, or neither the arg nor the description
/// mentions dictate.
fn parse_bind_object(object: &str) -> Option<HyprBind> {
    let bytes = object.as_bytes();
    let mut i = 0;
    skip_ws(bytes, &mut i);
    if bytes.get(i) != Some(&b'{') {
        return None;
    }
    i += 1;
    let (mut modmask, mut key, mut release, mut arg, mut description) =
        (None, None, None, None, None);
    loop {
        skip_ws(bytes, &mut i);
        match bytes.get(i) {
            None => return None,
            Some(b'}') => break,
            Some(b',') => {
                i += 1;
                continue;
            }
            Some(b'"') => {}
            _ => return None,
        }
        let name = read_json_string(bytes, &mut i)?;
        skip_ws(bytes, &mut i);
        if bytes.get(i) != Some(&b':') {
            return None;
        }
        i += 1;
        skip_ws(bytes, &mut i);
        match name.as_str() {
            "modmask" => modmask = read_u32(bytes, &mut i),
            "key" => key = read_json_string(bytes, &mut i),
            "release" => release = read_bool(bytes, &mut i),
            "arg" => arg = read_json_string(bytes, &mut i),
            "description" => description = read_json_string(bytes, &mut i),
            _ => {
                if !skip_json_value(bytes, &mut i) {
                    return None;
                }
            }
        }
    }
    let (modmask, key, release): (u32, String, bool) = (modmask?, key?, release?);
    let arg = arg.unwrap_or_default();
    let description = description.unwrap_or_default();
    if !mentions_dictate(&arg) && !mentions_dictate(&description) {
        return None;
    }
    let label = if description.is_empty() {
        command_from_arg(&arg)
    } else {
        description
    };
    Some(HyprBind {
        keys: format_keys(modmask, &key),
        label,
        release,
    })
}

/// True when the text mentions dictate, case-insensitively.
fn mentions_dictate(text: &str) -> bool {
    text.to_ascii_lowercase().contains("dictate")
}

/// `Super+Alt+D` from a Hyprland modmask and key name. Only the four common
/// modifiers are shown; a single lowercase letter is uppercased.
fn format_keys(modmask: u32, key: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if modmask & 64 != 0 {
        parts.push("Super");
    }
    if modmask & 4 != 0 {
        parts.push("Ctrl");
    }
    if modmask & 8 != 0 {
        parts.push("Alt");
    }
    if modmask & 1 != 0 {
        parts.push("Shift");
    }
    let upper;
    let key = if key.len() == 1 {
        upper = key.to_ascii_uppercase();
        upper.as_str()
    } else {
        key
    };
    if parts.is_empty() {
        key.to_string()
    } else {
        format!("{}+{key}", parts.join("+"))
    }
}

/// The dictate command from a bind arg (`dictate toggle` → `toggle`). The
/// `dictate` token (matched case-insensitively) may carry a path; when no
/// command follows it, the whole arg is shown.
fn command_from_arg(arg: &str) -> String {
    let mut tokens = arg.split_whitespace();
    while let Some(token) = tokens.next() {
        let name = token.rsplit('/').next().unwrap_or(token);
        if name.eq_ignore_ascii_case("dictate") {
            if let Some(command) = tokens.next() {
                return command.to_string();
            }
            break;
        }
    }
    arg.trim().to_string()
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
        wayland_shortcut_section(ui, view, actions);
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

/// Shortcut section on Wayland, where the app can't register its own
/// hotkey. On Hyprland the cached compositor binds are listed (re-queried
/// with Refresh); everywhere else the generic explanation is shown. The
/// configured in-app hotkey is never shown here: it isn't registered.
fn wayland_shortcut_section(ui: &mut egui::Ui, view: &PageView, actions: &mut Vec<PageAction>) {
    if !view.hyprland {
        generic_wayland_shortcut(ui);
        return;
    }
    match view.dictate_binds {
        Some(binds) if !binds.is_empty() => {
            ui.label("Your dictate shortcuts in Hyprland:");
            for line in binds {
                ui.label(line);
            }
        }
        Some(_) => {
            ui.label("No dictate shortcut is bound yet.");
            ui.hyperlink_to("How to set it", crate::tray::README_USE_IT_URL);
        }
        None => generic_wayland_shortcut(ui),
    }
    explain(
        ui,
        "Queried when the window opens; Refresh re-reads the binds.",
    );
    if ui.button("Refresh").clicked() {
        actions.push(PageAction::RefreshBinds);
    }
}

/// Generic Wayland shortcut text (other compositors, or a failed query).
fn generic_wayland_shortcut(ui: &mut egui::Ui) {
    ui.label(
        "Apps can't register global shortcuts on Wayland. Bind \"dictate toggle\" (or start/stop) in your compositor's keyboard settings:",
    );
    ui.hyperlink_to("How to set it", crate::tray::README_USE_IT_URL);
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

    /// Realistic `hyprctl binds -j` output: a hold-to-talk pair on F8, a
    /// toggle on Super+Alt+D, and an unrelated bind that must be filtered.
    const HYPRCTL_BINDS: &str = r#"[
        {"locked":false,"mouse":false,"release":false,"repeat":false,"modmask":0,"submap":"","key":"F8","keycode":0,"catch_all":false,"description":"","dispatcher":"exec","arg":"dictate start","enabled":true},
        {"locked":false,"mouse":false,"release":true,"repeat":false,"modmask":0,"submap":"","key":"F8","keycode":0,"catch_all":false,"description":"","dispatcher":"exec","arg":"dictate stop","enabled":true},
        {"locked":false,"mouse":false,"release":false,"repeat":false,"modmask":72,"submap":"","key":"d","keycode":0,"catch_all":false,"description":"","dispatcher":"exec","arg":"dictate toggle","enabled":true},
        {"locked":false,"mouse":false,"release":false,"repeat":false,"modmask":64,"submap":"","key":"Return","keycode":0,"catch_all":false,"description":"","dispatcher":"exec","arg":"foot","enabled":true}
    ]"#;

    #[test]
    fn parses_hyprland_dictate_binds() {
        let binds = parse_hyprland_binds(HYPRCTL_BINDS);
        assert_eq!(
            binds,
            vec![
                HyprBind {
                    keys: "F8".to_string(),
                    label: "start".to_string(),
                    release: false,
                },
                HyprBind {
                    keys: "F8".to_string(),
                    label: "stop".to_string(),
                    release: true,
                },
                HyprBind {
                    keys: "Super+Alt+D".to_string(),
                    label: "toggle".to_string(),
                    release: false,
                },
            ]
        );
    }

    #[test]
    fn formats_press_and_release_lines() {
        let binds = parse_hyprland_binds(HYPRCTL_BINDS);
        assert_eq!(
            format_hyprland_binds(&binds),
            vec![
                "F8 — start".to_string(),
                "F8 — stop (on release)".to_string(),
                "Super+Alt+D — toggle".to_string(),
            ]
        );
    }

    /// Omarchy defines binds in Lua, so `hyprctl binds -j` reports opaque
    /// numeric args (`__lua` dispatcher); the description names dictate.
    const HYPRCTL_LUA_BINDS: &str = r#"[
        {"locked":false,"mouse":false,"release":false,"repeat":false,"modmask":72,"submap":"","key":"D","keycode":0,"catch_all":false,"description":"Toggle dictate","dispatcher":"__lua","arg":"9","enabled":true},
        {"locked":false,"mouse":false,"release":false,"repeat":false,"modmask":0,"submap":"","key":"F8","keycode":0,"catch_all":false,"description":"Start dictate (push-to-talk)","dispatcher":"__lua","arg":"25","enabled":true},
        {"locked":false,"mouse":false,"release":true,"repeat":false,"modmask":0,"submap":"","key":"F8","keycode":0,"catch_all":false,"description":"Stop dictate (push-to-talk)","dispatcher":"__lua","arg":"41","enabled":true}
    ]"#;

    #[test]
    fn finds_lua_defined_binds_via_description() {
        let binds = parse_hyprland_binds(HYPRCTL_LUA_BINDS);
        assert_eq!(binds.len(), 3);
        assert_eq!(
            format_hyprland_binds(&binds),
            vec![
                "Super+Alt+D — Toggle dictate".to_string(),
                "F8 — Start dictate (push-to-talk)".to_string(),
                "F8 — Stop dictate (push-to-talk) (on release)".to_string(),
            ]
        );
    }

    #[test]
    fn only_release_binds_are_annotated() {
        let binds = vec![
            HyprBind {
                keys: "Ctrl+Shift+Space".to_string(),
                label: "toggle".to_string(),
                release: false,
            },
            HyprBind {
                keys: "F9".to_string(),
                label: "cancel".to_string(),
                release: true,
            },
        ];
        assert_eq!(
            format_hyprland_binds(&binds),
            vec![
                "Ctrl+Shift+Space — toggle".to_string(),
                "F9 — cancel (on release)".to_string(),
            ]
        );
    }

    #[test]
    fn modmask_names_and_key_case() {
        assert_eq!(format_keys(0, "F8"), "F8");
        assert_eq!(format_keys(72, "d"), "Super+Alt+D");
        assert_eq!(format_keys(5, "Space"), "Ctrl+Shift+Space");
        assert_eq!(format_keys(64, "Return"), "Super+Return");
    }

    #[test]
    fn command_from_arg_handles_paths_and_bare_dictate() {
        assert_eq!(
            command_from_arg("/home/u/.local/share/dictate/bin/dictate toggle"),
            "toggle"
        );
        assert_eq!(command_from_arg("dictate start"), "start");
        assert_eq!(command_from_arg("dictate"), "dictate");
    }

    #[test]
    fn malformed_entries_are_skipped() {
        assert!(parse_hyprland_binds("not json").is_empty());
        assert!(parse_hyprland_binds("[]").is_empty());
        // Missing `arg`, and a wrong type for `modmask`: both skipped, the
        // good entry survives.
        let json = r#"[
            {"modmask":0,"key":"F8","release":false,"dispatcher":"exec"},
            {"modmask":"x","key":"F8","release":false,"arg":"dictate toggle"},
            {"modmask":0,"key":"F8","release":false,"arg":"dictate toggle"}
        ]"#;
        let binds = parse_hyprland_binds(json);
        assert_eq!(binds.len(), 1);
        assert_eq!(binds[0].label, "toggle");
    }

    #[test]
    fn braces_and_escapes_inside_strings_parse() {
        let json = r#"[
            {"modmask":0,"key":"}","release":false,"arg":"notify-send \"dictate {done}\""}
        ]"#;
        let binds = parse_hyprland_binds(json);
        assert_eq!(binds.len(), 1);
        assert_eq!(binds[0].keys, "}");
        assert_eq!(binds[0].label, "notify-send \"dictate {done}\"");
    }

    #[test]
    fn matches_dictate_case_insensitively() {
        let json = r#"[
            {"modmask":0,"key":"F8","release":false,"arg":"DICTATE toggle"},
            {"modmask":0,"key":"F9","release":false,"arg":"9","description":"Toggle Dictate"}
        ]"#;
        let binds = parse_hyprland_binds(json);
        assert_eq!(binds.len(), 2);
        assert_eq!(binds[0].label, "toggle");
        assert_eq!(binds[1].label, "Toggle Dictate");
    }

    #[test]
    fn page_layout_numbers() {
        assert_eq!(PAGE_MARGIN, 16.0);
        assert_eq!(EXPLAIN_SIZE, 12.0);
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
