//! Settings page: one scrolling page with headed sections.
//!
//! The page renders borrowed daemon state ([`PageView`]) and returns user
//! intents ([`PageAction`]); the daemon applies them through the same code
//! paths the tray menu uses, so there is no duplicate logic.

use std::str::FromStr;

use global_hotkey::hotkey::HotKey;

use crate::config::{
    DictationMode as Mode, InferenceThreads, ModelChoice as SpeechModel, Settings,
};
use crate::daemon::{LiveSpeedCheck, SwitchTarget};
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
        "Best for accuracy and slower computers; text appears when you stop.",
    ),
    (
        Mode::Pauses,
        "Type at pauses",
        "Types when you pause; choose Parakeet or Fluid to avoid Whisper Tiny's extra delay.",
    ),
    (
        Mode::Live,
        "Live",
        "Text appears while you speak; requires Nemotron or Moonshine.",
    ),
];

/// All speech models offered on the page, with a radio label and choice guidance.
pub(crate) const SPEECH_MODEL_OPTIONS: [(SpeechModel, &str, &str); 5] = [
    (
        SpeechModel::Parakeet,
        SpeechModel::Parakeet.display_name(),
        "Fastest transcription and best accuracy in our test; needs more RAM and AVX2.",
    ),
    (
        SpeechModel::Fluid,
        SpeechModel::Fluid.display_name(),
        "Good accuracy with less RAM than Parakeet; a slower, balanced choice.",
    ),
    (
        SpeechModel::Whisper,
        SpeechModel::Whisper.display_name(),
        "For older CPUs without AVX2 or low RAM; loads fastest, with lower accuracy.",
    ),
    (
        SpeechModel::Nemotron,
        SpeechModel::Nemotron.display_name(),
        "Accurate live text; best for fast CPUs with plenty of RAM.",
    ),
    (
        SpeechModel::Moonshine,
        SpeechModel::Moonshine.display_name(),
        "Low-memory live dictation on Linux; faster final text, but lower accuracy.",
    ),
];

/// Inference-thread choices for Whisper and Moonshine, in page order.
pub(crate) const INFERENCE_THREAD_OPTIONS: [(InferenceThreads, &str); 3] = [
    (InferenceThreads::Auto, "Auto (1–2 by engine)"),
    (InferenceThreads::One, "1 thread"),
    (InferenceThreads::Two, "2 threads (Moonshine: default)"),
];

/// User intents from one page frame, applied by the daemon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PageAction {
    SelectMic(String),
    RefreshMics,
    RefreshBinds,
    SelectMode(Mode),
    SelectSpeechModel(SpeechModel),
    SelectInferenceThreads(InferenceThreads),
    SelectShortcut(String),
    ConfigureShortcut,
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
    /// A mode or speech-model switch is loading in the background, if any.
    pub switch_target: Option<SwitchTarget>,
    /// The switch is downloading the target model first.
    pub downloading: bool,
    /// Model download progress, 0–100.
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
    pub shortcut_status: Option<&'a str>,
    pub shortcut_busy: bool,
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

/// Status line while a selection's model downloads in the background.
pub(crate) fn download_line(target: SwitchTarget, percent: u64) -> String {
    format!("Downloading {}… {percent}%", target.model.display_name())
}

pub(crate) fn speech_model_loading_line(model: SpeechModel) -> String {
    format!("Loading {}…", model.display_name())
}

/// Status line for a finished Live speed check.
pub(crate) fn speed_check_line(check: LiveSpeedCheck) -> String {
    let verdict = if check.fast {
        "fast enough"
    } else {
        "too slow"
    };
    format!(
        "Nemotron speed check: {} ms per 560 ms of audio ({verdict})",
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

/// Render one page frame, collecting the user's intents. `capture` is the
/// window's shortcut-recording state, taken out for the frame.
pub(crate) fn show_page(
    ui: &mut egui::Ui,
    view: &PageView,
    capture: &mut crate::settings_ui::ShortcutCapture,
) -> Vec<PageAction> {
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
                speech_model_section(ui, view, &mut actions);
                section_gap(ui);
                mode_section(ui, view, &mut actions);
                section_gap(ui);
                shortcut_section(ui, view, &mut actions, capture);
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
    let selected = crate::config::effective_selection(view.settings);
    let busy = view.shortcut_busy || view.switch_target.is_some();
    for (mode, label, explanation) in MODE_OPTIONS {
        let supported = selected.model.supports(mode);
        if ui
            .add_enabled(
                !busy && supported,
                egui::RadioButton::new(selected.mode == mode, label),
            )
            .on_disabled_hover_text("Choose a compatible speech model first.")
            .clicked()
        {
            actions.push(PageAction::SelectMode(mode));
        }
        explain(ui, explanation);
    }
    explain(
        ui,
        "Unavailable modes are disabled for the selected speech model.",
    );
}

fn speech_model_section(ui: &mut egui::Ui, view: &PageView, actions: &mut Vec<PageAction>) {
    heading(ui, "Speech model");
    explain(
        ui,
        "Guidance is based on our CPU benchmark; results vary with your computer and voice.",
    );
    let selected = crate::config::effective_selection(view.settings);
    let busy = view.shortcut_busy || view.switch_target.is_some();
    for (model, label, explanation) in SPEECH_MODEL_OPTIONS {
        if ui
            .add_enabled(
                !busy && model.available(),
                egui::RadioButton::new(selected.model == model, label),
            )
            .on_disabled_hover_text(
                "Finish dictating or loading first. Moonshine requires Linux x86_64/arm64.",
            )
            .clicked()
        {
            actions.push(PageAction::SelectSpeechModel(model));
        }
        explain(ui, explanation);
    }
    explain(
        ui,
        "Choosing a model automatically selects a compatible mode when needed.",
    );
    if let Some(target) = view.switch_target {
        if view.downloading {
            ui.label(download_line(target, view.download_percent));
        } else {
            ui.label(speech_model_loading_line(target.model));
        }
    }
    if view.checking_speed {
        ui.label("Checking Nemotron speed…");
    }
    if let Some(check) = view.last_speed_check {
        ui.label(speed_check_line(check));
    }
    heading(ui, "Inference threads");
    for (threads, label) in INFERENCE_THREAD_OPTIONS {
        if ui
            .add_enabled(
                !busy,
                egui::RadioButton::new(view.settings.inference_threads == threads, label),
            )
            .clicked()
        {
            actions.push(PageAction::SelectInferenceThreads(threads));
        }
    }
    explain(
        ui,
        "Applies to Whisper and Moonshine. Auto: up to 2 threads for Whisper, single-thread for Moonshine. 1: one thread. 2: Whisper uses exactly 2; Moonshine uses its default threading, not exactly 2. Restart the daemon after changing this.",
    );
}

fn shortcut_section(
    ui: &mut egui::Ui,
    view: &PageView,
    actions: &mut Vec<PageAction>,
    capture: &mut crate::settings_ui::ShortcutCapture,
) {
    heading(ui, "Shortcut");
    let mac = cfg!(target_os = "macos");
    if capture.recording {
        // Read the frame input before any widget runs, so captured keys never
        // reach the rest of the page.
        poll_shortcut_capture(ui, actions, capture, mac);
    }
    let presets = crate::hotkey::shortcut_presets(mac);
    let current = view.settings.hotkey.as_str();
    if view.wayland && !view.hyprland {
        explain(
            ui,
            "Preferred shortcut. Your desktop confirms the actual binding in its shortcut dialog.",
        );
    }
    ui.add_enabled_ui(!view.shortcut_busy, |ui| {
        ui.horizontal(|ui| {
            egui::ComboBox::from_label("")
                .selected_text(crate::hotkey::shortcut_label(current, mac))
                .show_ui(ui, |ui| {
                    if !presets.contains(&current)
                        && ui
                            .selectable_label(true, crate::hotkey::shortcut_label(current, mac))
                            .clicked()
                    {
                        actions.push(PageAction::SelectShortcut(current.to_string()));
                    }
                    for preset in presets {
                        if ui
                            .selectable_label(
                                *preset == current,
                                crate::hotkey::shortcut_label(preset, mac),
                            )
                            .clicked()
                        {
                            actions.push(PageAction::SelectShortcut(preset.to_string()));
                        }
                    }
                });
            let label = if capture.recording {
                "Cancel"
            } else {
                "Record shortcut…"
            };
            if ui.button(label).clicked() {
                capture.recording = !capture.recording;
                capture.error = None;
                capture.held = None;
            }
        });
    });
    if capture.recording {
        ui.label("Press the new shortcut now (Esc to cancel).");
        if mac {
            ui.label("Tip: you can also press just modifier keys, like Command+Option.");
        }
    }
    explain(ui, "Tap to toggle dictation, hold to talk.");
    if view.wayland {
        wayland_shortcut_section(ui, view, actions);
    }
    inline_error(ui, capture.error.as_deref().or(view.shortcut_error));
}

/// Handle one frame of shortcut recording: Esc cancels, the first real key
/// press builds a value via [`shortcut_from_keys`], and a modifiers-held-then-
/// released span with no key records a modifier-only chord. Key events are frame
/// input (no widget focus or text field needed); handled ones are consumed so
/// the rest of the page ignores captured keys. Repeats never trigger.
/// (egui's clipboard shortcuts stay off while recording, so every key arrives
/// as a plain `Event::Key`.)
fn poll_shortcut_capture(
    ui: &mut egui::Ui,
    actions: &mut Vec<PageAction>,
    capture: &mut crate::settings_ui::ShortcutCapture,
    mac: bool,
) {
    let (pressed, modifiers) = ui.input(|input| {
        let frame = input.modifiers;
        let pressed: Vec<(egui::Key, egui::Modifiers)> = input
            .events
            .iter()
            .filter_map(|event| match event {
                egui::Event::Key {
                    key,
                    physical_key,
                    pressed: true,
                    repeat: false,
                    modifiers,
                    ..
                } => Some((capture_key(*key, *physical_key), *modifiers)),
                _ => None,
            })
            .collect();
        (pressed, frame)
    });
    let mut saw_key = false;
    for (key, key_modifiers) in &pressed {
        // Bare modifier presses are not capture attempts; they only feed the
        // modifier-only detection below.
        if is_modifier_key(*key) {
            continue;
        }
        saw_key = true;
        if *key == egui::Key::Escape {
            capture.recording = false;
            capture.error = None;
            capture.held = None;
            break;
        }
        match shortcut_from_keys(*key_modifiers, *key, mac) {
            Ok(value) => {
                capture.recording = false;
                capture.error = None;
                capture.held = None;
                actions.push(PageAction::SelectShortcut(value));
                break;
            }
            Err(message) => {
                capture.error = Some(message);
                // Keep recording; a later key this frame still applies.
            }
        }
    }
    if capture.recording {
        if saw_key {
            // A key breaks the modifier-only span.
            capture.held = None;
        } else {
            let mut current = modifier_set_from_egui(modifiers);
            for (_, event_modifiers) in &pressed {
                let set = modifier_set_from_egui(*event_modifiers);
                current.cmd |= set.cmd;
                current.ctrl |= set.ctrl;
                current.option |= set.option;
                current.shift |= set.shift;
            }
            if current.cmd || current.ctrl || current.option || current.shift {
                match &mut capture.held {
                    Some(held) => {
                        held.cmd |= current.cmd;
                        held.ctrl |= current.ctrl;
                        held.option |= current.option;
                        held.shift |= current.shift;
                    }
                    None => capture.held = Some(current),
                }
            } else if let Some(held) = capture.held.take() {
                // All modifiers released with no key in between: record the chord.
                match modifier_only_value(held, mac) {
                    Ok(value) => {
                        capture.recording = false;
                        capture.error = None;
                        actions.push(PageAction::SelectShortcut(value));
                    }
                    Err(message) => {
                        capture.error = Some(message);
                    }
                }
            }
        }
    }
    if !pressed.is_empty() {
        ui.input_mut(|input| {
            input.events.retain(|event| {
                !matches!(
                    event,
                    egui::Event::Key {
                        pressed: true,
                        repeat: false,
                        ..
                    }
                )
            });
        });
    }
}

/// Build a stored shortcut value from a captured key press: modifiers in
/// platform order plus the global-hotkey key name. The raw `ctrl` field (not
/// `command`) keeps Ctrl and Cmd separate on mac. Errors are user-facing
/// inline messages; recording continues.
pub(crate) fn shortcut_from_keys(
    modifiers: egui::Modifiers,
    key: egui::Key,
    mac: bool,
) -> Result<String, String> {
    let key_name = egui_key_name(key).ok_or_else(|| {
        format!(
            "dictate can't use {} as a shortcut key. Try a letter, digit, F key or Space.",
            key.name()
        )
    })?;
    let mut parts: Vec<&str> = Vec::new();
    if mac {
        if modifiers.mac_cmd {
            parts.push("Cmd");
        }
        if modifiers.ctrl {
            parts.push("Ctrl");
        }
        if modifiers.alt {
            parts.push("Option");
        }
        if modifiers.shift {
            parts.push("Shift");
        }
    } else {
        if modifiers.ctrl {
            parts.push("Ctrl");
        }
        if modifiers.alt {
            parts.push("Alt");
        }
        if modifiers.shift {
            parts.push("Shift");
        }
    }
    if parts.is_empty() && !is_f_key(key) {
        return Err(if mac {
            "Add Command, Control, Option or Shift (or use an F key), so normal typing doesn't start dictation."
        } else {
            "Add Ctrl, Alt or Shift (or use an F key), so normal typing doesn't start dictation."
        }
        .to_string());
    }
    parts.push(key_name);
    let value = parts.join("+");
    if let Err(err) = HotKey::from_str(&value) {
        return Err(format!(
            "Internal error: built shortcut \"{value}\" failed to parse: {err}"
        ));
    }
    Ok(value)
}

/// Global-hotkey key name for an egui key, or `None` when the parser has no
/// such key (F25+, Insert, Caps Lock, media keys, bare modifiers, ...).
fn egui_key_name(key: egui::Key) -> Option<&'static str> {
    use egui::Key;
    Some(match key {
        Key::A => "A",
        Key::B => "B",
        Key::C => "C",
        Key::D => "D",
        Key::E => "E",
        Key::F => "F",
        Key::G => "G",
        Key::H => "H",
        Key::I => "I",
        Key::J => "J",
        Key::K => "K",
        Key::L => "L",
        Key::M => "M",
        Key::N => "N",
        Key::O => "O",
        Key::P => "P",
        Key::Q => "Q",
        Key::R => "R",
        Key::S => "S",
        Key::T => "T",
        Key::U => "U",
        Key::V => "V",
        Key::W => "W",
        Key::X => "X",
        Key::Y => "Y",
        Key::Z => "Z",
        Key::Num0 => "0",
        Key::Num1 => "1",
        Key::Num2 => "2",
        Key::Num3 => "3",
        Key::Num4 => "4",
        Key::Num5 => "5",
        Key::Num6 => "6",
        Key::Num7 => "7",
        Key::Num8 => "8",
        Key::Num9 => "9",
        Key::F1 => "F1",
        Key::F2 => "F2",
        Key::F3 => "F3",
        Key::F4 => "F4",
        Key::F5 => "F5",
        Key::F6 => "F6",
        Key::F7 => "F7",
        Key::F8 => "F8",
        Key::F9 => "F9",
        Key::F10 => "F10",
        Key::F11 => "F11",
        Key::F12 => "F12",
        Key::F13 => "F13",
        Key::F14 => "F14",
        Key::F15 => "F15",
        Key::F16 => "F16",
        Key::F17 => "F17",
        Key::F18 => "F18",
        Key::F19 => "F19",
        Key::F20 => "F20",
        Key::F21 => "F21",
        Key::F22 => "F22",
        Key::F23 => "F23",
        Key::F24 => "F24",
        Key::Space => "Space",
        Key::Enter => "Enter",
        Key::Tab => "Tab",
        Key::Backspace => "Backspace",
        Key::Delete => "Delete",
        Key::ArrowDown => "ArrowDown",
        Key::ArrowLeft => "ArrowLeft",
        Key::ArrowRight => "ArrowRight",
        Key::ArrowUp => "ArrowUp",
        Key::Home => "Home",
        Key::End => "End",
        Key::PageUp => "PageUp",
        Key::PageDown => "PageDown",
        Key::Comma => "Comma",
        Key::Backslash => "Backslash",
        Key::Slash => "Slash",
        Key::Backtick => "Backquote",
        Key::Minus => "Minus",
        Key::Period => "Period",
        Key::Equals => "Equal",
        Key::Semicolon => "Semicolon",
        Key::Quote => "Quote",
        Key::OpenBracket => "BracketLeft",
        Key::CloseBracket => "BracketRight",
        _ => return None,
    })
}

/// True for F1–F24, which may stand alone without modifiers.
fn is_f_key(key: egui::Key) -> bool {
    use egui::Key;
    matches!(
        key,
        Key::F1
            | Key::F2
            | Key::F3
            | Key::F4
            | Key::F5
            | Key::F6
            | Key::F7
            | Key::F8
            | Key::F9
            | Key::F10
            | Key::F11
            | Key::F12
            | Key::F13
            | Key::F14
            | Key::F15
            | Key::F16
            | Key::F17
            | Key::F18
            | Key::F19
            | Key::F20
            | Key::F21
            | Key::F22
            | Key::F23
            | Key::F24
    )
}

/// The key a capture attempt uses: the physical position when known, else the
/// logical key. egui-winit reports the logical key first (Shift+1 arrives as
/// `!`), while global-hotkey registers physical key codes, so the physical key
/// is what was actually pressed.
fn capture_key(key: egui::Key, physical_key: Option<egui::Key>) -> egui::Key {
    physical_key.unwrap_or(key)
}

/// True for bare modifier keys, which never complete a capture on their own.
fn is_modifier_key(key: egui::Key) -> bool {
    use egui::Key;
    matches!(
        key,
        Key::ShiftLeft
            | Key::ShiftRight
            | Key::ControlLeft
            | Key::ControlRight
            | Key::AltLeft
            | Key::AltRight
            | Key::SuperLeft
            | Key::SuperRight
    )
}

/// The chord modifiers held in one egui modifier state. Uses the raw `ctrl`
/// field (not `command`) so Ctrl and Cmd stay separate on mac.
fn modifier_set_from_egui(modifiers: egui::Modifiers) -> crate::hotkey::ModifierSet {
    crate::hotkey::ModifierSet {
        cmd: modifiers.mac_cmd,
        ctrl: modifiers.ctrl,
        option: modifiers.alt,
        shift: modifiers.shift,
    }
}

/// Stored value (or inline error) for a modifier-only span: the held
/// modifiers in `Cmd, Ctrl, Option, Shift` order, rejected when unusable.
fn modifier_only_value(held: crate::hotkey::ModifierSet, mac: bool) -> Result<String, String> {
    let mut parts = Vec::new();
    if held.cmd {
        parts.push("Cmd");
    }
    if held.ctrl {
        parts.push("Ctrl");
    }
    if held.option {
        parts.push("Option");
    }
    if held.shift {
        parts.push("Shift");
    }
    let value = parts.join("+");
    if let Some(message) = crate::hotkey::modifier_only_error(&value, mac) {
        Err(message)
    } else {
        Ok(value)
    }
}

/// Show the portal's actual status separately from the preferred key. On
/// Hyprland, presets also install the compositor binding for that action.
fn wayland_shortcut_section(ui: &mut egui::Ui, view: &PageView, actions: &mut Vec<PageAction>) {
    if let Some(status) = view.shortcut_status {
        ui.label(status);
    }
    ui.add_enabled_ui(!view.shortcut_busy, |ui| {
        if view.hyprland {
            if ui.button("Enable shortcut / apply again").clicked() {
                actions.push(PageAction::SelectShortcut(view.settings.hotkey.clone()));
            }
            explain(ui, "Selecting a preset applies it immediately. Dictate checks for conflicts and backs up your Hyprland config before saving its own shortcut block.");
        } else if ui.button("Desktop shortcut settings…").clicked() {
            actions.push(PageAction::ConfigureShortcut);
        }
    });
    if view.hyprland {
        match view.dictate_binds {
            Some(binds) if !binds.is_empty() => {
                ui.label("Dictate bindings found in Hyprland:");
                for line in binds {
                    ui.label(line);
                }
            }
            Some(_) => {}
            None => {
                ui.label("Could not list compositor bindings.");
            }
        }
    }
    if ui.button("Refresh").clicked() {
        actions.push(PageAction::RefreshBinds);
    }
    explain(
        ui,
        "Requires a desktop with the Global Shortcuts portal. You can also bind the local dictate toggle or start/stop commands manually.",
    );
    ui.hyperlink_to("Manual setup", crate::tray::README_USE_IT_URL);
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
    use crate::config::Selection;

    #[test]
    fn mode_options_cover_every_mode_with_its_text() {
        let modes: Vec<Mode> = MODE_OPTIONS.iter().map(|(mode, _, _)| *mode).collect();
        assert_eq!(modes, vec![Mode::Whole, Mode::Pauses, Mode::Live]);
        assert_eq!(MODE_OPTIONS[0].1, "Whole recording");
        assert_eq!(
            MODE_OPTIONS[0].2,
            "Best for accuracy and slower computers; text appears when you stop."
        );
        assert_eq!(MODE_OPTIONS[1].1, "Type at pauses");
        assert_eq!(
            MODE_OPTIONS[1].2,
            "Types when you pause; choose Parakeet or Fluid to avoid Whisper Tiny's extra delay."
        );
        assert_eq!(MODE_OPTIONS[2].1, "Live");
        assert_eq!(
            MODE_OPTIONS[2].2,
            "Text appears while you speak; requires Nemotron or Moonshine."
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
    fn speech_model_options_cover_both_models_with_their_text() {
        let models: Vec<SpeechModel> = SPEECH_MODEL_OPTIONS
            .iter()
            .map(|(model, _, _)| *model)
            .collect();
        assert_eq!(
            models,
            vec![
                SpeechModel::Parakeet,
                SpeechModel::Fluid,
                SpeechModel::Whisper,
                SpeechModel::Nemotron,
                SpeechModel::Moonshine
            ]
        );
        assert_eq!(SPEECH_MODEL_OPTIONS[0].1, "Parakeet TDT 0.6B v2");
        assert_eq!(
            SPEECH_MODEL_OPTIONS[0].2,
            "Fastest transcription and best accuracy in our test; needs more RAM and AVX2."
        );
        assert_eq!(SPEECH_MODEL_OPTIONS[1].1, "Fluid Speech Mini");
        assert_eq!(
            SPEECH_MODEL_OPTIONS[1].2,
            "Good accuracy with less RAM than Parakeet; a slower, balanced choice."
        );
        assert_eq!(SPEECH_MODEL_OPTIONS[2].1, "Whisper Tiny English");
        assert_eq!(
            SPEECH_MODEL_OPTIONS[2].2,
            "For older CPUs without AVX2 or low RAM; loads fastest, with lower accuracy."
        );
    }

    #[test]
    fn inference_thread_options_cover_every_setting() {
        let threads: Vec<InferenceThreads> = INFERENCE_THREAD_OPTIONS
            .iter()
            .map(|(threads, _)| *threads)
            .collect();
        assert_eq!(
            threads,
            vec![
                InferenceThreads::Auto,
                InferenceThreads::One,
                InferenceThreads::Two
            ]
        );
        assert_eq!(INFERENCE_THREAD_OPTIONS[0].1, "Auto (1–2 by engine)");
    }

    #[test]
    fn mode_status_lines() {
        let live = Selection::new(Mode::Live, SpeechModel::Nemotron).unwrap();
        assert_eq!(
            download_line(live, 25),
            "Downloading Nemotron Streaming… 25%"
        );
        assert_eq!(
            download_line(live, 100),
            "Downloading Nemotron Streaming… 100%"
        );
        let moonshine = Selection::new(Mode::Live, SpeechModel::Moonshine).unwrap();
        assert_eq!(
            download_line(moonshine, 50),
            "Downloading Moonshine Tiny Streaming… 50%"
        );
        let fluid = Selection::new(Mode::Whole, SpeechModel::Fluid).unwrap();
        assert_eq!(
            download_line(fluid, 50),
            "Downloading Fluid Speech Mini… 50%"
        );
        assert_eq!(
            speech_model_loading_line(SpeechModel::Fluid),
            "Loading Fluid Speech Mini…"
        );
        assert_eq!(
            speech_model_loading_line(SpeechModel::Parakeet),
            "Loading Parakeet TDT 0.6B v2…"
        );
        assert_eq!(
            speech_model_loading_line(SpeechModel::Whisper),
            "Loading Whisper Tiny English…"
        );
        assert_eq!(
            speed_check_line(LiveSpeedCheck {
                ms: 120,
                fast: true
            }),
            "Nemotron speed check: 120 ms per 560 ms of audio (fast enough)"
        );
        assert_eq!(
            speed_check_line(LiveSpeedCheck {
                ms: 905,
                fast: false
            }),
            "Nemotron speed check: 905 ms per 560 ms of audio (too slow)"
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
    fn capture_builds_values_and_errors() {
        // Cmd+Option+Space on mac.
        let mods = egui::Modifiers {
            mac_cmd: true,
            command: true,
            alt: true,
            ..Default::default()
        };
        assert_eq!(
            shortcut_from_keys(mods, egui::Key::Space, true).unwrap(),
            "Cmd+Option+Space"
        );
        // Ctrl+Shift+D elsewhere.
        let mods = egui::Modifiers {
            ctrl: true,
            command: true,
            shift: true,
            ..Default::default()
        };
        assert_eq!(
            shortcut_from_keys(mods, egui::Key::D, false).unwrap(),
            "Ctrl+Shift+D"
        );
        // Plain A needs a modifier, with per-OS wording.
        assert_eq!(
            shortcut_from_keys(egui::Modifiers::NONE, egui::Key::A, true).unwrap_err(),
            "Add Command, Control, Option or Shift (or use an F key), so normal typing doesn't start dictation."
        );
        assert_eq!(
            shortcut_from_keys(egui::Modifiers::NONE, egui::Key::A, false).unwrap_err(),
            "Add Ctrl, Alt or Shift (or use an F key), so normal typing doesn't start dictation."
        );
        // Plain F8 stands alone on both.
        assert_eq!(
            shortcut_from_keys(egui::Modifiers::NONE, egui::Key::F8, true).unwrap(),
            "F8"
        );
        assert_eq!(
            shortcut_from_keys(egui::Modifiers::NONE, egui::Key::F8, false).unwrap(),
            "F8"
        );
        // Both Cmd and Ctrl kept on mac (raw ctrl, not merged command).
        let mods = egui::Modifiers {
            mac_cmd: true,
            command: true,
            ctrl: true,
            shift: true,
            ..Default::default()
        };
        assert_eq!(
            shortcut_from_keys(mods, egui::Key::D, true).unwrap(),
            "Cmd+Ctrl+Shift+D"
        );
        // Unknown keys name the key.
        let mods = egui::Modifiers {
            ctrl: true,
            ..Default::default()
        };
        assert_eq!(
            shortcut_from_keys(mods, egui::Key::CapsLock, false).unwrap_err(),
            "dictate can't use CapsLock as a shortcut key. Try a letter, digit, F key or Space."
        );
    }

    #[test]
    fn capture_prefers_the_physical_key() {
        assert_eq!(
            capture_key(egui::Key::Exclamationmark, Some(egui::Key::Num1)),
            egui::Key::Num1
        );
        assert_eq!(capture_key(egui::Key::A, None), egui::Key::A);
    }

    #[test]
    fn modifier_span_builds_chord_values_and_errors() {
        use crate::hotkey::ModifierSet;
        let cmd_option = ModifierSet {
            cmd: true,
            ctrl: false,
            option: true,
            shift: false,
        };
        assert_eq!(modifier_only_value(cmd_option, true).unwrap(), "Cmd+Option");
        let cmd_only = ModifierSet {
            cmd: true,
            ctrl: false,
            option: false,
            shift: false,
        };
        assert_eq!(
            modifier_only_value(cmd_only, true).unwrap_err(),
            "Use at least two modifier keys (for example Command+Option), or add a key."
        );
        let ctrl_alt = ModifierSet {
            cmd: false,
            ctrl: true,
            option: true,
            shift: false,
        };
        assert_eq!(
            modifier_only_value(ctrl_alt, false).unwrap_err(),
            "Shortcuts with only modifier keys work only on macOS. Add a key, for example Ctrl+Alt+Space."
        );
    }

    #[test]
    fn every_captured_value_parses() {
        use egui::{Key, Modifiers};
        let mod_sets = [
            Modifiers::NONE,
            Modifiers {
                ctrl: true,
                ..Default::default()
            },
            Modifiers {
                alt: true,
                shift: true,
                ..Default::default()
            },
            Modifiers {
                mac_cmd: true,
                ctrl: true,
                alt: true,
                shift: true,
                ..Default::default()
            },
        ];
        let keys = [
            Key::A,
            Key::Z,
            Key::Num0,
            Key::Num9,
            Key::F1,
            Key::F12,
            Key::F24,
            Key::Space,
            Key::Enter,
            Key::Tab,
            Key::Backspace,
            Key::Delete,
            Key::ArrowUp,
            Key::ArrowDown,
            Key::Home,
            Key::End,
            Key::PageUp,
            Key::PageDown,
            Key::Comma,
            Key::Period,
            Key::Semicolon,
            Key::Slash,
            Key::Backslash,
            Key::Quote,
            Key::Backtick,
            Key::Minus,
            Key::Equals,
            Key::OpenBracket,
            Key::CloseBracket,
        ];
        for mac in [true, false] {
            for mods in mod_sets {
                for key in keys {
                    if let Ok(value) = shortcut_from_keys(mods, key, mac) {
                        assert!(HotKey::from_str(&value).is_ok(), "{value} does not parse");
                    }
                }
            }
        }
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
