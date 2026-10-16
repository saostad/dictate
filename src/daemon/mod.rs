//! Long-running dictation daemon, controlled over a Unix socket or hotkey.
//!
//! `dictate daemon` keeps the speech model loaded. On Unix, `dictate
//! toggle|start|stop|cancel|status` sends one line to the socket and prints the
//! one-line reply; everywhere, a global hotkey (tap to toggle, hold to talk)
//! triggers the same commands without a terminal. On Windows there is no
//! socket yet, so the daemon is hotkey-only.
//!
//! The main thread runs a windowless winit event loop. It owns the state, the
//! [`Recorder`](crate::mic::Recorder) and the hotkey manager (cpal streams are
//! not `Send` everywhere, and the manager belongs on the event-loop thread).
//! The socket accept loop runs on a thread and forwards commands as user
//! events; one worker thread owns the [`Asr`](crate::asr::Asr) and transcribes
//! audio sent over an mpsc channel.
//!
//! Every daemon line goes through [`dlog!`]: stderr, plus `<data_dir>/daemon.log`
//! when stderr is not a terminal (compositor or on-demand autostart).

/// Print one daemon line: stderr, and the log file when active (see
/// [`init_file_log`]). Same syntax as `eprintln!`.
macro_rules! dlog {
    ($($arg:tt)*) => {
        $crate::daemon::log_line(format_args!($($arg)*))
    };
}

#[cfg(unix)]
pub(crate) mod socket;

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::time::{Duration, Instant};

use anyhow::Context;
use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use winit::application::ApplicationHandler;
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::WindowId;

/// A control command sent by a client (one line on the socket, or one hotkey
/// press). Some variants are only constructed by the Unix socket parser and
/// the tests, hence the allowance on other systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(unix), allow(dead_code))]
pub(crate) enum Command {
    Toggle,
    Start,
    Stop,
    Cancel,
    Status,
}

/// Recording state of the daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum State {
    Idle,
    Recording,
    Transcribing,
}

/// What the accept loop must do after a state transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    None,
    StartRecording,
    StopRecording,
    CancelRecording,
}

/// Pure state machine: the next state, the action to run, and the one-line
/// reply for every (state, command) pair.
pub(crate) fn next(state: State, cmd: Command) -> (State, Action, &'static str) {
    use Action::{CancelRecording, None, StartRecording, StopRecording};
    use Command::{Cancel, Start, Status, Stop, Toggle};
    use State::{Idle, Recording, Transcribing};
    match (state, cmd) {
        (Idle, Start) | (Idle, Toggle) => (Recording, StartRecording, "ok recording"),
        (Idle, Stop) => (Idle, None, "err not recording"),
        (Idle, Cancel) | (Idle, Status) => (Idle, None, "ok idle"),
        (Recording, Start) => (Recording, None, "err already recording"),
        (Recording, Stop) | (Recording, Toggle) => (Transcribing, StopRecording, "ok transcribing"),
        (Recording, Cancel) => (Idle, CancelRecording, "ok cancelled"),
        (Recording, Status) => (Recording, None, "ok recording"),
        (Transcribing, Status) => (Transcribing, None, "ok transcribing"),
        (Transcribing, Start)
        | (Transcribing, Stop)
        | (Transcribing, Toggle)
        | (Transcribing, Cancel) => (Transcribing, None, "err busy transcribing"),
    }
}

/// Display name of the default microphone in a listing, if any. Pure, for
/// the tray labels and tests.
fn default_mic_name(mics: &[crate::mic::Mic]) -> Option<String> {
    mics.iter()
        .find(|mic| mic.is_default)
        .map(|mic| mic.name.clone())
}

/// Display name of the selected microphone: the selection when it is listed,
/// else the default (what the next recording will use). Pure, for the tray
/// labels and tests.
fn selected_mic_name(selected: &str, mics: &[crate::mic::Mic]) -> Option<String> {
    if !selected.is_empty()
        && let Some(mic) = mics.iter().find(|mic| mic.id == selected)
    {
        return Some(mic.name.clone());
    }
    default_mic_name(mics)
}

/// True when a refreshed microphone listing changed (device ids or the
/// default), so the cached recording device must be re-resolved. Pure, for
/// tests. Names are display-only and don't affect the cache.
fn mic_listing_changed(old: &[crate::mic::Mic], new: &[crate::mic::Mic]) -> bool {
    if old.len() != new.len() {
        return true;
    }
    old.iter()
        .zip(new.iter())
        .any(|(before, after)| before.id != after.id || before.is_default != after.is_default)
}

/// One unit of work for the transcription worker.
pub(crate) enum Job {
    /// Whole recording at 16 kHz (whole mode): today's exact path.
    Full(Vec<f32>),
    /// One pause segment at 16 kHz plus its left context (pauses mode).
    Segment(crate::segment::Segment),
    /// End of a pauses recording: whole-recording duration plus the moment
    /// Stop was handled, so the worker can log latency after release.
    End {
        total_seconds: f32,
        stop_time: Instant,
    },
    /// 16 kHz audio for the Live engine (live mode, arbitrary length); the
    /// worker buffers it into model chunks.
    LiveChunk(Vec<f32>),
    /// End of a Live recording: the moment Stop was handled, so the worker
    /// can log latency after release.
    LiveEnd { stop_time: Instant },
    /// Discard a pauses/Live recording; already-typed text stays.
    Cancel,
    /// Swap in a freshly loaded TDT model (menu mode switch to whole/pauses);
    /// drops the Live model, then acks [`UserEvent::ModeSwapped`].
    SwapTdt(crate::asr::Asr),
    /// Swap in a freshly loaded Live model (menu mode switch to live); drops
    /// the TDT model, then acks [`UserEvent::ModeSwapped`].
    SwapLive(crate::live::Live),
}

impl std::fmt::Debug for Job {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The models are not `Debug`; variant names are enough for logs.
        let name = match self {
            Job::Full(_) => "Full",
            Job::Segment(_) => "Segment",
            Job::End { .. } => "End",
            Job::LiveChunk(_) => "LiveChunk",
            Job::LiveEnd { .. } => "LiveEnd",
            Job::Cancel => "Cancel",
            Job::SwapTdt(_) => "SwapTdt",
            Job::SwapLive(_) => "SwapLive",
        };
        write!(f, "{name}")
    }
}

/// Control message for a feeder thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeederCmd {
    Stop,
    Cancel,
}

/// A running feeder thread (pauses/live mode, while Recording only). The
/// pauses thread hands the Vad back on exit (`None` when feeder setup
/// consumed it); the daemon reuses one Vad across recordings.
enum FeederSession {
    Pauses {
        control_tx: mpsc::Sender<FeederCmd>,
        thread: Option<std::thread::JoinHandle<Option<crate::vad::Vad>>>,
    },
    Live {
        control_tx: mpsc::Sender<FeederCmd>,
        thread: Option<std::thread::JoinHandle<()>>,
    },
}

/// Effective dictation mode (settings file + `DICTATE_MODE`).
fn dictation_mode() -> crate::config::Mode {
    crate::config::mode()
}

/// Events handled on the daemon's event-loop thread.
#[derive(Debug)]
pub(crate) enum UserEvent {
    /// A command from a socket client; the reply goes back on the channel.
    #[cfg(unix)]
    Command(Command, mpsc::Sender<String>),
    /// The worker finished one transcription; back to Idle.
    WorkerDone,
    /// A global hotkey press or release.
    HotKey(GlobalHotKeyEvent),
    /// A tray menu click.
    Menu(tray_icon::menu::MenuEvent),
    /// The worker swapped in a pending mode switch's model; the switch can
    /// complete (save, notify).
    ModeSwapped,
    /// A background mode switch failed; already logged + notified, so the
    /// event loop only clears the loading state.
    ModeSwitchFailed,
    /// Open the Settings window (`dictate settings` from the socket; the tray
    /// menu calls `open_settings` directly).
    OpenSettings,
    /// Background mode-switch progress changed; repaint the Settings window
    /// if it is open.
    SettingsRefresh,
}

/// Progress of a background mode switch, shared between the loader thread and
/// the Settings page (which shows it inline; the tray label shows Loading).
#[derive(Debug, Clone, Default)]
pub(crate) struct SwitchProgress {
    /// Live-model download progress, 0–100.
    pub percent: u64,
    /// The Live speed check is running now.
    pub checking_speed: bool,
    /// The last Live speed-check outcome this session, if one ran.
    pub last_check: Option<LiveSpeedCheck>,
}

/// One Live speed-check outcome: measured ms per 560 ms chunk, and whether
/// the machine is fast enough for Live mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LiveSpeedCheck {
    pub ms: u64,
    pub fast: bool,
}

impl LiveSpeedCheck {
    /// Build from a benchmark median: measured ms plus the speed-gate verdict.
    pub(crate) fn new(median: Duration) -> Self {
        Self {
            ms: median.as_millis().min(u128::from(u64::MAX)) as u64,
            fast: crate::live::is_fast_enough(median),
        }
    }
}

/// Lock the shared switch progress. A poisoned lock yields its value: a
/// panicked loader thread already failed the switch loudly.
fn lock_progress(progress: &Mutex<SwitchProgress>) -> MutexGuard<'_, SwitchProgress> {
    progress
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A mode switch running on a background thread (the event loop stays
/// responsive). While one is in flight the tray label reads Loading and start
/// requests are refused.
enum ModeSwitch {
    /// Loading the target mode's model (Live loads always benchmark too).
    Loading { target: crate::config::Mode },
    /// Downloading the Live model, then loading it (always targets Live).
    Downloading,
}

impl ModeSwitch {
    /// The mode the switch is heading to.
    fn target(&self) -> crate::config::Mode {
        match self {
            ModeSwitch::Loading { target } => *target,
            ModeSwitch::Downloading => crate::config::Mode::Live,
        }
    }
}

/// Daemon log file, opened when stderr is not a terminal.
static LOG_FILE: std::sync::Mutex<Option<std::fs::File>> = std::sync::Mutex::new(None);

/// Pure daemon-log path: `<data_dir>/daemon.log`.
pub(crate) fn daemon_log_path_for(data_dir: &Path) -> PathBuf {
    data_dir.join("daemon.log")
}

/// Path of the daemon log file.
pub(crate) fn daemon_log_path() -> anyhow::Result<PathBuf> {
    let dirs = directories::ProjectDirs::from("", "", "dictate")
        .ok_or_else(|| anyhow::anyhow!("Could not find a home/data directory for this user"))?;
    Ok(daemon_log_path_for(dirs.data_dir()))
}

/// Start mirroring daemon lines to the log file when stderr is not a
/// terminal (autostarted daemons). Failures warn on stderr and keep
/// stderr-only logging. Skipped when the on-demand client already redirects
/// stderr to the log file (then every line, including panics, is logged via
/// the file descriptor and mirroring would duplicate it).
pub(crate) fn init_file_log() {
    use std::io::IsTerminal;
    if std::io::stderr().is_terminal()
        || std::env::var("DICTATE_STDERR_IS_LOG").as_deref() == Ok("1")
    {
        return;
    }
    let path = match daemon_log_path() {
        Ok(path) => path,
        Err(err) => {
            eprintln!("warning: could not open the daemon log: {err:#}");
            return;
        }
    };
    if let Some(parent) = path.parent()
        && let Err(err) = std::fs::create_dir_all(parent)
    {
        eprintln!(
            "warning: could not open daemon log {}: {err:#}",
            path.display()
        );
        return;
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(file) => {
            if let Ok(mut slot) = LOG_FILE.lock() {
                *slot = Some(file);
            }
        }
        Err(err) => eprintln!(
            "warning: could not open daemon log {}: {err:#}",
            path.display()
        ),
    }
}

/// Print one daemon line to stderr, and to the log file when active.
pub(crate) fn log_line(args: std::fmt::Arguments<'_>) {
    eprintln!("{args}");
    mirror_to_log(&args.to_string());
}

/// Append one already-printed line to the log file, when active. A failed
/// lock or write is skipped: stderr already shows the line.
pub(crate) fn mirror_to_log(line: &str) {
    use std::io::Write;
    let Ok(mut slot) = LOG_FILE.lock() else {
        return;
    };
    if let Some(file) = slot.as_mut()
        && writeln!(file, "{line}").is_err()
    {
        // The line is already on stderr; nothing more can be done.
    }
}

/// Run `dictate daemon`. A fatal startup failure is mirrored to the log
/// file too (its stderr twin is printed by `main`).
pub(crate) fn run_daemon(background: bool) -> anyhow::Result<()> {
    detach_console(background);
    init_file_log();
    repair_autostart();
    run_daemon_inner().map_err(|err| {
        mirror_to_log(&format!("error: {err:#}"));
        err
    })
}

/// Detach from the console for login launches (`daemon --background`); a
/// no-op on other platforms and without the flag.
#[cfg(target_os = "windows")]
fn detach_console(background: bool) {
    if background {
        // SAFETY: FreeConsole only detaches this process from its console; it
        // has no preconditions and its return value needs no handling here.
        unsafe { FreeConsole() };
    }
}

/// Detach from the console for login launches (`daemon --background`); a
/// no-op on other platforms and without the flag.
#[cfg(not(target_os = "windows"))]
fn detach_console(_background: bool) {}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn FreeConsole() -> i32;
}

/// Self-repair: when start-at-login is on but the entry points at a different
/// executable (the binary was moved), rewrite it for this one. A check or
/// rewrite failure is logged and never stops the daemon.
fn repair_autostart() {
    match crate::autostart::status() {
        Ok(state) if state.enabled && state.stale => match crate::autostart::enable() {
            Ok(()) => match std::env::current_exe() {
                Ok(exe) => dlog!("autostart: updated entry to {}", exe.display()),
                Err(_) => dlog!("autostart: updated the start-at-login entry"),
            },
            Err(err) => dlog!("error: Could not update the start-at-login entry: {err:#}"),
        },
        Ok(_) => {}
        Err(err) => dlog!("error: Could not check the start-at-login entry: {err:#}"),
    }
}

/// Load the model, bind the socket (Unix), serve commands.
fn run_daemon_inner() -> anyhow::Result<()> {
    let (file_settings, settings_problem) = crate::config::load();
    crate::config::init_effective(&file_settings);
    if let Some(problem) = settings_problem {
        let message = format!("{problem} Using defaults.");
        dlog!("error: {message}");
        crate::notify::show(
            crate::notify::Kind::Error,
            "dictate: settings file has an error",
            &message,
        );
    }
    // Env wins; an unset or empty DICTATE_HOTKEY falls back to the file.
    let hotkey = match std::env::var("DICTATE_HOTKEY").ok() {
        Some(value) if !value.is_empty() => crate::hotkey::resolve_hotkey(Some(&value))
            .map_err(|message| anyhow::anyhow!("{message}"))?,
        _ => crate::hotkey::resolve_file_hotkey(&file_settings.hotkey)
            .map_err(|message| anyhow::anyhow!("{message}"))?,
    };
    let load_start = Instant::now();
    let switch_progress = Arc::new(Mutex::new(SwitchProgress::default()));
    let (models, live_fallback) = load_daemon_models(dictation_mode(), &switch_progress)?;
    #[cfg(unix)]
    let socket_file = {
        let path = socket::socket_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Could not create daemon socket directory {}",
                    parent.display()
                )
            })?;
        }
        path
    };
    #[cfg(unix)]
    let listener = socket::bind_listener(&socket_file)?;
    // Ready only once the final model is loaded and the socket is bound, so
    // the line never precedes a fallback load or a missing socket.
    dlog!(
        "model loaded in {} ms. Ready: run \"dictate toggle\" to start/stop recording.",
        load_start.elapsed().as_millis()
    );
    let mut builder = EventLoop::<UserEvent>::with_user_event();
    // Menu bar app: tray icon only, no Dock icon or app menu.
    #[cfg(target_os = "macos")]
    {
        use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
        builder.with_activation_policy(ActivationPolicy::Accessory);
    }
    let event_loop = builder
        .build()
        .context("Could not start the daemon event loop")?;
    let (to_worker, worker_rx) = mpsc::channel::<Job>();
    let worker_proxy = event_loop.create_proxy();
    let worker_has_tdt = models.tdt.is_some();
    let worker = std::thread::spawn(move || transcribe_worker(worker_rx, worker_proxy, models));
    #[cfg(unix)]
    {
        let socket_proxy = event_loop.create_proxy();
        std::thread::spawn(move || socket::serve_socket(listener, socket_proxy));
    }
    #[cfg(not(unix))]
    dlog!(
        "Socket commands (dictate toggle/start/stop) are not available on Windows yet; use the hotkey."
    );
    let mut app = DaemonApp::new(
        to_worker,
        worker,
        file_settings,
        live_fallback,
        worker_has_tdt,
        switch_progress,
        event_loop.create_proxy(),
    );
    setup_hotkey(&event_loop, hotkey, &mut app);
    event_loop
        .run_app(&mut app)
        .context("The daemon event loop failed")?;
    #[cfg(unix)]
    socket::remove_socket_file(&socket_file);
    Ok(())
}

/// Models owned by the transcription worker. Only the startup mode's model
/// is loaded; the TDT model loads lazily if the tray later switches from
/// live to pauses (the only cross-family switch without a restart).
struct WorkerModels {
    tdt: Option<crate::asr::Asr>,
    live: Option<crate::live::Live>,
}

/// Load the model the startup mode needs. Live mode checks the model files
/// and the speed gate first; either failure logs + notifies and falls back to
/// whole mode for this session. The caller prints Ready once the socket is
/// bound too. A startup speed check records its outcome for the Settings page.
fn load_daemon_models(
    mode: crate::config::Mode,
    progress: &Mutex<SwitchProgress>,
) -> anyhow::Result<(WorkerModels, bool)> {
    if mode != crate::config::Mode::Live {
        return load_tdt_models(false);
    }
    if crate::model::live_model_dir_checked().is_err() {
        let body = "Run \"dictate download --live\" (about 900 MB). Using Whole recording.";
        dlog!("error: dictate: Live model missing: {body}");
        crate::notify::show(
            crate::notify::Kind::Error,
            "dictate: Live model missing",
            body,
        );
        return load_tdt_models(true);
    }
    let dir = crate::model::live_model_dir_checked()?;
    let env = crate::asr::shared_environment()?;
    let threads = crate::asr::intra_threads();
    let mut live = crate::live::Live::load(&dir, &env, threads)?;
    let median = check_live_speed(&mut live, progress);
    if !crate::live::is_fast_enough(median) {
        let body = format!(
            "Needs under 392 ms per 560 ms of audio, took {} ms. Using Whole recording.",
            median.as_millis()
        );
        dlog!("error: dictate: this computer is too slow for Live mode: {body}");
        crate::notify::show(
            crate::notify::Kind::Error,
            "dictate: this computer is too slow for Live mode",
            &body,
        );
        drop(live);
        return load_tdt_models(true);
    }
    Ok((
        WorkerModels {
            tdt: None,
            live: Some(live),
        },
        false,
    ))
}

/// Load the TDT model for whole/pauses mode, or as the Live fallback. The
/// caller prints Ready once the socket is bound too.
fn load_tdt_models(live_fallback: bool) -> anyhow::Result<(WorkerModels, bool)> {
    let dir = crate::model::model_dir_checked()?;
    let asr = crate::asr::Asr::load(&dir)?;
    Ok((
        WorkerModels {
            tdt: Some(asr),
            live: None,
        },
        live_fallback,
    ))
}

/// Register the global hotkey (except on Wayland) and keep its manager alive
/// in the app. Never fatal: the socket commands always keep working.
fn setup_hotkey(
    event_loop: &EventLoop<UserEvent>,
    config: crate::hotkey::HotkeyConfig,
    app: &mut DaemonApp,
) {
    app.hotkey_display = config.display.clone();
    if crate::hotkey::wayland_session() {
        dlog!(
            "Wayland session: global hotkeys are not available to apps. Bind \"dictate toggle\" (or start/stop) in your compositor; see README."
        );
        return;
    }
    let manager = match GlobalHotKeyManager::new().and_then(|manager| {
        manager.register(config.hotkey)?;
        Ok(manager)
    }) {
        Ok(manager) => manager,
        Err(err) => {
            dlog!(
                "Could not register hotkey {}: {err}. Socket commands still work.",
                config.display
            );
            return;
        }
    };
    app.hotkey = Some(config.hotkey);
    app._hotkey_manager = Some(manager);
    let proxy = event_loop.create_proxy();
    GlobalHotKeyEvent::set_event_handler(Some(move |event| {
        if proxy.send_event(UserEvent::HotKey(event)).is_err() {
            // The event loop is closed; the daemon is shutting down.
        }
    }));
    dlog!("Hotkey: {} (tap to toggle, hold to talk)", config.display);
}

/// Daemon state, owned by the event-loop thread: the recording state, the
/// `Recorder`, the hotkey manager, the tray icon and the tap-vs-hold tracking.
struct DaemonApp {
    state: State,
    recorder: Option<crate::mic::Recorder>,
    feeder: Option<FeederSession>,
    /// The shared voice detector, reused across streaming recordings. Moved
    /// into the feeder thread while Recording, handed back when it exits.
    vad: Option<crate::vad::Vad>,
    to_worker: mpsc::Sender<Job>,
    worker: std::thread::JoinHandle<()>,
    hotkey: Option<HotKey>,
    hotkey_display: String,
    press_started: Option<Instant>,
    tap_armed: bool,
    worker_gone_notified: bool,
    settings: crate::config::Settings,
    /// Live was requested but failed the startup checks: live recordings run
    /// whole for this session.
    live_fallback: bool,
    /// The worker holds the TDT model (whole/pauses need no load then).
    worker_has_tdt: bool,
    /// A mode switch loading on a background thread, if any.
    mode_switch: Option<ModeSwitch>,
    tray: Option<crate::tray::Tray>,
    tray_attempted: bool,
    /// The open Settings window, if any. `None` while closed: the window, GL
    /// context and egui state exist only on demand, at zero cost otherwise.
    settings_window: Option<crate::settings_ui::SettingsWindow>,
    /// Audio session: one cpal host plus the cached recording device and
    /// config, so `start` doesn't re-connect to the audio server.
    mic_session: crate::mic::MicSession,
    /// Cached microphone listing for the tray submenu and labels, refreshed
    /// at startup, on `Refresh list`, and when a recording starts.
    mics: Vec<crate::mic::Mic>,
    /// Display name of the microphone the current (or last) recording uses.
    active_mic: Option<String>,
    /// Progress of a background mode switch, shared with the loader thread
    /// and shown inline in the Settings page.
    switch_progress: Arc<Mutex<SwitchProgress>>,
    /// Last shortcut-switch error, shown inline in the Settings page
    /// (cleared on success and when the window opens).
    last_shortcut_error: Option<String>,
    /// Cached start-at-login state for the Settings page (queried when the
    /// window opens and after each toggle, never per frame).
    autostart_cache: Option<bool>,
    /// Last start-at-login error, shown inline in the Settings page.
    autostart_error: Option<String>,
    menu_proxy: EventLoopProxy<UserEvent>,
    /// Held so the registration stays alive; dropping it would unregister.
    /// Never read, hence the underscore.
    _hotkey_manager: Option<GlobalHotKeyManager>,
}

impl DaemonApp {
    fn new(
        to_worker: mpsc::Sender<Job>,
        worker: std::thread::JoinHandle<()>,
        settings: crate::config::Settings,
        live_fallback: bool,
        worker_has_tdt: bool,
        switch_progress: Arc<Mutex<SwitchProgress>>,
        menu_proxy: EventLoopProxy<UserEvent>,
    ) -> Self {
        Self {
            state: State::Idle,
            recorder: None,
            feeder: None,
            vad: None,
            to_worker,
            worker,
            hotkey: None,
            hotkey_display: String::new(),
            press_started: None,
            tap_armed: false,
            worker_gone_notified: false,
            settings,
            live_fallback,
            worker_has_tdt,
            mode_switch: None,
            tray: None,
            tray_attempted: false,
            settings_window: None,
            mic_session: crate::mic::MicSession::new(),
            mics: Vec::new(),
            active_mic: None,
            switch_progress,
            last_shortcut_error: None,
            autostart_cache: None,
            autostart_error: None,
            menu_proxy,
            _hotkey_manager: None,
        }
    }

    /// Dictation mode for the next recording: live maps to whole when the
    /// startup checks failed it for this session.
    fn current_mode(&self) -> crate::config::Mode {
        let mode = dictation_mode();
        if mode == crate::config::Mode::Live && self.live_fallback {
            crate::config::Mode::Whole
        } else {
            mode
        }
    }

    /// Run one command through the state machine and return the one-line
    /// reply. A dead worker is logged here and turns every reply into `err`.
    fn handle_command(&mut self, cmd: Command) -> String {
        if self.worker.is_finished() {
            dlog!("error: The transcription worker thread ended unexpectedly");
            if !self.worker_gone_notified {
                self.worker_gone_notified = true;
                crate::notify::show(
                    crate::notify::Kind::Error,
                    "dictate: transcription worker stopped",
                    "Restart \"dictate daemon\".",
                );
            }
            // A dead worker can't ack a pending swap; don't stay Loading.
            if self.mode_switch.take().is_some() {
                self.sync_mode_checks();
                self.sync_tray();
            }
            return "err transcription worker stopped; restart \"dictate daemon\"".to_string();
        }
        // A switch only starts from Idle, so every start here is a fresh
        // recording the worker has no model for yet.
        if self.mode_switch.is_some() && matches!(cmd, Command::Start | Command::Toggle) {
            return self.model_loading_refused();
        }
        let (new_state, action, ok_reply) = next(self.state, cmd);
        let reply = self.run_action(action, new_state, ok_reply);
        if self.state != State::Recording {
            self.tap_armed = false;
        }
        self.sync_tray();
        self.refresh_settings();
        reply
    }

    /// Map a hotkey press/release to a command and run it. A hotkey has no
    /// requester, so the reply is dropped; genuine failures are already
    /// logged by [`DaemonApp::run_action`].
    fn handle_hotkey(&mut self, event: GlobalHotKeyEvent) {
        if self.hotkey.map(|hotkey| hotkey.id()) != Some(event.id) {
            return;
        }
        match event.state {
            HotKeyState::Pressed => {
                self.press_started = Some(Instant::now());
                if let Some(cmd) = crate::hotkey::hotkey_action(
                    self.state,
                    HotKeyState::Pressed,
                    0,
                    self.tap_armed,
                ) {
                    self.tap_armed = false;
                    drop(self.handle_command(cmd));
                }
            }
            HotKeyState::Released => {
                let held_ms = self
                    .press_started
                    .map(|started| started.elapsed().as_millis() as u64)
                    .unwrap_or(0);
                self.press_started = None;
                if let Some(cmd) = crate::hotkey::hotkey_action(
                    self.state,
                    HotKeyState::Released,
                    held_ms,
                    self.tap_armed,
                ) {
                    self.tap_armed = false;
                    drop(self.handle_command(cmd));
                } else if self.state == State::Recording
                    && held_ms < crate::hotkey::HOLD_THRESHOLD_MS
                {
                    // Short tap while recording: keep recording; the next
                    // press stops it.
                    self.tap_armed = true;
                }
            }
        }
    }

    /// Microphone name for the tray label and tooltip: the recording's mic
    /// while busy, else the selected mic (or the default when nothing is
    /// selected or the selection is gone).
    fn tray_mic(&self) -> Option<String> {
        match self.state {
            State::Recording | State::Transcribing => self.active_mic.clone(),
            State::Idle => selected_mic_name(&self.settings.microphone, &self.mics),
        }
    }

    /// Refresh the tray icon and menu after a state or settings change
    /// (no tray: no-op).
    fn sync_tray(&self) {
        if let Some(tray) = &self.tray
            && let Err(err) = tray.sync_state(
                self.state,
                self.settings.mode,
                self.mode_switch.is_some(),
                self.tray_mic().as_deref(),
            )
        {
            dlog!("error: {err:#}");
        }
    }

    /// Re-list the microphones into the cache and rebuild the tray submenu.
    /// A listing failure is logged and keeps the old entries. When the
    /// listing changed, the cached recording device is dropped so the next
    /// `start` re-resolves (a removed mic falls back, a returned one is
    /// used again); an unchanged listing keeps the cache, which is what
    /// keeps `start` fast.
    fn refresh_microphones(&mut self) {
        match self.mic_session.list_microphones() {
            Ok(mics) => {
                if mic_listing_changed(&self.mics, &mics) {
                    self.mic_session.invalidate();
                }
                self.mics = mics;
                if let Some(tray) = self.tray.as_mut()
                    && let Err(err) =
                        tray.rebuild_microphones(&self.mics, &self.settings.microphone)
                {
                    dlog!("error: {err:#}");
                }
            }
            Err(err) => dlog!("error: {err:#}"),
        }
        self.sync_tray();
    }

    /// Switch the selected microphone from the tray menu: save + confirm.
    /// Refused while recording (the recorder already holds its device).
    fn switch_microphone(&mut self, id: &str) {
        if self.state == State::Recording {
            dlog!("Microphone switch refused: finish the current dictation first");
            crate::notify::show(
                crate::notify::Kind::Info,
                "dictate",
                "Finish the current dictation first",
            );
            return;
        }
        if id == self.settings.microphone {
            self.sync_microphone_checks();
            return;
        }
        let name = if id.is_empty() {
            crate::tray::system_default_label(default_mic_name(&self.mics).as_deref())
        } else {
            self.mics
                .iter()
                .find(|mic| mic.id == id)
                .map(|mic| mic.name.clone())
                .unwrap_or_else(|| id.to_string())
        };
        self.settings.microphone = id.to_string();
        self.mic_session.invalidate();
        self.report_save_error(crate::config::save(&self.settings));
        self.sync_microphone_checks();
        Self::confirm_settings_change(
            &crate::tray::SettingsChange::Microphone {
                id: id.to_string(),
                name,
            },
            true,
        );
        self.sync_tray();
    }

    /// Check exactly the selected microphone in the tray menu.
    fn sync_microphone_checks(&self) {
        if let Some(tray) = &self.tray {
            tray.sync_microphone_checks(&self.settings.microphone);
        }
    }

    /// Run one tray menu click.
    fn handle_menu(&mut self, event: tray_icon::menu::MenuEvent, event_loop: &ActiveEventLoop) {
        let action = self
            .tray
            .as_ref()
            .and_then(|tray| tray.match_event(event.id()));
        match action {
            None => {}
            Some(crate::tray::TrayAction::Toggle) => {
                drop(self.handle_command(Command::Toggle));
            }
            Some(crate::tray::TrayAction::Cancel) => {
                drop(self.handle_command(Command::Cancel));
            }
            Some(crate::tray::TrayAction::Sounds) => self.flip_sounds(),
            Some(crate::tray::TrayAction::Notifications) => self.flip_notifications(),
            Some(crate::tray::TrayAction::Microphone(id)) => self.switch_microphone(&id),
            Some(crate::tray::TrayAction::RefreshMics) => self.refresh_microphones(),
            Some(crate::tray::TrayAction::Mode(target)) => self.switch_mode(target),
            Some(crate::tray::TrayAction::Shortcut(display)) => self.switch_hotkey(&display),
            Some(crate::tray::TrayAction::HowTo) => self.open_how_to(),
            Some(crate::tray::TrayAction::OpenConfig) => self.open_settings_file(),
            Some(crate::tray::TrayAction::Settings) => self.open_settings(event_loop),
            Some(crate::tray::TrayAction::Quit) => event_loop.exit(),
        }
        self.refresh_settings();
    }

    /// Flip the `Sound cues` check, update the effective switch, save.
    fn flip_sounds(&mut self) {
        let new = !self.settings.sounds;
        self.settings.sounds = new;
        crate::config::refresh_sounds(new);
        if let Some(tray) = &self.tray {
            tray.set_sounds_checked(new);
        }
        self.report_save_error(crate::config::save(&self.settings));
        Self::confirm_settings_change(&crate::tray::SettingsChange::Sounds(new), false);
        self.sync_tray();
    }

    /// Flip the `Notifications` check, update the effective switch, save.
    fn flip_notifications(&mut self) {
        let new = !self.settings.notifications;
        self.settings.notifications = new;
        crate::config::refresh_notifications(new);
        if let Some(tray) = &self.tray {
            tray.set_notifications_checked(new);
        }
        self.report_save_error(crate::config::save(&self.settings));
        // Turning notifications off is confirmed even though they are now
        // off: the notification is the confirmation of that action.
        Self::confirm_settings_change(&crate::tray::SettingsChange::Notifications(new), !new);
        self.sync_tray();
    }

    /// Switch the dictation mode from the tray Mode submenu. Whole and pauses
    /// share the TDT model, so switching between them with it loaded is
    /// instant; anything else loads the needed model on a background thread
    /// (Live always benchmarks too) and completes on `ModeSwapped`. Picking
    /// Live with its model missing downloads it first, on the same thread.
    /// Refused while recording or transcribing, and while a switch is already
    /// running (only one download at a time).
    fn switch_mode(&mut self, target: crate::config::Mode) {
        if self.state != State::Idle {
            dlog!(
                "Mode switch to {} refused: finish the current dictation first",
                target.as_str()
            );
            crate::notify::show(
                crate::notify::Kind::Info,
                "dictate",
                "Finish the current dictation first",
            );
            self.sync_mode_checks();
            return;
        }
        if let Some(switch) = &self.mode_switch {
            let message = match switch {
                ModeSwitch::Downloading => "Already downloading",
                ModeSwitch::Loading { .. } => "Model loading…",
            };
            dlog!("Mode switch to {} refused: {message}", target.as_str());
            crate::notify::show(crate::notify::Kind::Info, "dictate", message);
            self.sync_mode_checks();
            return;
        }
        if target == self.settings.mode {
            self.sync_mode_checks();
            return;
        }
        if target != crate::config::Mode::Live && self.worker_has_tdt {
            self.complete_mode_switch(target);
            return;
        }
        let download_first =
            target == crate::config::Mode::Live && crate::model::live_model_dir_checked().is_err();
        self.mode_switch = Some(if download_first {
            ModeSwitch::Downloading
        } else {
            ModeSwitch::Loading { target }
        });
        self.sync_tray();
        let previous = self.settings.mode;
        let to_worker = self.to_worker.clone();
        let proxy = self.menu_proxy.clone();
        let progress = Arc::clone(&self.switch_progress);
        std::thread::spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_mode_switch_loader(
                    target,
                    previous,
                    download_first,
                    &to_worker,
                    &proxy,
                    &progress,
                );
            }));
            if outcome.is_err() {
                let message = "the model loader thread panicked";
                dlog!("error: {message}");
                crate::notify::show(
                    crate::notify::Kind::Error,
                    "dictate: could not switch mode",
                    message,
                );
                send_switch_failed(&proxy);
            }
        });
    }

    /// Finish a mode switch once the worker holds the new model: update the
    /// settings, save, and confirm. The previous mode is untouched on every
    /// failure path, so reaching here means the switch landed.
    fn complete_mode_switch(&mut self, target: crate::config::Mode) {
        self.settings.mode = target;
        crate::config::refresh_mode(target);
        self.worker_has_tdt = target != crate::config::Mode::Live;
        self.live_fallback = false;
        self.mode_switch = None;
        self.report_save_error(crate::config::save(&self.settings));
        Self::confirm_settings_change(&crate::tray::SettingsChange::Mode(target), true);
        self.sync_mode_checks();
        self.sync_tray();
    }

    /// Check exactly the current file mode in the Mode submenu.
    fn sync_mode_checks(&self) {
        if let Some(tray) = &self.tray {
            tray.sync_mode_checks(self.settings.mode);
        }
    }

    /// Log + notify a start request refused while a mode switch loads, and
    /// build its `err` reply.
    fn model_loading_refused(&self) -> String {
        let message = "The speech model is still loading.";
        dlog!("error: {message}");
        crate::notify::show(crate::notify::Kind::Error, "dictate", message);
        "err model loading".to_string()
    }

    /// Log a settings change and confirm it with an Info notification, so a
    /// menu that closes on click still gives visible feedback. `force`
    /// bypasses the notification switch: the "Notifications: off" toggle and
    /// mode switches (`Mode: <name>`, always shown) use it.
    fn confirm_settings_change(change: &crate::tray::SettingsChange, force: bool) {
        dlog!("{}", change.log_line());
        let body = change.notification_body();
        if force {
            crate::notify::show_forced(crate::notify::Kind::Info, "dictate", &body);
        } else {
            crate::notify::show(crate::notify::Kind::Info, "dictate", &body);
        }
    }

    /// Log and notify a settings-save failure (the in-memory value stays).
    fn report_save_error(&self, result: anyhow::Result<()>) {
        if let Err(err) = result {
            dlog!("error: {err:#}");
            crate::notify::show(
                crate::notify::Kind::Error,
                "dictate: could not save settings",
                &format!("{err:#}"),
            );
        }
    }

    /// Switch the global shortcut to `display`: register the new one first
    /// (so a failure keeps the old one working), then unregister the old one
    /// and save. Failures notify and keep the old shortcut checked.
    fn switch_hotkey(&mut self, display: &str) {
        if display == self.hotkey_display {
            self.last_shortcut_error = None;
            self.sync_shortcut_checks();
            return;
        }
        let new_key = match HotKey::from_str(display) {
            Ok(key) => key,
            Err(err) => {
                self.shortcut_failed(display, &format!("{err}"));
                return;
            }
        };
        if self._hotkey_manager.is_none() {
            match GlobalHotKeyManager::new() {
                Ok(manager) => self._hotkey_manager = Some(manager),
                Err(err) => {
                    self.shortcut_failed(display, &format!("{err}"));
                    return;
                }
            }
        }
        let Some(manager) = &self._hotkey_manager else {
            self.shortcut_failed(display, "the hotkey manager is gone");
            return;
        };
        if let Err(err) = manager.register(new_key) {
            self.shortcut_failed(display, &format!("{err}"));
            return;
        }
        if let Some(old) = self.hotkey
            && let Err(err) = manager.unregister(old)
        {
            dlog!(
                "error: Could not unregister shortcut {}: {err}",
                self.hotkey_display
            );
        }
        self.hotkey = Some(new_key);
        self.hotkey_display = display.to_string();
        self.press_started = None;
        self.tap_armed = false;
        self.last_shortcut_error = None;
        self.settings.hotkey = display.to_string();
        self.report_save_error(crate::config::save(&self.settings));
        self.sync_shortcut_checks();
        dlog!("Hotkey: {display} (tap to toggle, hold to talk)");
        Self::confirm_settings_change(
            &crate::tray::SettingsChange::Shortcut(display.to_string()),
            false,
        );
        self.sync_tray();
    }

    /// Record a failed shortcut switch for the Settings page's inline error,
    /// notify it, and re-check the current shortcut.
    fn shortcut_failed(&mut self, display: &str, err: &str) {
        self.last_shortcut_error = Some(err.to_string());
        crate::notify::show(
            crate::notify::Kind::Error,
            &format!("dictate: could not use shortcut {display}"),
            err,
        );
        self.sync_shortcut_checks();
    }

    /// Check exactly the current shortcut in the tray menu.
    fn sync_shortcut_checks(&self) {
        if let Some(tray) = &self.tray {
            tray.sync_shortcut_checks(&self.hotkey_display);
        }
    }

    /// Open the README keybind section in the default browser.
    fn open_how_to(&self) {
        if let Err(err) = crate::tray::open_target(crate::tray::README_USE_IT_URL) {
            let message = format!("{err:#}");
            dlog!("error: {message}");
            crate::notify::show(crate::notify::Kind::Error, "dictate", &message);
        }
    }

    /// Create the settings file (with the current settings, which are the
    /// defaults when the file is missing) and open it in the default app.
    fn open_settings_file(&self) {
        let Some(path) = crate::config::config_path() else {
            let message = "Could not find a home directory for this user";
            dlog!("error: {message}");
            crate::notify::show(crate::notify::Kind::Error, "dictate", message);
            return;
        };
        if !path.is_file()
            && let Err(err) = crate::config::save(&self.settings)
        {
            let message = format!("Could not create {}: {err:#}", path.display());
            dlog!("error: {message}");
            crate::notify::show(crate::notify::Kind::Error, "dictate", &message);
            return;
        }
        if let Err(err) = crate::tray::open_target(&path.to_string_lossy()) {
            let message = format!("{err:#}");
            dlog!("error: {message}");
            crate::notify::show(crate::notify::Kind::Error, "dictate", &message);
        }
    }

    /// Open the Settings window, or focus it when it is already open. A GL
    /// init failure is logged + notified; the daemon keeps running.
    fn open_settings(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.settings_window {
            window.focus();
            return;
        }
        // Fresh per-open page state: stale errors go, and start-at-login is
        // queried once (never per frame).
        self.last_shortcut_error = None;
        match crate::autostart::status() {
            Ok(status) => {
                self.autostart_cache = Some(status.enabled);
                self.autostart_error = None;
            }
            Err(err) => {
                let message = format!("{err:#}");
                dlog!("error: {message}");
                self.autostart_cache = None;
                self.autostart_error = Some(message);
            }
        }
        match crate::settings_ui::SettingsWindow::open(event_loop) {
            Ok(window) => {
                // Focus the new window too: on macOS the app is Accessory and
                // would otherwise stay behind other windows.
                window.focus();
                window.request_redraw();
                self.settings_window = Some(window);
            }
            Err(err) => {
                let message = format!("{err:#}");
                dlog!("error: dictate: could not open Settings: {message}");
                crate::notify::show(
                    crate::notify::Kind::Error,
                    "dictate: could not open Settings",
                    &message,
                );
            }
        }
    }

    /// Paint one Settings frame. The window is taken out while egui runs so
    /// the page can read daemon state; the page's intents apply after the
    /// frame through the tray's code paths. A frame failure drops the window
    /// (already out), and the daemon keeps running.
    fn redraw_settings(&mut self, event_loop: &ActiveEventLoop) {
        let Some(mut window) = self.settings_window.take() else {
            return;
        };
        let progress = lock_progress(&self.switch_progress).clone();
        let view = crate::settings_ui::page::PageView {
            settings: &self.settings,
            mics: &self.mics,
            active_mic: self.active_mic.as_deref(),
            switch_target: self.mode_switch.as_ref().map(ModeSwitch::target),
            downloading: matches!(self.mode_switch, Some(ModeSwitch::Downloading)),
            download_percent: progress.percent,
            checking_speed: progress.checking_speed,
            last_speed_check: progress.last_check,
            hotkey_display: &self.hotkey_display,
            wayland: crate::hotkey::wayland_session(),
            autostart_on: self.autostart_cache,
            autostart_error: self.autostart_error.as_deref(),
            shortcut_error: self.last_shortcut_error.as_deref(),
            version: env!("CARGO_PKG_VERSION"),
        };
        let mut actions = Vec::new();
        let outcome = window.redraw(event_loop, |ui| {
            actions = crate::settings_ui::page::show_page(ui, &view);
        });
        match outcome {
            Ok(delay) => {
                let changed = !actions.is_empty();
                for action in actions {
                    self.apply_page_action(action);
                }
                if changed || delay.is_zero() {
                    // Changed settings repaint at once; an animation or a
                    // follow-up layout repaints without leaving Wait.
                    window.request_redraw();
                } else if delay != Duration::MAX {
                    // A delayed repaint (a tooltip or blink): sleep until then.
                    event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + delay));
                }
                self.settings_window = Some(window);
            }
            Err(err) => {
                let message = format!("{err:#}");
                dlog!("error: dictate: Settings window failed: {message}");
                crate::notify::show(
                    crate::notify::Kind::Error,
                    "dictate: Settings window failed",
                    &message,
                );
                event_loop.set_control_flow(ControlFlow::Wait);
            }
        }
    }

    /// Repaint the Settings window if it is open (daemon state changed under
    /// it). Closed: no-op.
    fn refresh_settings(&self) {
        if let Some(window) = &self.settings_window {
            window.request_redraw();
        }
    }

    /// Apply one Settings-page intent through the same code path the tray menu
    /// uses for the same change.
    fn apply_page_action(&mut self, action: crate::settings_ui::page::PageAction) {
        use crate::settings_ui::page::PageAction;
        match action {
            PageAction::SelectMic(id) => self.switch_microphone(&id),
            PageAction::RefreshMics => self.refresh_microphones(),
            PageAction::SelectMode(mode) => self.switch_mode(mode),
            PageAction::SelectShortcut(display) => self.switch_hotkey(&display),
            PageAction::ToggleSounds => self.flip_sounds(),
            PageAction::ToggleNotifications => self.flip_notifications(),
            PageAction::SetAutostart(on) => self.apply_autostart(on),
            PageAction::OpenLog => self.open_daemon_log(),
            PageAction::OpenReleases => self.open_releases(),
        }
    }

    /// Turn start-at-login on or off from the Settings page, caching the
    /// outcome for the checkbox and the inline error.
    fn apply_autostart(&mut self, on: bool) {
        let outcome = if on {
            crate::autostart::enable()
        } else {
            crate::autostart::disable()
        }
        .and_then(|()| crate::autostart::status());
        match outcome {
            Ok(status) => {
                self.autostart_cache = Some(status.enabled);
                self.autostart_error = None;
                dlog!(
                    "Start at login: {}",
                    if status.enabled { "on" } else { "off" }
                );
            }
            Err(err) => {
                let message = format!("{err:#}");
                dlog!("error: {message}");
                self.autostart_error = Some(message);
                // Show the truth, whatever the toggle achieved.
                self.autostart_cache = crate::autostart::status().ok().map(|status| status.enabled);
            }
        }
    }

    /// Open the daemon log file in the default app.
    fn open_daemon_log(&self) {
        let result =
            daemon_log_path().and_then(|path| crate::tray::open_target(&path.to_string_lossy()));
        if let Err(err) = result {
            let message = format!("{err:#}");
            dlog!("error: {message}");
            crate::notify::show(crate::notify::Kind::Error, "dictate", &message);
        }
    }

    /// Open the GitHub releases page in the default browser.
    fn open_releases(&self) {
        if let Err(err) = crate::tray::open_target(crate::settings_ui::page::RELEASES_URL) {
            let message = format!("{err:#}");
            dlog!("error: {message}");
            crate::notify::show(crate::notify::Kind::Error, "dictate", &message);
        }
    }

    /// Run the state-machine action, updating `state`, and return the reply.
    /// I/O failures are logged and turn the reply into `err ...`.
    fn run_action(&mut self, action: Action, new_state: State, ok_reply: &str) -> String {
        match action {
            Action::None => {
                self.state = new_state;
                ok_reply.to_string()
            }
            Action::StartRecording => self.start_recording(new_state, ok_reply),
            Action::StopRecording => self.stop_recording(new_state, ok_reply),
            Action::CancelRecording => self.cancel_recording(new_state, ok_reply),
        }
    }

    /// Start the mic, plus a feeder thread in pauses/live mode. Pauses mode
    /// without a voice detector falls back to whole for the recording (logged
    /// + notified, not silent).
    fn start_recording(&mut self, new_state: State, ok_reply: &str) -> String {
        self.refresh_microphones();
        let mic = crate::config::resolve_microphone(
            &self.settings.microphone,
            std::env::var("DICTATE_MICROPHONE").ok().as_deref(),
        );
        match self.mic_session.start(mic.as_deref()) {
            Ok(started) => {
                self.active_mic = Some(started.mic_name().to_string());
                match self.current_mode() {
                    crate::config::Mode::Pauses => self.start_pauses_feeder(&started),
                    crate::config::Mode::Live => self.start_live_feeder(&started),
                    crate::config::Mode::Whole => {}
                }
                self.recorder = Some(started);
                self.state = new_state;
                crate::cue::play_start();
                ok_reply.to_string()
            }
            Err(err) => {
                dlog!("error: {err:#}");
                crate::notify::show(
                    crate::notify::Kind::Error,
                    "dictate: recording failed",
                    &format!("{err:#}"),
                );
                format!("err {err:#}")
            }
        }
    }

    /// Spawn the pauses feeder (VAD segmentation) for a recording. Without a
    /// voice detector this recording runs whole (already logged + notified).
    fn start_pauses_feeder(&mut self, started: &crate::mic::Recorder) {
        let Some(mut vad) = self.take_streaming_vad() else {
            return;
        };
        vad.reset();
        let buffer = started.buffer_handle();
        let sample_rate = started.sample_rate();
        let (control_tx, control_rx) = mpsc::channel::<FeederCmd>();
        let to_worker = self.to_worker.clone();
        let thread = std::thread::spawn(move || {
            feeder_loop(vad, buffer, sample_rate, control_rx, to_worker)
        });
        self.feeder = Some(FeederSession::Pauses {
            control_tx,
            thread: Some(thread),
        });
    }

    /// Spawn the live feeder (16 kHz audio, no VAD) for a recording.
    fn start_live_feeder(&mut self, started: &crate::mic::Recorder) {
        let buffer = started.buffer_handle();
        let sample_rate = started.sample_rate();
        let (control_tx, control_rx) = mpsc::channel::<FeederCmd>();
        let to_worker = self.to_worker.clone();
        let thread = std::thread::spawn(move || {
            live_feeder_loop(buffer, sample_rate, control_rx, to_worker)
        });
        self.feeder = Some(FeederSession::Live {
            control_tx,
            thread: Some(thread),
        });
    }

    /// A voice detector for a streaming recording, reusing the daemon's Vad
    /// across recordings (loaded lazily on the first one). `None` means
    /// whole-recording mode for this recording: the missing or broken file
    /// was already logged + notified, so the fallback is explicit.
    fn take_streaming_vad(&mut self) -> Option<crate::vad::Vad> {
        if let Some(vad) = self.vad.take() {
            return Some(vad);
        }
        if crate::model::vad_model_path_checked().is_err() {
            let body = "Run \"dictate download\" to enable Type while speaking. Using whole-recording mode.";
            dlog!("error: dictate: voice detector missing: {body}");
            crate::notify::show(
                crate::notify::Kind::Error,
                "dictate: voice detector missing",
                body,
            );
            return None;
        }
        match crate::vad::Vad::load_default() {
            Ok(vad) => Some(vad),
            Err(err) => {
                let body = format!("{err:#}. Using whole-recording mode.");
                dlog!("error: dictate: voice detector failed: {body}");
                crate::notify::show(
                    crate::notify::Kind::Error,
                    "dictate: voice detector failed",
                    &body,
                );
                None
            }
        }
    }

    /// Stop the mic. With a feeder (pauses/live): it sends the tail plus
    /// `End`/`LiveEnd` and the state stays Transcribing until the worker is
    /// done. Whole mode: today's exact path (one `Full` job at stop).
    fn stop_recording(&mut self, new_state: State, ok_reply: &str) -> String {
        if self.feeder.is_some() {
            return self.stop_streaming(new_state, ok_reply);
        }
        match self.recorder.take() {
            Some(started) => match started.stop() {
                Ok(samples) => match self.to_worker.send(Job::Full(samples)) {
                    Ok(()) => {
                        self.state = new_state;
                        crate::cue::play_stop();
                        ok_reply.to_string()
                    }
                    Err(err) => {
                        self.state = State::Idle;
                        self.recording_failed(format!(
                            "Could not queue audio for transcription: {err}"
                        ))
                    }
                },
                Err(err) => {
                    self.state = State::Idle;
                    self.recording_failed(format!("{err:#}"))
                }
            },
            None => {
                self.state = State::Idle;
                self.recording_failed(
                    "the daemon was recording but its recorder is gone".to_string(),
                );
                "err not recording".to_string()
            }
        }
    }

    /// Stop a pauses/live recording via the feeder thread.
    fn stop_streaming(&mut self, new_state: State, ok_reply: &str) -> String {
        // A mic failure discards the recording instead of finishing it.
        if let Some(recorder) = &self.recorder
            && let Some(text) = recorder.stream_error()
        {
            self.finish_feeder(FeederCmd::Cancel);
            self.recorder = None;
            self.state = State::Idle;
            return self.recording_failed(format!("Microphone stream error: {text}"));
        }
        let Some(recorder) = self.recorder.take() else {
            self.finish_feeder(FeederCmd::Cancel);
            self.state = State::Idle;
            self.recording_failed("the daemon was recording but its recorder is gone".to_string());
            return "err not recording".to_string();
        };
        // Let the device deliver its buffered tail before the stream is
        // dropped (the feeder drains the shared buffer next).
        recorder.drain();
        drop(recorder);
        self.finish_feeder(FeederCmd::Stop);
        self.state = new_state;
        crate::cue::play_stop();
        ok_reply.to_string()
    }

    /// Cancel the recording. Streaming on: the feeder sends `Cancel` and the
    /// worker discards queued segments (already-typed text stays).
    fn cancel_recording(&mut self, new_state: State, ok_reply: &str) -> String {
        self.finish_feeder(FeederCmd::Cancel);
        self.recorder = None;
        self.state = new_state;
        ok_reply.to_string()
    }

    /// Signal the feeder thread (`Stop`/`Cancel`) and wait for it to exit,
    /// taking the Vad back for the next pauses recording. No feeder: no-op.
    /// A dead feeder unblocks the worker with `Cancel`.
    fn finish_feeder(&mut self, cmd: FeederCmd) {
        let Some(session) = self.feeder.take() else {
            return;
        };
        match session {
            FeederSession::Pauses {
                control_tx,
                mut thread,
            } => {
                let signalled = control_tx.send(cmd).is_ok();
                if let Some(handle) = thread.take() {
                    match handle.join() {
                        Ok(vad) => self.vad = vad,
                        Err(_) => dlog!("error: the streaming feeder thread panicked"),
                    }
                }
                if !signalled && self.to_worker.send(Job::Cancel).is_err() {
                    // The worker is gone; `handle_command` reports that itself.
                }
            }
            FeederSession::Live {
                control_tx,
                mut thread,
            } => {
                let signalled = control_tx.send(cmd).is_ok();
                if let Some(handle) = thread.take()
                    && handle.join().is_err()
                {
                    dlog!("error: the live feeder thread panicked");
                }
                if !signalled && self.to_worker.send(Job::Cancel).is_err() {
                    // The worker is gone; `handle_command` reports that itself.
                }
            }
        }
    }

    /// Log + notify a recording failure and build its `err` reply.
    /// The caller sets the state; the message is returned for the reply.
    fn recording_failed(&self, message: String) -> String {
        dlog!("error: {message}");
        crate::notify::show(
            crate::notify::Kind::Error,
            "dictate: recording failed",
            &message,
        );
        format!("err {message}")
    }
}

impl ApplicationHandler<UserEvent> for DaemonApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // No windows, no polling: sleep until a socket, worker, hotkey or
        // menu event arrives.
        event_loop.set_control_flow(ControlFlow::Wait);
        // The tray must be created on the main thread after the loop
        // started; without a tray host the daemon keeps running headless.
        if !self.tray_attempted {
            self.tray_attempted = true;
            self.refresh_microphones();
            match crate::tray::build_tray(
                &self.settings,
                self.state,
                &self.mics,
                self.menu_proxy.clone(),
            ) {
                Ok(tray) => {
                    self.tray = Some(tray);
                    self.sync_tray();
                }
                Err(err) => {
                    dlog!("error: {err:#}");
                    crate::notify::show(
                        crate::notify::Kind::Error,
                        "dictate: no tray icon",
                        &format!("{err:#}"),
                    );
                }
            }
        }
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        // An egui delayed repaint came due: repaint once, back on plain Wait
        // (the frame re-arms a deadline only while egui still asks for one).
        if matches!(cause, StartCause::ResumeTimeReached { .. })
            && let Some(window) = &self.settings_window
        {
            event_loop.set_control_flow(ControlFlow::Wait);
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        // The Settings window is the daemon's only window; anything else is
        // not ours.
        if self
            .settings_window
            .as_ref()
            .is_none_or(|window| window.window_id() != window_id)
        {
            return;
        }
        match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                // Drop everything: window, GL context and egui state.
                self.settings_window = None;
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            WindowEvent::Resized(size) => {
                if let Some(window) = self.settings_window.as_mut() {
                    window.resize(size);
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.redraw_settings(event_loop),
            _ => {
                if let Some(window) = self.settings_window.as_mut()
                    && window.on_window_event(&event)
                {
                    window.request_redraw();
                }
            }
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            #[cfg(unix)]
            UserEvent::Command(cmd, reply_tx) => {
                let reply = self.handle_command(cmd);
                if reply_tx.send(reply).is_err() {
                    dlog!("error: Could not reply to a daemon client: the connection closed");
                }
            }
            UserEvent::WorkerDone => {
                self.state = State::Idle;
                self.tap_armed = false;
                self.sync_tray();
                self.refresh_settings();
            }
            UserEvent::ModeSwapped => {
                // One switch at a time; a stale ack with nothing pending is
                // ignored.
                if let Some(target) = self.mode_switch.as_ref().map(ModeSwitch::target) {
                    self.complete_mode_switch(target);
                }
                self.refresh_settings();
            }
            UserEvent::ModeSwitchFailed => {
                if self.mode_switch.take().is_some() {
                    self.sync_mode_checks();
                    self.sync_tray();
                }
                self.refresh_settings();
            }
            UserEvent::HotKey(event) => self.handle_hotkey(event),
            UserEvent::Menu(event) => self.handle_menu(event, event_loop),
            UserEvent::OpenSettings => self.open_settings(event_loop),
            UserEvent::SettingsRefresh => self.refresh_settings(),
        }
    }
}

/// Milestone percentages notified during a background Live download.
const DOWNLOAD_MILESTONES: [u64; 3] = [25, 50, 75];

/// How many of the 25/50/75 milestones `percent` has reached.
fn milestones_reached(percent: u64) -> usize {
    DOWNLOAD_MILESTONES
        .iter()
        .filter(|milestone| percent >= **milestone)
        .count()
}

/// Body of a Live-download progress notification.
fn live_download_body(percent: u64) -> String {
    format!("Downloading Live model… {percent}%")
}

/// Background model load for a menu mode switch: optionally downloads the
/// Live model first (with progress notifications), loads the target's model
/// (Live always benchmarks too) and hands it to the transcription worker for
/// the swap. Exactly one terminal outcome: a `Job::Swap*` sent to the worker
/// (which acks `UserEvent::ModeSwapped`), or a `UserEvent::ModeSwitchFailed`
/// after logging + notifying the specific failure. The previous mode is kept
/// on every failure; nothing is saved.
fn run_mode_switch_loader(
    target: crate::config::Mode,
    previous: crate::config::Mode,
    download_first: bool,
    to_worker: &mpsc::Sender<Job>,
    proxy: &EventLoopProxy<UserEvent>,
    progress: &Mutex<SwitchProgress>,
) {
    if download_first && download_live_for_switch(proxy, progress).is_err() {
        return;
    }
    let swap = match load_switch_model(target, previous, proxy, progress) {
        Ok(swap) => swap,
        Err(()) => {
            send_switch_failed(proxy);
            return;
        }
    };
    if to_worker.send(swap).is_err() {
        // The worker is gone; `handle_command` reports that itself. Fail the
        // switch so the next command's report isn't stuck behind Loading.
        switch_load_failed(anyhow::anyhow!("the transcription worker stopped"));
        send_switch_failed(proxy);
    }
    // Else the worker acks `ModeSwapped` once the swap lands.
}

/// Download the Live model for a menu switch, notifying at 25/50/75% (each
/// milestone replaces the previous one). Milestones already passed by
/// present files don't notify: only this download's progress does. A failure
/// is logged + notified with the real error and fails the switch.
fn download_live_for_switch(
    proxy: &EventLoopProxy<UserEvent>,
    progress: &Mutex<SwitchProgress>,
) -> Result<(), ()> {
    lock_progress(progress).percent = 0;
    let mut sent: Option<usize> = None;
    let result = crate::model::download_live_files_with_progress(|downloaded, total| {
        let percent = downloaded * 100 / total.max(1);
        lock_progress(progress).percent = percent.min(100);
        let reached = milestones_reached(percent);
        // Seed from the first report, so bytes already on disk (present
        // files count up front) don't fire stale milestones.
        let sent = sent.get_or_insert(reached);
        let before = *sent;
        while *sent < reached {
            let milestone = DOWNLOAD_MILESTONES[*sent];
            *sent += 1;
            crate::notify::show_progress(&live_download_body(milestone));
        }
        if *sent != before {
            send_settings_refresh(proxy);
        }
    });
    match result {
        Ok(()) => {
            lock_progress(progress).percent = 100;
            send_settings_refresh(proxy);
            Ok(())
        }
        Err(err) => {
            let message = format!("{err:#}");
            dlog!("error: dictate: Live download failed: {message}");
            crate::notify::show(
                crate::notify::Kind::Error,
                "dictate: Live download failed",
                &message,
            );
            send_switch_failed(proxy);
            Err(())
        }
    }
}

/// Load the target's model for a mode switch, as the swap job to send. Live
/// always runs the speed gate; failures are already logged + notified.
fn load_switch_model(
    target: crate::config::Mode,
    previous: crate::config::Mode,
    proxy: &EventLoopProxy<UserEvent>,
    progress: &Mutex<SwitchProgress>,
) -> Result<Job, ()> {
    match target {
        crate::config::Mode::Live => {
            load_live_for_switch(previous, proxy, progress).map(Job::SwapLive)
        }
        crate::config::Mode::Whole | crate::config::Mode::Pauses => {
            load_tdt_for_switch().map(Job::SwapTdt)
        }
    }
}

/// Load the TDT model for a switch to whole/pauses mode. Failures are logged
/// + notified with the real error (a missing model names `dictate download`).
fn load_tdt_for_switch() -> Result<crate::asr::Asr, ()> {
    let loaded = crate::model::model_dir_checked().and_then(|dir| crate::asr::Asr::load(&dir));
    match loaded {
        Ok(asr) => Ok(asr),
        Err(err) => {
            switch_load_failed(err);
            Err(())
        }
    }
}

/// Load the Live model for a switch to live mode and run the speed gate. Too
/// slow: the too-slow notification, staying on the previous mode.
fn load_live_for_switch(
    previous: crate::config::Mode,
    proxy: &EventLoopProxy<UserEvent>,
    progress: &Mutex<SwitchProgress>,
) -> Result<crate::live::Live, ()> {
    let loaded = (|| -> anyhow::Result<crate::live::Live> {
        let dir = crate::model::live_model_dir_checked()?;
        let env = crate::asr::shared_environment()?;
        let threads = crate::asr::intra_threads();
        crate::live::Live::load(&dir, &env, threads)
    })();
    let mut live = match loaded {
        Ok(live) => live,
        Err(err) => {
            switch_load_failed(err);
            return Err(());
        }
    };
    lock_progress(progress).checking_speed = true;
    send_settings_refresh(proxy);
    let median = check_live_speed(&mut live, progress);
    lock_progress(progress).checking_speed = false;
    send_settings_refresh(proxy);
    if crate::live::is_fast_enough(median) {
        return Ok(live);
    }
    let body = format!(
        "Needs under 392 ms per 560 ms of audio, took {} ms. Staying on {}.",
        median.as_millis(),
        previous.display_name()
    );
    dlog!("error: dictate: this computer is too slow for Live mode: {body}");
    crate::notify::show(
        crate::notify::Kind::Error,
        "dictate: this computer is too slow for Live mode",
        &body,
    );
    Err(())
}

/// Log + notify a mode-switch load failure with the real error. The previous
/// mode is kept and nothing is saved.
fn switch_load_failed(err: anyhow::Error) {
    let message = format!("{err:#}");
    dlog!("error: dictate: could not switch mode: {message}");
    crate::notify::show(
        crate::notify::Kind::Error,
        "dictate: could not switch mode",
        &message,
    );
}

/// Benchmark the Live model for the speed gate, logging the check first and
/// recording the outcome for the Settings page. Shared by the startup check
/// and menu switches.
fn check_live_speed(
    live: &mut crate::live::Live,
    progress: &Mutex<SwitchProgress>,
) -> std::time::Duration {
    dlog!("checking Live speed…");
    let median = live.benchmark();
    lock_progress(progress).last_check = Some(LiveSpeedCheck::new(median));
    median
}

/// Tell the event loop a background mode switch failed (already logged +
/// notified). A closed event loop means the daemon is shutting down.
fn send_switch_failed(proxy: &EventLoopProxy<UserEvent>) {
    if proxy.send_event(UserEvent::ModeSwitchFailed).is_err() {
        // The event loop is closed; the daemon is shutting down.
    }
}

/// Tell the event loop to repaint the Settings window (background progress
/// changed; a closed window ignores it). A closed loop means shutdown.
fn send_settings_refresh(proxy: &EventLoopProxy<UserEvent>) {
    if proxy.send_event(UserEvent::SettingsRefresh).is_err() {
        // The event loop is closed; the daemon is shutting down.
    }
}

/// Streaming feeder thread: while Recording, wake every 100 ms, drain new
/// device-rate samples and push them through the shared [`Feeder`](crate::segment::Feeder)
/// (continuous resample to 16 kHz, then VAD pause segmentation), sending each
/// finished segment to the worker. On `Stop` the tail plus `End` go out; on
/// `Cancel` a single `Cancel` goes out. Only the `Arc` buffer crosses
/// threads; the `Recorder` stays on the event-loop thread. Hands the Vad back
/// on exit (`None` when feeder setup consumed it).
fn feeder_loop(
    vad: crate::vad::Vad,
    buffer: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    sample_rate: u32,
    control_rx: mpsc::Receiver<FeederCmd>,
    to_worker: mpsc::Sender<Job>,
) -> Option<crate::vad::Vad> {
    let mut feeder = match crate::segment::Feeder::new(sample_rate, vad) {
        Ok(feeder) => feeder,
        Err(err) => {
            feeder_failed(&to_worker, err);
            return None;
        }
    };
    let mut total_device_samples: u64 = 0;
    let mut dump = SegmentDump::new();
    let mut full_device: Vec<f32> = Vec::new();
    loop {
        match control_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(FeederCmd::Stop) => {
                let new = crate::mic::take_buffer(&buffer);
                total_device_samples += new.len() as u64;
                if dump.is_some() {
                    full_device.extend_from_slice(&new);
                }
                let pushed = match feeder.push(&new) {
                    Ok(segments) => segments,
                    Err(err) => {
                        feeder_failed(&to_worker, err);
                        break;
                    }
                };
                for segment in pushed {
                    if let Some(dump) = dump.as_mut() {
                        dump.dump_segment(&segment);
                    }
                    send_segment(&to_worker, segment);
                }
                let total_seconds = total_device_samples as f32 / sample_rate.max(1) as f32;
                // Recordings under 0.3 s keep today's too-short message: drop
                // the tail so the worker sees only `End`.
                let too_short = total_device_samples * 1000 < u64::from(sample_rate.max(1)) * 300;
                if !too_short {
                    match feeder.finish() {
                        Ok(segments) => {
                            for segment in segments {
                                if let Some(dump) = dump.as_mut() {
                                    dump.dump_segment(&segment);
                                }
                                send_segment(&to_worker, segment);
                            }
                        }
                        Err(err) => {
                            feeder_failed(&to_worker, err);
                            break;
                        }
                    }
                }
                if to_worker
                    .send(Job::End {
                        total_seconds,
                        stop_time: Instant::now(),
                    })
                    .is_err()
                {
                    // The worker is gone; the event loop reports that itself.
                }
                if let Some(dump) = dump.as_mut() {
                    dump.dump_full(&full_device, sample_rate);
                }
                break;
            }
            Ok(FeederCmd::Cancel) => {
                if to_worker.send(Job::Cancel).is_err() {
                    // The worker is gone; the event loop reports that itself.
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let new = crate::mic::take_buffer(&buffer);
                if new.is_empty() {
                    continue;
                }
                total_device_samples += new.len() as u64;
                if dump.is_some() {
                    full_device.extend_from_slice(&new);
                }
                let pushed = match feeder.push(&new) {
                    Ok(segments) => segments,
                    Err(err) => {
                        feeder_failed(&to_worker, err);
                        break;
                    }
                };
                for segment in pushed {
                    if let Some(dump) = dump.as_mut() {
                        dump.dump_segment(&segment);
                    }
                    send_segment(&to_worker, segment);
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // The event loop went away without Stop/Cancel; unblock the
                // worker rather than leave it Transcribing forever.
                if to_worker.send(Job::Cancel).is_err() {
                    // The worker is gone too.
                }
                break;
            }
        }
    }
    Some(feeder.into_scorer())
}

/// Live feeder thread: while Recording, wake every 100 ms, drain new
/// device-rate samples, resample to 16 kHz (stateful, so the stream is
/// continuous) and send the audio to the worker, which buffers it into model
/// chunks. On `Stop` the tail plus `LiveEnd` go out; on `Cancel` a single
/// `Cancel` goes out. Only the `Arc` buffer crosses threads; the `Recorder`
/// stays on the event-loop thread.
fn live_feeder_loop(
    buffer: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    sample_rate: u32,
    control_rx: mpsc::Receiver<FeederCmd>,
    to_worker: mpsc::Sender<Job>,
) {
    let mut resampler = if sample_rate == crate::audio::TARGET_SAMPLE_RATE {
        None
    } else {
        match crate::audio::StreamResampler::new(sample_rate) {
            Ok(resampler) => Some(resampler),
            Err(err) => {
                feeder_failed(&to_worker, err);
                return;
            }
        }
    };
    loop {
        match control_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(FeederCmd::Stop) => {
                let new = crate::mic::take_buffer(&buffer);
                let tail = match push_live_audio(&mut resampler, &new) {
                    Ok(tail) => tail,
                    Err(err) => {
                        feeder_failed(&to_worker, err);
                        break;
                    }
                };
                send_live_chunk(&to_worker, tail);
                if let Some(resampler) = resampler.as_mut() {
                    match resampler.finish() {
                        Ok(flushed) => send_live_chunk(&to_worker, flushed),
                        Err(err) => {
                            feeder_failed(&to_worker, err);
                            break;
                        }
                    }
                }
                if to_worker
                    .send(Job::LiveEnd {
                        stop_time: Instant::now(),
                    })
                    .is_err()
                {
                    // The worker is gone; the event loop reports that itself.
                }
                break;
            }
            Ok(FeederCmd::Cancel) => {
                if to_worker.send(Job::Cancel).is_err() {
                    // The worker is gone; the event loop reports that itself.
                }
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let new = crate::mic::take_buffer(&buffer);
                if new.is_empty() {
                    continue;
                }
                match push_live_audio(&mut resampler, &new) {
                    Ok(ready) => send_live_chunk(&to_worker, ready),
                    Err(err) => {
                        feeder_failed(&to_worker, err);
                        break;
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // The event loop went away without Stop/Cancel; unblock the
                // worker rather than leave it Transcribing forever.
                if to_worker.send(Job::Cancel).is_err() {
                    // The worker is gone too.
                }
                break;
            }
        }
    }
}

/// Resample device-rate audio to 16 kHz (`None` resampler: already 16 kHz).
fn push_live_audio(
    resampler: &mut Option<crate::audio::StreamResampler>,
    device_samples: &[f32],
) -> anyhow::Result<Vec<f32>> {
    match resampler {
        Some(resampler) => resampler.push(device_samples),
        None => Ok(device_samples.to_vec()),
    }
}

/// Queue 16 kHz audio for Live transcription (empty pieces are skipped).
fn send_live_chunk(to_worker: &mpsc::Sender<Job>, samples_16k: Vec<f32>) {
    if samples_16k.is_empty() {
        return;
    }
    if to_worker.send(Job::LiveChunk(samples_16k)).is_err() {
        // The worker is gone; the event loop reports that itself.
    }
}

/// Queue one 16 kHz segment for transcription.
fn send_segment(to_worker: &mpsc::Sender<Job>, segment: crate::segment::Segment) {
    if segment.audio.is_empty() {
        return;
    }
    if to_worker.send(Job::Segment(segment)).is_err() {
        // The worker is gone; the event loop reports that itself.
    }
}

/// Per-recording segment dump for streaming diagnosis. When
/// `DICTATE_DEBUG_SEGMENTS` holds a non-empty directory, every emitted
/// segment is written as `<id>-<N>.wav` (16 kHz mono f32, exactly the samples
/// sent to the model) plus `<id>-<N>-ctx.wav` for its context prefix, and the
/// whole recording as `<id>-full.wav`. A write failure is logged once and
/// disables the rest of the recording; dictation itself never fails.
struct SegmentDump {
    dir: PathBuf,
    recording_ms: u128,
    next_n: usize,
    failed: bool,
}

impl SegmentDump {
    /// Enable dumping for one recording when the env var holds a non-empty
    /// directory. Creates the directory; on failure logs once and stays off.
    fn new() -> Option<Self> {
        let dir = std::env::var_os("DICTATE_DEBUG_SEGMENTS")?;
        if dir.is_empty() {
            return None;
        }
        let dir = PathBuf::from(dir);
        if let Err(err) = std::fs::create_dir_all(&dir) {
            dlog!(
                "warning: could not create segment dump dir {}: {err:#}",
                dir.display()
            );
            return None;
        }
        let recording_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis())
            .unwrap_or(0);
        Some(Self {
            dir,
            recording_ms,
            next_n: 1,
            failed: false,
        })
    }

    /// Dump one emitted segment (empty audio is skipped, like the send path,
    /// so N matches the worker's per-segment log).
    fn dump_segment(&mut self, segment: &crate::segment::Segment) {
        if self.failed || segment.audio.is_empty() {
            return;
        }
        let n = self.next_n;
        self.next_n += 1;
        let (path, ctx_path) = segment_dump_paths(&self.dir, self.recording_ms, n);
        if let Err(err) = write_dump_wav(&path, &segment.audio) {
            self.fail_once(err);
            return;
        }
        if !segment.context.is_empty()
            && let Err(err) = write_dump_wav(&ctx_path, &segment.context)
        {
            self.fail_once(err);
        }
    }

    /// Dump the whole device-rate recording, resampled to 16 kHz.
    fn dump_full(&mut self, device_samples: &[f32], sample_rate: u32) {
        if self.failed {
            return;
        }
        let samples_16k =
            if device_samples.is_empty() || sample_rate == crate::audio::TARGET_SAMPLE_RATE {
                device_samples.to_vec()
            } else {
                match crate::audio::resample(device_samples, sample_rate) {
                    Ok(samples) => samples,
                    Err(err) => {
                        self.fail_once(err);
                        return;
                    }
                }
            };
        let path = full_dump_path(&self.dir, self.recording_ms);
        if let Err(err) = write_dump_wav(&path, &samples_16k) {
            self.fail_once(err);
        }
    }

    /// Log the first dump failure of this recording and skip the rest.
    fn fail_once(&mut self, err: anyhow::Error) {
        if !self.failed {
            self.failed = true;
            dlog!("warning: could not write segment dump: {err:#}");
        }
    }
}

/// Pure dump file names for segment N (1-based): the segment plus its
/// context prefix.
fn segment_dump_paths(dir: &Path, recording_ms: u128, n: usize) -> (PathBuf, PathBuf) {
    (
        dir.join(format!("{recording_ms}-{n}.wav")),
        dir.join(format!("{recording_ms}-{n}-ctx.wav")),
    )
}

/// Pure dump file name for the whole recording.
fn full_dump_path(dir: &Path, recording_ms: u128) -> PathBuf {
    dir.join(format!("{recording_ms}-full.wav"))
}

/// Write 16 kHz mono f32 samples as a WAV file.
fn write_dump_wav(path: &Path, samples: &[f32]) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: crate::audio::TARGET_SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("Could not write {}", path.display()))?;
    for sample in samples {
        writer
            .write_sample(*sample)
            .with_context(|| format!("Could not write {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("Could not write {}", path.display()))?;
    Ok(())
}

/// Abort a streaming recording after a feeder failure: log + notify, and send
/// `Cancel` so the worker discards queued segments and returns to Idle
/// (already-typed text stays).
fn feeder_failed(to_worker: &mpsc::Sender<Job>, err: anyhow::Error) {
    let message = format!("{err:#}");
    dlog!("error: {message}");
    crate::notify::show(
        crate::notify::Kind::Error,
        "dictate: recording failed",
        &message,
    );
    if to_worker.send(Job::Cancel).is_err() {
        // The worker is gone; the event loop reports that itself.
    }
}

/// Per-recording Live state in the worker.
#[derive(Default)]
struct LiveRun {
    /// Buffered 16 kHz audio awaiting full model chunks.
    buffer: Vec<f32>,
    /// Transcribed chunks (successful only).
    chunks: usize,
    /// Total chunk transcription time in ms.
    total_ms: u128,
    /// Typed characters.
    typed_chars: usize,
    /// The next non-empty output is the recording's first (trim its space).
    first: bool,
}

impl LiveRun {
    fn new() -> Self {
        Self {
            first: true,
            ..Self::default()
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// The TDT model, loading it on first need (the tray can switch a live
/// daemon to pauses without a restart).
fn ensure_tdt(models: &mut WorkerModels) -> anyhow::Result<&mut crate::asr::Asr> {
    if models.tdt.is_none() {
        let dir = crate::model::model_dir_checked()?;
        let start = Instant::now();
        let asr = crate::asr::Asr::load(&dir)?;
        dlog!("model loaded in {} ms", start.elapsed().as_millis());
        models.tdt = Some(asr);
    }
    models
        .tdt
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("Internal error: speech model missing after load"))
}

/// Drain newly arrived jobs into the local queue; when a `Cancel` is queued,
/// discard everything up to and including it and report true.
fn pop_cancel(queue: &mut std::collections::VecDeque<Job>, rx: &mpsc::Receiver<Job>) -> bool {
    while let Ok(pending) = rx.try_recv() {
        queue.push_back(pending);
    }
    if let Some(position) = queue.iter().position(|job| matches!(job, Job::Cancel)) {
        for _ in 0..=position {
            queue.pop_front();
        }
        return true;
    }
    false
}

/// Owns the models: transcribes queued recordings and signals completion.
/// Never exits while the daemon runs; every failure is logged.
///
/// `Full` is the whole-mode path (today's exact behavior).
/// `Segment`/`End` are the pauses path: each segment is typed as it finishes,
/// and at `End` the pending punctuation plus the latency line go out.
/// `LiveChunk`/`LiveEnd` are the live path: chunk outputs are typed as they
/// arrive, and at `LiveEnd` the remainder is padded, the model is flushed and
/// the summary line goes out. `Cancel` resets both streaming states.
fn transcribe_worker(
    rx: mpsc::Receiver<Job>,
    proxy: EventLoopProxy<UserEvent>,
    mut models: WorkerModels,
) {
    let mut typer = crate::typer::Typer::new(true);
    if let Err(err) = &typer {
        dlog!("error: {err:#}");
    }
    // Local queue: segments drained while a transcription was running wait
    // here, in order, so a `Cancel` spotted mid-transcription can discard
    // them before anything more is typed.
    let mut queue: std::collections::VecDeque<Job> = std::collections::VecDeque::new();
    // Per-recording pauses state.
    let mut typed_words: Vec<String> = Vec::new();
    let mut pending_punct: Option<char> = None;
    let mut segments: usize = 0;
    let mut failures: usize = 0;
    // Per-recording live state.
    let mut live = LiveRun::new();
    let worker_done =
        |proxy: &EventLoopProxy<UserEvent>| proxy.send_event(UserEvent::WorkerDone).is_err();
    'worker: loop {
        let job = if let Some(job) = queue.pop_front() {
            job
        } else {
            match rx.recv() {
                Ok(job) => job,
                Err(_) => break,
            }
        };
        // Retry a typer that failed at startup (e.g. macOS Accessibility
        // granted while the daemon runs), so no restart is needed.
        if typer.is_err() {
            typer = crate::typer::Typer::new(false);
        }
        match job {
            Job::Full(samples) => {
                match ensure_tdt(&mut models) {
                    Ok(asr) => transcribe_full(&samples, asr, typer.as_mut()),
                    Err(err) => {
                        dlog!("error: {err:#}");
                        crate::notify::show(
                            crate::notify::Kind::Error,
                            "dictate: transcription failed",
                            &format!("{err:#}"),
                        );
                    }
                }
                if worker_done(&proxy) {
                    break;
                }
            }
            Job::Segment(segment) => {
                segments += 1;
                let seq = segments;
                let seconds = segment.audio.len() as f32 / 16_000.0;
                let context_s = segment.context.len() as f32 / 16_000.0;
                let asr = match ensure_tdt(&mut models) {
                    Ok(asr) => asr,
                    Err(err) => {
                        dlog!("error: {err:#}");
                        crate::notify::show(
                            crate::notify::Kind::Error,
                            "dictate: transcription failed",
                            &format!("{err:#}"),
                        );
                        failures += 1;
                        continue;
                    }
                };
                let start = Instant::now();
                let (kept_texts, boundary) =
                    match asr.transcribe_segment_pieces(&segment.context, &segment.audio) {
                        Ok(pieces) => pieces,
                        Err(err) => {
                            dlog!("error: {err:#}");
                            crate::notify::show(
                                crate::notify::Kind::Error,
                                "dictate: transcription failed",
                                &format!("{err:#}"),
                            );
                            failures += 1;
                            continue;
                        }
                    };
                let transcribe_ms = start.elapsed().as_millis();
                if kept_texts.is_empty() {
                    continue;
                }
                // A `Cancel` that arrived mid-transcription discards this
                // segment's text and every queued job up to it.
                while let Ok(pending) = rx.try_recv() {
                    queue.push_back(pending);
                }
                if let Some(position) = queue.iter().position(|job| matches!(job, Job::Cancel)) {
                    for _ in 0..=position {
                        queue.pop_front();
                    }
                    typed_words.clear();
                    pending_punct = None;
                    segments = 0;
                    failures = 0;
                    if worker_done(&proxy) {
                        break;
                    }
                    continue;
                }
                let tail_start = typed_words.len().saturating_sub(2);
                let (to_type, new_pending) = crate::asr::join_piece(
                    &typed_words[tail_start..],
                    pending_punct,
                    boundary,
                    &kept_texts,
                );
                if to_type.is_empty() {
                    // Deduped to nothing with no boundary: drop the old
                    // pending without typing.
                    pending_punct = new_pending;
                    continue;
                }
                let type_start = Instant::now();
                let typed = match typer.as_mut() {
                    Ok(typer) => typer.type_text(&to_type),
                    Err(err) => Err(anyhow::anyhow!("{err:#}")),
                };
                let type_ms = type_start.elapsed().as_millis();
                match typed {
                    Ok(()) => {
                        crate::asr::append_typed_words(&mut typed_words, &to_type);
                        pending_punct = new_pending;
                        dlog!(
                            "segment {seq}: {seconds:.1} s audio (context {context_s:.1} s), transcribed in {transcribe_ms} ms, typed {} chars in {type_ms} ms",
                            to_type.chars().count()
                        );
                    }
                    Err(err) => {
                        failures += 1;
                        dlog!("error: {err:#}");
                        dlog!("Text that was not typed: {to_type}");
                        crate::notify::show(
                            crate::notify::Kind::Error,
                            "dictate: could not type the text",
                            &crate::notify::typing_failure_body(&format!("{err:#}"), &to_type),
                        );
                    }
                }
            }
            Job::End {
                total_seconds,
                stop_time,
            } => {
                if !typed_words.is_empty() {
                    if let Some(punct) = pending_punct {
                        let text = punct.to_string();
                        let typed = match typer.as_mut() {
                            Ok(typer) => typer.type_text(&text),
                            Err(err) => Err(anyhow::anyhow!("{err:#}")),
                        };
                        if let Err(err) = typed {
                            dlog!("error: {err:#}");
                            dlog!("Text that was not typed: {text}");
                            crate::notify::show(
                                crate::notify::Kind::Error,
                                "dictate: could not type the text",
                                &crate::notify::typing_failure_body(&format!("{err:#}"), &text),
                            );
                        }
                    }
                    dlog!(
                        "done {} ms after stop (total {total_seconds:.1} s audio, {segments} segments)",
                        stop_time.elapsed().as_millis()
                    );
                } else if failures == 0 {
                    if total_seconds * 1000.0
                        < crate::mic::MIN_RECORDING_SAMPLES as f32 / 16_000.0 * 1000.0
                    {
                        let ms = (total_seconds * 1000.0) as u64;
                        let message =
                            format!("Recording too short ({ms} ms); nothing to transcribe.");
                        dlog!("{message}");
                        crate::notify::show(crate::notify::Kind::Info, "dictate", &message);
                    } else {
                        let message = crate::asr::no_speech_message(total_seconds);
                        dlog!("{message}");
                        crate::notify::show(crate::notify::Kind::Info, "dictate", &message);
                    }
                }
                typed_words.clear();
                pending_punct = None;
                segments = 0;
                failures = 0;
                if worker_done(&proxy) {
                    break;
                }
            }
            Job::LiveChunk(samples) => {
                let Some(live_model) = models.live.as_mut() else {
                    dlog!("error: Internal error: live audio arrived without a live model");
                    continue;
                };
                live.buffer.extend_from_slice(&samples);
                let chunk = live_model.chunk_samples().max(1);
                while live.buffer.len() >= chunk {
                    let piece: Vec<f32> = live.buffer.drain(..chunk).collect();
                    let start = Instant::now();
                    let out = match live_model.transcribe_chunk(&piece) {
                        Ok(out) => out,
                        Err(err) => {
                            dlog!("error: {err:#}");
                            crate::notify::show(
                                crate::notify::Kind::Error,
                                "dictate: transcription failed",
                                &format!("{err:#}"),
                            );
                            continue;
                        }
                    };
                    live.chunks += 1;
                    live.total_ms += start.elapsed().as_millis();
                    if out.is_empty() {
                        continue;
                    }
                    // A `Cancel` that arrived mid-transcription discards this
                    // output and the rest of the recording.
                    if pop_cancel(&mut queue, &rx) {
                        live.reset();
                        live_model.reset();
                        if worker_done(&proxy) {
                            break 'worker;
                        }
                        continue 'worker;
                    }
                    let to_type = if live.first {
                        live.first = false;
                        out.trim_start().to_string()
                    } else {
                        out
                    };
                    if to_type.is_empty() {
                        continue;
                    }
                    match typer.as_mut() {
                        Ok(typer) => match typer.type_text(&to_type) {
                            Ok(()) => live.typed_chars += to_type.chars().count(),
                            Err(err) => {
                                live_type_failed(&to_type, err);
                            }
                        },
                        Err(err) => {
                            live_type_failed(&to_type, anyhow::anyhow!("{err:#}"));
                        }
                    }
                }
            }
            Job::LiveEnd { stop_time } => {
                let Some(live_model) = models.live.as_mut() else {
                    dlog!("error: Internal error: live recording ended without a live model");
                    live.reset();
                    if worker_done(&proxy) {
                        break;
                    }
                    continue;
                };
                let chunk = live_model.chunk_samples().max(1);
                if !live.buffer.is_empty() {
                    let mut last = std::mem::take(&mut live.buffer);
                    last.resize(chunk, 0.0);
                    transcribe_live_piece(live_model, &mut live, typer.as_mut(), &last);
                }
                let zeros = vec![0.0; chunk];
                for _ in 0..crate::live::FLUSH_CHUNKS {
                    transcribe_live_piece(live_model, &mut live, typer.as_mut(), &zeros);
                }
                live_model.reset();
                let avg = if live.chunks > 0 {
                    live.total_ms / live.chunks as u128
                } else {
                    0
                };
                dlog!(
                    "live: {} chunks, avg {} ms/chunk, typed {} chars, done {} ms after stop",
                    live.chunks,
                    avg,
                    live.typed_chars,
                    stop_time.elapsed().as_millis()
                );
                live.reset();
                if worker_done(&proxy) {
                    break;
                }
            }
            Job::Cancel => {
                typed_words.clear();
                pending_punct = None;
                segments = 0;
                failures = 0;
                live.reset();
                if let Some(live_model) = models.live.as_mut() {
                    live_model.reset();
                }
                if worker_done(&proxy) {
                    break;
                }
            }
            Job::SwapTdt(model) => {
                models.tdt = Some(model);
                models.live = None;
                if proxy.send_event(UserEvent::ModeSwapped).is_err() {
                    break;
                }
            }
            Job::SwapLive(model) => {
                models.live = Some(model);
                models.tdt = None;
                if proxy.send_event(UserEvent::ModeSwapped).is_err() {
                    break;
                }
            }
        }
    }
}

/// Transcribe one full Live chunk (remainder or flush) and type its output.
/// Failures are logged + notified; the chunk is skipped.
fn transcribe_live_piece(
    live_model: &mut crate::live::Live,
    run: &mut LiveRun,
    typer: Result<&mut crate::typer::Typer, &mut anyhow::Error>,
    piece: &[f32],
) {
    let start = Instant::now();
    let out = match live_model.transcribe_chunk(piece) {
        Ok(out) => out,
        Err(err) => {
            dlog!("error: {err:#}");
            crate::notify::show(
                crate::notify::Kind::Error,
                "dictate: transcription failed",
                &format!("{err:#}"),
            );
            return;
        }
    };
    run.chunks += 1;
    run.total_ms += start.elapsed().as_millis();
    if out.is_empty() {
        return;
    }
    let to_type = if run.first {
        run.first = false;
        out.trim_start().to_string()
    } else {
        out
    };
    if to_type.is_empty() {
        return;
    }
    match typer {
        Ok(typer) => match typer.type_text(&to_type) {
            Ok(()) => run.typed_chars += to_type.chars().count(),
            Err(err) => live_type_failed(&to_type, err),
        },
        Err(err) => live_type_failed(&to_type, anyhow::anyhow!("{err:#}")),
    }
}

/// Log + notify a Live typing failure; the text stays on stderr, never lost.
fn live_type_failed(to_type: &str, err: anyhow::Error) {
    dlog!("error: {err:#}");
    dlog!("Text that was not typed: {to_type}");
    crate::notify::show(
        crate::notify::Kind::Error,
        "dictate: could not type the text",
        &crate::notify::typing_failure_body(&format!("{err:#}"), to_type),
    );
}

/// Transcribe one whole recording (whole mode): today's exact behavior.
fn transcribe_full(
    samples: &[f32],
    asr: &mut crate::asr::Asr,
    typer: Result<&mut crate::typer::Typer, &mut anyhow::Error>,
) {
    let seconds = samples.len() as f32 / 16_000.0;
    if crate::mic::recording_too_short(samples) {
        let ms = samples.len() as u64 * 1000 / 16_000;
        let message = format!("Recording too short ({ms} ms); nothing to transcribe.");
        dlog!("{message}");
        crate::notify::show(crate::notify::Kind::Info, "dictate", &message);
    } else {
        let start = Instant::now();
        match asr.transcribe(samples) {
            Ok(text) => {
                if crate::asr::is_empty_transcript(&text) {
                    let message = crate::asr::no_speech_message(seconds);
                    dlog!("{message}");
                    crate::notify::show(crate::notify::Kind::Info, "dictate", &message);
                } else {
                    type_and_log(&text, seconds, start, typer);
                }
            }
            Err(err) => {
                dlog!("error: {err:#}");
                crate::notify::show(
                    crate::notify::Kind::Error,
                    "dictate: transcription failed",
                    &format!("{err:#}"),
                );
            }
        }
    }
}

/// Type one transcript and log the outcome. A failed setup or a failed
/// keystroke keeps the text on stderr so it is never lost.
fn type_and_log(
    text: &str,
    seconds: f32,
    start: Instant,
    typer: Result<&mut crate::typer::Typer, &mut anyhow::Error>,
) {
    let typed = match typer {
        Ok(typer) => typer.type_text(text),
        Err(err) => Err(anyhow::anyhow!("{err:#}")),
    };
    match typed {
        Ok(()) => dlog!(
            "recorded {seconds:.1} s, transcribed in {} ms, typed {} chars",
            start.elapsed().as_millis(),
            text.chars().count()
        ),
        Err(err) => {
            dlog!("error: {err:#}");
            dlog!("Text that was not typed: {text}");
            crate::notify::show(
                crate::notify::Kind::Error,
                "dictate: could not type the text",
                &crate::notify::typing_failure_body(&format!("{err:#}"), text),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_machine_covers_every_pair() {
        use Action::{CancelRecording, None, StartRecording, StopRecording};
        use Command::{Cancel, Start, Status, Stop, Toggle};
        use State::{Idle, Recording, Transcribing};
        let cases = [
            (Idle, Start, (Recording, StartRecording, "ok recording")),
            (Idle, Toggle, (Recording, StartRecording, "ok recording")),
            (Idle, Stop, (Idle, None, "err not recording")),
            (Idle, Cancel, (Idle, None, "ok idle")),
            (Idle, Status, (Idle, None, "ok idle")),
            (Recording, Start, (Recording, None, "err already recording")),
            (
                Recording,
                Stop,
                (Transcribing, StopRecording, "ok transcribing"),
            ),
            (
                Recording,
                Toggle,
                (Transcribing, StopRecording, "ok transcribing"),
            ),
            (Recording, Cancel, (Idle, CancelRecording, "ok cancelled")),
            (Recording, Status, (Recording, None, "ok recording")),
            (
                Transcribing,
                Start,
                (Transcribing, None, "err busy transcribing"),
            ),
            (
                Transcribing,
                Stop,
                (Transcribing, None, "err busy transcribing"),
            ),
            (
                Transcribing,
                Toggle,
                (Transcribing, None, "err busy transcribing"),
            ),
            (
                Transcribing,
                Cancel,
                (Transcribing, None, "err busy transcribing"),
            ),
            (
                Transcribing,
                Status,
                (Transcribing, None, "ok transcribing"),
            ),
        ];
        assert_eq!(cases.len(), 15);
        for (state, cmd, expected) in cases {
            assert_eq!(next(state, cmd), expected, "next({state:?}, {cmd:?})");
        }
    }

    fn test_mics() -> Vec<crate::mic::Mic> {
        vec![
            crate::mic::Mic {
                id: "pulseaudio:alsa_input.usb".to_string(),
                name: "USB Microphone".to_string(),
                is_default: false,
            },
            crate::mic::Mic {
                id: "pulseaudio:alsa_input.pci".to_string(),
                name: "Built-in Audio Analog Stereo".to_string(),
                is_default: true,
            },
        ]
    }

    #[test]
    fn default_mic_name_finds_default() {
        assert_eq!(
            default_mic_name(&test_mics()),
            Some("Built-in Audio Analog Stereo".to_string())
        );
        assert_eq!(default_mic_name(&[]), None);
    }

    #[test]
    fn selected_mic_name_prefers_selection_then_default() {
        let mics = test_mics();
        assert_eq!(
            selected_mic_name("pulseaudio:alsa_input.usb", &mics),
            Some("USB Microphone".to_string())
        );
        // Nothing selected: the default.
        assert_eq!(
            selected_mic_name("", &mics),
            Some("Built-in Audio Analog Stereo".to_string())
        );
        // A selection that is gone: the default (what records next).
        assert_eq!(
            selected_mic_name("pulseaudio:gone", &mics),
            Some("Built-in Audio Analog Stereo".to_string())
        );
        assert_eq!(selected_mic_name("", &[]), None);
    }

    #[test]
    fn mic_listing_change_tracks_ids_and_default() {
        let mics = test_mics();
        assert!(!mic_listing_changed(&mics, &test_mics()));
        assert!(!mic_listing_changed(&[], &[]));
        // A renamed mic is display-only: the cache stays.
        let mut renamed = test_mics();
        renamed[0].name = "USB Mic Renamed".to_string();
        assert!(!mic_listing_changed(&mics, &renamed));
        // A removed or added mic invalidates.
        assert!(mic_listing_changed(&mics, &mics[1..]));
        let mut added = test_mics();
        added.push(crate::mic::Mic {
            id: "pulseaudio:alsa_input.hdmi".to_string(),
            name: "HDMI".to_string(),
            is_default: false,
        });
        assert!(mic_listing_changed(&mics, &added));
        // A moved default invalidates.
        let mut moved = test_mics();
        moved[0].is_default = true;
        moved[1].is_default = false;
        assert!(mic_listing_changed(&mics, &moved));
    }

    #[test]
    fn daemon_log_path_appends_file_name() {
        assert_eq!(
            daemon_log_path_for(Path::new("/data")),
            PathBuf::from("/data/daemon.log")
        );
    }

    #[test]
    fn download_milestones_fire_at_25_50_75() {
        assert_eq!(milestones_reached(0), 0);
        assert_eq!(milestones_reached(24), 0);
        assert_eq!(milestones_reached(25), 1);
        assert_eq!(milestones_reached(49), 1);
        assert_eq!(milestones_reached(50), 2);
        assert_eq!(milestones_reached(74), 2);
        assert_eq!(milestones_reached(75), 3);
        assert_eq!(milestones_reached(100), 3);
        assert_eq!(DOWNLOAD_MILESTONES, [25, 50, 75]);
    }

    #[test]
    fn speed_check_records_ms_and_verdict() {
        assert_eq!(
            LiveSpeedCheck::new(Duration::from_millis(120)),
            LiveSpeedCheck {
                ms: 120,
                fast: true
            }
        );
        assert_eq!(
            LiveSpeedCheck::new(Duration::from_millis(905)),
            LiveSpeedCheck {
                ms: 905,
                fast: false
            }
        );
        // The gate is 392 ms per 560 ms chunk, inclusive.
        assert!(LiveSpeedCheck::new(Duration::from_millis(392)).fast);
        assert!(!LiveSpeedCheck::new(Duration::from_millis(393)).fast);
    }

    #[test]
    fn download_progress_body_names_the_milestone() {
        assert_eq!(live_download_body(25), "Downloading Live model… 25%");
        assert_eq!(live_download_body(50), "Downloading Live model… 50%");
        assert_eq!(live_download_body(75), "Downloading Live model… 75%");
    }

    #[test]
    fn segment_dump_paths_name_segment_context_and_full() {
        let dir = Path::new("/tmp/seg");
        let (segment, ctx) = segment_dump_paths(dir, 123, 2);
        assert_eq!(segment, PathBuf::from("/tmp/seg/123-2.wav"));
        assert_eq!(ctx, PathBuf::from("/tmp/seg/123-2-ctx.wav"));
        assert_eq!(
            full_dump_path(dir, 123),
            PathBuf::from("/tmp/seg/123-full.wav")
        );
    }
}
