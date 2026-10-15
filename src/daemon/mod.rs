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
use std::sync::mpsc;
use std::time::Instant;

use anyhow::Context;
use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
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

/// One unit of work for the transcription worker.
#[derive(Debug)]
pub(crate) enum Job {
    /// Whole recording at 16 kHz (streaming off): today's exact path.
    Full(Vec<f32>),
    /// One pause segment at 16 kHz plus its left context (streaming on).
    Segment(crate::segment::Segment),
    /// End of a streaming recording: whole-recording duration plus the moment
    /// Stop was handled, so the worker can log latency after release.
    End {
        total_seconds: f32,
        stop_time: Instant,
    },
    /// Discard a streaming recording; already-typed text stays.
    Cancel,
}

/// Control message for the streaming feeder thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeederCmd {
    Stop,
    Cancel,
}

/// A running feeder thread (streaming on, while Recording only).
struct FeederSession {
    control_tx: mpsc::Sender<FeederCmd>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Effective streaming switch (settings file + `DICTATE_STREAMING`).
fn streaming_enabled() -> bool {
    crate::config::streaming_on()
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
pub(crate) fn run_daemon() -> anyhow::Result<()> {
    init_file_log();
    run_daemon_inner().map_err(|err| {
        mirror_to_log(&format!("error: {err:#}"));
        err
    })
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
    let dir = crate::model::model_dir_checked()?;
    // Env wins; an unset or empty DICTATE_HOTKEY falls back to the file.
    let hotkey = match std::env::var("DICTATE_HOTKEY").ok() {
        Some(value) if !value.is_empty() => crate::hotkey::resolve_hotkey(Some(&value))
            .map_err(|message| anyhow::anyhow!("{message}"))?,
        _ => crate::hotkey::resolve_file_hotkey(&file_settings.hotkey)
            .map_err(|message| anyhow::anyhow!("{message}"))?,
    };
    let start = Instant::now();
    let asr = crate::asr::Asr::load(&dir)?;
    dlog!(
        "model loaded in {} ms. Ready: run \"dictate toggle\" to start/stop recording.",
        start.elapsed().as_millis()
    );
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
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .context("Could not start the daemon event loop")?;
    let (to_worker, worker_rx) = mpsc::channel::<Job>();
    let worker_proxy = event_loop.create_proxy();
    let worker = std::thread::spawn(move || transcribe_worker(worker_rx, worker_proxy, asr));
    #[cfg(unix)]
    {
        let socket_proxy = event_loop.create_proxy();
        std::thread::spawn(move || socket::serve_socket(listener, socket_proxy));
    }
    #[cfg(not(unix))]
    dlog!(
        "Socket commands (dictate toggle/start/stop) are not available on Windows yet; use the hotkey."
    );
    let mut app = DaemonApp::new(to_worker, worker, file_settings, event_loop.create_proxy());
    setup_hotkey(&event_loop, hotkey, &mut app);
    event_loop
        .run_app(&mut app)
        .context("The daemon event loop failed")?;
    #[cfg(unix)]
    socket::remove_socket_file(&socket_file);
    Ok(())
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
    to_worker: mpsc::Sender<Job>,
    worker: std::thread::JoinHandle<()>,
    hotkey: Option<HotKey>,
    hotkey_display: String,
    press_started: Option<Instant>,
    tap_armed: bool,
    worker_gone_notified: bool,
    settings: crate::config::Settings,
    tray: Option<crate::tray::Tray>,
    tray_attempted: bool,
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
        menu_proxy: EventLoopProxy<UserEvent>,
    ) -> Self {
        Self {
            state: State::Idle,
            recorder: None,
            feeder: None,
            to_worker,
            worker,
            hotkey: None,
            hotkey_display: String::new(),
            press_started: None,
            tap_armed: false,
            worker_gone_notified: false,
            settings,
            tray: None,
            tray_attempted: false,
            menu_proxy,
            _hotkey_manager: None,
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
            return "err transcription worker stopped; restart \"dictate daemon\"".to_string();
        }
        let (new_state, action, ok_reply) = next(self.state, cmd);
        let reply = self.run_action(action, new_state, ok_reply);
        if self.state != State::Recording {
            self.tap_armed = false;
        }
        self.sync_tray();
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

    /// Refresh the tray icon and menu after a state change (no tray: no-op).
    fn sync_tray(&self) {
        if let Some(tray) = &self.tray
            && let Err(err) = tray.sync_state(self.state)
        {
            dlog!("error: {err:#}");
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
            Some(crate::tray::TrayAction::Streaming) => self.flip_streaming(),
            Some(crate::tray::TrayAction::Shortcut(display)) => self.switch_hotkey(&display),
            Some(crate::tray::TrayAction::HowTo) => self.open_how_to(),
            Some(crate::tray::TrayAction::OpenConfig) => self.open_settings_file(),
            Some(crate::tray::TrayAction::Quit) => event_loop.exit(),
        }
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
    }

    /// Flip the `Type while speaking` check, update the effective switch, save.
    fn flip_streaming(&mut self) {
        let new = !self.settings.streaming;
        self.settings.streaming = new;
        crate::config::refresh_streaming(new);
        if let Some(tray) = &self.tray {
            tray.set_streaming_checked(new);
        }
        self.report_save_error(crate::config::save(&self.settings));
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
        self.settings.hotkey = display.to_string();
        self.report_save_error(crate::config::save(&self.settings));
        self.sync_shortcut_checks();
        dlog!("Hotkey: {display} (tap to toggle, hold to talk)");
    }

    /// Notify a failed shortcut switch and re-check the current shortcut.
    fn shortcut_failed(&self, display: &str, err: &str) {
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

    /// Start the mic, and the feeder thread when streaming is on.
    fn start_recording(&mut self, new_state: State, ok_reply: &str) -> String {
        match crate::mic::Recorder::start() {
            Ok(started) => {
                if streaming_enabled() {
                    let buffer = started.buffer_handle();
                    let sample_rate = started.sample_rate();
                    let (control_tx, control_rx) = mpsc::channel::<FeederCmd>();
                    let to_worker = self.to_worker.clone();
                    let thread = std::thread::spawn(move || {
                        feeder_loop(buffer, sample_rate, control_rx, to_worker);
                    });
                    self.feeder = Some(FeederSession {
                        control_tx,
                        thread: Some(thread),
                    });
                    self.recorder = Some(started);
                } else {
                    self.recorder = Some(started);
                }
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

    /// Stop the mic. Streaming on: the feeder sends the tail plus `End` and
    /// the state stays Transcribing until the worker is done. Streaming off:
    /// today's exact path (one `Full` job at stop).
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

    /// Stop a streaming recording via the feeder thread.
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

    /// Signal the feeder thread (`Stop`/`Cancel`) and wait for it to exit.
    /// No feeder: no-op. A dead feeder unblocks the worker with `Cancel`.
    fn finish_feeder(&mut self, cmd: FeederCmd) {
        let Some(mut session) = self.feeder.take() else {
            return;
        };
        let signalled = session.control_tx.send(cmd).is_ok();
        if let Some(thread) = session.thread.take()
            && thread.join().is_err()
        {
            dlog!("error: the streaming feeder thread panicked");
        }
        if !signalled && self.to_worker.send(Job::Cancel).is_err() {
            // The worker is gone; `handle_command` reports that itself.
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
            match crate::tray::build_tray(&self.settings, self.state, self.menu_proxy.clone()) {
                Ok(tray) => self.tray = Some(tray),
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

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        _event: WindowEvent,
    ) {
        // The daemon never creates a window.
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
            }
            UserEvent::HotKey(event) => self.handle_hotkey(event),
            UserEvent::Menu(event) => self.handle_menu(event, event_loop),
        }
    }
}

/// Streaming feeder thread: while Recording, wake every 100 ms, drain new
/// device-rate samples and push them through the shared [`Feeder`](crate::segment::Feeder)
/// (continuous resample to 16 kHz, then pause segmentation), sending each
/// finished segment to the worker. On `Stop` the tail plus `End` go out; on
/// `Cancel` a single `Cancel` goes out. Only the `Arc` buffer crosses
/// threads; the `Recorder` stays on the event-loop thread.
fn feeder_loop(
    buffer: std::sync::Arc<std::sync::Mutex<Vec<f32>>>,
    sample_rate: u32,
    control_rx: mpsc::Receiver<FeederCmd>,
    to_worker: mpsc::Sender<Job>,
) {
    let mut feeder = match crate::segment::Feeder::new(sample_rate) {
        Ok(feeder) => feeder,
        Err(err) => {
            feeder_failed(&to_worker, err);
            return;
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

/// Owns the `Asr`: transcribes queued recordings and signals completion.
/// Never exits while the daemon runs; every failure is logged.
///
/// `Full` is the streaming-off path (today's exact behavior).
/// `Segment`/`End`/`Cancel` are the streaming path: each segment is typed
/// as it finishes, at `End` the pending punctuation plus the latency line go
/// out, and `Cancel` discards queued segments without typing anything more.
fn transcribe_worker(
    rx: mpsc::Receiver<Job>,
    proxy: EventLoopProxy<UserEvent>,
    mut asr: crate::asr::Asr,
) {
    let mut typer = crate::typer::Typer::new();
    if let Err(err) = &typer {
        dlog!("error: {err:#}");
    }
    // Local queue: segments drained while a transcription was running wait
    // here, in order, so a `Cancel` spotted mid-transcription can discard
    // them before anything more is typed.
    let mut queue: std::collections::VecDeque<Job> = std::collections::VecDeque::new();
    // Per-recording streaming state.
    let mut typed_words: Vec<String> = Vec::new();
    let mut pending_punct: Option<char> = None;
    let mut segments: usize = 0;
    let mut failures: usize = 0;
    let worker_done =
        |proxy: &EventLoopProxy<UserEvent>| proxy.send_event(UserEvent::WorkerDone).is_err();
    loop {
        let job = if let Some(job) = queue.pop_front() {
            job
        } else {
            match rx.recv() {
                Ok(job) => job,
                Err(_) => break,
            }
        };
        match job {
            Job::Full(samples) => {
                transcribe_full(&samples, &mut asr, typer.as_mut());
                if worker_done(&proxy) {
                    break;
                }
            }
            Job::Segment(segment) => {
                segments += 1;
                let seq = segments;
                let seconds = segment.audio.len() as f32 / 16_000.0;
                let context_s = segment.context.len() as f32 / 16_000.0;
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
            Job::Cancel => {
                typed_words.clear();
                pending_punct = None;
                segments = 0;
                failures = 0;
                if worker_done(&proxy) {
                    break;
                }
            }
        }
    }
}

/// Transcribe one whole recording (streaming off): today's exact behavior.
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

    #[test]
    fn daemon_log_path_appends_file_name() {
        assert_eq!(
            daemon_log_path_for(Path::new("/data")),
            PathBuf::from("/data/daemon.log")
        );
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
