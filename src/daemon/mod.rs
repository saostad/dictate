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

#[cfg(unix)]
pub(crate) mod socket;

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

/// Run `dictate daemon`: load the model, bind the socket (Unix), serve commands.
pub(crate) fn run_daemon() -> anyhow::Result<()> {
    let (file_settings, settings_problem) = crate::config::load();
    crate::config::init_effective(&file_settings);
    if let Some(problem) = settings_problem {
        let message = format!("{problem} Using defaults.");
        eprintln!("error: {message}");
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
    eprintln!(
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
    let (to_worker, worker_rx) = mpsc::channel::<Vec<f32>>();
    let worker_proxy = event_loop.create_proxy();
    let worker = std::thread::spawn(move || transcribe_worker(worker_rx, worker_proxy, asr));
    #[cfg(unix)]
    {
        let socket_proxy = event_loop.create_proxy();
        std::thread::spawn(move || socket::serve_socket(listener, socket_proxy));
    }
    #[cfg(not(unix))]
    eprintln!(
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
        eprintln!(
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
            eprintln!(
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
    eprintln!("Hotkey: {} (tap to toggle, hold to talk)", config.display);
}

/// Daemon state, owned by the event-loop thread: the recording state, the
/// `Recorder`, the hotkey manager, the tray icon and the tap-vs-hold tracking.
struct DaemonApp {
    state: State,
    recorder: Option<crate::mic::Recorder>,
    to_worker: mpsc::Sender<Vec<f32>>,
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
        to_worker: mpsc::Sender<Vec<f32>>,
        worker: std::thread::JoinHandle<()>,
        settings: crate::config::Settings,
        menu_proxy: EventLoopProxy<UserEvent>,
    ) -> Self {
        Self {
            state: State::Idle,
            recorder: None,
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
            eprintln!("error: The transcription worker thread ended unexpectedly");
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
        let reply = run_action(
            action,
            new_state,
            ok_reply,
            &mut self.state,
            &mut self.recorder,
            &self.to_worker,
        );
        if self.state != State::Recording {
            self.tap_armed = false;
        }
        self.sync_tray();
        reply
    }

    /// Map a hotkey press/release to a command and run it. A hotkey has no
    /// requester, so the reply is dropped; genuine failures are already
    /// logged by [`run_action`].
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
            eprintln!("error: {err:#}");
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

    /// Log and notify a settings-save failure (the in-memory value stays).
    fn report_save_error(&self, result: anyhow::Result<()>) {
        if let Err(err) = result {
            eprintln!("error: {err:#}");
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
            eprintln!(
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
        eprintln!("Hotkey: {display} (tap to toggle, hold to talk)");
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
            eprintln!("error: {message}");
            crate::notify::show(crate::notify::Kind::Error, "dictate", &message);
        }
    }

    /// Create the settings file (with the current settings, which are the
    /// defaults when the file is missing) and open it in the default app.
    fn open_settings_file(&self) {
        let Some(path) = crate::config::config_path() else {
            let message = "Could not find a home directory for this user";
            eprintln!("error: {message}");
            crate::notify::show(crate::notify::Kind::Error, "dictate", message);
            return;
        };
        if !path.is_file()
            && let Err(err) = crate::config::save(&self.settings)
        {
            let message = format!("Could not create {}: {err:#}", path.display());
            eprintln!("error: {message}");
            crate::notify::show(crate::notify::Kind::Error, "dictate", &message);
            return;
        }
        if let Err(err) = crate::tray::open_target(&path.to_string_lossy()) {
            let message = format!("{err:#}");
            eprintln!("error: {message}");
            crate::notify::show(crate::notify::Kind::Error, "dictate", &message);
        }
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
                    eprintln!("error: {err:#}");
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
                    eprintln!("error: Could not reply to a daemon client: the connection closed");
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

/// Run the state-machine action, updating `state`, and return the reply.
/// I/O failures are logged and turn the reply into `err ...`.
fn run_action(
    action: Action,
    new_state: State,
    ok_reply: &str,
    state: &mut State,
    recorder: &mut Option<crate::mic::Recorder>,
    to_worker: &mpsc::Sender<Vec<f32>>,
) -> String {
    match action {
        Action::None => {
            *state = new_state;
            ok_reply.to_string()
        }
        Action::StartRecording => match crate::mic::Recorder::start() {
            Ok(started) => {
                *recorder = Some(started);
                *state = new_state;
                crate::cue::play_start();
                ok_reply.to_string()
            }
            Err(err) => {
                eprintln!("error: {err:#}");
                crate::notify::show(
                    crate::notify::Kind::Error,
                    "dictate: recording failed",
                    &format!("{err:#}"),
                );
                format!("err {err:#}")
            }
        },
        Action::StopRecording => match recorder.take() {
            Some(started) => match started.stop() {
                Ok(samples) => match to_worker.send(samples) {
                    Ok(()) => {
                        *state = new_state;
                        crate::cue::play_stop();
                        ok_reply.to_string()
                    }
                    Err(err) => {
                        *state = State::Idle;
                        let message = format!("Could not queue audio for transcription: {err}");
                        eprintln!("error: {message}");
                        crate::notify::show(
                            crate::notify::Kind::Error,
                            "dictate: recording failed",
                            &message,
                        );
                        format!("err {message}")
                    }
                },
                Err(err) => {
                    *state = State::Idle;
                    eprintln!("error: {err:#}");
                    crate::notify::show(
                        crate::notify::Kind::Error,
                        "dictate: recording failed",
                        &format!("{err:#}"),
                    );
                    format!("err {err:#}")
                }
            },
            None => {
                *state = State::Idle;
                eprintln!("error: the daemon was recording but its recorder is gone");
                crate::notify::show(
                    crate::notify::Kind::Error,
                    "dictate: recording failed",
                    "the daemon was recording but its recorder is gone",
                );
                "err not recording".to_string()
            }
        },
        Action::CancelRecording => {
            *recorder = None;
            *state = new_state;
            ok_reply.to_string()
        }
    }
}

/// Owns the `Asr`: transcribes queued recordings and signals completion.
/// Never exits while the daemon runs; every failure is logged.
fn transcribe_worker(
    rx: mpsc::Receiver<Vec<f32>>,
    proxy: EventLoopProxy<UserEvent>,
    mut asr: crate::asr::Asr,
) {
    let mut typer = crate::typer::Typer::new();
    if let Err(err) = &typer {
        eprintln!("error: {err:#}");
    }
    for samples in rx {
        let seconds = samples.len() as f32 / 16_000.0;
        if crate::mic::recording_too_short(&samples) {
            let ms = samples.len() as u64 * 1000 / 16_000;
            let message = format!("Recording too short ({ms} ms); nothing to transcribe.");
            eprintln!("{message}");
            crate::notify::show(crate::notify::Kind::Info, "dictate", &message);
        } else {
            let start = Instant::now();
            match asr.transcribe(&samples) {
                Ok(text) => {
                    if crate::asr::is_empty_transcript(&text) {
                        let message = crate::asr::no_speech_message(seconds);
                        eprintln!("{message}");
                        crate::notify::show(crate::notify::Kind::Info, "dictate", &message);
                    } else {
                        type_and_log(&text, seconds, start, typer.as_mut());
                    }
                }
                Err(err) => {
                    eprintln!("error: {err:#}");
                    crate::notify::show(
                        crate::notify::Kind::Error,
                        "dictate: transcription failed",
                        &format!("{err:#}"),
                    );
                }
            }
        }
        if proxy.send_event(UserEvent::WorkerDone).is_err() {
            break;
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
        Ok(()) => eprintln!(
            "recorded {seconds:.1} s, transcribed in {} ms, typed {} chars",
            start.elapsed().as_millis(),
            text.chars().count()
        ),
        Err(err) => {
            eprintln!("error: {err:#}");
            eprintln!("Text that was not typed: {text}");
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
}
