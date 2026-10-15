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

use std::sync::mpsc;
use std::time::Instant;

use anyhow::Context;
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
enum UserEvent {
    /// A command from a socket client; the reply goes back on the channel.
    #[cfg(unix)]
    Command(Command, mpsc::Sender<String>),
    /// The worker finished one transcription; back to Idle.
    WorkerDone,
    /// A global hotkey press or release.
    HotKey(GlobalHotKeyEvent),
}

/// Run `dictate daemon`: load the model, bind the socket (Unix), serve commands.
pub(crate) fn run_daemon() -> anyhow::Result<()> {
    let dir = crate::model::model_dir_checked()?;
    let hotkey = crate::hotkey::resolve_hotkey(std::env::var("DICTATE_HOTKEY").ok().as_deref())
        .map_err(|message| anyhow::anyhow!("{message}"))?;
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
    let mut app = DaemonApp::new(to_worker, worker);
    setup_hotkey(&event_loop, &hotkey, &mut app);
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
    config: &crate::hotkey::HotkeyConfig,
    app: &mut DaemonApp,
) {
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
    app.hotkey_id = Some(config.hotkey.id());
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
/// `Recorder`, the hotkey manager and the tap-vs-hold tracking.
struct DaemonApp {
    state: State,
    recorder: Option<crate::mic::Recorder>,
    to_worker: mpsc::Sender<Vec<f32>>,
    worker: std::thread::JoinHandle<()>,
    hotkey_id: Option<u32>,
    press_started: Option<Instant>,
    tap_armed: bool,
    /// Held so the registration stays alive; dropping it would unregister.
    /// Never read, hence the underscore.
    _hotkey_manager: Option<GlobalHotKeyManager>,
}

impl DaemonApp {
    fn new(to_worker: mpsc::Sender<Vec<f32>>, worker: std::thread::JoinHandle<()>) -> Self {
        Self {
            state: State::Idle,
            recorder: None,
            to_worker,
            worker,
            hotkey_id: None,
            press_started: None,
            tap_armed: false,
            _hotkey_manager: None,
        }
    }

    /// Run one command through the state machine and return the one-line
    /// reply. A dead worker is logged here and turns every reply into `err`.
    fn handle_command(&mut self, cmd: Command) -> String {
        if self.worker.is_finished() {
            eprintln!("error: The transcription worker thread ended unexpectedly");
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
        reply
    }

    /// Map a hotkey press/release to a command and run it. A hotkey has no
    /// requester, so the reply is dropped; genuine failures are already
    /// logged by [`run_action`].
    fn handle_hotkey(&mut self, event: GlobalHotKeyEvent) {
        if Some(event.id) != self.hotkey_id {
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
}

impl ApplicationHandler<UserEvent> for DaemonApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // No windows, no polling: sleep until a socket, worker or hotkey
        // event arrives.
        event_loop.set_control_flow(ControlFlow::Wait);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        _event: WindowEvent,
    ) {
        // The daemon never creates a window.
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
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
            }
            UserEvent::HotKey(event) => self.handle_hotkey(event),
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
                ok_reply.to_string()
            }
            Err(err) => {
                eprintln!("error: {err:#}");
                format!("err {err:#}")
            }
        },
        Action::StopRecording => match recorder.take() {
            Some(started) => match started.stop() {
                Ok(samples) => match to_worker.send(samples) {
                    Ok(()) => {
                        *state = new_state;
                        ok_reply.to_string()
                    }
                    Err(err) => {
                        *state = State::Idle;
                        let message = format!("Could not queue audio for transcription: {err}");
                        eprintln!("error: {message}");
                        format!("err {message}")
                    }
                },
                Err(err) => {
                    *state = State::Idle;
                    eprintln!("error: {err:#}");
                    format!("err {err:#}")
                }
            },
            None => {
                *state = State::Idle;
                eprintln!("error: the daemon was recording but its recorder is gone");
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
            eprintln!("Recording too short ({ms} ms); nothing to transcribe.");
        } else {
            let start = Instant::now();
            match asr.transcribe(&samples) {
                Ok(text) => {
                    if crate::asr::is_empty_transcript(&text) {
                        eprintln!("{}", crate::asr::no_speech_message(seconds));
                    } else {
                        type_and_log(&text, seconds, start, typer.as_mut());
                    }
                }
                Err(err) => eprintln!("error: {err:#}"),
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
