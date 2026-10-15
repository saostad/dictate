//! Long-running dictation daemon, controlled over a Unix socket or hotkey.
//!
//! `dictate daemon` keeps the speech model loaded. `dictate
//! toggle|start|stop|cancel|status` sends one line to the socket and prints the
//! one-line reply; a global hotkey (tap to toggle, hold to talk) triggers the
//! same commands without a terminal.
//!
//! The main thread runs a windowless winit event loop. It owns the state, the
//! [`Recorder`](crate::mic::Recorder) and the hotkey manager (cpal streams are
//! not `Send` everywhere, and the manager belongs on the event-loop thread).
//! The socket accept loop runs on a thread and forwards commands as user
//! events; one worker thread owns the [`Asr`](crate::asr::Asr) and transcribes
//! audio sent over an mpsc channel.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::Context;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::WindowId;

/// A control command sent by a client (one line on the socket).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Parse one client line into a command.
pub(crate) fn parse_command(line: &str) -> Option<Command> {
    match line.trim() {
        "toggle" => Some(Command::Toggle),
        "start" => Some(Command::Start),
        "stop" => Some(Command::Stop),
        "cancel" => Some(Command::Cancel),
        "status" => Some(Command::Status),
        _ => None,
    }
}

/// The error message of a daemon reply, if it is one (`err <message>`).
fn reply_error(reply: &str) -> Option<&str> {
    if reply == "err" {
        return Some("");
    }
    reply.strip_prefix("err ")
}

/// Pure socket-path choice: the runtime dir when there is one, else the data
/// dir, joined with `dictate.sock`.
pub(crate) fn socket_path_for(runtime_dir: Option<&Path>, data_dir: &Path) -> PathBuf {
    match runtime_dir {
        Some(dir) => dir.join("dictate.sock"),
        None => data_dir.join("dictate.sock"),
    }
}

/// Path of the daemon socket.
pub(crate) fn socket_path() -> anyhow::Result<PathBuf> {
    let dirs = directories::ProjectDirs::from("", "", "dictate")
        .ok_or_else(|| anyhow::anyhow!("Could not find a home/data directory for this user"))?;
    Ok(socket_path_for(dirs.runtime_dir(), dirs.data_dir()))
}

/// Events handled on the daemon's event-loop thread.
#[derive(Debug)]
enum UserEvent {
    /// A command from a socket client; the reply goes back on the channel.
    Command(Command, mpsc::Sender<String>),
    /// The worker finished one transcription; back to Idle.
    WorkerDone,
    /// A global hotkey press or release.
    HotKey(GlobalHotKeyEvent),
}

/// Run `dictate daemon`: load the model, bind the socket, serve commands.
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
    let path = socket_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "Could not create daemon socket directory {}",
                parent.display()
            )
        })?;
    }
    let listener = bind_listener(&path)?;
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .context("Could not start the daemon event loop")?;
    let (to_worker, worker_rx) = mpsc::channel::<Vec<f32>>();
    let worker_proxy = event_loop.create_proxy();
    let worker = std::thread::spawn(move || transcribe_worker(worker_rx, worker_proxy, asr));
    let socket_proxy = event_loop.create_proxy();
    std::thread::spawn(move || serve_socket(listener, socket_proxy));
    let mut app = DaemonApp::new(to_worker, worker);
    setup_hotkey(&event_loop, &hotkey, &mut app);
    event_loop
        .run_app(&mut app)
        .context("The daemon event loop failed")?;
    remove_socket_file(&path);
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

/// Run one client command: send it to the daemon and print the reply.
pub(crate) fn client_command(cmd: &str) -> anyhow::Result<()> {
    let path = socket_path()?;
    let mut stream = UnixStream::connect(&path).map_err(|err| {
        anyhow::anyhow!(
            "Could not reach the dictate daemon at {}: {err}. Start it with \"dictate daemon\".",
            path.display()
        )
    })?;
    writeln!(stream, "{cmd}").with_context(|| {
        format!(
            "Could not send {cmd} to the dictate daemon at {}",
            path.display()
        )
    })?;
    let reader = stream.try_clone().with_context(|| {
        format!(
            "Could not read the reply from the dictate daemon at {}",
            path.display()
        )
    })?;
    let mut line = String::new();
    BufReader::new(reader)
        .read_line(&mut line)
        .with_context(|| {
            format!(
                "Could not read the reply from the dictate daemon at {}",
                path.display()
            )
        })?;
    if line.is_empty() {
        anyhow::bail!(
            "The dictate daemon at {} closed the connection without a reply.",
            path.display()
        );
    }
    let reply = line.trim();
    if let Some(message) = reply_error(reply) {
        if message.is_empty() {
            anyhow::bail!(
                "The dictate daemon at {} returned an error without details.",
                path.display()
            );
        }
        anyhow::bail!("{message}");
    }
    println!("{reply}");
    Ok(())
}

/// Bind the daemon socket, handling the already-running and stale cases.
fn bind_listener(path: &Path) -> anyhow::Result<UnixListener> {
    match UnixListener::bind(path) {
        Ok(listener) => Ok(listener),
        Err(err) if err.kind() == std::io::ErrorKind::AddrInUse => {
            if UnixStream::connect(path).is_ok() {
                anyhow::bail!(
                    "dictate daemon is already running (socket {}).",
                    path.display()
                );
            }
            std::fs::remove_file(path)
                .with_context(|| format!("Could not remove stale socket {}", path.display()))?;
            eprintln!("Removed stale socket {}", path.display());
            UnixListener::bind(path)
                .with_context(|| format!("Could not bind the daemon socket {}", path.display()))
        }
        Err(err) => {
            Err(err).with_context(|| format!("Could not bind the daemon socket {}", path.display()))
        }
    }
}

/// Delete the socket file; failures are warnings, never silent.
fn remove_socket_file(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => eprintln!(
            "warning: could not delete daemon socket {}: {err}",
            path.display()
        ),
    }
}

/// Socket accept loop, on its own thread: blocking accept (an idle daemon
/// sleeps), one command per connection, forwarded to the event loop.
fn serve_socket(listener: UnixListener, proxy: EventLoopProxy<UserEvent>) {
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => handle_connection(stream, &proxy),
            Err(err) => eprintln!("error: Could not accept a daemon connection: {err:#}"),
        }
    }
}

/// Serve one client connection: read one line, forward the command to the
/// event loop, write back the reply. Failures are logged, never fatal.
fn handle_connection(stream: UnixStream, proxy: &EventLoopProxy<UserEvent>) {
    if let Err(err) = handle_connection_inner(stream, proxy) {
        eprintln!("error: {err:#}");
    }
}

fn handle_connection_inner(
    stream: UnixStream,
    proxy: &EventLoopProxy<UserEvent>,
) -> anyhow::Result<()> {
    // `accept()` inherits O_NONBLOCK from the listener on macOS/BSD, which
    // would make `read_line` fail with WouldBlock; force blocking mode. The
    // read timeout keeps a client that sends nothing from blocking us forever.
    stream
        .set_nonblocking(false)
        .context("Could not use blocking mode on a daemon client connection")?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .context("Could not set a read timeout on a daemon client connection")?;
    let reader = stream
        .try_clone()
        .context("Could not read from a daemon client connection")?;
    let mut line = String::new();
    let bytes = BufReader::new(reader)
        .read_line(&mut line)
        .context("Could not read from a daemon client connection")?;
    if bytes == 0 {
        // EOF with zero bytes: an already-running probe from a second
        // `dictate daemon`. Close it with no reply and no log.
        return Ok(());
    }
    let reply = match parse_command(&line) {
        None => {
            let text = line.trim();
            format!("err unknown command \"{text}\"")
        }
        Some(cmd) => forward_command(cmd, proxy),
    };
    let mut stream = stream;
    writeln!(stream, "{reply}").context("Could not reply to a daemon client")?;
    Ok(())
}

/// Send a command to the event loop and wait for its reply.
fn forward_command(cmd: Command, proxy: &EventLoopProxy<UserEvent>) -> String {
    let (reply_tx, reply_rx) = mpsc::channel::<String>();
    if proxy.send_event(UserEvent::Command(cmd, reply_tx)).is_err() {
        return "err daemon did not answer".to_string();
    }
    match reply_rx.recv_timeout(Duration::from_secs(2)) {
        Ok(reply) => reply,
        Err(_) => "err daemon did not answer".to_string(),
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
    fn socket_path_prefers_runtime_dir() {
        let path = socket_path_for(Some(Path::new("/run/user/1000")), Path::new("/data"));
        assert_eq!(path, PathBuf::from("/run/user/1000/dictate.sock"));
    }

    #[test]
    fn socket_path_falls_back_to_data_dir() {
        let path = socket_path_for(None, Path::new("/data"));
        assert_eq!(path, PathBuf::from("/data/dictate.sock"));
    }

    #[test]
    fn commands_parse_and_reject() {
        assert_eq!(parse_command("toggle\n"), Some(Command::Toggle));
        assert_eq!(parse_command("start"), Some(Command::Start));
        assert_eq!(parse_command("stop"), Some(Command::Stop));
        assert_eq!(parse_command("cancel"), Some(Command::Cancel));
        assert_eq!(parse_command("status"), Some(Command::Status));
        assert_eq!(parse_command("bogus"), None);
        assert_eq!(parse_command(""), None);
    }

    #[test]
    fn reply_errors_split() {
        assert_eq!(reply_error("ok recording"), None);
        assert_eq!(
            reply_error("err busy transcribing"),
            Some("busy transcribing")
        );
        assert_eq!(reply_error("err"), Some(""));
    }

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
