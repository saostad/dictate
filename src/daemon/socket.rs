//! Unix-socket control for the daemon (Unix only).
//!
//! `dictate toggle|start|stop|cancel|status` sends one line to the socket and
//! prints the one-line reply. The accept loop runs on its own thread and
//! forwards commands to the event loop in [`super`].

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::Context;
use winit::event_loop::EventLoopProxy;

use super::{Command, UserEvent};

/// Parse one client line into a command.
fn parse_command(line: &str) -> Option<Command> {
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

/// Run one client command: send it to the daemon and print the reply.
/// Failures (except `status`) also pop a desktop notification, since keybind
/// runs show no stderr; the error is still returned for the exit code.
pub(crate) fn client_command(cmd: &str) -> anyhow::Result<()> {
    // Short-lived process: no settings file, but DICTATE_NOTIFY=0 must apply.
    crate::config::init_effective(&crate::config::Settings::default());
    if let Err(err) = client_command_inner(cmd) {
        if cmd != "status" {
            crate::notify::show_now(crate::notify::Kind::Error, "dictate", &format!("{err:#}"));
        }
        return Err(err);
    }
    Ok(())
}

fn client_command_inner(cmd: &str) -> anyhow::Result<()> {
    let path = socket_path()?;
    let mut stream = match UnixStream::connect(&path) {
        Ok(stream) => stream,
        Err(err)
            if cmd != "status"
                && matches!(
                    err.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                ) =>
        {
            autostart_and_connect(&path)?
        }
        Err(err) => {
            return Err(anyhow::anyhow!(
                "Could not reach the dictate daemon at {}: {err}. Start it with \"dictate daemon\".",
                path.display()
            ));
        }
    };
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

/// Start the daemon on demand and connect to it: spawn `dictate daemon`
/// detached with its output appended to the daemon log, notify, then poll
/// the socket every 200 ms for up to 30 s.
fn autostart_and_connect(path: &Path) -> anyhow::Result<UnixStream> {
    let log_path = super::daemon_log_path()?;
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("Could not create daemon log directory {}", parent.display())
        })?;
    }
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("Could not open daemon log {}", log_path.display()))?;
    let out = log
        .try_clone()
        .with_context(|| format!("Could not open daemon log {}", log_path.display()))?;
    let exe =
        std::env::current_exe().context("Could not find the dictate binary to start the daemon")?;
    std::process::Command::new(exe)
        .arg("daemon")
        .stdin(std::process::Stdio::null())
        .stdout(out)
        .stderr(log)
        .env("DICTATE_STDERR_IS_LOG", "1")
        .process_group(0)
        .spawn()
        .context("Could not start the dictate daemon")?;
    crate::notify::show_now(
        crate::notify::Kind::Info,
        "dictate",
        "Starting dictate (loading the speech model)…",
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match UnixStream::connect(path) {
            Ok(stream) => return Ok(stream),
            Err(_) => {
                if std::time::Instant::now() >= deadline {
                    anyhow::bail!(
                        "The dictate daemon did not start within 30 s. See its log: {}",
                        log_path.display()
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
    }
}

/// Bind the daemon socket, handling the already-running and stale cases.
pub(crate) fn bind_listener(path: &Path) -> anyhow::Result<UnixListener> {
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
            dlog!("Removed stale socket {}", path.display());
            UnixListener::bind(path)
                .with_context(|| format!("Could not bind the daemon socket {}", path.display()))
        }
        Err(err) => {
            Err(err).with_context(|| format!("Could not bind the daemon socket {}", path.display()))
        }
    }
}

/// Delete the socket file; failures are warnings, never silent.
pub(crate) fn remove_socket_file(path: &Path) {
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => dlog!(
            "warning: could not delete daemon socket {}: {err}",
            path.display()
        ),
    }
}

/// Socket accept loop, on its own thread: blocking accept (an idle daemon
/// sleeps), one command per connection, forwarded to the event loop.
pub(super) fn serve_socket(listener: UnixListener, proxy: EventLoopProxy<UserEvent>) {
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => handle_connection(stream, &proxy),
            Err(err) => dlog!("error: Could not accept a daemon connection: {err:#}"),
        }
    }
}

/// Serve one client connection: read one line, forward the command to the
/// event loop, write back the reply. Failures are logged, never fatal.
fn handle_connection(stream: UnixStream, proxy: &EventLoopProxy<UserEvent>) {
    if let Err(err) = handle_connection_inner(stream, proxy) {
        dlog!("error: {err:#}");
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

/// How long the accept thread waits for the event loop's reply. Starts open
/// the microphone, which can take a moment; everything else answers at once.
fn reply_timeout(cmd: Command) -> Duration {
    match cmd {
        Command::Start | Command::Toggle => Duration::from_secs(5),
        Command::Stop | Command::Cancel | Command::Status => Duration::from_secs(2),
    }
}

/// Send a command to the event loop and wait for its reply.
fn forward_command(cmd: Command, proxy: &EventLoopProxy<UserEvent>) -> String {
    let (reply_tx, reply_rx) = mpsc::channel::<String>();
    if proxy.send_event(UserEvent::Command(cmd, reply_tx)).is_err() {
        return "err daemon did not answer".to_string();
    }
    match reply_rx.recv_timeout(reply_timeout(cmd)) {
        Ok(reply) => reply,
        Err(_) => "err daemon did not answer".to_string(),
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
    fn starts_wait_longer_for_a_reply() {
        assert_eq!(reply_timeout(Command::Start), Duration::from_secs(5));
        assert_eq!(reply_timeout(Command::Toggle), Duration::from_secs(5));
        assert_eq!(reply_timeout(Command::Stop), Duration::from_secs(2));
        assert_eq!(reply_timeout(Command::Cancel), Duration::from_secs(2));
        assert_eq!(reply_timeout(Command::Status), Duration::from_secs(2));
    }
}
