//! Unix-socket control for the daemon (Unix only).
//!
//! `dictate toggle|start|stop|cancel|status` sends one line to the socket and
//! prints the one-line reply. The accept loop runs on its own thread and
//! forwards commands to the event loop in [`super`].

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
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
pub(crate) fn remove_socket_file(path: &Path) {
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
pub(super) fn serve_socket(listener: UnixListener, proxy: EventLoopProxy<UserEvent>) {
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
}
