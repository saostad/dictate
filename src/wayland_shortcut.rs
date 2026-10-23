//! Local XDG GlobalShortcuts session. D-Bus and compositor IPC never leave
//! this machine. The worker owns the session and closes its connection on
//! cancellation; no portal request blocks the window/event-loop thread.

pub(crate) mod hyprland;

use std::collections::HashMap;
use std::future::Future;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_channel::{Receiver, Sender};
use futures_lite::{StreamExt, future};
use global_hotkey::HotKeyState;
use zbus::zvariant::{OwnedObjectPath, OwnedValue, Value};
use zbus::{Connection, MatchRule, Message, MessageStream, Proxy};

pub(crate) const APP_ID: &str = "io.github.saostad.Dictate";
pub(crate) const ACTION_ID: &str = "dictation";
const DESKTOP: &str = "org.freedesktop.portal.Desktop";
const PATH: &str = "/org/freedesktop/portal/desktop";
const INTERFACE: &str = "org.freedesktop.portal.GlobalShortcuts";
type Properties = HashMap<String, OwnedValue>;
type Shortcuts = Vec<(String, Properties)>;

#[derive(Debug)]
pub(crate) enum Event {
    Ready(Option<String>),
    Unavailable(String),
    Notice(String),
    Trigger(HotKeyState),
}

/// One connection/session per daemon. Dropping it cancels pending requests,
/// closes the bus connection, and thereby unregisters all session actions.
pub(crate) struct Portal {
    stop: Sender<()>,
    configure: Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Portal {
    pub(crate) fn start(
        preferred: String,
        previous: Option<Self>,
        emit: impl Fn(Event) + Send + 'static,
    ) -> Self {
        let (stop, stopped) = async_channel::bounded(1);
        let (configure, requests) = async_channel::bounded(1);
        let thread = std::thread::spawn(move || {
            // A replacement must wait for the previous session to disappear,
            // while the daemon's window thread stays responsive.
            if let Some(mut previous) = previous {
                previous.stop.close();
                if let Some(thread) = previous.thread.take() {
                    let _ = thread.join();
                }
            }
            async_io::block_on(async {
                if let Err(err) = install_identity() {
                    emit(Event::Unavailable(format!(
                        "Could not register Dictate: {err:#}"
                    )));
                    return;
                }
                loop {
                    if stopped.is_closed() {
                        return;
                    }
                    let attempt = async {
                        let connection =
                            deadline(async { Ok(Connection::session().await?) }, 10).await?;
                        let mut session = None;
                        let result = future::or(
                            serve(&connection, &preferred, &requests, &emit, &mut session),
                            async {
                                let _ = stopped.recv().await;
                                Ok(())
                            },
                        )
                        .await;
                        if let Some(session) = session {
                            let _ = deadline(
                                async {
                                    connection
                                        .call_method(
                                            Some(DESKTOP),
                                            &session,
                                            Some("org.freedesktop.portal.Session"),
                                            "Close",
                                            &(),
                                        )
                                        .await?;
                                    Ok(())
                                },
                                3,
                            )
                            .await;
                        }
                        // Closing the connection also cancels a request/session
                        // whose handle has not arrived yet.
                        let _ = deadline(async { Ok(connection.close().await?) }, 3).await;
                        result
                    };
                    let result = attempt.await;
                    if stopped.is_closed() || result.is_ok() {
                        return;
                    }
                    if let Err(err) = result {
                        let rejected = err.downcast_ref::<SetupRejected>().is_some();
                        emit(Event::Unavailable(format!(
                            "Global shortcut unavailable: {err:#}"
                        )));
                        // A dismissed/rejected dialog requires an explicit
                        // retry; never reopen it every five seconds.
                        if rejected {
                            return;
                        }
                    }
                    // Reconnect after portal/DBus restarts without spinning.
                    future::or(
                        async {
                            async_io::Timer::after(Duration::from_secs(5)).await;
                        },
                        async {
                            let _ = stopped.recv().await;
                        },
                    )
                    .await;
                }
            });
        });
        Self {
            stop,
            configure,
            thread: Some(thread),
        }
    }

    pub(crate) fn configure(&self) {
        let _ = self.configure.try_send(());
    }

    pub(crate) fn shutdown(mut self) {
        self.stop.close();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Portal {
    fn drop(&mut self) {
        self.stop.close();
    }
}

async fn deadline<T>(task: impl Future<Output = Result<T>>, seconds: u64) -> Result<T> {
    future::or(task, async {
        async_io::Timer::after(Duration::from_secs(seconds)).await;
        Err(anyhow!(
            "The desktop portal did not answer within {seconds} seconds"
        ))
    })
    .await
}

async fn signals(connection: &Connection, interface: &str, sender: &str) -> Result<MessageStream> {
    let rule = MatchRule::builder()
        .msg_type(zbus::message::Type::Signal)
        .sender(sender)?
        .interface(interface)?
        .build();
    Ok(MessageStream::for_match_rule(rule, connection, Some(64)).await?)
}

async fn response(stream: &mut MessageStream, request: &OwnedObjectPath) -> Result<Properties> {
    deadline(
        async {
            while let Some(message) = stream.next().await {
                let message = message?;
                let header = message.header();
                if header
                    .path()
                    .is_some_and(|path| path.as_str() == request.as_str())
                    && header
                        .member()
                        .is_some_and(|member| member.as_str() == "Response")
                {
                    let (code, data): (u32, Properties) = message.body().deserialize()?;
                    return match code {
                        0 => Ok(data),
                        _ => Err(SetupRejected(code).into()),
                    };
                }
            }
            bail!("The desktop portal connection closed")
        },
        120,
    )
    .await
}

#[derive(Debug)]
struct SetupRejected(u32);
impl std::fmt::Display for SetupRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 1 {
            write!(
                f,
                "Shortcut setup was cancelled in the desktop dialog; select a shortcut to retry"
            )
        } else {
            write!(
                f,
                "The desktop rejected shortcut setup (response {}); select a shortcut to retry",
                self.0
            )
        }
    }
}
impl std::error::Error for SetupRejected {}

async fn serve(
    connection: &Connection,
    preferred: &str,
    configure: &Receiver<()>,
    emit: &impl Fn(Event),
    session_handle: &mut Option<OwnedObjectPath>,
) -> Result<()> {
    let registry = Proxy::new(
        connection,
        DESKTOP,
        PATH,
        "org.freedesktop.host.portal.Registry",
    )
    .await?;
    // Older portals identify host applications themselves and have no Registry.
    if let Err(err) = registry
        .call::<_, _, ()>("Register", &(APP_ID, HashMap::<&str, Value>::new()))
        .await
    {
        match &err {
            zbus::Error::MethodError(name, _, _)
                if matches!(
                    name.as_str(),
                    "org.freedesktop.DBus.Error.UnknownInterface"
                        | "org.freedesktop.DBus.Error.UnknownMethod"
                ) => {}
            _ => return Err(err).context("Registering the desktop application identity"),
        }
    }
    let proxy = Proxy::new(connection, DESKTOP, PATH, INTERFACE).await?;
    // Subscribe BEFORE invoking methods: portals may emit Response before the
    // method return. Keep separate queues for request and session signals.
    let mut replies = signals(connection, "org.freedesktop.portal.Request", DESKTOP).await?;
    let mut events = signals(connection, INTERFACE, DESKTOP).await?;
    let mut sessions = signals(connection, "org.freedesktop.portal.Session", DESKTOP).await?;
    let owner_rule = MatchRule::builder()
        .msg_type(zbus::message::Type::Signal)
        .sender("org.freedesktop.DBus")?
        .interface("org.freedesktop.DBus")?
        .member("NameOwnerChanged")?
        .add_arg(DESKTOP)?
        .build();
    let mut owners = MessageStream::for_match_rule(owner_rule, connection, Some(4)).await?;
    let options = HashMap::from([
        ("handle_token", Value::from("dictate_create")),
        ("session_handle_token", Value::from("dictate_shortcut")),
    ]);
    let request: OwnedObjectPath = proxy.call("CreateSession", &(options,)).await?;
    let mut data = response(&mut replies, &request).await?;
    let session = data
        .remove("session_handle")
        .context("Missing portal session handle")?;
    // The portal specification historically declares this value as a string.
    let session = if let Ok(path) = <OwnedObjectPath>::try_from(session.try_clone()?) {
        path
    } else {
        OwnedObjectPath::try_from(String::try_from(session)?)?
    };
    *session_handle = Some(session.clone());
    let preferred = hyprland::shortcut(preferred)?.portal;
    let shortcuts = vec![(
        ACTION_ID,
        HashMap::from([
            (
                "description",
                Value::from("Dictate: tap to toggle, hold to talk"),
            ),
            ("preferred_trigger", Value::from(preferred.as_str())),
        ]),
    )];
    let options = HashMap::from([("handle_token", Value::from("dictate_bind"))]);
    let request: OwnedObjectPath = proxy
        .call("BindShortcuts", &(&session, shortcuts, "", options))
        .await?;
    let mut data = response(&mut replies, &request).await?;
    let shortcuts: Shortcuts = data
        .remove("shortcuts")
        .context("Missing portal shortcut list")?
        .try_into()?;
    let mut gate = PressGate::default();
    emit(Event::Ready(trigger_description(shortcuts)?));
    loop {
        enum Input {
            Signal(Option<zbus::Result<Message>>),
            Owner(Option<zbus::Result<Message>>),
            Closed(Option<zbus::Result<Message>>),
            Configure,
        }
        let input = future::or(
            future::or(async { Input::Signal(events.next().await) }, async {
                Input::Owner(owners.next().await)
            }),
            future::or(async { Input::Closed(sessions.next().await) }, async {
                let _ = configure.recv().await;
                Input::Configure
            }),
        )
        .await;
        match input {
            Input::Signal(Some(message)) => {
                let message = message?;
                let header = message.header();
                match header.member().map(|member| member.as_str()) {
                    Some("Activated" | "Deactivated") => {
                        let (source, action, _, _): (OwnedObjectPath, String, u64, Properties) =
                            message.body().deserialize()?;
                        if let Some(state) = gate.feed(
                            source.as_str(),
                            session.as_str(),
                            &action,
                            header
                                .member()
                                .is_some_and(|member| member.as_str() == "Activated"),
                        ) {
                            emit(Event::Trigger(state));
                        }
                    }
                    Some("ShortcutsChanged") => {
                        let (source, shortcuts): (OwnedObjectPath, Shortcuts) =
                            message.body().deserialize()?;
                        if source == session {
                            emit(Event::Ready(trigger_description(shortcuts)?));
                        }
                    }
                    _ => {}
                }
            }
            Input::Owner(Some(message)) => {
                let message = message?;
                if message
                    .header()
                    .member()
                    .is_some_and(|m| m.as_str() == "NameOwnerChanged")
                {
                    let (name, _, _): (String, String, String) = message.body().deserialize()?;
                    if name == DESKTOP {
                        bail!("The desktop portal restarted; reconnecting");
                    }
                }
            }
            Input::Closed(Some(message)) => {
                let message = message?;
                if message
                    .header()
                    .path()
                    .is_some_and(|path| path.as_str() == session.as_str())
                {
                    bail!("The desktop closed the shortcut session; reconnecting");
                }
            }
            Input::Configure => {
                if let Err(err) = proxy
                    .call::<_, _, ()>(
                        "ConfigureShortcuts",
                        &(&session, "", HashMap::<&str, Value>::new()),
                    )
                    .await
                {
                    emit(Event::Notice(format!(
                        "Open your desktop's Keyboard Shortcuts settings to change the binding: {err}"
                    )));
                }
            }
            _ => bail!("The desktop portal disconnected; reconnecting"),
        }
    }
}

fn trigger_description(shortcuts: Shortcuts) -> Result<Option<String>> {
    let (_, mut data) = shortcuts
        .into_iter()
        .find(|(id, _)| id == ACTION_ID)
        .context("The desktop did not register the dictation action")?;
    Ok(data
        .remove("trigger_description")
        .and_then(|value| String::try_from(value).ok())
        .filter(|value| !value.trim().is_empty()))
}

/// Ignore foreign sessions/actions, duplicate keydown, and unmatched release.
#[derive(Default)]
struct PressGate {
    down: bool,
}
impl PressGate {
    fn feed(
        &mut self,
        source: &str,
        session: &str,
        action: &str,
        down: bool,
    ) -> Option<HotKeyState> {
        if source != session || action != ACTION_ID || self.down == down {
            return None;
        }
        self.down = down;
        Some(if down {
            HotKeyState::Pressed
        } else {
            HotKeyState::Released
        })
    }
}

fn install_identity() -> Result<()> {
    let dirs = directories::BaseDirs::new()
        .context("Could not find the desktop applications directory")?;
    let path = dirs
        .data_dir()
        .join("applications")
        .join(format!("{APP_ID}.desktop"));
    const MANAGED: &str = "X-Dictate-Managed=true";
    if path.exists()
        && !std::fs::read_to_string(&path)?
            .lines()
            .any(|line| line == MANAGED)
    {
        // Packaged desktop entries are owned by the package manager/user.
        return Ok(());
    }
    std::fs::create_dir_all(path.parent().unwrap())?;
    // A matching .desktop basename is required by the host portal Registry.
    // No autostart is enabled by this application identity entry.
    let exe = std::env::current_exe()?;
    let exe = exe.to_str().context("The executable path is not UTF-8")?;
    let mut quoted = String::from("\"");
    for c in exe.chars() {
        if matches!(c, '\\' | '"' | '`' | '$') {
            quoted.push('\\');
        }
        if matches!(c, '\n' | '\r') {
            bail!("Unsupported newline in executable path");
        }
        if c == '%' {
            quoted.push('%');
        }
        quoted.push(c);
    }
    quoted.push('"');
    let text = format!(
        "[Desktop Entry]\nType=Application\nName=Dictate\nComment=Local voice typing\nExec={quoted} settings\nTerminal=false\nNoDisplay=true\n{MANAGED}\n"
    );
    use std::io::Write;
    let mut file = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
    file.write_all(text.as_bytes())?;
    file.as_file().sync_all()?;
    file.persist(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn portal_events_require_session_action_and_balanced_edges() {
        let mut gate = PressGate::default();
        assert_eq!(gate.feed("foreign", "ours", ACTION_ID, true), None);
        assert_eq!(gate.feed("ours", "ours", "other", true), None);
        assert_eq!(gate.feed("ours", "ours", ACTION_ID, false), None);
        assert_eq!(
            gate.feed("ours", "ours", ACTION_ID, true),
            Some(HotKeyState::Pressed)
        );
        assert_eq!(gate.feed("ours", "ours", ACTION_ID, true), None);
        assert_eq!(gate.feed("foreign", "ours", ACTION_ID, false), None);
        assert_eq!(
            gate.feed("ours", "ours", ACTION_ID, false),
            Some(HotKeyState::Released)
        );
        assert_eq!(gate.feed("ours", "ours", ACTION_ID, false), None);
    }
    #[test]
    fn only_actual_portal_description_is_reported() {
        let data = HashMap::from([(
            "trigger_description".into(),
            OwnedValue::from(zbus::zvariant::Str::from("Ctrl+Alt+D")),
        )]);
        assert_eq!(
            trigger_description(vec![(ACTION_ID.into(), data)])
                .unwrap()
                .as_deref(),
            Some("Ctrl+Alt+D")
        );
        assert_eq!(
            trigger_description(vec![(ACTION_ID.into(), HashMap::new())]).unwrap(),
            None
        );
        assert!(trigger_description(vec![("other".into(), HashMap::new())]).is_err());
    }

    #[test]
    #[ignore = "requires a running desktop GlobalShortcuts portal"]
    fn native_portal_registration_and_cleanup() {
        let (tx, rx) = std::sync::mpsc::channel();
        let portal = Portal::start(crate::hotkey::DEFAULT_HOTKEY.into(), None, move |event| {
            let _ = tx.send(event);
        });
        match rx
            .recv_timeout(Duration::from_secs(20))
            .expect("portal response")
        {
            Event::Ready(trigger) => eprintln!("Real portal registered: {trigger:?}"),
            event => panic!("Portal did not register: {event:?}"),
        }
        if crate::hotkey::hyprland_session() {
            let output = std::process::Command::new("hyprctl")
                .arg("globalshortcuts")
                .output()
                .unwrap();
            let text = String::from_utf8(output.stdout).unwrap();
            assert!(text.contains(APP_ID) && text.contains(ACTION_ID), "{text}");
        }
        drop(portal);
        // Sender disappears only after the worker has closed its connection.
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(5)),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
        ));
    }
}
