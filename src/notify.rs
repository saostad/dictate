//! Desktop notifications for failures the user would otherwise never see.
//!
//! The daemon and the client commands run from compositor keybinds and
//! autostart, where nobody reads stderr. Failures are still logged there, and
//! additionally shown as a desktop notification unless `DICTATE_NOTIFY=0`.
//! Showing runs on a spawned thread so it never blocks the event loop or the
//! worker (the short-lived client commands use a blocking call instead, so
//! the notification is delivered before the process exits), and notification
//! failures only warn once per process.

use std::sync::atomic::{AtomicBool, Ordering};

/// Severity of a notification: a critical error, or a passing info.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Error,
    Info,
}

/// Set once the first notification failure has been reported.
static WARNED: AtomicBool = AtomicBool::new(false);

/// Body of the typing-failure notification: the error plus the untyped text,
/// so the text can be recovered from the notification.
pub(crate) fn typing_failure_body(err: &str, text: &str) -> String {
    format!("{err}\n\nText: {text}")
}

/// The effective notification switch (file + env, resolved at startup).
/// Both `show` and `show_now` share this check.
fn enabled_now() -> bool {
    crate::config::notifications_on()
}

/// Warn about a notification failure, at most once per process.
fn warn_once(err: &anyhow::Error) {
    if !WARNED.swap(true, Ordering::SeqCst) {
        let message = format!("warning: could not show desktop notification: {err:#}");
        eprintln!("{message}");
        crate::daemon::mirror_to_log(&message);
    }
}

/// Show a desktop notification, unless disabled. Returns immediately; failures
/// warn once per process and never reach the caller.
pub(crate) fn show(kind: Kind, summary: &str, body: &str) {
    if !enabled_now() {
        return;
    }
    let owned = (summary.to_owned(), body.to_owned());
    std::thread::spawn(move || {
        if let Err(err) = show_blocking(kind, &owned.0, &owned.1) {
            warn_once(&err);
        }
    });
}

/// Show a desktop notification even when notifications are disabled. Only
/// for user-action confirmations the menu can't show otherwise: the
/// "Notifications: off" toggle itself (the confirmation of that action) and
/// mode switches (`Mode: <name>`). Returns immediately; failures warn once per
/// process.
pub(crate) fn show_forced(kind: Kind, summary: &str, body: &str) {
    let owned = (summary.to_owned(), body.to_owned());
    std::thread::spawn(move || {
        if let Err(err) = show_blocking(kind, &owned.0, &owned.1) {
            warn_once(&err);
        }
    });
}

/// Fixed id for the Live-download progress notification: re-showing with
/// the same id replaces the previous milestone instead of stacking (notify-rust
/// `Notification::id`, supported on XDG, Windows and non-preview macOS).
/// Large enough to never collide with server-issued ids.
const LIVE_DOWNLOAD_ID: u32 = 0xD1C7_A7E5;

/// Show the Live-download progress, replacing the previous milestone.
/// Blocking: the download thread is already a background thread, and blocking
/// keeps milestones ordered. Respects the notification switch; failures warn
/// once per process.
pub(crate) fn show_progress(body: &str) {
    if !enabled_now() {
        return;
    }
    if let Err(err) = show_progress_blocking(body) {
        warn_once(&err);
    }
}

/// Show one progress notification on the calling thread.
fn show_progress_blocking(body: &str) -> anyhow::Result<()> {
    let mut notification = notify_rust::Notification::new();
    notification
        .appname("dictate")
        .summary("dictate")
        .body(body)
        .timeout(notify_rust::Timeout::Milliseconds(6_000))
        .id(LIVE_DOWNLOAD_ID);
    #[cfg(all(unix, not(target_os = "macos")))]
    notification.urgency(notify_rust::Urgency::Normal);
    notification
        .show()
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    Ok(())
}

/// Show a desktop notification on the calling thread, unless disabled.
/// For short-lived processes (the client commands) that would exit before a
/// spawned thread delivers the message. Failures warn once per process.
/// Unix-only: the only caller is the `#[cfg(unix)]` socket client.
#[cfg(unix)]
pub(crate) fn show_now(kind: Kind, summary: &str, body: &str) {
    if !enabled_now() {
        return;
    }
    if let Err(err) = show_blocking(kind, summary, body) {
        warn_once(&err);
    }
}

/// Show one notification on the calling thread.
#[cfg_attr(not(all(unix, not(target_os = "macos"))), allow(unused_variables))]
fn show_blocking(kind: Kind, summary: &str, body: &str) -> anyhow::Result<()> {
    let mut notification = notify_rust::Notification::new();
    notification
        .appname("dictate")
        .summary(summary)
        .body(body)
        .timeout(notify_rust::Timeout::Milliseconds(6_000));
    #[cfg(all(unix, not(target_os = "macos")))]
    notification.urgency(match kind {
        Kind::Error => notify_rust::Urgency::Critical,
        Kind::Info => notify_rust::Urgency::Normal,
    });
    notification
        .show()
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typing_failure_body_keeps_error_and_text() {
        assert_eq!(
            typing_failure_body("oops", "hello world"),
            "oops\n\nText: hello world"
        );
    }
}
