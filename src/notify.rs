//! Desktop notifications for failures the user would otherwise never see.
//!
//! The daemon and the client commands run from compositor keybinds and
//! autostart, where nobody reads stderr. Failures are still logged there, and
//! additionally shown as a desktop notification unless `DICTATE_NOTIFY=0`.
//! Showing runs on a spawned thread so it never blocks the event loop or the
//! worker (the short-lived client commands use a blocking call instead, so
//! the notification is delivered before the process exits), and notification
//! failures only warn once per process.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

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

/// Fixed id for background-status notifications (download progress, loading,
/// transcribing): re-showing with the same id replaces the previous status
/// instead of stacking (notify-rust `Notification::id`, supported on XDG,
/// Windows and non-preview macOS). Large enough to never collide with
/// server-issued ids.
const STATUS_ID: u32 = 0xD1C7_A7E5;

/// Show the download progress, replacing the previous status.
/// Blocking: the download thread is already a background thread, and blocking
/// keeps milestones ordered. Respects the notification switch; failures warn
/// once per process.
pub(crate) fn show_progress(body: &str) {
    if !enabled_now() {
        return;
    }
    if let Err(err) = show_status_blocking(body) {
        warn_once(&err);
    }
}

/// A status notification that appears only if the work outlasts its delay.
/// Dropping it before then cancels the notification.
pub(crate) struct DelayedStatus(Arc<AtomicBool>);

impl Drop for DelayedStatus {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Show `body` as a status after `delay`, unless the returned guard is
/// dropped first. Keeps quick operations silent while slow ones explain the
/// wait.
pub(crate) fn delayed_status(delay: Duration, body: &str) -> DelayedStatus {
    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancelled);
    let body = body.to_owned();
    std::thread::spawn(move || {
        std::thread::sleep(delay);
        if flag.load(Ordering::SeqCst) || !enabled_now() {
            return;
        }
        if let Err(err) = show_status_blocking(&body) {
            warn_once(&err);
        }
    });
    DelayedStatus(cancelled)
}

/// Show one status notification on the calling thread.
fn show_status_blocking(body: &str) -> anyhow::Result<()> {
    let mut notification = notify_rust::Notification::new();
    notification
        .appname("dictate")
        .summary("dictate")
        .body(body)
        .timeout(notify_rust::Timeout::Milliseconds(6_000))
        .id(STATUS_ID);
    #[cfg(all(unix, not(target_os = "macos")))]
    notification.urgency(notify_rust::Urgency::Normal);
    notification
        .show()
        .map(|_| ())
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    Ok(())
}

/// Show a desktop notification on the calling thread, unless disabled.
/// For short-lived processes (the client commands, a daemon that exits at
/// start) that would exit before a spawned thread delivers the message.
/// Failures warn once per process.
pub(crate) fn show_now(kind: Kind, summary: &str, body: &str) {
    if !enabled_now() {
        return;
    }
    if let Err(err) = show_blocking(kind, summary, body) {
        warn_once(&err);
    }
}

/// Notification timeout in milliseconds: errors stay a bit longer than
/// info. Both use Normal urgency: Critical notifications stay until
/// dismissed on some desktops and would pile up over the screen.
fn timeout_ms(kind: Kind) -> u32 {
    match kind {
        Kind::Error => 8_000,
        Kind::Info => 6_000,
    }
}

/// Show one notification on the calling thread.
fn show_blocking(kind: Kind, summary: &str, body: &str) -> anyhow::Result<()> {
    let mut notification = notify_rust::Notification::new();
    notification
        .appname("dictate")
        .summary(summary)
        .body(body)
        .timeout(notify_rust::Timeout::Milliseconds(timeout_ms(kind)));
    #[cfg(all(unix, not(target_os = "macos")))]
    notification.urgency(notify_rust::Urgency::Normal);
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
    fn error_notifications_stay_longer_than_info() {
        assert_eq!(timeout_ms(Kind::Error), 8_000);
        assert_eq!(timeout_ms(Kind::Info), 6_000);
    }

    #[test]
    fn typing_failure_body_keeps_error_and_text() {
        assert_eq!(
            typing_failure_body("oops", "hello world"),
            "oops\n\nText: hello world"
        );
    }
}
