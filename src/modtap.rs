//! Modifier-only shortcut listener for macOS: a listen-only CGEventTap.
//!
//! global-hotkey can't register shortcuts made of only modifier keys (like
//! Command+Option), so on macOS a background thread watches one chord through
//! a session event tap and reports presses to the daemon. The tap needs Input
//! Monitoring permission; without it [`ModifierTap::start`] explains how to
//! grant it.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use objc2_core_foundation::{CFMachPort, CFRetained, CFRunLoop, kCFRunLoopCommonModes};
use objc2_core_graphics::{
    CGEvent, CGEventFlags, CGEventMask, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement,
    CGEventTapProxy, CGEventType, CGPreflightListenEventAccess, CGRequestListenEventAccess,
};

use crate::hotkey::{ChordEvent, ChordTracker, ModifierSet};

/// The modifier-key listener: the watched chord (`None` watches nothing) plus
/// the tap thread.
pub(crate) struct ModifierTap {
    chord: Arc<Mutex<Option<ModifierSet>>>,
    /// Held so the tap thread lives as long as the tap. Never read, hence
    /// the underscore.
    _thread: JoinHandle<()>,
}

impl ModifierTap {
    /// Start the listener thread: a listen-only session tap for flags-changed
    /// and key-down events, reporting chord events to `on_event`.
    pub(crate) fn start(on_event: impl Fn(ChordEvent) + Send + 'static) -> Result<Self, String> {
        let chord = Arc::new(Mutex::new(None));
        let thread_chord = Arc::clone(&chord);
        let (ready_tx, ready_rx) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("dictate-modtap".to_string())
            .spawn(move || run_tap_thread(thread_chord, Box::new(on_event), ready_tx))
            .map_err(|err| format!("Could not start the modifier-key listener: {err}"))?;
        match ready_rx.recv_timeout(Duration::from_secs(2)) {
            Ok(Ok(())) => Ok(Self {
                chord,
                _thread: thread,
            }),
            Ok(Err(message)) => Err(message),
            Err(_) => Err(
                "Could not start the modifier-key listener: it did not start within 2 seconds."
                    .to_string(),
            ),
        }
    }

    /// Watch a new chord (`None` watches nothing). The tap thread adopts it
    /// before its next event, resetting the tracker.
    pub(crate) fn set_chord(&self, chord: Option<ModifierSet>) {
        match self.chord.lock() {
            Ok(mut current) => {
                *current = chord;
            }
            Err(err) => {
                crate::dlog!(
                    "dictate: modifier-key listener: chord lock poisoned ({err}); chord unchanged"
                );
            }
        }
    }
}

/// Callback state, owned by the tap thread: the watched chord (shared with
/// `ModifierTap`), the tracker for it, the tap itself (for re-enabling), and
/// the daemon callback.
struct TapState {
    chord: Arc<Mutex<Option<ModifierSet>>>,
    tracker: Option<ChordTracker>,
    tap: Option<CFRetained<CFMachPort>>,
    on_event: Box<dyn Fn(ChordEvent) + Send>,
}

impl TapState {
    /// Adopt the currently watched chord, rebuilding the tracker when it
    /// changed (including to or from "watch nothing").
    fn sync_chord(&mut self) {
        let current = match self.chord.lock() {
            Ok(current) => *current,
            Err(err) => {
                crate::dlog!(
                    "dictate: modifier-key listener: chord lock poisoned ({err}); keeping the previous chord"
                );
                return;
            }
        };
        let tracked = self.tracker.as_ref().map(|tracker| tracker.chord);
        if tracked != current {
            self.tracker = current.map(|chord| ChordTracker {
                chord,
                active: false,
            });
        }
    }
}

/// Tap thread body: create the tap, attach it to this thread's run loop, then
/// run it. Reports the setup outcome over `ready`.
fn run_tap_thread(
    chord: Arc<Mutex<Option<ModifierSet>>>,
    on_event: Box<dyn Fn(ChordEvent) + Send>,
    ready: mpsc::Sender<Result<(), String>>,
) {
    let mut state = Box::new(TapState {
        chord,
        tracker: None,
        tap: None,
        on_event,
    });
    // Raw pointer for the callback; the box outlives the run loop below and is
    // dropped (with the tap) on every return path.
    let state_ptr: *mut TapState = std::ptr::addr_of_mut!(*state);
    let mask: CGEventMask = (1 << CGEventType::FlagsChanged.0) | (1 << CGEventType::KeyDown.0);
    // SAFETY: the callback matches the tap signature; `state_ptr` stays valid
    // until the box drops after the run loop exits, longer than the tap.
    let tap = unsafe {
        CGEvent::tap_create(
            CGEventTapLocation::SessionEventTap,
            CGEventTapPlacement::HeadInsertEventTap,
            CGEventTapOptions::ListenOnly,
            mask,
            Some(tap_callback),
            state_ptr.cast::<c_void>(),
        )
    };
    let Some(tap) = tap else {
        let _ = ready.send(Err(tap_create_error()));
        return;
    };
    // SAFETY: `state_ptr` is alive and only this thread touches it; the
    // callback cannot run before the source joins the run loop below.
    unsafe {
        (*state_ptr).tap = Some(tap.clone());
    }
    let Some(source) = CFMachPort::new_run_loop_source(None, Some(&tap), 0) else {
        let _ = ready.send(Err(
            "Could not start the modifier-key listener: macOS refused to create the run-loop source."
                .to_string(),
        ));
        return;
    };
    let Some(run_loop) = CFRunLoop::current() else {
        let _ = ready.send(Err(
            "Could not start the modifier-key listener: no run loop on the listener thread."
                .to_string(),
        ));
        return;
    };
    // SAFETY: reading the system-provided run-loop mode constant.
    let mode = unsafe { kCFRunLoopCommonModes };
    run_loop.add_source(Some(&source), mode);
    let _ = ready.send(Ok(()));
    CFRunLoop::run();
}

/// Event-tap callback: track the watched chord and report presses. Runs on the
/// tap thread; returns the event unchanged.
unsafe extern "C-unwind" fn tap_callback(
    _proxy: CGEventTapProxy,
    event_type: CGEventType,
    event: NonNull<CGEvent>,
    user_info: *mut c_void,
) -> *mut CGEvent {
    // SAFETY: `user_info` is the tap thread's state box, alive until the run
    // loop exits, and the callback only runs on that thread while installed.
    let state = unsafe { &mut *(user_info as *mut TapState) };
    if event_type == CGEventType::TapDisabledByTimeout
        || event_type == CGEventType::TapDisabledByUserInput
    {
        if let Some(tap) = &state.tap {
            CGEvent::tap_enable(tap, true);
            crate::dlog!(
                "dictate: modifier-key listener: the event tap was disabled by macOS; re-enabled it"
            );
        }
        return event.as_ptr();
    }
    if event_type == CGEventType::FlagsChanged {
        // SAFETY: the callback event pointer is valid for the call.
        let flags = CGEvent::flags(Some(unsafe { event.as_ref() }));
        state.sync_chord();
        let chord_event = state
            .tracker
            .as_mut()
            .and_then(|tracker| tracker.on_flags(modifiers_from_flags(flags)));
        if let Some(chord_event) = chord_event {
            (state.on_event)(chord_event);
        }
    } else if event_type == CGEventType::KeyDown {
        state.sync_chord();
        let chord_event = state
            .tracker
            .as_mut()
            .and_then(|tracker| tracker.on_key_down());
        if let Some(chord_event) = chord_event {
            (state.on_event)(chord_event);
        }
    }
    event.as_ptr()
}

/// Error for a refused event tap: without Input Monitoring permission the tap
/// cannot be created, so ask for it and explain how to grant it.
fn tap_create_error() -> String {
    if CGPreflightListenEventAccess() {
        "Could not start the modifier-key listener: macOS refused to create the event tap."
            .to_string()
    } else {
        CGRequestListenEventAccess();
        "dictate needs Input Monitoring permission for modifier-only shortcuts. Open System Settings → Privacy & Security → Input Monitoring, turn on Dictate (or the terminal that runs it), then quit and reopen dictate."
            .to_string()
    }
}

/// Modifier keys held in a tap event's flags. Only the four chord modifiers
/// map; caps lock, fn and numeric-pad bits are ignored.
fn modifiers_from_flags(flags: CGEventFlags) -> ModifierSet {
    ModifierSet {
        cmd: flags.contains(CGEventFlags::MaskCommand),
        ctrl: flags.contains(CGEventFlags::MaskControl),
        option: flags.contains(CGEventFlags::MaskAlternate),
        shift: flags.contains(CGEventFlags::MaskShift),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_map_to_the_chord_modifiers() {
        let flags = CGEventFlags::MaskCommand | CGEventFlags::MaskAlternate;
        assert_eq!(
            modifiers_from_flags(flags),
            ModifierSet {
                cmd: true,
                ctrl: false,
                option: true,
                shift: false,
            }
        );
        // Caps lock, fn and numeric-pad bits are ignored.
        let flags = CGEventFlags::MaskControl
            | CGEventFlags::MaskShift
            | CGEventFlags::MaskAlphaShift
            | CGEventFlags::MaskSecondaryFn
            | CGEventFlags::MaskNumericPad;
        assert_eq!(
            modifiers_from_flags(flags),
            ModifierSet {
                cmd: false,
                ctrl: true,
                option: false,
                shift: true,
            }
        );
        assert_eq!(
            modifiers_from_flags(CGEventFlags::empty()),
            ModifierSet {
                cmd: false,
                ctrl: false,
                option: false,
                shift: false,
            }
        );
    }
}
