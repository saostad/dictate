# Wayland shortcuts — implementation and machine verification

Date: 2026-10-22. Checkout: `main`. Machine: AsusGL552,
Hyprland 0.56.2 on Omarchy; xdg-desktop-portal 1.22.1 and
xdg-desktop-portal-hyprland 1.4.1.

## Behavior

Settings now offers the existing shortcut presets and custom key capture on
Wayland. The local XDG GlobalShortcuts portal owns a persistent dictation
action, `io.github.saostad.Dictate:dictation`. Activated/Deactivated events
feed the daemon's existing tap/hold state machine. No subprocess is launched
per key event and no remote service is involved.

On Hyprland, selecting a preset or clicking **Enable shortcut / apply again**
checks the compositor's current bindings, backs up the user config, updates
only a marked Dictate block, reloads, checks config errors, and verifies the
resulting binding. A failed change restores the previous text unless another
process edited it during validation. Existing conflicting actions are kept.
Omarchy's Lua `bindings.lua` and legacy `hyprland.conf` are supported. Custom
Lua configurations must load `require("hypr.bindings")` for automatic setup.

The portal worker subscribes before making requests, filters session/action
IDs, ignores duplicate presses and unmatched releases, cancels pending work
on shutdown, explicitly closes its session, and reconnects after service
loss. A replacement waits for its predecessor off the window thread. User
cancellation does not repeatedly reopen desktop dialogs. Losing the portal
during a hold recording cancels that recording; completed tap recordings
remain controllable through the tray/socket.

Settings displays actual portal/compositor status separately from the saved
preference. Other Wayland desktops control the final binding through their
portal dialog. Failed attempts to open desktop shortcut settings preserve an
existing working listener. Shortcut changes are blocked during recording or
an existing shortcut setup.

## Validation

Final source checks passed:

- `cargo fmt --all -- --check`
- `cargo test --offline --quiet`: **264 passed, 2 ignored**
- `cargo test --offline --no-default-features --features system-ort --quiet`:
  **271 passed, 2 ignored**
- `cargo clippy --offline --all-targets -- -D warnings`
- `cargo clippy --offline --all-targets --no-default-features --features system-ort -- -D warnings`
- `cargo build --offline --release --locked`
- Explicit ignored `native_portal_registration_and_cleanup` test: passed.

Desktop checks:

- Real host Registry / GlobalShortcuts creation, binding, and worker shutdown
  succeeded. Native `hl.dsp.global(...)` dispatch reached the isolated test
  listener as `Trigger(Pressed)` with the recording daemon stopped.
- Switched Ctrl+Shift+Space → Ctrl+Alt+D → Ctrl+Shift+Space through the daemon's
  actual settings path. Each switch left exactly one managed compositor bind
  and saved the requested preference.
- Attempting Super+Space was rejected because it belongs to **Omarchy menu**.
  The user config was byte-for-byte unchanged and Dictate's default remained
  active. This check did not create another portal session.
- Hyprland reported no config errors after changes and reloads.
- The final release daemon started without a model-directory environment
  override, loaded the user's selected Moonshine mode, reported **Active:
  Ctrl+Shift+Space**, and repeatedly answered `ok idle`. It is left running
  in the tray with no Settings window while the desktop remains locked.
- Existing downloaded models were cloned with Btrfs reflinks into
  `~/.local/share/dictate/models`, allowing the already-enabled login entry
  to work without the earlier verification-cache environment override.

Machine-only backups are next to `~/.config/hypr/bindings.lua`, named
`dictate-shortcut-backup-*.txt`. Detailed command outputs and temporary
isolated probes are in `/tmp/dictate-shortcut-verification/`.

## Limits

The desktop was locked (`hyprctl locked` returned `true`) during attempted
keyboard injection. Hyprland correctly blocked ordinary shortcuts and window
focus changes. Physical tap/hold, microphone capture, and final text insertion
therefore remain for the user's unlocked desktop test. No lock-screen
exception was added. Native portal dispatch, the existing tap/hold unit
tests, and preset configuration were verified independently.

This installed Hyprland portal can leave native shortcut metadata visible in
`hyprctl globalshortcuts` after client termination. New sessions still
registered successfully and received native dispatch. The app explicitly
closes sessions during orderly shutdown; it does not restart the desktop's
portal services. KDE/GNOME and legacy Hyprland config setup were not tested
on their actual desktop environments.

Opening Settings while the desktop was locked stalled its renderer and the
daemon event loop. The daemon was restarted without that window and repeated
status checks passed. Open Settings from the tray after unlocking. This
existing rendering limitation was not changed as part of shortcut handling.
