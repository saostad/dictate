//! wtype-style text typing over `zwp_virtual_keyboard_v1` (Linux Wayland).
//!
//! Like wtype: upload one custom xkb keymap holding exactly the needed
//! characters to our own virtual keyboard, then press those keycodes with no
//! modifiers. Every character has its own key, so nothing can be dropped or
//! need a modifier.

use std::collections::HashMap;
use std::io::Write;
use std::os::fd::AsFd;

use wayland_client::{
    Connection, Dispatch, EventQueue, QueueHandle,
    protocol::{wl_keyboard, wl_registry, wl_seat},
};
use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{
    zwp_virtual_keyboard_manager_v1, zwp_virtual_keyboard_v1,
};

/// First xkb keycode used in our keymaps.
const FIRST_KEYCODE: u32 = 9;
/// Last usable xkb keycode (xkb maximum is 255).
const LAST_KEYCODE: u32 = 255;
/// Unique characters per keymap upload.
const KEYMAP_CAPACITY: usize = (LAST_KEYCODE - FIRST_KEYCODE + 1) as usize;
/// xkb keycode = evdev keycode + 8; the `key` request takes evdev codes.
const EVDEV_OFFSET: u32 = 8;

/// Compositor globals we bind: a seat and the virtual keyboard manager.
struct State {
    seat: Option<wl_seat::WlSeat>,
    manager: Option<zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1>,
}

/// One virtual keyboard on a Wayland connection, reused across `type_text`.
pub(crate) struct WaylandTyper {
    queue: EventQueue<State>,
    state: State,
    keyboard: Option<zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1>,
    time: u32,
}

impl WaylandTyper {
    /// Connect to the compositor and create one virtual keyboard.
    pub(crate) fn new() -> anyhow::Result<Self> {
        let display = std::env::var("WAYLAND_DISPLAY").unwrap_or_default();
        let connection = Connection::connect_to_env().map_err(|err| {
            anyhow::anyhow!("Could not connect to Wayland display \"{display}\": {err}")
        })?;
        let mut state = State {
            seat: None,
            manager: None,
        };
        let mut queue: EventQueue<State> = connection.new_event_queue();
        connection.display().get_registry(&queue.handle(), ());
        queue
            .roundtrip(&mut state)
            .map_err(|err| anyhow::anyhow!("Wayland roundtrip failed: {err}"))?;
        let (seat, manager) = match (state.seat, state.manager) {
            (Some(seat), Some(manager)) => (seat, manager),
            (None, _) => {
                anyhow::bail!("Could not type on Wayland: the compositor offers no seat (wl_seat)")
            }
            (_, None) => anyhow::bail!(
                "This compositor does not support virtual keyboards (zwp_virtual_keyboard_v1)"
            ),
        };
        let keyboard = manager.create_virtual_keyboard(&seat, &queue.handle(), ());
        // The seat and manager proxies must stay alive: dropping them would
        // release the binding the keyboard was created from.
        let state = State {
            seat: Some(seat),
            manager: Some(manager),
        };
        Ok(Self {
            queue,
            state,
            keyboard: Some(keyboard),
            time: 0,
        })
    }

    /// Type `text`: one custom keymap upload per batch of unique characters,
    /// then key press + release per character.
    pub(crate) fn type_text(&mut self, text: &str) -> anyhow::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        // Owned (one Arc bump per call): a borrow would conflict with the
        // timestamp counter below.
        let Some(keyboard) = self.keyboard.clone() else {
            anyhow::bail!("the Wayland virtual keyboard is gone")
        };
        for batch in split_batches(text) {
            let (keymap, keycodes) = build_keymap(batch);
            self.upload_keymap(&keyboard, &keymap)?;
            let mut keys = Vec::with_capacity(batch.chars().count());
            for ch in batch.chars() {
                let Some(code) = keycodes.get(&ch) else {
                    anyhow::bail!("no keycode for {ch:?} in the Wayland keymap")
                };
                debug_assert!(*code > EVDEV_OFFSET);
                keys.push(*code - EVDEV_OFFSET);
            }
            for key in keys {
                keyboard.key(self.next_time(), key, wl_keyboard::KeyState::Pressed as u32);
                keyboard.key(
                    self.next_time(),
                    key,
                    wl_keyboard::KeyState::Released as u32,
                );
            }
        }
        self.queue.flush().map_err(|err| {
            anyhow::anyhow!("Could not send key events to the Wayland compositor: {err}")
        })?;
        self.roundtrip()
    }

    /// Upload one keymap: write it to an anonymous file, send it, roundtrip
    /// so the key events below use it. The file must outlive the roundtrip:
    /// the fd is sent on flush.
    fn upload_keymap(
        &mut self,
        keyboard: &zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
        keymap: &str,
    ) -> anyhow::Result<()> {
        let mut file = tempfile::tempfile().map_err(|err| {
            anyhow::anyhow!("Could not create a temporary file for the Wayland keymap: {err}")
        })?;
        file.write_all(keymap.as_bytes())
            .map_err(|err| anyhow::anyhow!("Could not write the Wayland keymap: {err}"))?;
        let size = u32::try_from(keymap.len())
            .map_err(|err| anyhow::anyhow!("The Wayland keymap is too large ({err})"))?;
        keyboard.keymap(wl_keyboard::KeymapFormat::XkbV1 as u32, file.as_fd(), size);
        self.roundtrip()
    }

    /// One Wayland roundtrip (flush + dispatch).
    fn roundtrip(&mut self) -> anyhow::Result<()> {
        self.queue
            .roundtrip(&mut self.state)
            .map_err(|err| anyhow::anyhow!("Wayland roundtrip failed: {err}"))?;
        Ok(())
    }

    /// Next key-event timestamp: strictly increasing (the protocol clock has
    /// an undefined base; only order matters).
    fn next_time(&mut self) -> u32 {
        self.time = self.time.wrapping_add(1);
        self.time
    }
}

impl Drop for WaylandTyper {
    fn drop(&mut self) {
        if let Some(keyboard) = self.keyboard.take() {
            keyboard.destroy();
        }
        if self.queue.flush().is_err() {
            // The connection is closing anyway; nothing more can be done.
        }
    }
}

impl Dispatch<wl_registry::WlRegistry, ()> for State {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        (): &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_registry::Event::Global {
                name, interface, ..
            } => {
                // Version 1 of both is all we need (no seat requests).
                if interface == "wl_seat" && state.seat.is_none() {
                    state.seat = Some(registry.bind::<wl_seat::WlSeat, _, _>(name, 1, qh, ()));
                } else if interface == "zwp_virtual_keyboard_manager_v1" && state.manager.is_none()
                {
                    state.manager = Some(
                        registry.bind::<
                            zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
                            _,
                            _,
                        >(name, 1, qh, ()),
                    );
                }
            }
            wl_registry::Event::GlobalRemove { .. } => {
                // A vanished seat/manager surfaces as roundtrip/flush errors.
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for State {
    fn event(
        _state: &mut Self,
        _seat: &wl_seat::WlSeat,
        _event: wl_seat::Event,
        (): &(),
        _: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // We make no seat requests; capabilities/name are irrelevant.
    }
}

impl Dispatch<zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1, ()> for State {
    fn event(
        _state: &mut Self,
        _manager: &zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
        _event: zwp_virtual_keyboard_manager_v1::Event,
        (): &(),
        _: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // The manager sends no events.
    }
}

impl Dispatch<zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1, ()> for State {
    fn event(
        _state: &mut Self,
        _keyboard: &zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
        _event: zwp_virtual_keyboard_v1::Event,
        (): &(),
        _: &Connection,
        _qh: &QueueHandle<Self>,
    ) {
        // The keyboard sends no events.
    }
}

/// Split `text` into batches of at most [`KEYMAP_CAPACITY`] unique characters
/// each (one keymap upload per batch), on char boundaries, in order.
fn split_batches(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut batches = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut start = 0;
    for (byte, ch) in text.char_indices() {
        if !seen.contains(&ch) && seen.len() >= KEYMAP_CAPACITY {
            batches.push(&text[start..byte]);
            start = byte;
            seen.clear();
        }
        seen.insert(ch);
    }
    batches.push(&text[start..]);
    batches
}

/// Build an xkb keymap string mapping keycodes [`FIRST_KEYCODE`].. to the
/// unique characters of `batch` (first-seen order), plus the char→keycode
/// map. Keysyms are `U{:04X}` (`Return`/`Tab` for newline/tab), so no
/// modifiers are needed. Callers must keep batches within [`KEYMAP_CAPACITY`]
/// unique characters (see [`split_batches`]).
fn build_keymap(batch: &str) -> (String, HashMap<char, u32>) {
    let mut chars: Vec<char> = Vec::new();
    for ch in batch.chars() {
        if !chars.contains(&ch) {
            chars.push(ch);
        }
    }
    let mut keymap = String::from(
        "xkb_keymap {\n    xkb_keycodes \"dictate\" {\n        minimum = 8;\n        maximum = 255;\n",
    );
    let mut keycodes = HashMap::with_capacity(chars.len());
    for (index, ch) in chars.iter().enumerate() {
        let code = FIRST_KEYCODE + index as u32;
        keymap.push_str(&format!("        <K{code}> = {code};\n"));
        keycodes.insert(*ch, code);
    }
    keymap.push_str("    };\n    xkb_types \"dictate\" {\n        include \"complete\"\n    };\n    xkb_compat \"dictate\" {\n        include \"complete\"\n    };\n    xkb_symbols \"dictate\" {\n");
    for (index, ch) in chars.iter().enumerate() {
        let code = FIRST_KEYCODE + index as u32;
        let keysym = keysym_name(*ch);
        keymap.push_str(&format!("        key <K{code}> {{ [ {keysym} ] }};\n"));
    }
    keymap.push_str("    };\n};\n");
    (keymap, keycodes)
}

/// xkb keysym name for one character: `U{:04X}` (`Return`/`Tab` for `\n`/`\t`).
fn keysym_name(ch: char) -> String {
    match ch {
        '\n' => "Return".to_string(),
        '\t' => "Tab".to_string(),
        _ => format!("U{:04X}", ch as u32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keymap_assigns_keycodes_in_first_seen_order() {
        let (keymap, codes) = build_keymap("bbaac");
        assert_eq!(codes.len(), 3);
        assert_eq!(codes[&'b'], 9);
        assert_eq!(codes[&'a'], 10);
        assert_eq!(codes[&'c'], 11);
        assert!(keymap.contains("<K9> = 9;"), "missing keycode:\n{keymap}");
        assert!(
            keymap.contains("key <K9> { [ U0062 ] };"),
            "missing key:\n{keymap}"
        );
        assert!(
            keymap.contains("key <K10> { [ U0061 ] };"),
            "missing key:\n{keymap}"
        );
    }

    #[test]
    fn keymap_names_newline_tab_and_non_ascii() {
        let (keymap, codes) = build_keymap("a\n\té—");
        assert_eq!(codes[&'\n'], 10);
        assert_eq!(codes[&'\t'], 11);
        assert_eq!(codes[&'é'], 12);
        assert_eq!(codes[&'—'], 13);
        assert!(
            keymap.contains("key <K10> { [ Return ] };"),
            "missing Return:\n{keymap}"
        );
        assert!(
            keymap.contains("key <K11> { [ Tab ] };"),
            "missing Tab:\n{keymap}"
        );
        assert!(
            keymap.contains("key <K12> { [ U00E9 ] };"),
            "missing é:\n{keymap}"
        );
        assert!(
            keymap.contains("key <K13> { [ U2014 ] };"),
            "missing —:\n{keymap}"
        );
    }

    #[test]
    fn keymap_has_all_sections() {
        let (keymap, _) = build_keymap("hi");
        for piece in [
            "xkb_keymap {",
            "xkb_keycodes",
            "xkb_types",
            "xkb_compat",
            "xkb_symbols",
            "minimum = 8;",
            "maximum = 255;",
        ] {
            assert!(keymap.contains(piece), "missing {piece}:\n{keymap}");
        }
    }

    #[test]
    fn batches_split_beyond_capacity_on_char_boundaries() {
        // 300 unique chars (private-use range) → 247 + 53, chars in order.
        let text: String = (0..300)
            .map(|i| char::from_u32(0xE000 + i).unwrap())
            .collect();
        let batches = split_batches(&text);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches.concat(), text);
        assert_eq!(batches[0].chars().count(), 247);
        assert_eq!(batches[1].chars().count(), 53);
        // Repeated chars don't consume capacity.
        assert_eq!(split_batches(&"ab".repeat(1000)).len(), 1);
        assert!(split_batches("").is_empty());
    }

    #[test]
    fn empty_text_builds_an_empty_keymap() {
        let (keymap, codes) = build_keymap("");
        assert!(codes.is_empty());
        assert!(keymap.contains("xkb_keymap {"), "not a keymap:\n{keymap}");
    }

    #[test]
    fn capacity_fits_the_xkb_keycode_range() {
        assert_eq!(KEYMAP_CAPACITY, 247);
        assert_eq!(FIRST_KEYCODE + KEYMAP_CAPACITY as u32 - 1, LAST_KEYCODE);
    }
}
