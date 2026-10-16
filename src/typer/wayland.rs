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

/// xkb keycode for `' '` (evdev 57 KEY_SPACE).
const SPACE_KEYCODE: u32 = 65;
/// xkb keycode for `'\n'` (evdev 28 KEY_ENTER).
const ENTER_KEYCODE: u32 = 36;
/// xkb keycode for `'\t'` (evdev 15 KEY_TAB).
const TAB_KEYCODE: u32 = 23;
/// Pool of xkb keycodes for every other character: ordinary character keys
/// only (evdev 2-13, 16-27, 30-41, 43-53, i.e. the main alphanumeric and
/// punctuation block). Text must never land on ESC, BACKSPACE, TAB, ENTER,
/// modifiers, CAPSLOCK, F-keys, navigation, keypad or media keys: apps that
/// interpret keys by their physical evdev code would misread them (e.g. a
/// leading space on evdev 1 reads as Escape and gets dropped).
const POOL_KEYCODES: [u32; 47] = [
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, // evdev 2-13
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, // evdev 16-27
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, // evdev 30-41
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // evdev 43-53
];
/// Unique characters per keymap upload. The fixed-code characters (space, tab,
/// newline) don't consume pool codes, so counting every unique char against
/// the pool size is conservative and always safe.
const KEYMAP_CAPACITY: usize = POOL_KEYCODES.len();
/// xkb keycode = evdev keycode + 8; the `key` request takes evdev codes.
const EVDEV_OFFSET: u32 = 8;

/// Compositor globals we bind: a seat and the virtual keyboard manager.
struct State {
    seat: Option<wl_seat::WlSeat>,
    manager: Option<zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1>,
}

/// Wayland typing state: the connection, seat and manager are reused across
/// `type_text` calls; the virtual keyboard itself is fresh per call (see
/// [`WaylandTyper::type_text`]).
pub(crate) struct WaylandTyper {
    queue: EventQueue<State>,
    state: State,
    time: u32,
}

impl WaylandTyper {
    /// Connect to the compositor and bind the seat and manager.
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
        match (&state.seat, &state.manager) {
            (Some(_), Some(_)) => {}
            (None, _) => {
                anyhow::bail!("Could not type on Wayland: the compositor offers no seat (wl_seat)")
            }
            (_, None) => anyhow::bail!(
                "This compositor does not support virtual keyboards (zwp_virtual_keyboard_v1)"
            ),
        }
        // The seat and manager proxies must stay alive: dropping them would
        // release the bindings each keyboard is created from.
        Ok(Self {
            queue,
            state,
            time: 0,
        })
    }

    /// Type `text`: one custom keymap upload per batch of unique characters,
    /// then key press + release per character.
    pub(crate) fn type_text(&mut self, text: &str) -> anyhow::Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        for batch in split_batches(text) {
            // A fresh virtual keyboard per call (and per batch): the
            // compositor applies a new keymap only after already-queued keys,
            // so reusing one keyboard decodes the next call with the previous
            // keymap. wtype avoids this with one keyboard per run.
            let (Some(seat), Some(manager)) = (self.state.seat.clone(), self.state.manager.clone())
            else {
                anyhow::bail!("the Wayland seat or virtual keyboard manager is gone")
            };
            let keyboard = manager.create_virtual_keyboard(&seat, &self.queue.handle(), ());
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
            self.queue.flush().map_err(|err| {
                anyhow::anyhow!("Could not send key events to the Wayland compositor: {err}")
            })?;
            self.roundtrip()?;
            keyboard.destroy();
            self.queue.flush().map_err(|err| {
                anyhow::anyhow!("Could not release the Wayland virtual keyboard: {err}")
            })?;
        }
        Ok(())
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

/// Build an xkb keymap string mapping the unique characters of `batch` to
/// physical-safe keycodes (first-seen order within the pool), plus the
/// char→keycode map. `' '`/`\n`/`\t` get the real KEY_SPACE/KEY_ENTER/KEY_TAB
/// codes; every other char gets a code from [`POOL_KEYCODES`]. Keysyms are the
/// xkb names for space/Return/Tab and `U{:04X}` otherwise, so no modifiers are
/// needed. Callers must keep batches within [`KEYMAP_CAPACITY`] unique
/// characters (see [`split_batches`]); a char past the end of the pool is left
/// out and fails loudly in `type_text`, which names it.
fn build_keymap(batch: &str) -> (String, HashMap<char, u32>) {
    let mut chars: Vec<char> = Vec::new();
    for ch in batch.chars() {
        if !chars.contains(&ch) {
            chars.push(ch);
        }
    }
    let mut pairs: Vec<(char, u32)> = Vec::with_capacity(chars.len());
    let mut pool = POOL_KEYCODES.iter();
    for ch in chars {
        let code = match ch {
            ' ' => SPACE_KEYCODE,
            '\n' => ENTER_KEYCODE,
            '\t' => TAB_KEYCODE,
            _ => {
                let Some(code) = pool.next() else {
                    continue;
                };
                *code
            }
        };
        pairs.push((ch, code));
    }
    let mut keymap = String::from(
        "xkb_keymap {\n    xkb_keycodes \"dictate\" {\n        minimum = 8;\n        maximum = 255;\n",
    );
    let mut keycodes = HashMap::with_capacity(pairs.len());
    for (ch, code) in &pairs {
        keymap.push_str(&format!("        <K{code}> = {code};\n"));
        keycodes.insert(*ch, *code);
    }
    keymap.push_str("    };\n    xkb_types \"dictate\" {\n        include \"complete\"\n    };\n    xkb_compat \"dictate\" {\n        include \"complete\"\n    };\n    xkb_symbols \"dictate\" {\n");
    for (ch, code) in &pairs {
        let keysym = keysym_name(*ch);
        keymap.push_str(&format!("        key <K{code}> {{ [ {keysym} ] }};\n"));
    }
    keymap.push_str("    };\n};\n");
    (keymap, keycodes)
}

/// xkb keysym name for one character: `space`/`Return`/`Tab` for `' '`/`\n`/`\t`,
/// `U{:04X}` otherwise.
fn keysym_name(ch: char) -> String {
    match ch {
        ' ' => "space".to_string(),
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
        assert_eq!(codes[&'b'], 10);
        assert_eq!(codes[&'a'], 11);
        assert_eq!(codes[&'c'], 12);
        assert!(keymap.contains("<K10> = 10;"), "missing keycode:\n{keymap}");
        assert!(
            keymap.contains("key <K10> { [ U0062 ] };"),
            "missing key:\n{keymap}"
        );
        assert!(
            keymap.contains("key <K11> { [ U0061 ] };"),
            "missing key:\n{keymap}"
        );
    }

    #[test]
    fn keymap_names_newline_tab_and_non_ascii() {
        let (keymap, codes) = build_keymap("a\n\té—");
        assert_eq!(codes[&'a'], 10);
        assert_eq!(codes[&'\n'], 36);
        assert_eq!(codes[&'\t'], 23);
        assert_eq!(codes[&'é'], 11);
        assert_eq!(codes[&'—'], 12);
        assert!(
            keymap.contains("key <K36> { [ Return ] };"),
            "missing Return:\n{keymap}"
        );
        assert!(
            keymap.contains("key <K23> { [ Tab ] };"),
            "missing Tab:\n{keymap}"
        );
        assert!(
            keymap.contains("key <K11> { [ U00E9 ] };"),
            "missing é:\n{keymap}"
        );
        assert!(
            keymap.contains("key <K12> { [ U2014 ] };"),
            "missing —:\n{keymap}"
        );
    }

    #[test]
    fn space_uses_key_space_even_when_first() {
        let (keymap, codes) = build_keymap(" abc");
        assert_eq!(codes[&' '], 65);
        assert_eq!(codes[&'a'], 10);
        assert!(
            keymap.contains("key <K65> { [ space ] };"),
            "missing space key:\n{keymap}"
        );
    }

    #[test]
    fn pool_uses_only_ordinary_character_keys() {
        // Forbidden physical keys: ESC, BACKSPACE, TAB, ENTER, modifiers,
        // CAPSLOCK, F-keys, navigation/arrows, keypad, media (evdev codes).
        const FORBIDDEN_EVDEV: &[u32] = &[
            1, 14, 15, 28, // ESC, BACKSPACE, TAB, ENTER
            29, 42, 54, 56, 97, 100, 125, 126, 127, // modifiers
            58,  // CAPSLOCK
            59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 87, 88, // F1-F12
            102, 103, 104, 105, 106, 107, 108, 109, 110, 111, // nav/arrows
            55, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 96, 98, // keypad
            113, 114, 115, 116, // media/power
        ];
        for code in POOL_KEYCODES {
            let evdev = code - EVDEV_OFFSET;
            assert!(
                !FORBIDDEN_EVDEV.contains(&evdev),
                "pool keycode {code} is forbidden evdev {evdev}"
            );
            assert!(
                (2..=13).contains(&evdev)
                    || (16..=27).contains(&evdev)
                    || (30..=41).contains(&evdev)
                    || (43..=53).contains(&evdev),
                "pool keycode {code} is outside the character block (evdev {evdev})"
            );
        }
        // The fixed codes are the one exception: each matches its own key.
        assert_eq!(SPACE_KEYCODE - EVDEV_OFFSET, 57); // KEY_SPACE
        assert_eq!(ENTER_KEYCODE - EVDEV_OFFSET, 28); // KEY_ENTER
        assert_eq!(TAB_KEYCODE - EVDEV_OFFSET, 15); // KEY_TAB
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
        // 300 unique chars (private-use range) → 6×47 + 18, chars in order.
        let text: String = (0..300)
            .map(|i| char::from_u32(0xE000 + i).unwrap())
            .collect();
        let batches = split_batches(&text);
        assert_eq!(batches.len(), 7);
        assert_eq!(batches.concat(), text);
        for batch in &batches[..6] {
            assert_eq!(batch.chars().count(), 47);
        }
        assert_eq!(batches[6].chars().count(), 18);
        // Repeated chars don't consume capacity.
        assert_eq!(split_batches(&"ab".repeat(1000)).len(), 1);
        assert!(split_batches("").is_empty());
    }

    #[test]
    fn sixty_unique_chars_split_into_two_batches() {
        let text: String = (0..60)
            .map(|i| char::from_u32(0xE100 + i).unwrap())
            .collect();
        let batches = split_batches(&text);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches.concat(), text);
        assert_eq!(batches[0].chars().count(), 47);
        assert_eq!(batches[1].chars().count(), 13);
    }

    #[test]
    fn empty_text_builds_an_empty_keymap() {
        let (keymap, codes) = build_keymap("");
        assert!(codes.is_empty());
        assert!(keymap.contains("xkb_keymap {"), "not a keymap:\n{keymap}");
    }

    #[test]
    fn keymap_capacity_matches_the_pool() {
        assert_eq!(KEYMAP_CAPACITY, 47);
        assert_eq!(POOL_KEYCODES.len(), KEYMAP_CAPACITY);
        let mut seen = std::collections::HashSet::new();
        for code in POOL_KEYCODES {
            assert!(
                (9..=255).contains(&code),
                "pool code {code} out of xkb range"
            );
            assert!(seen.insert(code), "pool code {code} duplicated");
        }
        for fixed in [SPACE_KEYCODE, ENTER_KEYCODE, TAB_KEYCODE] {
            assert!(
                (9..=255).contains(&fixed),
                "fixed code {fixed} out of xkb range"
            );
            assert!(
                !POOL_KEYCODES.contains(&fixed),
                "fixed code {fixed} collides with the pool"
            );
        }
    }
}
