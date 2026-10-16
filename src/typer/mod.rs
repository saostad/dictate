//! Types transcribed text into the focused window.
//!
//! On Wayland, enigo's key simulation drops characters (notably capitals) and
//! is slow, so Linux Wayland sessions use a wtype-style virtual keyboard
//! instead ([`wayland`]); everywhere else this wraps enigo.

#[cfg(target_os = "linux")]
mod wayland;

use enigo::{Enigo, Keyboard, Settings};

pub(crate) struct Typer {
    #[cfg(target_os = "linux")]
    wayland: Option<wayland::WaylandTyper>,
    enigo: Option<Enigo>,
}

impl Typer {
    /// `prompt` lets macOS show its Accessibility permission dialog when the
    /// permission is missing; retries pass false so it isn't shown repeatedly.
    pub(crate) fn new(prompt: bool) -> anyhow::Result<Self> {
        #[cfg(target_os = "linux")]
        if crate::hotkey::wayland_session() {
            return Ok(Self {
                wayland: Some(wayland::WaylandTyper::new()?),
                enigo: None,
            });
        }
        let settings = Settings {
            open_prompt_to_get_permissions: prompt,
            ..Settings::default()
        };
        let enigo = Enigo::new(&settings)
            .map_err(|err| anyhow::anyhow!("Could not set up keyboard input simulation: {err}"))?;
        Ok(Self {
            #[cfg(target_os = "linux")]
            wayland: None,
            enigo: Some(enigo),
        })
    }

    pub(crate) fn type_text(&mut self, text: &str) -> anyhow::Result<()> {
        #[cfg(target_os = "linux")]
        if let Some(typer) = &mut self.wayland {
            return typer.type_text(text);
        }
        match &mut self.enigo {
            Some(enigo) => enigo.text(text).map_err(|err| {
                anyhow::anyhow!("Could not type the text into the focused window: {err}")
            }),
            // Unreachable by construction (`new` sets exactly one typer).
            None => Err(anyhow::anyhow!("the keyboard typer is gone")),
        }
    }
}
