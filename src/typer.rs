//! Types transcribed text into the focused window via enigo.

use enigo::{Enigo, Keyboard, Settings};

pub(crate) struct Typer {
    enigo: Enigo,
}

impl Typer {
    pub(crate) fn new() -> anyhow::Result<Self> {
        let enigo = Enigo::new(&Settings::default())
            .map_err(|err| anyhow::anyhow!("Could not set up keyboard input simulation: {err}"))?;
        Ok(Self { enigo })
    }

    pub(crate) fn type_text(&mut self, text: &str) -> anyhow::Result<()> {
        self.enigo.text(text).map_err(|err| {
            anyhow::anyhow!("Could not type the text into the focused window: {err}")
        })
    }
}
