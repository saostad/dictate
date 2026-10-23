//! Persistent Hyprland binding for the portal action. Only a marked block in
//! the user's config is managed; unrelated bindings are never unbound.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use global_hotkey::hotkey::{HotKey, Modifiers};
use serde_json::Value;

use super::{ACTION_ID, APP_ID};

const DESCRIPTION: &str = "Dictate voice typing (tap or hold)";
const BEGIN_LUA: &str = "-- BEGIN Dictate shortcut (managed by Dictate Settings)";
const END_LUA: &str = "-- END Dictate shortcut";
const BEGIN_CONF: &str = "# BEGIN Dictate shortcut (managed by Dictate Settings)";
const END_CONF: &str = "# END Dictate shortcut";

pub(crate) struct Shortcut {
    mask: u64,
    key: String,
    lua: String,
    conf_mods: String,
    pub(super) portal: String,
}

pub(crate) fn shortcut(display: &str) -> Result<Shortcut> {
    let parsed =
        HotKey::from_str(display).map_err(|err| anyhow::anyhow!("Invalid shortcut: {err}"))?;
    let code = format!("{:?}", parsed.key);
    let key = if let Some(letter) = code.strip_prefix("Key") {
        letter.to_ascii_lowercase()
    } else if let Some(digit) = code.strip_prefix("Digit") {
        digit.to_string()
    } else {
        match code.as_str() {
            "Space" => "space",
            "Enter" => "Return",
            "Escape" => "Escape",
            "ArrowUp" => "Up",
            "ArrowDown" => "Down",
            "ArrowLeft" => "Left",
            "ArrowRight" => "Right",
            "Backspace" => "BackSpace",
            "Tab" => "Tab",
            "Delete" => "Delete",
            "Insert" => "Insert",
            "Home" => "Home",
            "End" => "End",
            "PageUp" => "Prior",
            "PageDown" => "Next",
            "Minus" => "minus",
            "Equal" => "equal",
            "BracketLeft" => "bracketleft",
            "BracketRight" => "bracketright",
            "Backslash" => "backslash",
            "Semicolon" => "semicolon",
            "Quote" => "apostrophe",
            "Backquote" => "grave",
            "Comma" => "comma",
            "Period" => "period",
            "Slash" => "slash",
            value
                if value
                    .strip_prefix('F')
                    .and_then(|n| n.parse::<u8>().ok())
                    .is_some_and(|n| (1..=35).contains(&n)) =>
            {
                value
            }
            _ => bail!("This key is not supported by the Wayland shortcut picker: {code}"),
        }
        .to_string()
    };
    let mut mods = Vec::new();
    let mut portal = Vec::new();
    let mut mask = 0;
    for (modifier, label, trigger, bit) in [
        (Modifiers::CONTROL, "CTRL", "CTRL", 4),
        (Modifiers::ALT, "ALT", "ALT", 8),
        (Modifiers::SHIFT, "SHIFT", "SHIFT", 1),
        (Modifiers::SUPER, "SUPER", "LOGO", 64),
    ] {
        if parsed.mods.contains(modifier) {
            mods.push(label);
            portal.push(trigger);
            mask |= bit;
        }
    }
    let conf_mods = mods.join(" ");
    mods.push(&key);
    portal.push(&key);
    Ok(Shortcut {
        mask,
        lua: mods.join(" + "),
        conf_mods,
        portal: portal.join("+"),
        key,
    })
}

fn binding_matches(binding: &Value, shortcut: &Shortcut) -> bool {
    binding["modmask"].as_u64() == Some(shortcut.mask)
        && binding["key"]
            .as_str()
            .is_some_and(|key| key.eq_ignore_ascii_case(&shortcut.key))
}

fn owned(binding: &Value) -> bool {
    (binding["dispatcher"] == "global" && binding["arg"] == format!("{APP_ID}:{ACTION_ID}"))
        || (binding["dispatcher"] == "__lua" && binding["description"] == DESCRIPTION)
}

fn check_conflicts(bindings: &[Value], shortcut: &Shortcut) -> Result<()> {
    for binding in bindings {
        // Checking every submap is deliberately conservative: never take a
        // shortcut belonging to another action, including release bindings.
        if binding_matches(binding, shortcut) && !owned(binding) {
            let description = binding["description"]
                .as_str()
                .filter(|s| !s.is_empty())
                .or_else(|| binding["dispatcher"].as_str())
                .unwrap_or("another action");
            bail!(
                "This shortcut is already assigned to {description}. Choose another shortcut; the existing binding was kept."
            );
        }
    }
    Ok(())
}

fn hyprctl(args: &[&str]) -> Result<String> {
    let output = std::process::Command::new("hyprctl")
        .args(args)
        .output()
        .context("Could not run hyprctl")?;
    if !output.status.success() {
        bail!(
            "hyprctl {} failed: {}{}",
            args.join(" "),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn bindings() -> Result<Vec<Value>> {
    Ok(serde_json::from_str(&hyprctl(&["binds", "-j"])?)?)
}

pub(crate) fn active(display: &str) -> Result<bool> {
    let shortcut = shortcut(display)?;
    let bindings = bindings()?;
    check_conflicts(&bindings, &shortcut)?;
    Ok(bindings
        .iter()
        .any(|binding| owned(binding) && binding_matches(binding, &shortcut)))
}

#[derive(Clone, Copy)]
enum Syntax {
    Lua,
    Conf,
}
impl Syntax {
    fn markers(self) -> (&'static str, &'static str) {
        match self {
            Self::Lua => (BEGIN_LUA, END_LUA),
            Self::Conf => (BEGIN_CONF, END_CONF),
        }
    }
    fn block(self, shortcut: &Shortcut) -> String {
        let (begin, end) = self.markers();
        let action = format!("{APP_ID}:{ACTION_ID}");
        match self {
            Self::Lua => format!(
                "{begin}\nhl.bind(\"{}\", hl.dsp.global(\"{action}\"), {{ description = \"{DESCRIPTION}\" }})\n{end}\n",
                shortcut.lua
            ),
            Self::Conf => format!(
                "{begin}\nbindd = {}, {}, {DESCRIPTION}, global, {action}\n{end}\n",
                shortcut.conf_mods, shortcut.key
            ),
        }
    }
}

/// Refuse malformed/ambiguous markers rather than risk rewriting user text.
fn replace_block(original: &str, block: &str, syntax: Syntax) -> Result<String> {
    let (begin, end) = syntax.markers();
    let starts: Vec<_> = original.match_indices(begin).map(|(i, _)| i).collect();
    let ends: Vec<_> = original.match_indices(end).map(|(i, _)| i).collect();
    match (starts.as_slice(), ends.as_slice()) {
        ([], []) => Ok(format!(
            "{original}{}{block}",
            if original.ends_with('\n') || original.is_empty() {
                ""
            } else {
                "\n"
            }
        )),
        ([start], [end]) if start < end => {
            let after = end + syntax.markers().1.len();
            let after = after + usize::from(original[after..].starts_with('\n'));
            Ok(format!(
                "{}{block}{}",
                &original[..*start],
                &original[after..]
            ))
        }
        _ => bail!(
            "The managed Dictate shortcut block is incomplete or duplicated. Repair its BEGIN/END markers before trying again."
        ),
    }
}

fn config_path() -> Result<(PathBuf, Syntax)> {
    let dirs = directories::BaseDirs::new().context("Could not locate the Hyprland config")?;
    let dir = dirs.config_dir().join("hypr");
    let main = dir.join("hyprland.lua");
    if main.is_file() {
        let text = std::fs::read_to_string(&main)?;
        if !text.contains("require(\"hypr.bindings\")")
            && !text.contains("require('hypr.bindings')")
        {
            bail!(
                "Automatic setup requires hyprland.lua to load require(\"hypr.bindings\"). Add the Dictate global action in your compositor config instead."
            );
        }
        Ok((
            dir.join("bindings.lua")
                .canonicalize()
                .context("Could not find the user bindings.lua")?,
            Syntax::Lua,
        ))
    } else {
        let main = dir.join("hyprland.conf");
        Ok((
            main.canonicalize()
                .context("Could not find the user hyprland.conf")?,
            Syntax::Conf,
        ))
    }
}

fn atomic_write(path: &Path, content: &str) -> Result<()> {
    use std::io::Write;
    let mut temp =
        tempfile::NamedTempFile::new_in(path.parent().context("Config has no parent directory")?)?;
    temp.as_file()
        .set_permissions(std::fs::metadata(path)?.permissions())?;
    temp.write_all(content.as_bytes())?;
    temp.as_file().sync_all()?;
    temp.persist(path)
        .context("Could not atomically save the Hyprland shortcut")?;
    Ok(())
}

fn validate_reload() -> Result<()> {
    hyprctl(&["reload"])?;
    let errors = hyprctl(&["configerrors"])?;
    if !errors.trim().is_empty() && errors.trim() != "ok" {
        bail!("Hyprland reported configuration errors: {}", errors.trim());
    }
    Ok(())
}

/// Run on a background thread. Validate first, back up, change only our block,
/// reload, then verify the actual binding. Roll back our edit on any failure.
pub(crate) fn install(display: &str) -> Result<PathBuf> {
    let shortcut = shortcut(display)?;
    check_conflicts(&bindings()?, &shortcut)?;
    let errors = hyprctl(&["configerrors"])?;
    if !errors.trim().is_empty() && errors.trim() != "ok" {
        bail!(
            "Fix the existing Hyprland configuration errors before setting a shortcut: {}",
            errors.trim()
        );
    }
    let (path, syntax) = config_path()?;
    let original = std::fs::read_to_string(&path)?;
    let updated = replace_block(&original, &syntax.block(&shortcut), syntax)?;
    // Backups have unique names and never replace a previous backup.
    use std::io::Write;
    let mut backup = tempfile::Builder::new()
        .prefix("dictate-shortcut-backup-")
        .suffix(".txt")
        .tempfile_in(path.parent().unwrap())?;
    backup.write_all(original.as_bytes())?;
    backup.as_file().sync_all()?;
    let (_, backup_path) = backup.keep()?;
    // Catch edits made while we were checking the compositor/creating backup.
    if std::fs::read_to_string(&path)? != original {
        bail!("The Hyprland config changed during setup. Try again; no edit was applied.");
    }
    atomic_write(&path, &updated)?;
    let validation = validate_reload().and_then(|()| {
        if !active(display)? { bail!("Hyprland did not load the Dictate shortcut. Check that your config loads the user bindings file."); }
        Ok(())
    });
    if let Err(err) = validation {
        if std::fs::read_to_string(&path)? == updated {
            atomic_write(&path, &original)?;
            validate_reload()
                .context("The previous config was restored but Hyprland still reports errors")?;
            bail!("{err:#} The previous config was restored.");
        }
        bail!(
            "{err:#} The config was edited during validation; restore it using {} if needed.",
            backup_path.display()
        );
    }
    Ok(backup_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_and_presets_map_modifiers_and_keysyms() {
        let default = shortcut("Ctrl+Shift+Space").unwrap();
        assert_eq!(default.mask, 5);
        assert_eq!(default.lua, "CTRL + SHIFT + space");
        assert_eq!(default.portal, "CTRL+SHIFT+space");
        assert_eq!(shortcut("Ctrl+Alt+D").unwrap().portal, "CTRL+ALT+d");
        assert_eq!(shortcut("Super+F8").unwrap().mask, 64);
        for preset in crate::hotkey::shortcut_presets(false) {
            shortcut(preset).unwrap();
        }
        assert!(shortcut("Ctrl+Alt").is_err());
        assert!(shortcut("Ctrl+Shift+Space\"\nmalicious").is_err());
    }
    #[test]
    fn conflicts_cover_case_release_and_submaps_but_allow_our_binding() {
        let shortcut = shortcut("Ctrl+Shift+Space").unwrap();
        let binding = serde_json::json!({"modmask":5,"key":"SPACE","description":"Another app", "release":true,"submap":"special","dispatcher":"exec"});
        assert!(check_conflicts(&[binding], &shortcut).is_err());
        let own = serde_json::json!({"modmask":5,"key":"space","dispatcher":"global","arg":format!("{APP_ID}:{ACTION_ID}")});
        assert!(check_conflicts(&[own], &shortcut).is_ok());
        let unrelated = serde_json::json!({"modmask":4,"key":"space","dispatcher":"exec"});
        assert!(check_conflicts(&[unrelated], &shortcut).is_ok());
    }
    #[test]
    fn managed_block_is_idempotent_and_preserves_surrounding_config() {
        for syntax in [Syntax::Lua, Syntax::Conf] {
            let a = syntax.block(&shortcut("Ctrl+Shift+Space").unwrap());
            let b = syntax.block(&shortcut("F8").unwrap());
            let original = "personal binding\n";
            let installed = replace_block(original, &a, syntax).unwrap();
            assert_eq!(replace_block(&installed, &a, syntax).unwrap(), installed);
            let surrounded = format!("{installed}another binding\n");
            assert_eq!(
                replace_block(&surrounded, &b, syntax).unwrap(),
                format!("{original}{b}another binding\n")
            );
            let (begin, end) = syntax.markers();
            assert!(replace_block(begin, &a, syntax).is_err());
            assert!(replace_block(&format!("{end}\n{begin}"), &a, syntax).is_err());
            assert!(replace_block(&format!("{installed}{a}"), &b, syntax).is_err());
        }
    }
}
