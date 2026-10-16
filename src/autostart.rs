//! Start dictate at login: a `.desktop` entry on Linux/BSD, a LaunchAgent
//! plist on macOS, a `Run` registry value on Windows.
//!
//! The per-OS string builders and parsers are pure (unit-tested on every OS;
//! only the small `enable`/`disable`/`status` glue touches the OS).

use anyhow::Context;

/// Whether start-at-login is on, what the OS entry launches, and whether that
/// entry points at a different executable than this one.
pub(crate) struct AutostartStatus {
    pub enabled: bool,
    pub target: Option<String>,
    pub stale: bool,
}

#[cfg(all(unix, not(target_os = "macos")))]
use desktop as imp;
#[cfg(target_os = "macos")]
use macos as imp;
#[cfg(target_os = "windows")]
use windows as imp;

/// Turn start-at-login on for this executable (rewrites any existing entry).
pub(crate) fn enable() -> anyhow::Result<()> {
    imp::enable()
}

/// Turn start-at-login off (already-off is not an error).
pub(crate) fn disable() -> anyhow::Result<()> {
    imp::disable()
}

/// Whether start-at-login is on, and what the entry launches.
pub(crate) fn status() -> anyhow::Result<AutostartStatus> {
    imp::status()
}

/// The status for an entry that exists and launches `target` (`None` when the
/// entry gives no parseable target; such an entry is reported as-is and
/// `enable` rewrites it).
fn status_for_target(target: Option<String>) -> anyhow::Result<AutostartStatus> {
    let stale = match &target {
        Some(target) => {
            let exe =
                std::env::current_exe().context("Could not determine this executable's path")?;
            *target != exe.to_string_lossy().into_owned()
        }
        None => false,
    };
    Ok(AutostartStatus {
        enabled: true,
        target,
        stale,
    })
}

/// Write an autostart entry atomically (write `.tmp`, rename), creating the
/// parent directory. Mirrors `config::save_to`.
#[cfg(unix)]
fn write_entry_atomically(path: &std::path::Path, text: &str) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Could not create directory {}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .map(|name| format!("{}.tmp", name.to_string_lossy()))
        .unwrap_or_else(|| "entry.tmp".to_string());
    let tmp = path.with_file_name(file_name);
    std::fs::write(&tmp, text).with_context(|| format!("Could not write {}", tmp.display()))?;
    if let Err(err) = std::fs::rename(&tmp, path)
        .with_context(|| format!("Could not move {} to {}", tmp.display(), path.display()))
    {
        if let Err(cleanup) = std::fs::remove_file(&tmp) {
            eprintln!("warning: could not delete {}: {cleanup}", tmp.display());
        }
        return Err(err);
    }
    Ok(())
}

/// Remove an autostart entry file; a missing file is not an error.
#[cfg(unix)]
fn remove_entry_file(path: &std::path::Path) -> anyhow::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(anyhow::anyhow!(
            "Could not remove {}: {err}",
            path.display()
        )),
    }
}

/// Read an autostart entry file; a missing file means start-at-login is off.
#[cfg(unix)]
fn read_entry_file(path: &std::path::Path) -> anyhow::Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(content) => Ok(Some(content)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(anyhow::anyhow!("Could not read {}: {err}", path.display())),
    }
}

/// Quote one argument for a desktop-entry `Exec` line: wrap in double quotes,
/// escaping `\`, `"`, `` ` `` and `$` per the desktop-entry spec.
#[cfg(any(test, all(unix, not(target_os = "macos"))))]
fn quote_exec_arg(arg: &str) -> String {
    let mut out = String::with_capacity(arg.len() + 2);
    out.push('"');
    for c in arg.chars() {
        if matches!(c, '\\' | '"' | '`' | '$') {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('"');
    out
}

/// The full text of the Linux/BSD autostart desktop entry for `exe`.
#[cfg(any(test, all(unix, not(target_os = "macos"))))]
fn desktop_entry_text(exe: &str) -> String {
    format!(
        "[Desktop Entry]\nType=Application\nName=Dictate\nComment=Voice typing\nExec={} daemon\nTerminal=false\nX-GNOME-Autostart-enabled=true\n",
        quote_exec_arg(exe)
    )
}

/// The executable a desktop entry launches: its `Exec=` first token. `None`
/// when the entry has no `Exec=` line or the token is malformed.
#[cfg(any(test, all(unix, not(target_os = "macos"))))]
fn parse_desktop_exec(content: &str) -> Option<String> {
    content
        .lines()
        .find_map(|line| parse_exec_token(line.strip_prefix("Exec=")?.trim()))
}

/// The first whitespace-separated token, honoring double quotes and backslash
/// escapes (the quoting `quote_exec_arg` writes).
#[cfg(any(test, all(unix, not(target_os = "macos"))))]
fn parse_exec_token(value: &str) -> Option<String> {
    let value = value.trim_start();
    if let Some(rest) = value.strip_prefix('"') {
        let mut out = String::new();
        let mut chars = rest.chars();
        loop {
            match chars.next()? {
                '\\' => out.push(chars.next()?),
                '"' => return Some(out),
                c => out.push(c),
            }
        }
    }
    let mut out = String::new();
    let mut chars = value.chars();
    loop {
        match chars.next()? {
            '\\' => out.push(chars.next()?),
            c if c.is_whitespace() => return Some(out),
            c => out.push(c),
        }
    }
}

/// Escape XML text (`&` first, so the other entities survive).
#[cfg(any(test, target_os = "macos"))]
fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Reverse of `escape_xml` (`&amp;` last, so `&amp;lt;` unescapes once).
#[cfg(any(test, target_os = "macos"))]
fn unescape_xml(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// The full text of the macOS LaunchAgent plist. An app-bundle executable
/// runs the daemon with no arguments (see `launched_as_macos_app`), so it
/// gets `[exe]`; anything else gets `[exe, "daemon"]`.
#[cfg(any(test, target_os = "macos"))]
fn launch_agent_plist(exe: &str, app_bundle: bool) -> String {
    let daemon_arg = if app_bundle {
        String::new()
    } else {
        "        <string>daemon</string>\n".to_string()
    };
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n    <key>Label</key>\n    <string>com.saostad.dictate</string>\n    <key>ProgramArguments</key>\n    <array>\n        <string>{}</string>\n{}    </array>\n    <key>RunAtLoad</key>\n    <true/>\n    <key>ProcessType</key>\n    <string>Interactive</string>\n</dict>\n</plist>\n",
        escape_xml(exe),
        daemon_arg
    )
}

/// The executable a LaunchAgent plist launches: the first `<string>` under
/// `ProgramArguments`, XML-unescaped. `None` when missing or malformed.
#[cfg(any(test, target_os = "macos"))]
fn parse_plist_exe(content: &str) -> Option<String> {
    let args = content.find("<key>ProgramArguments</key>")?;
    let after = &content[args..];
    let start = after.find("<string>")? + "<string>".len();
    let end = after[start..].find("</string>")?;
    Some(unescape_xml(&after[start..start + end]))
}

/// The Windows `Run` value data for `exe`: `"<exe>" daemon --background`.
/// (Double quotes cannot appear in Windows paths, so no escaping is needed.)
#[cfg(any(test, target_os = "windows"))]
fn windows_run_value(exe: &str) -> String {
    format!("\"{exe}\" daemon --background")
}

/// Parse `reg query <key>` output: the value data of the `dictate REG_SZ`
/// line, if present. (A key query succeeds whether or not the value exists,
/// so absence here means off — no locale-dependent error parsing.)
#[cfg(any(test, target_os = "windows"))]
fn parse_reg_query_value(output: &str) -> Option<String> {
    for line in output.lines() {
        let mut tokens = line.split_whitespace();
        if tokens.next() != Some("dictate") || tokens.next() != Some("REG_SZ") {
            continue;
        }
        let data_at = line.find("REG_SZ")? + "REG_SZ".len();
        return Some(line[data_at..].trim().to_string());
    }
    None
}

/// The executable a `Run` value launches: its first token (quoted or bare).
/// `None` when empty or the quote is unterminated.
#[cfg(any(test, target_os = "windows"))]
fn parse_command_exe(data: &str) -> Option<String> {
    let data = data.trim_start();
    if let Some(rest) = data.strip_prefix('"') {
        let end = rest.find('"')?;
        return Some(rest[..end].to_string());
    }
    data.split_whitespace().next().map(str::to_string)
}

/// Linux/BSD: `$XDG_CONFIG_HOME/autostart/dictate.desktop`.
#[cfg(all(unix, not(target_os = "macos")))]
mod desktop {
    use anyhow::Context;

    fn entry_path() -> anyhow::Result<std::path::PathBuf> {
        let base = directories::BaseDirs::new().ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find a home directory for this user (needed for the start-at-login entry)"
            )
        })?;
        Ok(base.config_dir().join("autostart").join("dictate.desktop"))
    }

    pub(super) fn enable() -> anyhow::Result<()> {
        let path = entry_path()?;
        let exe = std::env::current_exe().context("Could not determine this executable's path")?;
        let text = super::desktop_entry_text(&exe.to_string_lossy());
        super::write_entry_atomically(&path, &text)
    }

    pub(super) fn disable() -> anyhow::Result<()> {
        super::remove_entry_file(&entry_path()?)
    }

    pub(super) fn status() -> anyhow::Result<super::AutostartStatus> {
        let Some(content) = super::read_entry_file(&entry_path()?)? else {
            return Ok(super::AutostartStatus {
                enabled: false,
                target: None,
                stale: false,
            });
        };
        super::status_for_target(super::parse_desktop_exec(&content))
    }
}

/// macOS: `~/Library/LaunchAgents/com.saostad.dictate.plist` (no launchctl
/// calls; the agent applies at next login).
#[cfg(target_os = "macos")]
mod macos {
    use anyhow::Context;

    const FILE_NAME: &str = "com.saostad.dictate.plist";

    fn entry_path() -> anyhow::Result<std::path::PathBuf> {
        let base = directories::BaseDirs::new().ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find a home directory for this user (needed for the start-at-login entry)"
            )
        })?;
        Ok(base
            .home_dir()
            .join("Library")
            .join("LaunchAgents")
            .join(FILE_NAME))
    }

    pub(super) fn enable() -> anyhow::Result<()> {
        let path = entry_path()?;
        let exe = std::env::current_exe().context("Could not determine this executable's path")?;
        let text = super::launch_agent_plist(
            &exe.to_string_lossy(),
            crate::is_app_bundle_executable(&exe),
        );
        super::write_entry_atomically(&path, &text)
    }

    pub(super) fn disable() -> anyhow::Result<()> {
        super::remove_entry_file(&entry_path()?)
    }

    pub(super) fn status() -> anyhow::Result<super::AutostartStatus> {
        let Some(content) = super::read_entry_file(&entry_path()?)? else {
            return Ok(super::AutostartStatus {
                enabled: false,
                target: None,
                stale: false,
            });
        };
        super::status_for_target(super::parse_plist_exe(&content))
    }
}

/// Windows: `HKCU\...\Run`, value `dictate`, via reg.exe.
#[cfg(target_os = "windows")]
mod windows {
    use anyhow::Context;
    use std::os::windows::process::CommandExt;

    const KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";
    const VALUE_NAME: &str = "dictate";
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    fn reg(args: &[&str]) -> anyhow::Result<std::process::Output> {
        std::process::Command::new("reg.exe")
            .args(args)
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .with_context(|| format!("Could not run reg.exe on {KEY}"))
    }

    pub(super) fn enable() -> anyhow::Result<()> {
        let exe = std::env::current_exe().context("Could not determine this executable's path")?;
        let value = super::windows_run_value(&exe.to_string_lossy());
        let output = reg(&[
            "add",
            KEY,
            "/v",
            VALUE_NAME,
            "/t",
            "REG_SZ",
            "/d",
            value.as_str(),
            "/f",
        ])?;
        if !output.status.success() {
            anyhow::bail!(
                "Could not write the start-at-login entry ({KEY}\\{VALUE_NAME}): {}",
                reg_output_text(&output)
            );
        }
        Ok(())
    }

    pub(super) fn disable() -> anyhow::Result<()> {
        if !status()?.enabled {
            return Ok(());
        }
        let output = reg(&["delete", KEY, "/v", VALUE_NAME, "/f"])?;
        if !output.status.success() {
            anyhow::bail!(
                "Could not remove the start-at-login entry ({KEY}\\{VALUE_NAME}): {}",
                reg_output_text(&output)
            );
        }
        Ok(())
    }

    pub(super) fn status() -> anyhow::Result<super::AutostartStatus> {
        let output = reg(&["query", KEY])?;
        if !output.status.success() {
            anyhow::bail!(
                "Could not read the start-at-login entry ({KEY}): {}",
                reg_output_text(&output)
            );
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let Some(data) = super::parse_reg_query_value(&text) else {
            return Ok(super::AutostartStatus {
                enabled: false,
                target: None,
                stale: false,
            });
        };
        super::status_for_target(super::parse_command_exe(&data))
    }

    /// reg.exe output (stdout plus stderr) as one line for error context.
    fn reg_output_text(output: &std::process::Output) -> String {
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            if !text.trim().is_empty() {
                text.push_str("; ");
            }
            text.push_str(stderr.trim());
        }
        let text = text.trim().to_string();
        if text.is_empty() {
            format!("reg.exe failed with {}", output.status)
        } else {
            text
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_entry_quotes_paths_with_spaces() {
        assert_eq!(
            desktop_entry_text("/opt/my apps/dictate"),
            "[Desktop Entry]\nType=Application\nName=Dictate\nComment=Voice typing\nExec=\"/opt/my apps/dictate\" daemon\nTerminal=false\nX-GNOME-Autostart-enabled=true\n"
        );
    }

    #[test]
    fn desktop_entry_escapes_special_chars() {
        let text = desktop_entry_text("/opt/a$b\\c`d\"e/dictate");
        let exec = text.lines().find(|line| line.starts_with("Exec=")).unwrap();
        assert_eq!(exec, "Exec=\"/opt/a\\$b\\\\c\\`d\\\"e/dictate\" daemon");
        // ... and the parser reads it back exactly.
        assert_eq!(
            parse_desktop_exec(&text).as_deref(),
            Some("/opt/a$b\\c`d\"e/dictate")
        );
    }

    #[test]
    fn desktop_exec_parse_handles_bare_and_broken_lines() {
        assert_eq!(
            parse_desktop_exec("[Desktop Entry]\nExec=/usr/bin/dictate daemon\n").as_deref(),
            Some("/usr/bin/dictate")
        );
        assert_eq!(
            parse_desktop_exec("[Desktop Entry]\nExec=\"/opt/a b/d\" daemon\n").as_deref(),
            Some("/opt/a b/d")
        );
        assert_eq!(parse_desktop_exec("[Desktop Entry]\nName=x\n"), None);
        assert_eq!(parse_desktop_exec("Exec=\"/unterminated daemon\n"), None);
        assert_eq!(parse_desktop_exec("Exec=\n"), None);
    }

    #[test]
    fn plist_has_both_argument_forms() {
        let plain = launch_agent_plist("/Users/me/.local/bin/dictate", false);
        assert!(plain.contains("<key>Label</key>\n    <string>com.saostad.dictate</string>"));
        assert!(plain.contains(
            "<string>/Users/me/.local/bin/dictate</string>\n        <string>daemon</string>"
        ));
        assert!(plain.contains("<key>RunAtLoad</key>\n    <true/>"));
        assert!(plain.contains("<key>ProcessType</key>\n    <string>Interactive</string>"));
        let app = launch_agent_plist("/Applications/Dictate.app/Contents/MacOS/dictate", true);
        assert!(
            app.contains("<string>/Applications/Dictate.app/Contents/MacOS/dictate</string>\n")
        );
        assert!(!app.contains("<string>daemon</string>"));
    }

    #[test]
    fn plist_escapes_and_parses_back() {
        let exe = "/Users/a&b/<dir>/dictate\"x'y";
        let text = launch_agent_plist(exe, false);
        assert!(text.contains("<string>/Users/a&amp;b/&lt;dir&gt;/dictate&quot;x&apos;y</string>"));
        assert_eq!(parse_plist_exe(&text).as_deref(), Some(exe));
        assert_eq!(parse_plist_exe("<dict></dict>"), None);
        // The Label string before ProgramArguments is not mistaken for the exe.
        assert_eq!(
            parse_plist_exe(&launch_agent_plist("/x/dictate", false)).as_deref(),
            Some("/x/dictate")
        );
    }

    #[test]
    fn windows_run_value_shape() {
        assert_eq!(
            windows_run_value("C:\\tools\\dictate.exe"),
            "\"C:\\tools\\dictate.exe\" daemon --background"
        );
    }

    #[test]
    fn reg_query_parser_finds_the_dictate_line() {
        let output = "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\r\n    dictate    REG_SZ    \"C:\\tools\\dictate.exe\" daemon --background\r\n    OneDrive    REG_SZ    \"C:\\Program Files\\x.exe\" /background\r\n";
        assert_eq!(
            parse_reg_query_value(output).as_deref(),
            Some("\"C:\\tools\\dictate.exe\" daemon --background")
        );
        assert_eq!(
            parse_command_exe(&parse_reg_query_value(output).unwrap()).as_deref(),
            Some("C:\\tools\\dictate.exe")
        );
    }

    #[test]
    fn reg_query_parser_rejects_absent_and_foreign_lines() {
        let output = "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\r\n    OneDrive    REG_SZ    \"C:\\x.exe\"\r\n";
        assert_eq!(parse_reg_query_value(output), None);
        let other_type = "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\r\n    dictate    REG_EXPAND_SZ    \"%X%\\dictate.exe\"\r\n";
        assert_eq!(parse_reg_query_value(other_type), None);
        assert_eq!(parse_reg_query_value(""), None);
    }

    #[test]
    fn command_exe_handles_bare_and_broken_values() {
        assert_eq!(
            parse_command_exe("C:\\tools\\dictate.exe daemon").as_deref(),
            Some("C:\\tools\\dictate.exe")
        );
        assert_eq!(parse_command_exe(""), None);
        assert_eq!(parse_command_exe("\"C:\\unterminated"), None);
    }

    #[test]
    fn status_for_target_compares_with_current_exe() {
        let current = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let fresh = status_for_target(Some(current)).unwrap();
        assert!(fresh.enabled);
        assert!(!fresh.stale);
        let stale = status_for_target(Some("/definitely/not/the/test/binary".to_string())).unwrap();
        assert!(stale.enabled);
        assert!(stale.stale);
        let unknown = status_for_target(None).unwrap();
        assert!(unknown.enabled);
        assert!(!unknown.stale);
    }
}
