//! Custom dictionary: word-aware replacements applied to dictated text just
//! before it is typed.
//!
//! Rules live in `dictionary.txt` next to `config.toml`, one per line:
//!
//! ```text
//! # Lines starting with # are comments.
//! cooper netties -> Kubernetes
//! post gress -> PostgreSQL
//! ```
//!
//! Matching ignores case and only replaces whole words: `post gress` matches
//! "Post gress" but not "compost gress". The replacement is typed exactly as
//! written. The longest matching rule wins, and replaced text is never matched
//! again. In Pauses and Live modes each typed piece is replaced on its own, so
//! a multi-word phrase split across two pieces is not replaced.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use anyhow::Context;

pub(crate) const FILE_NAME: &str = "dictionary.txt";
const SEPARATOR: &str = "->";
const TEMPLATE: &str = "\
# Dictate dictionary: one replacement per line, written as
#   what the model types -> what you want typed
# Matching ignores case and only replaces whole words. Examples:
#   cooper netties -> Kubernetes
#   post gress -> PostgreSQL
";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rule {
    pub from: String,
    pub to: String,
}

/// A line of `dictionary.txt` that is not a valid rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LineError {
    pub line: usize,
    pub message: &'static str,
}

impl std::fmt::Display for LineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{FILE_NAME} line {}: {}", self.line, self.message)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Dictionary {
    /// Patterns as lowercase chars, longest first.
    rules: Vec<(Vec<char>, Rule)>,
    /// The same rules in file order, for listing.
    ordered: Vec<Rule>,
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '\''
}

fn lower(c: char) -> char {
    // Single-char lowercase keeps the pattern and the text aligned char by
    // char; the few chars that lowercase to several (e.g. 'İ') compare as-is.
    let mut it = c.to_lowercase();
    match (it.next(), it.next()) {
        (Some(l), None) => l,
        _ => c,
    }
}

/// Parse `dictionary.txt`. Invalid lines are returned alongside the valid
/// rules so the caller can report them.
pub(crate) fn parse(text: &str) -> (Dictionary, Vec<LineError>) {
    let mut rules = Vec::new();
    let mut errors = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line_number = index + 1;
        match parse_rule(line) {
            Ok(rule) => rules.push(rule),
            Err(message) => errors.push(LineError {
                line: line_number,
                message,
            }),
        }
    }
    (Dictionary::new(rules), errors)
}

fn parse_rule(line: &str) -> Result<Rule, &'static str> {
    let Some((from, to)) = line.split_once(SEPARATOR) else {
        return Err("expected \"heard -> written\"");
    };
    let from = normalize_spaces(from);
    let to = to.trim().to_string();
    if from.is_empty() {
        return Err("nothing before \"->\"");
    }
    if to.contains(SEPARATOR) {
        return Err("more than one \"->\"");
    }
    Ok(Rule { from, to })
}

fn normalize_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl Dictionary {
    pub(crate) fn new(ordered: Vec<Rule>) -> Self {
        let mut rules: Vec<(Vec<char>, Rule)> = ordered
            .iter()
            .cloned()
            .map(|rule| (rule.from.chars().map(lower).collect(), rule))
            .collect();
        // Longest first; stable, so equal lengths keep file order.
        rules.sort_by_key(|(pattern, _)| std::cmp::Reverse(pattern.len()));
        Self { rules, ordered }
    }

    pub(crate) fn len(&self) -> usize {
        self.rules.len()
    }

    pub(crate) fn rules(&self) -> impl Iterator<Item = &Rule> {
        self.ordered.iter()
    }

    /// Replace every whole-word, case-insensitive match. A space in a pattern
    /// matches any run of whitespace in the text.
    pub(crate) fn apply(&self, text: &str) -> String {
        if self.rules.is_empty() {
            return text.to_string();
        }
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::with_capacity(text.len());
        let mut i = 0;
        while i < chars.len() {
            let matched = self
                .rules
                .iter()
                .find_map(|(pattern, rule)| match_at(&chars, i, pattern).map(|end| (end, rule)));
            match matched {
                Some((mut end, rule)) => {
                    out.push_str(&rule.to);
                    // Deleting a word between two spaces leaves one space.
                    if rule.to.is_empty()
                        && (out.is_empty() || out.ends_with(char::is_whitespace))
                        && chars.get(end).is_some_and(|c| *c == ' ')
                    {
                        end += 1;
                    }
                    i = end;
                }
                None => {
                    out.push(chars[i]);
                    i += 1;
                }
            }
        }
        out
    }
}

/// End index of `pattern` matched at `start`, honoring word boundaries.
fn match_at(chars: &[char], start: usize, pattern: &[char]) -> Option<usize> {
    let mut i = start;
    let mut p = 0;
    while p < pattern.len() {
        if pattern[p] == ' ' {
            let run = chars[i..].iter().take_while(|c| c.is_whitespace()).count();
            if run == 0 {
                return None;
            }
            i += run;
        } else {
            if i >= chars.len() || lower(chars[i]) != pattern[p] {
                return None;
            }
            i += 1;
        }
        p += 1;
    }
    // A pattern edge that is a word char must sit on a word boundary.
    let first = pattern[0];
    let last = pattern[pattern.len() - 1];
    if is_word(first) && start > 0 && is_word(chars[start - 1]) {
        return None;
    }
    if is_word(last) && i < chars.len() && is_word(chars[i]) {
        return None;
    }
    Some(i)
}

// ---------------------------------------------------------------------------
// The file
// ---------------------------------------------------------------------------

/// Path of `dictionary.txt`, next to `config.toml`.
pub(crate) fn path() -> anyhow::Result<PathBuf> {
    let config = crate::config::config_path()
        .ok_or_else(|| anyhow::anyhow!("No home directory, so there is no dictionary file"))?;
    Ok(config.with_file_name(FILE_NAME))
}

/// Read and parse the file. A missing file is an empty dictionary.
pub(crate) fn read(path: &Path) -> anyhow::Result<(Dictionary, Vec<LineError>)> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(parse(&text)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Default::default()),
        Err(err) => Err(err).with_context(|| format!("Could not read {}", path.display())),
    }
}

/// Create the file with an explanatory header if it does not exist yet.
pub(crate) fn ensure_file(path: &Path) -> anyhow::Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Could not create {}", dir.display()))?;
    }
    std::fs::write(path, TEMPLATE).with_context(|| format!("Could not write {}", path.display()))
}

/// Append a rule, replacing an existing rule for the same phrase.
pub(crate) fn add(path: &Path, from: &str, to: &str) -> anyhow::Result<()> {
    let line = format!("{} {SEPARATOR} {}", normalize_spaces(from), to.trim());
    let rule = parse_rule(&line).map_err(|message| anyhow::anyhow!("Invalid rule: {message}"))?;
    ensure_file(path)?;
    remove_lines(path, &rule.from)?;
    let mut text = std::fs::read_to_string(path)
        .with_context(|| format!("Could not read {}", path.display()))?;
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&line);
    text.push('\n');
    std::fs::write(path, text).with_context(|| format!("Could not write {}", path.display()))
}

/// Remove every rule for `from` (case-insensitive). Returns how many went.
pub(crate) fn remove(path: &Path, from: &str) -> anyhow::Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    remove_lines(path, &normalize_spaces(from))
}

fn remove_lines(path: &Path, from: &str) -> anyhow::Result<usize> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Could not read {}", path.display()))?;
    let key = from.to_lowercase();
    let mut removed = 0;
    let mut kept = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim();
        let same = !trimmed.starts_with('#')
            && parse_rule(trimmed).is_ok_and(|rule| rule.from.to_lowercase() == key);
        if same {
            removed += 1;
        } else {
            kept.push_str(line);
            kept.push('\n');
        }
    }
    if removed > 0 {
        std::fs::write(path, kept)
            .with_context(|| format!("Could not write {}", path.display()))?;
    }
    Ok(removed)
}

// ---------------------------------------------------------------------------
// Daemon use: cached by modification time, problems reported once per edit
// ---------------------------------------------------------------------------

/// What the daemon currently knows about the file.
#[derive(Clone, Debug, Default)]
pub(crate) struct Status {
    pub dictionary: Arc<Dictionary>,
    /// Why some or all rules are not applied, if anything is wrong.
    pub problem: Option<String>,
}

struct Cache {
    path: PathBuf,
    modified: Option<SystemTime>,
    status: Status,
}

static CACHE: Mutex<Option<Cache>> = Mutex::new(None);

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Current dictionary, reloaded when the file changes. A file problem is
/// logged and shown once per edit of the file; the valid rules still apply.
pub(crate) fn current() -> Status {
    let (status, fresh) = load_cached();
    if fresh && let Some(problem) = &status.problem {
        crate::dlog!("warning: dictionary fallback=true: {problem}");
        crate::notify::show(crate::notify::Kind::Error, "dictate: dictionary", problem);
    }
    status
}

/// The cached status, plus whether this call (re)loaded the file.
fn load_cached() -> (Status, bool) {
    let path = match path() {
        Ok(path) => path,
        Err(err) => {
            return (
                Status {
                    dictionary: Arc::default(),
                    problem: Some(format!("{err:#}")),
                },
                false,
            );
        }
    };
    let stamp = modified(&path);
    let mut cache = CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(cache) = cache.as_ref()
        && cache.path == path
        && cache.modified == stamp
    {
        return (cache.status.clone(), false);
    }
    let status = match read(&path) {
        Ok((dictionary, errors)) => Status {
            dictionary: Arc::new(dictionary),
            problem: problem_text(&errors),
        },
        Err(err) => Status {
            dictionary: Arc::default(),
            problem: Some(format!("{err:#}; no replacements are applied")),
        },
    };
    *cache = Some(Cache {
        path,
        modified: stamp,
        status: status.clone(),
    });
    (status, true)
}

fn problem_text(errors: &[LineError]) -> Option<String> {
    let first = errors.first()?;
    let more = match errors.len() {
        1 => String::new(),
        n => format!(" (and {} more)", n - 1),
    };
    Some(format!("Skipped invalid rule: {first}{more}"))
}

/// Apply the current dictionary to text about to be typed.
pub(crate) fn for_typing(text: &str) -> String {
    current().dictionary.apply(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict(rules: &[(&str, &str)]) -> Dictionary {
        Dictionary::new(
            rules
                .iter()
                .map(|(from, to)| Rule {
                    from: from.to_string(),
                    to: to.to_string(),
                })
                .collect(),
        )
    }

    #[test]
    fn replaces_whole_words_ignoring_case() {
        let d = dict(&[("post gress", "PostgreSQL"), ("jira", "Jira")]);
        assert_eq!(
            d.apply("Post gress and JIRA, not compost gress or jiraffe."),
            "PostgreSQL and Jira, not compost gress or jiraffe."
        );
    }

    #[test]
    fn pattern_space_matches_any_whitespace() {
        let d = dict(&[("cooper netties", "Kubernetes")]);
        assert_eq!(d.apply("cooper\n  netties!"), "Kubernetes!");
        assert_eq!(d.apply("coopernetties"), "coopernetties");
    }

    #[test]
    fn longest_rule_wins_and_output_is_not_rematched() {
        let d = dict(&[("new", "old"), ("new york", "NYC"), ("nyc", "never")]);
        assert_eq!(d.apply("new york is new"), "NYC is old");
    }

    #[test]
    fn keeps_streaming_piece_spacing_and_unicode() {
        let d = dict(&[("straße", "Strasse"), ("c++", "C++")]);
        assert_eq!(d.apply(" STRASSE straße, c++."), " STRASSE Strasse, C++.");
        assert_eq!(dict(&[]).apply(" hi"), " hi");
    }

    #[test]
    fn apostrophes_stay_inside_words() {
        let d = dict(&[("don", "Don")]);
        assert_eq!(d.apply("don't call don"), "don't call Don");
    }

    #[test]
    fn parse_reports_invalid_lines_and_keeps_valid_ones() {
        let (d, errors) = parse("# comment\n\nfoo -> Foo\nno arrow\n -> x\na -> b -> c\n");
        assert_eq!(d.len(), 1);
        assert_eq!(
            errors.iter().map(|e| e.line).collect::<Vec<_>>(),
            vec![4, 5, 6]
        );
        assert!(
            problem_text(&errors)
                .unwrap()
                .contains("line 4: expected \"heard -> written\" (and 2 more)")
        );
    }

    #[test]
    fn empty_replacement_deletes_the_phrase() {
        let (d, errors) = parse("um ->\n");
        assert!(errors.is_empty());
        assert_eq!(d.apply("so um yes um"), "so yes ");
        assert_eq!(d.apply("um, yes"), ", yes");
    }

    #[test]
    fn add_replaces_and_remove_deletes_rules() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE_NAME);
        add(&path, "post  gress", "Postgres").unwrap();
        add(&path, "Post gress", "PostgreSQL").unwrap();
        add(&path, "jira", "Jira").unwrap();
        add(&path, "a long phrase here", "x").unwrap();
        let (d, errors) = read(&path).unwrap();
        assert!(errors.is_empty());
        let rules: Vec<_> = d
            .rules()
            .map(|r| (r.from.as_str(), r.to.as_str()))
            .collect();
        assert_eq!(
            rules,
            vec![
                ("Post gress", "PostgreSQL"),
                ("jira", "Jira"),
                ("a long phrase here", "x")
            ]
        );
        assert!(
            std::fs::read_to_string(&path)
                .unwrap()
                .starts_with("# Dictate")
        );
        assert_eq!(remove(&path, "POST GRESS").unwrap(), 1);
        assert_eq!(remove(&path, "missing").unwrap(), 0);
        assert_eq!(read(&path).unwrap().0.len(), 2);
        assert!(add(&path, " ", "x").is_err());
    }

    #[test]
    fn missing_file_is_an_empty_dictionary() {
        let dir = tempfile::tempdir().unwrap();
        let (d, errors) = read(&dir.path().join(FILE_NAME)).unwrap();
        assert_eq!(d.len(), 0);
        assert!(errors.is_empty());
    }
}
