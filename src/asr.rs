use std::path::Path;
use std::sync::OnceLock;

use anyhow::Context;
use parakeet_rs::ort::environment::Environment;
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber};

use crate::config::SpeechModel;

/// The loaded Whole/Pauses recognizer. Both engines expose the same two
/// operations, so every call site keeps calling [`Asr::transcribe`] and
/// [`Asr::transcribe_words`] regardless of the selected model.
enum Engine {
    Parakeet(ParakeetTDT),
    Fluid(crate::fluid::FluidModel),
}

pub struct Asr {
    engine: Engine,
}

pub(crate) fn shared_environment() -> anyhow::Result<Environment> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env.clone());
    }
    let env = build_environment()?;
    if ENV.set(env.clone()).is_err() {
        // Another thread initialized first; prefer its environment.
        if let Some(first) = ENV.get() {
            return Ok(first.clone());
        }
    }
    Ok(env)
}

#[cfg(feature = "bundled-ort")]
fn build_environment() -> anyhow::Result<Environment> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !std::arch::is_x86_feature_detected!("avx2") {
        anyhow::bail!(
            "This CPU has no AVX2, but this build bundles an ONNX Runtime that needs AVX2. Download the prebuilt Linux release from https://github.com/saostad/dictate/releases/latest (it works on any x86_64 CPU), or rebuild with: cargo build --release --no-default-features --features system-ort"
        );
    }
    Ok(parakeet_rs::ort::init().build()?)
}

#[cfg(feature = "system-ort")]
fn build_environment() -> anyhow::Result<Environment> {
    let exe = std::env::current_exe().ok();
    let exe_dir = exe.as_ref().and_then(|p| p.parent());
    let path = ort_dylib_path(std::env::var_os("ORT_DYLIB_PATH"), exe_dir, |p| p.exists());
    let builder = parakeet_rs::ort::init_from(&path).map_err(|err| {
        anyhow::anyhow!(
            "Could not load ONNX Runtime from {}: {err:#}. Put the onnxruntime library next to the dictate binary, install it (Arch: sudo pacman -S onnxruntime-cpu), or set ORT_DYLIB_PATH to its full path.",
            path.display()
        )
    })?;
    Ok(builder.build()?)
}

/// Path of the ONNX Runtime library for system-ort builds: `ORT_DYLIB_PATH`
/// when it is set and non-empty, otherwise the library file next to the
/// running binary when it exists, otherwise the platform's default library
/// file name (resolved through the system search path).
#[cfg(feature = "system-ort")]
fn ort_dylib_path(
    env: Option<std::ffi::OsString>,
    exe_dir: Option<&Path>,
    mut exists: impl FnMut(&Path) -> bool,
) -> std::path::PathBuf {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
    if let Some(path) = env
        && !path.is_empty()
    {
        return std::path::PathBuf::from(path);
    }
    let file_name = format!("{DLL_PREFIX}onnxruntime{DLL_SUFFIX}");
    if let Some(dir) = exe_dir {
        let next_to_binary = dir.join(&file_name);
        if exists(&next_to_binary) {
            return next_to_binary;
        }
    }
    std::path::PathBuf::from(file_name)
}

/// True when the transcript holds no recognized speech.
pub(crate) fn is_empty_transcript(text: &str) -> bool {
    text.trim().is_empty()
}

/// Message for `stderr` when no speech was recognized in `seconds` of audio.
pub(crate) fn no_speech_message(seconds: f32) -> String {
    format!("No speech recognized in {seconds:.1} s of audio.")
}

/// One word with its timestamps in seconds.
#[derive(Debug, Clone)]
pub struct Word {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

/// Keep the words decoded at/after the context boundary: a word belongs to
/// the segment when its midpoint is at or past `context_s` (seconds).
pub(crate) fn keep_words_after(words: &[Word], context_s: f32) -> Vec<Word> {
    words
        .iter()
        .filter(|word| (word.start + word.end) / 2.0 >= context_s)
        .cloned()
        .collect()
}

/// Join kept words exactly as the model spells them: single spaces, with
/// standalone punctuation attached to the preceding word (mirrors parakeet-rs
/// `rebuild_text` for `TimestampMode::Words`).
pub(crate) fn join_words(words: &[Word]) -> String {
    let mut out = String::new();
    for (index, word) in words.iter().enumerate() {
        let text = word.text.as_str();
        let is_standalone_punct = text.len() == 1
            && text
                .chars()
                .all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));
        if index > 0 && !is_standalone_punct {
            out.push(' ');
        }
        out.push_str(text);
    }
    out
}

/// Punctuation held back between streaming pieces.
const BOUNDARY_PUNCT: &[char] = &['.', ',', '!', '?', ';', ':'];

/// Trailing punctuation of one word (`.,?!;:`), if any.
pub(crate) fn trailing_punct(word: &str) -> Option<char> {
    word.chars()
        .last()
        .filter(|last| BOUNDARY_PUNCT.contains(last))
}

/// Punctuation the context decode attaches to the last context word before
/// the boundary (None when the context region holds no words or no trailing
/// punctuation).
pub(crate) fn boundary_punct_before(words: &[Word], context_s: f32) -> Option<char> {
    words
        .iter()
        .rfind(|word| (word.start + word.end) / 2.0 < context_s)
        .and_then(|word| trailing_punct(&word.text))
}

/// Join one streaming piece onto already-typed text.
///
/// `prev_tail_words` are the last up to 2 typed words (for overlap dedupe).
/// `pending_punct` is the previous piece's held trailing punctuation: it is
/// superseded by the boundary (typed instead, or dropped when there is none)
/// and only reaches the page at `End`, so this function takes it to make the
/// replacement explicit and then ignores it. `context_boundary_punct` is the
/// punctuation from the context decode on the last context word before the
/// boundary; punctuation leading the kept words supersedes it (the fresher
/// decode). `kept_words` are the current piece's decoded word texts (at/after
/// the boundary).
///
/// Returns the text to type (boundary punctuation plus spacing included) and
/// the new pending trailing punctuation (held for the next boundary or `End`).
/// Kept words are typed exactly as decoded: no lowercasing, since the context
/// decode already capitalizes correctly.
pub(crate) fn join_piece(
    prev_tail_words: &[String],
    pending_punct: Option<char>,
    context_boundary_punct: Option<char>,
    kept_words: &[String],
) -> (String, Option<char>) {
    let _ = pending_punct;
    let mut kept: Vec<String> = kept_words.to_vec();
    let mut boundary = context_boundary_punct;
    // Leading punctuation (standalone tokens and prefixes) is the boundary:
    // the context decode placed it at/after the cut.
    strip_leading_punct(&mut kept, &mut boundary);
    // Overlap dedupe: drop the first 1-2 kept words when they equal the last
    // 1-2 typed words (case and punctuation-insensitive).
    dedupe_overlap(prev_tail_words, &mut kept);
    // Dedupe can expose a new leading punctuation token; it is a boundary
    // too (and never typed with a space before it).
    strip_leading_punct(&mut kept, &mut boundary);
    // Hold back the new trailing punctuation.
    let mut new_pending = None;
    if let Some(last) = kept.last_mut()
        && let Some(punct) = trailing_punct(last)
    {
        new_pending = Some(punct);
        last.pop();
        if last.is_empty() {
            kept.pop();
        }
    }
    let dummy: Vec<Word> = kept
        .iter()
        .map(|text| Word {
            text: text.clone(),
            start: 0.0,
            end: 0.0,
        })
        .collect();
    let joined = join_words(&dummy);
    if prev_tail_words.is_empty() {
        return (joined, new_pending);
    }
    if joined.is_empty() {
        let boundary = boundary.map(|c| c.to_string()).unwrap_or_default();
        return (boundary, new_pending);
    }
    match boundary {
        Some(boundary) => (format!("{boundary} {joined}"), new_pending),
        None => (format!(" {joined}"), new_pending),
    }
}

/// Move leading punctuation (standalone tokens and prefixes) into the
/// boundary; the last one wins. Boundary punctuation is ASCII, so the byte
/// split below always lands on a char boundary.
fn strip_leading_punct(kept: &mut Vec<String>, boundary: &mut Option<char>) {
    while !kept.is_empty() {
        let stripped = kept[0].trim_start_matches(BOUNDARY_PUNCT).to_string();
        if stripped.len() == kept[0].len() {
            break;
        }
        *boundary = kept[0][..kept[0].len() - stripped.len()].chars().last();
        if stripped.is_empty() {
            kept.remove(0);
        } else {
            kept[0] = stripped;
            break;
        }
    }
}

/// Drop the first 1-2 kept words when they repeat the last typed words.
/// Comparison is lowercase with punctuation removed; empty normalizations
/// never match.
fn dedupe_overlap(prev_tail_words: &[String], kept: &mut Vec<String>) {
    fn norm(word: &str) -> String {
        word.chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase()
    }
    if kept.is_empty() || prev_tail_words.is_empty() {
        return;
    }
    if kept.len() >= 2 && prev_tail_words.len() >= 2 {
        let prev_len = prev_tail_words.len();
        let kept0 = norm(&kept[0]);
        let kept1 = norm(&kept[1]);
        let prev0 = norm(&prev_tail_words[prev_len - 2]);
        let prev1 = norm(&prev_tail_words[prev_len - 1]);
        if !kept0.is_empty() && !kept1.is_empty() && kept0 == prev0 && kept1 == prev1 {
            kept.drain(..2);
            return;
        }
    }
    let Some(prev_last) = prev_tail_words.last() else {
        return;
    };
    let kept0 = norm(&kept[0]);
    if !kept0.is_empty() && kept0 == norm(prev_last) {
        kept.drain(..1);
    }
}

/// Record the words a successful `type_text` call added: split on whitespace,
/// dropping a leading boundary-punctuation token (it attaches to the previous
/// word rather than starting a new one).
pub(crate) fn append_typed_words(typed_words: &mut Vec<String>, to_type: &str) {
    let mut words = to_type.split_whitespace();
    if let Some(first) = words.clone().next()
        && first.chars().all(|c| BOUNDARY_PUNCT.contains(&c))
    {
        words.next();
    }
    typed_words.extend(words.map(|word| word.to_string()));
}

/// ONNX Runtime intra-op threads: use all cores, clamped to a sane range.
/// Falls back to parakeet-rs's default (4) when the count is unavailable.
pub(crate) fn thread_count(parallelism: Option<usize>) -> usize {
    parallelism.unwrap_or(4).clamp(1, 16)
}

pub(crate) fn intra_threads() -> usize {
    thread_count(
        std::thread::available_parallelism()
            .ok()
            .map(|threads| threads.get()),
    )
}

impl Asr {
    /// Load the standard (NVIDIA Parakeet) model from `dir`.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let threads = intra_threads();
        let model = load_inner(dir, threads)
            .with_context(|| format!("Loading speech model from {} failed", dir.display()))?;
        eprintln!("using {threads} threads");
        Ok(Self {
            engine: Engine::Parakeet(model),
        })
    }

    /// Load Fluid Speech Mini from `dir`.
    pub fn load_fluid(dir: &Path) -> anyhow::Result<Self> {
        let model = crate::fluid::FluidModel::load(dir)
            .with_context(|| format!("Loading Fluid Speech Mini from {} failed", dir.display()))?;
        eprintln!("using {} threads", intra_threads());
        Ok(Self {
            engine: Engine::Fluid(model),
        })
    }

    /// Load the speech model selected in the settings (`model = "standard"` or
    /// `"fluid"`, with `DICTATE_SPEECH_MODEL` winning).
    pub fn load_configured() -> anyhow::Result<Self> {
        match crate::config::speech_model() {
            SpeechModel::Standard => Self::load(&crate::model::model_dir_checked()?),
            SpeechModel::Fluid => Self::load_fluid(&crate::model::fluid_model_dir_checked()?),
        }
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> anyhow::Result<String> {
        match &mut self.engine {
            Engine::Parakeet(model) => {
                let result = model.transcribe_samples(samples.to_vec(), 16_000, 1, None)?;
                Ok(result.text.trim().to_string())
            }
            Engine::Fluid(model) => model.transcribe(samples),
        }
    }

    /// Transcribe with word-level timestamps (seconds).
    pub fn transcribe_words(&mut self, samples: &[f32]) -> anyhow::Result<Vec<Word>> {
        match &mut self.engine {
            Engine::Parakeet(model) => {
                let result = model.transcribe_samples(
                    samples.to_vec(),
                    16_000,
                    1,
                    Some(TimestampMode::Words),
                )?;
                Ok(result
                    .tokens
                    .into_iter()
                    .map(|token| Word {
                        text: token.text,
                        start: token.start,
                        end: token.end,
                    })
                    .collect())
            }
            Engine::Fluid(model) => model.transcribe_words(samples),
        }
    }

    /// Transcribe one streaming segment, returning the kept word texts plus
    /// the boundary punctuation from the context decode. Segments after the
    /// first are decoded with left `context` audio for accuracy at the cut,
    /// then trimmed to the words at/after the context boundary; the boundary
    /// is the trailing punctuation on the last context word before it. The
    /// first segment (empty context) decodes alone via plain transcribe, split
    /// into words. Empty results are `(empty, None)` and skipped silently.
    pub(crate) fn transcribe_segment_pieces(
        &mut self,
        context: &[f32],
        segment: &[f32],
    ) -> anyhow::Result<(Vec<String>, Option<char>)> {
        if context.is_empty() {
            let text = self.transcribe(segment)?;
            if is_empty_transcript(&text) {
                return Ok((Vec::new(), None));
            }
            let words = text.split_whitespace().map(str::to_string).collect();
            return Ok((words, None));
        }
        let mut audio = Vec::with_capacity(context.len() + segment.len());
        audio.extend_from_slice(context);
        audio.extend_from_slice(segment);
        let words = self.transcribe_words(&audio)?;
        let context_s = context.len() as f32 / 16_000.0;
        let boundary = boundary_punct_before(&words, context_s);
        let kept = keep_words_after(&words, context_s);
        let texts = kept.into_iter().map(|word| word.text).collect();
        Ok((texts, boundary))
    }
}

fn load_inner(dir: &Path, threads: usize) -> anyhow::Result<ParakeetTDT> {
    let env = shared_environment()?;
    let config = ExecutionConfig::new()
        .with_environment(env)
        .with_intra_threads(threads);
    Ok(ParakeetTDT::from_pretrained(dir, Some(config))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_transcript_check_ignores_whitespace() {
        assert!(is_empty_transcript(""));
        assert!(is_empty_transcript("   "));
        assert!(!is_empty_transcript("hello"));
    }

    #[test]
    fn no_speech_message_names_duration() {
        assert_eq!(
            no_speech_message(3.9),
            "No speech recognized in 3.9 s of audio."
        );
    }

    #[test]
    fn thread_count_clamps_parallelism() {
        assert_eq!(thread_count(Some(8)), 8);
        assert_eq!(thread_count(Some(1)), 1);
        assert_eq!(thread_count(Some(16)), 16);
        assert_eq!(thread_count(Some(64)), 16);
        assert_eq!(thread_count(Some(0)), 1);
        assert_eq!(thread_count(None), 4);
    }

    #[test]
    fn intra_threads_stays_in_range() {
        assert!((1..=16).contains(&intra_threads()));
    }

    fn word(text: &str, start: f32, end: f32) -> Word {
        Word {
            text: text.to_string(),
            start,
            end,
        }
    }

    fn kept_texts(words: &[Word], context_s: f32) -> Vec<String> {
        keep_words_after(words, context_s)
            .iter()
            .map(|word| word.text.clone())
            .collect()
    }

    #[test]
    fn keep_words_after_drops_context_words() {
        let words = vec![
            word("country", 0.5, 0.9),
            word("can", 1.0, 1.2),
            word("ask", 1.6, 1.9),
            word("what", 2.0, 2.2),
        ];
        assert_eq!(kept_texts(&words, 1.5), vec!["ask", "what"]);
    }

    #[test]
    fn keep_words_after_keeps_boundary_word() {
        // Midpoint exactly on the boundary belongs to the segment.
        let words = vec![word("ask", 1.4, 1.6)];
        assert_eq!(kept_texts(&words, 1.5), vec!["ask"]);
    }

    #[test]
    fn keep_words_after_drops_straddling_word() {
        // A word straddling the cut with its midpoint before it is context.
        let words = vec![word("you,", 1.2, 1.6), word("ask", 1.7, 1.9)];
        assert_eq!(kept_texts(&words, 1.5), vec!["ask"]);
    }

    #[test]
    fn keep_words_after_empty_is_empty() {
        assert!(keep_words_after(&[], 1.5).is_empty());
    }

    #[test]
    fn join_words_attaches_standalone_punctuation() {
        let words = vec![
            word("you", 1.6, 1.8),
            word(",", 1.8, 1.9),
            word("ask", 1.9, 2.1),
        ];
        assert_eq!(join_words(&words), "you, ask");
        assert_eq!(join_words(&[]), "");
    }

    fn texts(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| (*word).to_string()).collect()
    }

    #[test]
    fn trailing_punct_finds_final_punctuation() {
        assert_eq!(trailing_punct("you,"), Some(','));
        assert_eq!(trailing_punct("you."), Some('.'));
        assert_eq!(trailing_punct(","), Some(','));
        assert_eq!(trailing_punct("you"), None);
        assert_eq!(trailing_punct(""), None);
    }

    #[test]
    fn boundary_punct_comes_from_last_context_word() {
        let words = vec![
            word("country", 0.5, 0.9),
            word("you,", 1.0, 1.4),
            word("ask", 1.6, 1.9),
        ];
        assert_eq!(boundary_punct_before(&words, 1.5), Some(','));
        assert_eq!(boundary_punct_before(&words, 0.4), None);
        assert!(boundary_punct_before(&[], 1.5).is_none());
    }

    #[test]
    fn join_piece_first_piece_holds_trailing_punctuation() {
        let (text, pending) = join_piece(&[], None, None, &texts(&["And", "so"]));
        assert_eq!((text.as_str(), pending), ("And so", None));
        let (text, pending) = join_piece(&[], None, None, &texts(&["hello", "you."]));
        assert_eq!((text.as_str(), pending), ("hello you", Some('.')));
    }

    #[test]
    fn join_piece_segmented_fixture_has_no_double_punctuation() {
        // `--segmented jfk.wav` joined as `... for you., ask what ...`: the
        // pending `.` plus a leading `,` both reached the page.
        let prev = texts(&["for", "you"]);
        let kept = texts(&[",", "ask", "what"]);
        let (text, pending) = join_piece(&prev, Some('.'), Some(','), &kept);
        assert_eq!((text.as_str(), pending), (", ask what", None));
    }

    #[test]
    fn join_piece_live_run_keeps_non_overlap_words() {
        // Live run 1: `... for you, and what ...` ("and" is mis-heard, but the
        // join must keep it rather than drop or garble it).
        let prev = texts(&["for", "you"]);
        let kept = texts(&["and", "what"]);
        let (text, pending) = join_piece(&prev, Some(','), Some(','), &kept);
        assert_eq!((text.as_str(), pending), (", and what", None));
    }

    #[test]
    fn join_piece_live_run_dedupes_repeated_word() {
        // Live run 2: `... for you. you, and what ...` (the leading `you,`
        // repeats the typed `you`).
        let prev = texts(&["for", "you"]);
        let kept = texts(&["you,", "and", "what"]);
        let (text, pending) = join_piece(&prev, Some('.'), Some('.'), &kept);
        assert_eq!((text.as_str(), pending), (". and what", None));
    }

    #[test]
    fn join_piece_live_run_drops_pending_without_context_punct() {
        // Live run 3: `... country. can do ...` (a period typed mid-sentence;
        // the context decode has no punctuation, so none is typed).
        let prev = texts(&["your", "country"]);
        let kept = texts(&["can", "do"]);
        let (text, pending) = join_piece(&prev, Some('.'), None, &kept);
        assert_eq!((text.as_str(), pending), (" can do", None));
    }

    #[test]
    fn join_piece_dedupes_two_word_overlap() {
        let prev = texts(&["can", "do"]);
        let kept = texts(&["Can", "do", "for"]);
        let (text, pending) = join_piece(&prev, None, None, &kept);
        assert_eq!((text.as_str(), pending), (" for", None));
    }

    #[test]
    fn join_piece_keeps_model_capitalization() {
        // No lowercasing: the context decode already capitalizes correctly
        // (lowercasing turned "Americans" into "americans" mid-sentence).
        let prev = texts(&["my", "fellow"]);
        let (text, _) = join_piece(&prev, None, Some(','), &texts(&["Americans", "ask"]));
        assert_eq!(text, ", Americans ask");
        let (text, _) = join_piece(&prev, None, Some('.'), &texts(&["Ask"]));
        assert_eq!(text, ". Ask");
    }

    #[test]
    fn join_piece_leading_punctuation_becomes_boundary() {
        // `for you` + `, ask`: the kept comma sits past the boundary, so it
        // is the boundary instead of being stripped (it used to be lost).
        let prev = texts(&["for", "you"]);
        let (text, pending) = join_piece(&prev, None, None, &texts(&[",", "ask"]));
        assert_eq!((text.as_str(), pending), (", ask", None));
        // The kept punctuation is the fresher decode, so it wins over the
        // context boundary.
        let (text, _) = join_piece(&prev, None, Some('.'), &texts(&[",", "ask"]));
        assert_eq!(text, ", ask");
        // Prefix form.
        let (text, _) = join_piece(&prev, None, None, &texts(&[",ask"]));
        assert_eq!(text, ", ask");
    }

    #[test]
    fn join_piece_dedupe_exposed_punctuation_becomes_boundary() {
        // Dedupe dropping "you" exposes a standalone `,`: it becomes the
        // boundary (`, and`), never a space-separated `you , and`.
        let prev = texts(&["for", "you"]);
        let (text, pending) = join_piece(&prev, None, Some('.'), &texts(&["you", ",", "and"]));
        assert_eq!((text.as_str(), pending), (", and", None));
    }

    #[test]
    fn join_piece_deduped_to_empty_types_boundary_only() {
        let prev = texts(&["for", "you"]);
        let (text, pending) = join_piece(&prev, Some('.'), Some(','), &texts(&["you"]));
        assert_eq!((text.as_str(), pending), (",", None));
    }

    #[test]
    fn append_typed_words_skips_boundary_token() {
        let mut typed = texts(&["hello"]);
        append_typed_words(&mut typed, ", ask what");
        assert_eq!(typed, texts(&["hello", "ask", "what"]));
        append_typed_words(&mut typed, ",");
        assert_eq!(typed, texts(&["hello", "ask", "what"]));
        let mut fresh = Vec::new();
        append_typed_words(&mut fresh, "And so");
        assert_eq!(fresh, texts(&["And", "so"]));
    }
}

#[cfg(all(test, feature = "system-ort"))]
mod system_ort_tests {
    use super::*;
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};
    use std::path::PathBuf;

    fn default_lib_name() -> PathBuf {
        PathBuf::from(format!("{DLL_PREFIX}onnxruntime{DLL_SUFFIX}"))
    }

    #[test]
    fn ort_dylib_path_uses_env_when_set() {
        let path = ort_dylib_path(Some("/opt/ort/libonnxruntime.so".into()), None, |_| true);
        assert_eq!(path, PathBuf::from("/opt/ort/libonnxruntime.so"));
    }

    #[test]
    fn ort_dylib_path_env_wins_over_bundled_lib() {
        let path = ort_dylib_path(
            Some("/opt/ort/libonnxruntime.so".into()),
            Some(Path::new("/opt/dictate")),
            |_| true,
        );
        assert_eq!(path, PathBuf::from("/opt/ort/libonnxruntime.so"));
    }

    #[test]
    fn ort_dylib_path_uses_lib_next_to_binary() {
        let dir = Path::new("/opt/dictate");
        let path = ort_dylib_path(None, Some(dir), |_| true);
        assert_eq!(path, dir.join(default_lib_name()));
    }

    #[test]
    fn ort_dylib_path_empty_env_falls_through_to_bundled_lib() {
        let dir = Path::new("/opt/dictate");
        let path = ort_dylib_path(Some("".into()), Some(dir), |_| true);
        assert_eq!(path, dir.join(default_lib_name()));
    }

    #[test]
    fn ort_dylib_path_probes_exe_dir() {
        let dir = Path::new("/opt/dictate");
        let mut probed = Vec::new();
        let path = ort_dylib_path(None, Some(dir), |p| {
            probed.push(p.to_path_buf());
            false
        });
        assert_eq!(probed, vec![dir.join(default_lib_name())]);
        assert_eq!(path, default_lib_name());
    }

    #[test]
    fn ort_dylib_path_falls_back_when_unset() {
        assert_eq!(ort_dylib_path(None, None, |_| true), default_lib_name());
    }

    #[test]
    fn ort_dylib_path_falls_back_on_empty_env() {
        let path = ort_dylib_path(Some("".into()), None, |_| true);
        assert_eq!(path, default_lib_name());
    }
}
