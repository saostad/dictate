use std::path::Path;
use std::sync::OnceLock;

use anyhow::Context;
use parakeet_rs::ort::environment::Environment;
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber};

pub struct Asr {
    model: ParakeetTDT,
}

fn shared_environment() -> anyhow::Result<Environment> {
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

/// Lowercase the first word of a mid-sentence continuation piece, except "I"
/// and I-contractions.
pub(crate) fn decapitalize_continuation(piece: &str) -> String {
    match piece.split_once(' ') {
        None => decapitalize_word(piece),
        Some((first, rest)) => format!("{} {rest}", decapitalize_word(first)),
    }
}

/// True when the piece ends a sentence (previous piece ended, so the next one
/// keeps the model's capitalization).
pub(crate) fn ends_sentence(piece: &str) -> bool {
    piece
        .chars()
        .last()
        .is_some_and(|last| matches!(last, '.' | '?' | '!'))
}

fn decapitalize_word(word: &str) -> String {
    if is_capital_i(word) {
        word.to_string()
    } else {
        word.to_lowercase()
    }
}

/// "I" (possibly with trailing punctuation attached) or an I-contraction.
fn is_capital_i(word: &str) -> bool {
    if word == "I" || word.starts_with("I'") || word.starts_with("I\u{2019}") {
        return true;
    }
    word.strip_prefix('I')
        .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| !c.is_alphabetic()))
}

impl Asr {
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        load_inner(dir)
            .with_context(|| format!("Loading speech model from {} failed", dir.display()))
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> anyhow::Result<String> {
        let result = self
            .model
            .transcribe_samples(samples.to_vec(), 16_000, 1, None)?;
        Ok(result.text.trim().to_string())
    }

    /// Transcribe with word-level timestamps (seconds).
    pub fn transcribe_words(&mut self, samples: &[f32]) -> anyhow::Result<Vec<Word>> {
        let result = self.model.transcribe_samples(
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

    /// Transcribe one streaming segment. Segments after the first are decoded
    /// with left `context` audio (the trailing audio before the segment) for
    /// accuracy at the cut, then trimmed to the words at/after the context
    /// boundary. The first segment (empty context) decodes alone, exactly as
    /// before. `mid_sentence` lowercases the first word unless it is "I" or
    /// an I-contraction.
    pub(crate) fn transcribe_segment(
        &mut self,
        context: &[f32],
        segment: &[f32],
        mid_sentence: bool,
    ) -> anyhow::Result<String> {
        if context.is_empty() {
            return self.transcribe(segment);
        }
        let mut audio = Vec::with_capacity(context.len() + segment.len());
        audio.extend_from_slice(context);
        audio.extend_from_slice(segment);
        let words = self.transcribe_words(&audio)?;
        let context_s = context.len() as f32 / 16_000.0;
        let kept = keep_words_after(&words, context_s);
        let piece = join_words(&kept);
        if mid_sentence {
            Ok(decapitalize_continuation(&piece))
        } else {
            Ok(piece)
        }
    }
}

fn load_inner(dir: &Path) -> anyhow::Result<Asr> {
    let env = shared_environment()?;
    let config = ExecutionConfig::new().with_environment(env);
    let model = ParakeetTDT::from_pretrained(dir, Some(config))?;
    Ok(Asr { model })
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

    #[test]
    fn decapitalize_continuation_lowercases_first_word() {
        assert_eq!(decapitalize_continuation("Ask what"), "ask what");
        assert_eq!(decapitalize_continuation("Ask"), "ask");
        assert_eq!(decapitalize_continuation("ask what"), "ask what");
        assert_eq!(decapitalize_continuation(""), "");
    }

    #[test]
    fn decapitalize_continuation_keeps_capital_i() {
        assert_eq!(decapitalize_continuation("I ask"), "I ask");
        assert_eq!(decapitalize_continuation("I'm asking"), "I'm asking");
        assert_eq!(decapitalize_continuation("I'll go"), "I'll go");
        assert_eq!(decapitalize_continuation("I, for one"), "I, for one");
    }

    #[test]
    fn ends_sentence_checks_final_punctuation() {
        assert!(ends_sentence("you."));
        assert!(ends_sentence("you?"));
        assert!(ends_sentence("you!"));
        assert!(!ends_sentence("you,"));
        assert!(!ends_sentence("you"));
        assert!(!ends_sentence(""));
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
