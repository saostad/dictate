//! File transcription: bounded audio, shared pause feeder, incremental output.
use crate::{asr, audio, cli::UsageError, config, live, media, model, moonshine, segment, vad};
use anyhow::Context;
use std::{
    io::{IsTerminal, Write},
    path::{Path, PathBuf},
};

pub(crate) struct Options<'a> {
    path: &'a str,
    engine: Option<&'a str>,
    format: &'a str,
    output: Option<&'a str>,
    segmented: bool,
}
impl<'a> Options<'a> {
    pub(crate) fn parse(args: &'a [String]) -> Result<Self, UsageError> {
        let bad = || UsageError(crate::cli::command_usage("transcribe").unwrap().into());
        let mut options = Self {
            path: "",
            engine: None,
            format: "txt",
            output: None,
            segmented: false,
        };
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--fluid" | "--whisper" | "--live" | "--moonshine" => {
                    if options.engine.replace(arg.as_str()).is_some() {
                        return Err(bad());
                    }
                }
                "--segmented" => options.segmented = true,
                "--format" => {
                    options.format = args.next().ok_or_else(bad)?;
                    if !matches!(options.format, "txt" | "json" | "srt" | "vtt") {
                        return Err(bad());
                    }
                }
                "-o" | "--output" => {
                    if options
                        .output
                        .replace(args.next().ok_or_else(bad)?.as_str())
                        .is_some()
                    {
                        return Err(bad());
                    }
                }
                "-" => {
                    if !options.path.is_empty() {
                        return Err(bad());
                    }
                    options.path = arg;
                }
                value if !value.starts_with('-') && !value.is_empty() => {
                    if !options.path.is_empty() {
                        return Err(bad());
                    }
                    options.path = value;
                }
                _ => return Err(bad()),
            }
        }
        if options.path.is_empty() {
            return Err(bad());
        }
        if options.output == Some(options.path) && options.path != "-" {
            return Err(UsageError(
                "Output must not overwrite the input file".into(),
            ));
        }
        Ok(options)
    }
}

fn same_file(input: &str, output: &Path) -> anyhow::Result<bool> {
    if input == "-" || !output.exists() {
        return Ok(false);
    }
    let input = Path::new(input);
    if input.canonicalize()? == output.canonicalize()? {
        return Ok(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let a = input.metadata()?;
        let b = output.metadata()?;
        Ok(a.dev() == b.dev() && a.ino() == b.ino())
    }
    #[cfg(not(unix))]
    Ok(false)
}

struct Output {
    temp: Option<tempfile::NamedTempFile>,
    path: Option<PathBuf>,
}
impl Output {
    fn new(input: &str, path: Option<&str>) -> anyhow::Result<Self> {
        if let Some(path) = path {
            let path = PathBuf::from(path);
            anyhow::ensure!(
                !same_file(input, &path)?,
                "Output must not overwrite input {input}"
            );
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            Ok(Self {
                temp: Some(tempfile::NamedTempFile::new_in(parent)?),
                path: Some(path),
            })
        } else {
            Ok(Self {
                temp: None,
                path: None,
            })
        }
    }
    fn write(&mut self, text: &str) -> anyhow::Result<()> {
        if let Some(temp) = &mut self.temp {
            temp.write_all(text.as_bytes())?;
            temp.flush()?;
        } else {
            let mut out = std::io::stdout().lock();
            out.write_all(text.as_bytes())?;
            out.flush()?;
        }
        Ok(())
    }
    fn finish(mut self) -> anyhow::Result<()> {
        if let Some(temp) = self.temp.take() {
            temp.as_file().sync_all()?;
            temp.persist(self.path.as_ref().unwrap())
                .context("Could not atomically save transcript")?;
        }
        Ok(())
    }
}

#[allow(clippy::large_enum_variant)]
enum Engine {
    Batch(asr::Asr, bool),
    Live(live::Live, LiveTiming),
    Moonshine(moonshine::Session<moonshine::Moonshine>),
}
impl Engine {
    fn load(name: &str) -> anyhow::Result<Self> {
        Ok(match name {
            "nemotron" => Self::Live(
                live::Live::load(
                    &model::live_model_dir_checked()?,
                    &asr::shared_environment()?,
                    asr::intra_threads(),
                )?,
                LiveTiming::default(),
            ),
            "moonshine" => {
                let threads = moonshine::Threads::for_setting(config::inference_threads());
                moonshine::configure_threads(threads);
                model::check_moonshine_runtime()?;
                Self::Moonshine(moonshine::Session::new(moonshine::Moonshine::load(
                    &model::moonshine_model_dir_checked()?,
                    threads,
                )?))
            }
            "fluid" => Self::Batch(
                asr::Asr::load_fluid(&model::fluid_model_dir_checked()?)?,
                true,
            ),
            "whisper" => Self::Batch(
                asr::Asr::load_whisper(&model::whisper_model_dir_checked()?)?,
                false,
            ),
            _ => Self::Batch(asr::Asr::load(&model::model_dir_checked()?)?, true),
        })
    }
    fn decode(
        &mut self,
        segment: &segment::Segment,
        timestamps: bool,
    ) -> anyhow::Result<(String, Option<Vec<asr::Word>>)> {
        match self {
            Self::Batch(asr, words) => {
                if timestamps && *words {
                    let mut audio = segment.context.clone();
                    audio.extend_from_slice(&segment.audio);
                    let words = asr.transcribe_words(&audio)?;
                    let context_s = segment.context.len() as f32 / audio::TARGET_SAMPLE_RATE as f32;
                    let mut words = asr::keep_words_after(&words, context_s);
                    trim_leading_punctuation(&mut words);
                    offset_words(
                        &mut words,
                        segment.offset as f32 / audio::TARGET_SAMPLE_RATE as f32 - context_s,
                    );
                    Ok((asr::join_words(&words), Some(words)))
                } else if segment.context.is_empty() {
                    Ok((asr.transcribe(&segment.audio)?, None))
                } else {
                    let piece = asr.transcribe_segment_pieces(
                        &segment.context,
                        &segment.audio,
                        segment.start,
                    )?;
                    Ok((piece.words.join(" "), None))
                }
            }
            Self::Live(live, timing) => {
                let mut text = String::new();
                let mut first = true;
                let chunk = live.chunk_samples();
                for samples in segment.audio.chunks(chunk) {
                    let mut padded = samples.to_vec();
                    padded.resize(chunk, 0.0);
                    let out = timing.time(|| live.transcribe_chunk(&padded))?;
                    crate::append_live_output(&mut text, &mut first, &out);
                }
                let zeros = vec![0.0; chunk];
                for _ in 0..live::FLUSH_CHUNKS {
                    let out = timing.time(|| live.transcribe_chunk(&zeros))?;
                    crate::append_live_output(&mut text, &mut first, &out);
                }
                live.reset();
                Ok((text, None))
            }
            Self::Moonshine(session) => {
                let mut text = String::new();
                // The existing VAD frame is also a bounded native streaming input.
                for samples in segment.audio.chunks(vad::FRAME_LEN) {
                    text.push_str(&session.feed(samples)?.completed);
                }
                text.push_str(&session.finish()?.completed);
                Ok((text, None))
            }
        }
    }
}

/// Per-chunk timing of the Live (Nemotron) engine, reported as its real-time
/// factor once the file is done (scripts/check-real-models.py reads it).
#[derive(Default)]
struct LiveTiming {
    chunks: u128,
    total_ms: u128,
}
impl LiveTiming {
    fn time<T>(&mut self, f: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
        let start = std::time::Instant::now();
        let out = f()?;
        self.total_ms += start.elapsed().as_millis();
        self.chunks += 1;
        Ok(out)
    }
    fn line(&self) -> String {
        let avg_ms = self.total_ms / self.chunks.max(1);
        format!(
            "live: avg {avg_ms} ms per {} ms chunk (RTF {:.2})",
            live::CHUNK_MS,
            live::rtf(self.total_ms as f64 / self.chunks.max(1) as f64)
        )
    }
}

// Context punctuation belongs to the preceding cue, never a new cue by itself.
fn trim_leading_punctuation(words: &mut Vec<asr::Word>) {
    while let Some(first) = words.first_mut() {
        first.text = first
            .text
            .trim_start_matches(|c: char| asr::trailing_punct(&c.to_string()).is_some())
            .trim_start()
            .to_owned();
        if !first.text.is_empty() {
            break;
        }
        words.remove(0);
    }
}

fn offset_words(words: &mut [asr::Word], offset: f32) {
    for word in words {
        word.start = (word.start + offset).max(0.0);
        word.end = (word.end + offset).max(word.start);
    }
}
fn document(
    file: &str,
    engine: &str,
    duration: f64,
    segments: &[serde_json::Value],
    text: &str,
) -> serde_json::Value {
    serde_json::json!({"file":file,"engine":engine,"duration_s":duration,"segments":segments,"text":text})
}

fn timestamp(seconds: f64, separator: char) -> String {
    let millis = (seconds.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{separator}{:03}",
        millis / 3_600_000,
        millis / 60_000 % 60,
        millis / 1000 % 60,
        millis % 1000
    )
}
fn cue(index: usize, start: f64, end: f64, text: &str, vtt: bool) -> String {
    let separator = if vtt { '.' } else { ',' };
    // Escape cue markup; normalize line breaks so text cannot create extra cues.
    let text = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    format!(
        "{index}\n{} --> {}\n{text}\n\n",
        timestamp(start, separator),
        timestamp(end, separator)
    )
}
struct Run<'a> {
    format: &'a str,
    output: Output,
    text: String,
    segments: Vec<serde_json::Value>,
    count: usize,
    expected_duration: Option<f64>,
    typed: Vec<String>,
    pending: Option<char>,
    /// Segments come from the pause feeder (long file or --segmented).
    segmented: bool,
    /// Feeder segments seen so far, including ones without speech.
    seq: usize,
}
/// Typed words kept for joining the next segment (the old feeder window).
const TYPED_TAIL_WORDS: usize = 8;
impl Run<'_> {
    fn segment(&mut self, engine: &mut Engine, segment: segment::Segment) -> anyhow::Result<()> {
        if segment.audio.is_empty() {
            return Ok(());
        }
        let joined_batch = self.format == "txt" && matches!(engine, Engine::Batch(_, _));
        self.seq += 1;
        let started = std::time::Instant::now();
        let mut decode = match engine {
            Engine::Batch(..) if self.format != "txt" => "word-timestamp decode",
            Engine::Batch(..) => "context decode",
            Engine::Live(..) | Engine::Moonshine(..) => "streaming decode",
        };
        let (text, words) = if self.format == "txt" {
            if let Engine::Batch(asr, _) = engine {
                let piece =
                    asr.transcribe_segment_pieces(&segment.context, &segment.audio, segment.start)?;
                if piece.disjoint {
                    decode = "disjoint decode";
                }
                let (text, pending) = asr::join_piece(
                    &self.typed,
                    self.pending,
                    piece.boundary,
                    &piece.words,
                    asr::JoinMode::from_disjoint(piece.disjoint),
                );
                self.pending = pending;
                asr::append_typed_words(&mut self.typed, &text);
                // join_piece only needs the last few typed words; keep memory flat.
                let excess = self.typed.len().saturating_sub(TYPED_TAIL_WORDS);
                self.typed.drain(..excess);
                (text, None)
            } else {
                engine.decode(&segment, false)?
            }
        } else {
            engine.decode(&segment, true)?
        };
        if asr::is_empty_transcript(&text) {
            return Ok(());
        }
        if self.segmented {
            // Same per-segment diagnostics as the daemon's Pauses path.
            eprintln!(
                "segment {}: {:.1} s audio (context {:.1} s, {}, {decode}), transcribed in {} ms",
                self.seq,
                segment.audio.len() as f32 / audio::TARGET_SAMPLE_RATE as f32,
                segment.context.len() as f32 / audio::TARGET_SAMPLE_RATE as f32,
                segment.start.label(),
                started.elapsed().as_millis()
            );
        }
        let text = if joined_batch {
            text.as_str()
        } else {
            text.trim()
        };
        let audio_start = segment.offset as f64 / audio::TARGET_SAMPLE_RATE as f64;
        let audio_end = audio_start + segment.audio.len() as f64 / audio::TARGET_SAMPLE_RATE as f64;
        let (start, end) = words
            .as_ref()
            .filter(|words| !words.is_empty())
            .map(|words| {
                (
                    words
                        .iter()
                        .map(|w| f64::from(w.start))
                        .fold(f64::INFINITY, f64::min),
                    words.iter().map(|w| f64::from(w.end)).fold(0.0, f64::max),
                )
            })
            .unwrap_or((audio_start, audio_end));
        self.count += 1;
        let prefix = if joined_batch || self.text.is_empty() {
            ""
        } else {
            " "
        };
        match self.format {
            "txt" => self.output.write(&format!("{prefix}{text}"))?,
            "srt" | "vtt" => {
                self.output
                    .write(&cue(self.count, start, end, text, self.format == "vtt"))?
            }
            _ => {}
        }
        if self.format == "json" {
            let mut entry = serde_json::json!({"start":start,"end":end,"text":text});
            if let Some(words) = words {
                entry["words"] = words
                    .into_iter()
                    .map(|w| serde_json::json!({"start":w.start,"end":w.end,"text":w.text}))
                    .collect();
            }
            self.segments.push(entry);
        }
        self.text.push_str(prefix);
        self.text.push_str(text);
        if std::io::stderr().is_terminal() {
            eprintln!(
                "transcribed {}{}",
                timestamp(end, '.'),
                self.expected_duration
                    .map(|s| format!(" / {}", timestamp(s, '.')))
                    .unwrap_or_default()
            );
        }
        Ok(())
    }
}

pub(crate) fn run(args: &[String]) -> anyhow::Result<()> {
    let options = Options::parse(args)?;
    crate::init_config();
    // Unflagged runs use the Whole/Pauses (batch) model, even when Live is
    // selected for dictation; --live / --moonshine pick a streaming engine.
    let name = match options.engine {
        Some("--live") => "nemotron",
        Some(flag) => &flag[2..],
        None => config::speech_model().as_str(),
    };
    let mut run = Run {
        format: options.format,
        output: Output::new(options.path, options.output)?,
        text: String::new(),
        segments: Vec::new(),
        count: 0,
        expected_duration: None,
        typed: Vec::new(),
        pending: None,
        segmented: false,
        seq: 0,
    };
    let mut source = media::Audio::open(options.path)?;
    run.expected_duration = source.duration;
    let mut engine = Engine::load(name)?;
    if options.format != "txt" && matches!(name, "whisper" | "nemotron" | "moonshine") {
        eprintln!(
            "warning: {}",
            serde_json::json!({
                "fallback": true,
                "engine": name,
                "attempted": "word timestamps",
                "reason": "engine does not expose word timestamps",
                "message": "Using audio segment start/end timestamps"
            })
        );
    }
    if options.format == "vtt" {
        run.output.write("WEBVTT\n\n")?;
    }
    let mut resampler = if source.rate == audio::TARGET_SAMPLE_RATE {
        None
    } else {
        Some(audio::StreamResampler::new(source.rate)?)
    };
    let mut buffer = Vec::new();
    let mut feeder = None;
    let mut total = 0u64;
    loop {
        let decoded = source
            .next()
            .with_context(|| format!("Could not decode {}", options.path))?;
        let done = decoded.is_none();
        let samples = match (&mut resampler, decoded) {
            (Some(r), Some(samples)) => r.push(&samples)?,
            (Some(r), None) => r.finish()?,
            (None, Some(samples)) => samples,
            (None, None) => Vec::new(),
        };
        total += samples.len() as u64;
        buffer.extend(samples);
        if feeder.is_none() && (options.segmented || buffer.len() > segment::MAX_SEGMENT_SAMPLES) {
            feeder = Some(segment::Feeder::new(
                audio::TARGET_SAMPLE_RATE,
                vad::Vad::load_default()?,
            )?);
            run.segmented = true;
        }
        if let Some(feeder) = &mut feeder {
            for piece in feeder.push(&buffer)? {
                run.segment(&mut engine, piece)?;
            }
            buffer.clear();
            if done {
                for piece in feeder.finish()? {
                    run.segment(&mut engine, piece)?;
                }
            }
        } else if done {
            run.segment(
                &mut engine,
                segment::Segment {
                    audio: std::mem::take(&mut buffer),
                    context: Vec::new(),
                    start: segment::SegmentStart::Recording,
                    offset: 0,
                },
            )?;
        }
        if done {
            break;
        }
    }
    if let Engine::Live(_, timing) = &engine {
        eprintln!("{}", timing.line());
    }
    if let Some(punct) = run.pending
        && options.format == "txt"
        && !run.text.is_empty()
    {
        run.output.write(&punct.to_string())?;
        run.text.push(punct);
    }
    let duration = total as f64 / audio::TARGET_SAMPLE_RATE as f64;
    if run.text.is_empty() {
        eprintln!("{}", asr::no_speech_message(duration as f32));
    }
    match options.format {
        "json" => run.output.write(&format!(
            "{}\n",
            document(options.path, name, duration, &run.segments, &run.text)
        ))?,
        "txt" if !run.text.is_empty() => run.output.write("\n")?,
        _ => {}
    }
    run.output.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn options_any_order_and_usage_errors() {
        for args in [
            vec!["clip.mp3", "--format", "json", "--fluid", "-o", "out.json"],
            vec![
                "--fluid", "--output", "out.json", "--format", "json", "clip.mp3",
            ],
        ] {
            let args: Vec<String> = args.into_iter().map(str::to_owned).collect();
            let o = Options::parse(&args).unwrap();
            assert_eq!(
                (o.path, o.engine, o.format, o.output),
                ("clip.mp3", Some("--fluid"), "json", Some("out.json"))
            );
        }
        for args in [
            vec!["--format", "bad", "x"],
            vec!["--output"],
            vec!["x", "-o", "x"],
            vec!["--live", "--fluid", "x"],
        ] {
            assert!(
                Options::parse(&args.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err()
            );
        }
        assert!(Options::parse(&["-".into()]).is_ok());
    }
    #[test]
    fn live_timing_line_matches_the_check_script() {
        let mut timing = LiveTiming {
            chunks: 4,
            total_ms: 1120,
        };
        assert_eq!(
            timing.line(),
            "live: avg 280 ms per 560 ms chunk (RTF 0.50)"
        );
        timing.time(|| Ok(())).unwrap();
        assert_eq!(timing.chunks, 5);
        assert!(LiveTiming::default().line().starts_with("live: avg 0 ms"));
    }
    #[test]
    fn standard_subtitle_times_and_escaping() {
        assert_eq!(timestamp(3723.456, ','), "01:02:03,456");
        assert_eq!(timestamp(59.9996, '.'), "00:01:00.000");
        assert_eq!(
            cue(2, 12.5, 14.0, "a < b & c\nnext", false),
            "2\n00:00:12,500 --> 00:00:14,000\na &lt; b &amp; c next\n\n"
        );
        assert!(cue(1, 0.0, 1.0, "hi", true).contains("00:00:00.000 --> 00:00:01.000"));
    }
    #[test]
    fn context_punctuation_cannot_create_an_empty_caption() {
        let mut words = vec![
            asr::Word {
                text: ",".into(),
                start: 0.0,
                end: 0.1,
            },
            asr::Word {
                text: ": hello".into(),
                start: 0.2,
                end: 0.5,
            },
        ];
        trim_leading_punctuation(&mut words);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[0].start, 0.2);
        words[0].text = ".".into();
        trim_leading_punctuation(&mut words);
        assert!(words.is_empty());
    }

    #[test]
    fn json_empty_escaping_and_word_offsets() {
        let empty = document("-", "standard", 0.0, &[], "");
        assert_eq!(empty["segments"], serde_json::json!([]));
        assert_eq!(empty["text"], "");
        assert_eq!(empty["duration_s"], 0.0);
        let mut words = vec![asr::Word {
            start: 1.5,
            end: 2.0,
            text: "quoted \"word\"".into(),
        }];
        offset_words(&mut words, 12.0 - 1.5);
        assert_eq!((words[0].start, words[0].end), (12.0, 12.5));
        let entry = serde_json::json!({"start":12.0,"end":13.0,"text":words[0].text,"words":[{"start":words[0].start,"end":words[0].end,"text":words[0].text}]});
        let value = document("a\"b.mp3", "standard", 13.0, &[entry], "quoted \"word\"");
        let encoded = serde_json::to_string(&value).unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&encoded).unwrap(),
            value
        );
        assert!(encoded.contains("\\\""));
        assert_eq!(value["segments"][0]["words"][0]["start"], 12.0);
    }

    #[test]
    fn atomic_output_and_refuse_input_alias() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.wav");
        let output = dir.path().join("out.txt");
        std::fs::write(&input, b"audio").unwrap();
        std::fs::write(&output, b"old").unwrap();
        let mut writer = Output::new(input.to_str().unwrap(), output.to_str()).unwrap();
        writer.write("new").unwrap();
        assert_eq!(std::fs::read(&output).unwrap(), b"old");
        writer.finish().unwrap();
        assert_eq!(std::fs::read(&output).unwrap(), b"new");
        {
            let mut abandoned = Output::new(input.to_str().unwrap(), output.to_str()).unwrap();
            abandoned.write("incomplete").unwrap();
        }
        assert_eq!(std::fs::read(&output).unwrap(), b"new");
        assert!(Output::new(input.to_str().unwrap(), input.to_str()).is_err());
        #[cfg(unix)]
        {
            let link = dir.path().join("alias");
            std::fs::hard_link(&input, &link).unwrap();
            assert!(Output::new(input.to_str().unwrap(), link.to_str()).is_err());
        }
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            if cfg!(unix) { 3 } else { 2 }
        );
    }
}
