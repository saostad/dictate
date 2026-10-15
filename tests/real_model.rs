//! Transcribes a real clip with the downloaded model. Ignored by default;
//! run with `cargo test -- --ignored`. Uses `$DICTATE_TEST_WAV` when set,
//! otherwise `tests/fixtures/jfk.wav`.

use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn test_wav() -> String {
    std::env::var("DICTATE_TEST_WAV").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jfk.wav").to_string()
    })
}

#[test]
#[ignore]
fn transcribe_with_downloaded_model() {
    let wav = test_wav();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_dictate"))
        .args(["transcribe", &wav])
        .output()
        .expect("run dictate transcribe");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(
        text.to_lowercase()
            .contains("ask not what your country can do for you"),
        "unexpected transcription: {text}"
    );
}

/// Read any WAV as mono f32 samples plus its sample rate (i16/f32, downmixed).
fn read_wav_mono(path: &str) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("open test WAV");
    let spec = reader.spec();
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.expect("f32 sample"))
            .collect(),
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.expect("i16 sample") as f32 / 32768.0)
            .collect(),
        (format, bits) => panic!("unsupported test WAV format: {format:?} {bits}-bit"),
    };
    assert!(spec.channels > 0, "test WAV has no channels");
    let channels = usize::from(spec.channels);
    let mono: Vec<f32> = samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect();
    (mono, spec.sample_rate)
}

/// Linear-interpolate `mono` from `from_rate` to `to_rate`.
fn resample_linear(mono: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if mono.is_empty() || from_rate == to_rate {
        return mono.to_vec();
    }
    let out_len = (mono.len() as u64 * u64::from(to_rate) / u64::from(from_rate)) as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * f64::from(from_rate) / f64::from(to_rate);
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(mono.len() - 1);
            let frac = (pos - lo as f64) as f32;
            mono[lo] * (1.0 - frac) + mono[hi] * frac
        })
        .collect()
}

#[test]
#[ignore]
fn segmented_transcribe_at_48k_keeps_word_onsets() {
    // Device-rate path: resample the 16 kHz fixture to 48 kHz, then run the
    // same feeder path the streaming daemon uses.
    let (mono, rate) = read_wav_mono(&test_wav());
    let up = resample_linear(&mono, rate, 48_000);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp =
        std::env::temp_dir().join(format!("dictate-test-seg48-{}-{n}.wav", std::process::id()));
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&tmp, spec).expect("create temp WAV");
    for sample in &up {
        writer
            .write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)
            .expect("write sample");
    }
    writer.finalize().expect("finalize temp WAV");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_dictate"))
        .args(["transcribe", "--segmented"])
        .arg(&tmp)
        .output()
        .expect("run dictate transcribe --segmented");
    std::fs::remove_file(&tmp).ok();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout);
    // Exact sentence: lowercased, with `.`/`,` normalized away (punctuation
    // at the cut may differ between runs).
    let normalize = |s: &str| {
        s.to_lowercase()
            .chars()
            .filter(|c| *c != '.' && *c != ',')
            .collect::<String>()
    };
    assert_eq!(
        normalize(text.trim()),
        normalize(
            "and so, my fellow americans, ask not what your country can do for you, ask what you can do for your country."
        ),
        "segmented transcription differs: {text}"
    );
    // The onset check is only meaningful if segmentation actually happened
    // (one segment would equal plain transcribe, already covered above).
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("segment 1:"),
        "expected segmented output, stderr: {stderr}"
    );
    assert!(
        stderr.contains("segment 2:"),
        "expected at least two segments, stderr: {stderr}"
    );
}
