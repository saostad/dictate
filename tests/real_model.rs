//! Transcribes a real WAV with the downloaded model. Ignored by default so CI
//! never downloads the 660 MB model; run with:
//! `DICTATE_TEST_WAV=/path/to/speech.wav cargo test -- --ignored`.

#[test]
#[ignore]
fn transcribe_with_downloaded_model() {
    let wav = std::env::var("DICTATE_TEST_WAV")
        .expect("set DICTATE_TEST_WAV to a WAV file to run this test");
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
    assert!(!text.trim().is_empty(), "transcription was empty");
}
