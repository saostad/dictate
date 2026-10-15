//! Transcribes a real clip with the downloaded model. Ignored by default;
//! run with `cargo test -- --ignored`. Uses `$DICTATE_TEST_WAV` when set,
//! otherwise `tests/fixtures/jfk.wav`.

#[test]
#[ignore]
fn transcribe_with_downloaded_model() {
    let wav = std::env::var("DICTATE_TEST_WAV").unwrap_or_else(|_| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jfk.wav").to_string()
    });
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
