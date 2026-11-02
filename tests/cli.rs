use std::process::Command;

#[test]
fn cli_help_version_and_usage_errors() {
    for args in [
        vec!["--version"],
        vec!["-V"],
        vec!["version"],
        vec!["-h"],
        vec!["download", "--help"],
        vec!["transcribe", "--help"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_dictate"))
            .args(&args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty(), "{args:?}: unexpected stderr");
        assert!(!output.stdout.is_empty(), "{args:?}: missing output");
        if matches!(args[0], "--version" | "-V" | "version") {
            assert_eq!(
                String::from_utf8(output.stdout).unwrap(),
                if args[0] == "version" {
                    format!(
                        "dictate {}\nhttps://github.com/saostad/dictate/releases/latest\n",
                        env!("CARGO_PKG_VERSION")
                    )
                } else {
                    format!("dictate {}\n", env!("CARGO_PKG_VERSION"))
                }
            );
        }
    }
    for args in [
        vec![],
        vec!["transcribe", "--bogus", "x.wav"],
        vec!["transcribe", "--format", "yaml", "x.wav"],
        vec!["transcribe", "x.wav", "-o", "x.wav"],
        vec!["transcribe", "x.wav", "extra.wav"],
        vec!["benchmark", "--bogus"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_dictate"))
            .args(&args)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty(), "{args:?}: unexpected stdout");
        assert!(!output.stderr.is_empty(), "{args:?}: missing diagnostic");
    }
}

#[test]
fn unreadable_wav_is_a_runtime_failure_with_no_stdout() {
    // An existing non-WAV file fails before any model is loaded.
    let output = Command::new(env!("CARGO_BIN_EXE_dictate"))
        .args([
            "transcribe",
            concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Could not read audio"));
}
