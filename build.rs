// Link the Clang OS X runtime on macOS release builds.
//
// Clang emits a reference to `___isPlatformVersionAtLeast` (for `@available`
// in ONNX Runtime's Obj-C/CoreML code) that lives in `libclang_rt.osx.a`,
// which rustc's `-nodefaultlibs` link leaves out, failing the release link
// with `Undefined symbols for architecture arm64`. Pointing rustc at the
// static runtime fixes it. Failures only warn (debug builds link without
// it), and non-macOS targets are untouched.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    link_clang_runtime();
}

fn link_clang_runtime() {
    let output = match std::process::Command::new("clang")
        .arg("--print-runtime-dir")
        .output()
    {
        Ok(output) => output,
        Err(err) => {
            println!(
                "cargo:warning=macOS Clang runtime link skipped: could not run `clang --print-runtime-dir`: {err}"
            );
            return;
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        if stderr.is_empty() {
            println!(
                "cargo:warning=macOS Clang runtime link skipped: `clang --print-runtime-dir` exited with {status}",
                status = output.status
            );
        } else {
            println!(
                "cargo:warning=macOS Clang runtime link skipped: `clang --print-runtime-dir` failed: {stderr}"
            );
        }
        return;
    }
    let dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if dir.is_empty() {
        println!(
            "cargo:warning=macOS Clang runtime link skipped: `clang --print-runtime-dir` printed nothing"
        );
        return;
    }
    let lib = std::path::Path::new(&dir).join("libclang_rt.osx.a");
    if !lib.is_file() {
        println!(
            "cargo:warning=macOS Clang runtime link skipped: {} not found in {dir}",
            lib.display()
        );
        return;
    }
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
}
