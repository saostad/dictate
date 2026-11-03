#!/bin/sh
# Fail if whisper.cpp was built with instructions beyond the x86_64 baseline
# that .cargo/config.toml asks for (GGML_AVX/AVX2/FMA/F16C/BMI2 all OFF).
# A cached or stale native build can silently keep them, and the release then
# dies with SIGILL on older CPUs (e.g. Sandy Bridge has no BMI2).
#
#   sh scripts/check-baseline-isa.sh target/<triple>/release
#
# It scans whisper-rs-sys's static libraries only: Rust crates such as memchr
# legitimately ship AVX2 paths that they select at run time.
set -eu

dir="${1:?usage: check-baseline-isa.sh <cargo profile dir, e.g. target/x86_64-unknown-linux-gnu/release>}"
libs="$(find "$dir/build" -path '*whisper-rs-sys-*/out/lib/*' -name '*.a' | sort)"
[ -n "$libs" ] || { echo "FAIL: no whisper-rs-sys static libraries under $dir/build"; exit 1; }

# VEX/EVEX (AVX, AVX2, FMA, F16C, AVX-512: every mnemonic starts with v) and
# BMI1/BMI2. tzcnt/lzcnt are left out: objdump prints the baseline `rep bsf`
# as tzcnt.
pattern='[[:space:]](v[a-z0-9]+|shlx|shrx|sarx|rorx|mulx|pdep|pext|bzhi|andn|blsi|blsr|blsmsk)[[:space:]]'
status=0
for lib in $libs; do
    hits="$(objdump -d --no-show-raw-insn "$lib" | grep -Ec "$pattern" || true)"
    if [ "$hits" -ne 0 ]; then
        echo "FAIL: $lib has $hits instructions beyond the x86_64 baseline, e.g.:"
        objdump -d --no-show-raw-insn "$lib" | grep -E "$pattern" | sort -u -k2,2 | head -5
        status=1
    else
        echo "OK: $lib"
    fi
done
exit "$status"
