#!/bin/sh
# Install the latest dictate release for this machine.
#
#   curl -fsSL https://raw.githubusercontent.com/saostad/dictate/main/install.sh | sh
#
# Files fetched with curl get no quarantine flag, so macOS Gatekeeper does not
# block the unsigned binary the way it does after a browser download.
#
# DICTATE_VERSION picks a release tag (default: latest). DICTATE_INSTALL_DIR
# picks where the unpacked release lives (default: ~/.local/lib/dictate); the
# `dictate` command is linked into DICTATE_BIN_DIR (default: ~/.local/bin).
set -eu

REPO="saostad/dictate"
INSTALL_DIR="${DICTATE_INSTALL_DIR:-$HOME/.local/lib/dictate}"
BIN_DIR="${DICTATE_BIN_DIR:-$HOME/.local/bin}"

fail() {
    echo "dictate install: $*" >&2
    exit 1
}

os="$(uname -s)"
arch="$(uname -m)"
case "$os/$arch" in
    Linux/x86_64 | Linux/amd64) asset="dictate-linux-x86_64" ;;
    Linux/aarch64 | Linux/arm64) asset="dictate-linux-aarch64" ;;
    Darwin/arm64) asset="dictate-macos-aarch64" ;;
    Darwin/x86_64) fail "there is no Intel macOS build (ONNX Runtime ships no prebuilt library for it)" ;;
    *) fail "no release for $os $arch; build from source with: cargo build --release" ;;
esac

if [ -n "${DICTATE_VERSION:-}" ]; then
    base="https://github.com/$REPO/releases/download/$DICTATE_VERSION"
else
    base="https://github.com/$REPO/releases/latest/download"
fi

command -v curl >/dev/null 2>&1 || fail "curl is required"
if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    fail "sha256sum or shasum is required"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM

echo "Downloading $asset.tar.gz..."
curl -fSL --progress-bar -o "$tmp/$asset.tar.gz" "$base/$asset.tar.gz" ||
    fail "download failed: $base/$asset.tar.gz"
curl -fsSL -o "$tmp/$asset.tar.gz.sha256" "$base/$asset.tar.gz.sha256" ||
    fail "download failed: $base/$asset.tar.gz.sha256"

expected="$(awk '{print $1}' "$tmp/$asset.tar.gz.sha256")"
actual="$(sha256 "$tmp/$asset.tar.gz")"
[ "$expected" = "$actual" ] || fail "SHA256 mismatch: expected $expected, got $actual"

tar -xzf "$tmp/$asset.tar.gz" -C "$tmp"
[ -x "$tmp/$asset/dictate" ] || fail "archive has no dictate binary"

# Clear any quarantine flag (e.g. inherited from a quarantined parent process).
if [ "$os" = "Darwin" ]; then
    xattr -dr com.apple.quarantine "$tmp/$asset" 2>/dev/null || true
fi

rm -rf "$INSTALL_DIR"
mkdir -p "$(dirname "$INSTALL_DIR")" "$BIN_DIR"
mv "$tmp/$asset" "$INSTALL_DIR"
# Linux x86_64 loads libonnxruntime.so from next to the real binary, so link
# the binary rather than copying it out of its folder.
ln -sf "$INSTALL_DIR/dictate" "$BIN_DIR/dictate"

"$BIN_DIR/dictate" help >/dev/null 2>&1 || fail "installed binary does not run: $BIN_DIR/dictate"

echo "Installed dictate to $BIN_DIR/dictate"
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) echo "Add $BIN_DIR to your PATH, e.g.: echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc" ;;
esac
echo "Next: dictate download, then dictate daemon"
