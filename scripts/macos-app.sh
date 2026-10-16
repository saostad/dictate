#!/bin/sh
# Build Dictate.app (a menu bar app that runs `dictate daemon`) and install it.
#
#   scripts/macos-app.sh            build target/release/Dictate.app and copy it
#                                   to /Applications (~/Applications if that is
#                                   not writable)
#   scripts/macos-app.sh --no-install   only build the bundle
#
# The bundle is signed with a local self-signed certificate (no Apple Developer
# account needed), so macOS gives it its own Microphone and Accessibility
# entries named "Dictate", and they survive rebuilds.
set -eu

install=1
case "${1:-}" in
    "") ;;
    --no-install) install=0 ;;
    *)
        echo "Usage: scripts/macos-app.sh [--no-install]" >&2
        exit 2
        ;;
esac

[ "$(uname -s)" = "Darwin" ] || {
    echo "macos-app.sh: macOS only" >&2
    exit 1
}

identity="Dictate Local Signing"
keychain="$HOME/Library/Keychains/dictate-signing.keychain-db"

# Create a self-signed code-signing certificate in its own password-less
# keychain (first run only). Signing every build with the same certificate
# keeps the app's identity stable, so macOS keeps its permissions across
# rebuilds; ad-hoc signatures change with every build.
ensure_signing_identity() {
    if [ ! -f "$keychain" ]; then
        tmp="$(mktemp -d)"
        openssl req -x509 -newkey rsa:2048 -nodes -days 7300 \
            -keyout "$tmp/key.pem" -out "$tmp/cert.pem" -subj "/CN=$identity" \
            -addext "keyUsage=critical,digitalSignature" \
            -addext "extendedKeyUsage=critical,codeSigning" \
            -addext "basicConstraints=critical,CA:false" 2>/dev/null
        # 3DES/SHA1 PKCS#12: the encryption `security import` understands.
        openssl pkcs12 -export -inkey "$tmp/key.pem" -in "$tmp/cert.pem" \
            -out "$tmp/id.p12" -passout pass:dictate \
            -keypbe PBE-SHA1-3DES -certpbe PBE-SHA1-3DES -macalg sha1
        security create-keychain -p "" "$keychain"
        security set-keychain-settings "$keychain"
        security unlock-keychain -p "" "$keychain"
        security import "$tmp/id.p12" -k "$keychain" -P dictate -T /usr/bin/codesign
        security set-key-partition-list -S apple-tool:,apple: -s -k "" "$keychain" >/dev/null
        rm -rf "$tmp"
    fi
    security unlock-keychain -p "" "$keychain"
    # codesign only finds identities in keychains on the user search list.
    if ! security list-keychains -d user | grep -q "dictate-signing"; then
        # shellcheck disable=SC2046
        security list-keychains -d user -s $(security list-keychains -d user | tr -d '"') "$keychain"
    fi
}

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)"
bundle_id="com.saostad.dictate"

cargo build --release

app="target/release/Dictate.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS"
cp target/release/dictate "$app/Contents/MacOS/dictate"

cat >"$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Dictate</string>
    <key>CFBundleDisplayName</key>
    <string>Dictate</string>
    <key>CFBundleIdentifier</key>
    <string>$bundle_id</string>
    <key>CFBundleExecutable</key>
    <string>dictate</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$version</string>
    <key>CFBundleVersion</key>
    <string>$version</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Dictate records your voice while you dictate and turns it into text.</string>
</dict>
</plist>
EOF

ensure_signing_identity
codesign --force --sign "$identity" --identifier "$bundle_id" "$app"
codesign --verify --strict "$app"
echo "Built $app"

[ "$install" = 1 ] || exit 0

dest_dir="/Applications"
[ -w "$dest_dir" ] || dest_dir="$HOME/Applications"
mkdir -p "$dest_dir"
dest="$dest_dir/Dictate.app"

# Quit a running copy so the new build replaces it cleanly.
pkill -f "$dest/Contents/MacOS/dictate" 2>/dev/null || true
while pgrep -f "$dest/Contents/MacOS/dictate" >/dev/null 2>&1; do sleep 0.2; done
rm -rf "$dest"
cp -R "$app" "$dest"
echo "Installed $dest"
echo "Start it with: open \"$dest\""
