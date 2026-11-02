#!/bin/sh
# Build the release's pinned Linux Vulkan loader without installing system-wide.
# Usage: sh scripts/build-vulkan-loader.sh <release directory>
set -eu
[ "$#" = 1 ] || { echo "usage: $0 <release directory>" >&2; exit 1; }
DEST="$(realpath "$1")"
VERSION=1.4.363
LOADER_SHA=cc520304a4d8de9a6a1699553d6090aef0c25f87ef6cbc16b7c84ebd99093264
HEADERS_SHA=cbaf687d3c59b9666fe080e7f8396b6b4f4d344a768ee6b337c59852f4526b68
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM
curl -fsSL "https://codeload.github.com/KhronosGroup/Vulkan-Loader/tar.gz/refs/tags/v$VERSION" -o "$WORK/loader.tar.gz"
curl -fsSL "https://codeload.github.com/KhronosGroup/Vulkan-Headers/tar.gz/refs/tags/v$VERSION" -o "$WORK/headers.tar.gz"
printf '%s  %s\n' "$LOADER_SHA" "$WORK/loader.tar.gz" "$HEADERS_SHA" "$WORK/headers.tar.gz" | sha256sum -c -
tar -xzf "$WORK/loader.tar.gz" -C "$WORK"
tar -xzf "$WORK/headers.tar.gz" -C "$WORK"
cmake -S "$WORK/Vulkan-Headers-$VERSION" -B "$WORK/headers-build" \
    -DCMAKE_INSTALL_PREFIX="$WORK/install" -DVULKAN_HEADERS_ENABLE_TESTS=OFF
cmake --install "$WORK/headers-build"
# Whisper only needs compute. Avoid window-system dependencies in the loader.
# Use /usr as the loader's logical install prefix so system ICD discovery still
# searches the normal /etc and /usr/share paths, not this temporary build tree.
cmake -S "$WORK/Vulkan-Loader-$VERSION" -B "$WORK/loader-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_PREFIX_PATH="$WORK/install" -DBUILD_TESTS=OFF \
    -DBUILD_WSI_XCB_SUPPORT=OFF -DBUILD_WSI_XLIB_SUPPORT=OFF \
    -DBUILD_WSI_WAYLAND_SUPPORT=OFF
cmake --build "$WORK/loader-build" --parallel
cp -L "$WORK/loader-build/loader/libvulkan.so.1" "$DEST/libvulkan.so.1"
