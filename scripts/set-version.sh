#!/bin/sh
# Set the package version in Cargo.toml and Cargo.lock.
set -eu

version="${1:?usage: set-version.sh <version>}"

awk -v v="$version" '
    !done && /^version = / { $0 = "version = \"" v "\""; done = 1 }
    { print }
' Cargo.toml > Cargo.toml.new
mv Cargo.toml.new Cargo.toml

awk -v v="$version" '
    prev == "name = \"dictate\"" && /^version = / { $0 = "version = \"" v "\"" }
    { print; prev = $0 }
' Cargo.lock > Cargo.lock.new
mv Cargo.lock.new Cargo.lock

grep -q "^version = \"$version\"$" Cargo.toml
grep -A1 '^name = "dictate"$' Cargo.lock | grep -q "^version = \"$version\"$"
