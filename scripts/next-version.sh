#!/bin/sh
# Print the next release version, from the commits since the current version's tag.
# feat or breaking change -> minor (major once at 1.0), anything else -> patch.
set -eu

current="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)"
[ -n "$current" ] || { echo "no version in Cargo.toml" >&2; exit 1; }

if git rev-parse -q --verify "refs/tags/v$current" >/dev/null; then
    range="v$current..HEAD"
else
    range="HEAD"
fi

subjects="$(git log --format=%s "$range")"
bodies="$(git log --format=%b "$range")"

if printf '%s\n' "$subjects" | grep -Eq '^[a-z]+(\([^)]*\))?!:' ||
    printf '%s\n' "$bodies" | grep -q '^BREAKING CHANGE'; then
    level=breaking
elif printf '%s\n' "$subjects" | grep -Eq '^feat(\([^)]*\))?:'; then
    level=feature
else
    level=fix
fi

IFS=. read -r major minor patch <<EOF
$current
EOF

case "$level" in
    breaking)
        if [ "$major" = 0 ]; then minor=$((minor + 1)); patch=0
        else major=$((major + 1)); minor=0; patch=0; fi ;;
    feature)
        if [ "$major" = 0 ]; then minor=$((minor + 1)); patch=0
        else minor=$((minor + 1)); patch=0; fi ;;
    fix)
        patch=$((patch + 1)) ;;
esac

echo "$major.$minor.$patch"
