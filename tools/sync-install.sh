#!/bin/bash
# Copy the public repo's renderer and library over the local Omarchy install.
#
# WHY THIS EXISTS: the install at ~/.local/share/omarchy-cymatics-screensaver is
# what the screensaver launcher actually runs, so that is where edits naturally
# happen -- and twice in one night ten-plus fixes accumulated there and never
# reached this repo, leaving the public code describing a screensaver that no
# longer existed. The repo is the source of truth; the install is a deployment.
# Edit here, run this, and the two cannot silently disagree.
#
# The only intended difference is the module docstring's first line, which names
# the Omarchy integration in the install.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$HOME/.local/share/omarchy-cymatics-screensaver}"
[ -d "$DEST" ] || { echo "no install at $DEST" >&2; exit 1; }
cp "$REPO/src/library.py" "$DEST/library.py"
sed '1s|^"""chladni-tui: a Chladni sand plate for the terminal\.|"""Cymatics screensaver for Omarchy.|' \
    "$REPO/src/screensaver.py" > "$DEST/cyscreen.py"
n=$(diff "$REPO/src/screensaver.py" "$DEST/cyscreen.py" | grep -c '^[<>]' || true)
[ "$n" -le 2 ] || { echo "unexpected drift after sync: $n lines" >&2; exit 1; }
echo "synced $REPO -> $DEST (docstring line only differs)"
