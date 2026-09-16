#!/usr/bin/env bash
# Install chladni-tui into ~/.local (no root, nothing outside your home).
set -euo pipefail

PREFIX="${PREFIX:-$HOME/.local}"
SHARE="$PREFIX/share/chladni-tui"
BIN="$PREFIX/bin"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

need() { command -v "$1" >/dev/null || { echo "missing: $1" >&2; return 1; }; }
for c in python3 ffmpeg curl; do need "$c" || exit 1; done
for c in fzf yt-dlp gum; do need "$c" || echo "optional tool not found: $c" >&2; done

mkdir -p "$SHARE" "$BIN"
cp "$HERE"/src/*.py "$SHARE/"
mv -f "$SHARE/screensaver.py" "$SHARE/cyscreen.py"

if [[ ! -x $SHARE/.venv/bin/python ]]; then
  echo "creating venv..."
  python3 -m venv "$SHARE/.venv" 2>/dev/null || {
    command -v uv >/dev/null && uv venv "$SHARE/.venv"; }
fi
"$SHARE/.venv/bin/python" -m pip install --quiet --upgrade numpy scipy Pillow 2>/dev/null \
  || uv pip install --python "$SHARE/.venv/bin/python" --quiet numpy scipy Pillow

for f in "$HERE"/bin/*; do
  dst="$BIN/$(basename "$f")"
  sed "s|\$HOME/.local/share/chladni-tui/.venv|$SHARE/.venv|g; s|\$HOME/.local/share/chladni-tui|$SHARE|g" \
    "$f" > "$dst"
  chmod +x "$dst"
done

CONF="${XDG_CONFIG_HOME:-$HOME/.config}/chladni-tui"
mkdir -p "$CONF"
cp -n "$HERE"/config/* "$CONF"/ 2>/dev/null || true

echo
echo "installed to $SHARE"
echo "add some audio:  chladni-add"
echo "then run:        chladni-screensaver"
