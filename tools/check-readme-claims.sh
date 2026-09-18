#!/usr/bin/env bash
# Fail if the README claims something the repo does not have.
#
# This exists because the README once documented --anonymise while the shipped
# source defined no such flag: it was added to a working copy, the screenshots
# were regenerated from that copy, and only the images were committed. Viewing
# the screenshots confirmed they were clean -- a fact about the screenshots, not
# about whether the mechanism shipped.
#
# Note on flag extraction. The first version of this script flagged every --foo
# anywhere in the README, and immediately produced a false positive on
# --unshare-net: a bubblewrap flag discussed in prose, not one this tool accepts.
# A checker that cannot tell "a flag we define" from "a flag we mention" measures
# the wrong thing -- the same defect it was written to catch. So a flag counts as
# CLAIMED only where the README shows it passed to a command this repo ships.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
fail=0

# EVERY flag in the README counts, minus an explicit exemption list. Narrowing
# this to flags shown beside one of our binaries was tried and silently stopped
# catching the very bug this was written for: the flag was documented in prose,
# not in a usage block. Fail closed -- a new external flag must be declared.
exempt=tools/external-flags.txt
claimed=$(grep -oE -- '(^|[^-])--[a-z][a-z0-9-]+' README.md 2>/dev/null \
          | grep -oE -- '--[a-z][a-z0-9-]+' | sort -u)
for f in $claimed; do
  grep -rqF -- "\"$f\"" src/ bin/ 2>/dev/null && continue
  grep -rqF -- "$f" bin/ 2>/dev/null && continue
  grep -qE "^\\s*$f(\\s|$)" "$exempt" 2>/dev/null && continue
  echo "README documents $f, nothing defines it, and it is not declared in $exempt" >&2
  fail=1
done

for p in $(grep -oE '`(src|bin|docs|tools|config)/[A-Za-z0-9_.-]+`' README.md 2>/dev/null | tr -d '`' | sort -u); do
  [ -e "$p" ] || { echo "README references $p, which does not exist" >&2; fail=1; }
done
for i in $(grep -oE '!\[[^]]*\]\(([^)]+)\)' README.md 2>/dev/null | sed 's/.*(\(.*\))/\1/'); do
  case "$i" in http*) continue;; esac
  [ -f "$i" ] || { echo "README embeds $i, which does not exist" >&2; fail=1; }
done
if grep -qi '\bMIT\b' README.md && [ ! -f LICENSE ]; then
  echo "README claims MIT, no LICENSE file" >&2; fail=1
fi
for p in src/*.py; do [ -f "$p" ] || continue
  python3 -c "import ast;ast.parse(open('$p').read())" 2>/dev/null \
    || { echo "$p does not parse" >&2; fail=1; }
done
for s in bin/* tools/* install.sh; do [ -f "$s" ] || continue
  head -1 "$s" | grep -q bash || continue
  bash -n "$s" 2>/dev/null || { echo "$s does not parse" >&2; fail=1; }
done

# NUMBERS DECAY SILENTLY. A path that stops existing is caught above; a figure
# that stops being true is not, because nothing about "16,044 frames" looks
# wrong once it is 12,988. Every number below is recomputed from its SOURCE and
# the README must still contain it. Only claims with a reproducible source
# belong here -- prose and screenshots are still on the author.
# Matched against a whitespace-flattened copy: README prose wraps, so "760
# verses" is routinely split across two lines and a literal grep reports a stale
# number that is merely hyphenated by the margin.
README_FLAT=$(tr "\n" " " < README.md | tr -s " ")
claim() {  # claim <what> <expected> [alt]
  local what="$1" want="$2" alt="${3:-}"
  if [[ $README_FLAT == *"$want"* ]] || { [ -n "$alt" ] && [[ $README_FLAT == *"$alt"* ]]; }; then
    return 0
  fi
  echo "README no longer states $what -- it is now $want" >&2
  fail=1
}

PY_BIN=${PY_BIN:-$HOME/.local/share/cymatics/.venv/bin/python}
if [ -x "$PY_BIN" ]; then
  # library scale, straight out of the timeline the screensaver reads
  eval "$("$PY_BIN" - <<'EOF' 2>/dev/null
import json, os, numpy as np
tl = os.path.expanduser("~/.local/share/omarchy-cymatics-screensaver/timeline.npz")
if os.path.exists(tl):
    z = np.load(tl, allow_pickle=False)
    m = json.loads(str(z["meta"])); t = m["tracks"] if isinstance(m, dict) else m
    ayat = sum(len(x.get("text") or []) for x in t)
    print("N_TRACKS=%d" % len(t))
    print("N_MODES=%d" % z["mn"].shape[0])
    print("N_AYAT=%d" % ayat)
EOF
)"
  [ -n "${N_MODES:-}" ] && claim "the mode count" "$N_MODES modes" "${N_MODES} of ${N_MODES}"
  [ -n "${N_AYAT:-}" ]  && claim "the verse count" "$N_AYAT verses" "all $N_AYAT verses"
  [ -n "${N_TRACKS:-}" ] && claim "the track count" "$N_TRACKS tracks"

  # the invariant harness reports its own frame count; the README quotes it
  LC=$("$PY_BIN" tools/layout-check.py 2>/dev/null | grep -oE '^layout-check: [0-9]+' | grep -oE '[0-9]+')
  if [ -n "$LC" ]; then
    printf -v LCC "%'d" "$LC" 2>/dev/null || LCC=$LC
    claim "the layout-check frame count" "$LCC frames" "$LC frames"
  fi
fi

# constants the README quotes by value
DMC=$(grep -oE '^DOTS_MIN_COLS = [0-9]+' src/screensaver.py | grep -oE '[0-9]+')
[ -n "$DMC" ] && claim "the dither threshold" "$DMC up" "$DMC columns"

((fail)) && { echo "FAIL: the README claims something this repo does not have" >&2; exit 1; }
echo "ok: README claims check out ($(echo "$claimed" | wc -w) flags, all referenced paths and images)"
