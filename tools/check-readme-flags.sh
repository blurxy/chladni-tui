#!/usr/bin/env bash
# Every command-line flag the README promises must exist in the source.
#
# This exists because the README once documented --anonymise while the shipped
# source had no such flag: the flag was added to a working copy, the screenshots
# were regenerated from that copy, and only the images were committed. Viewing
# the screenshots confirmed they were clean -- which is a fact about the
# screenshots, not about whether the mechanism shipped. The published repo
# claimed a capability it did not have.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
fail=0
flags=$(grep -oE -- '(^|[^-])--[a-z][a-z0-9-]+' README.md | grep -oE -- '--[a-z][a-z0-9-]+' | sort -u)
for f in $flags; do
  if ! grep -rqF -- "\"$f\"" src/ bin/ 2>/dev/null && ! grep -rqF -- "$f" bin/ 2>/dev/null; then
    echo "README documents $f but nothing in src/ or bin/ defines it" >&2
    fail=1
  fi
done
if ((fail)); then echo "FAIL: README promises flags the code does not have" >&2; exit 1; fi
echo "ok: all $(echo "$flags" | wc -w) README flags exist in the source"
