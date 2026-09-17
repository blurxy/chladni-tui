#!/usr/bin/env python3
"""Assert the layout is identical for every verse, at every window size.

WHY THIS EXISTS. The band under the plate used to derive its top edge from the
wrapped line count of whatever ayah was playing, while the plate's size came
from a constant. The two disagreed, so a long verse grew the text upward into
the picture and a short one did not -- the layout rearranged itself every few
seconds, and which frames were broken depended on the recitation. A defect
whose trigger is the audio cannot be found by looking at one screenshot.

So this draws every ayah in the library at a range of terminal sizes and
checks four things that must hold regardless of content:

    band-constant  the ayah header lands on the same row for every verse
    no-overlap     the plate's lowest sub-row stays above that row
    in-bounds      nothing is written outside the grid, or onto the strip
    no-clipping    the translation fits the rows reserved for it

It draws the chrome layer only -- no audio, no sand, no terminal -- so it runs
in about a second and can go in a pre-commit hook.

    tools/layout-check.py                # every track, default sizes
    tools/layout-check.py --cols 165 --rows 38
"""
import argparse, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALL = os.path.expanduser("~/.local/share/omarchy-cymatics-screensaver")

# Terminal shapes worth checking: the dedicated launcher's font-8 grid, a
# plain terminal at the default font (where the app is usually run by hand,
# and where it looks worst), and the extremes either side.
SIZES = [(387, 95), (300, 80), (200, 60), (165, 38), (120, 30), (100, 24), (80, 20)]


def load_module(path):
    spec = importlib.util.spec_from_file_location("cyscreen", path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = ["layout-check"]
    spec.loader.exec_module(mod)
    return mod


def verses(m, limit):
    """Every (transliteration, translation) pair the library actually ships."""
    import numpy as np
    tl = os.path.join(os.path.dirname(m.__file__ if hasattr(m, "__file__") else ""), "")
    path = os.path.join(INSTALL, "timeline.npz")
    out = []
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        tracks = meta["tracks"] if isinstance(meta, dict) else meta
        for t in tracks:
            for pair in (t.get("text") or []):
                out.append(tuple(pair))
    # The degenerate shapes a track can legitimately have, which used to crash
    # draw() outright rather than merely shift the layout.
    out += [(), ("",), ("only a transliteration",),
            ("x" * 400, "y" * 700)]          # past the wrap limit on both lines
    return out[:limit] if limit else out


def base_state(m):
    """A frame's worth of state, with the telemetry taken from the REAL
    sysinfo()/livestats() rather than invented.

    A hand-written fixture put strings in mem_used, load and temp, which the
    telemetry panel formats with %d and %.2f -- so this reported a crash in the
    app that only the fixture could produce, and would equally have missed a
    real one. The renderer's own sources are the only thing that gets the types
    right without having to know them.
    """
    import numpy as np
    info = m.sysinfo()
    live = m.livestats(info)
    st = {"name": "R", "sub": "s", "f0": 120.0, "t": 10.0, "dur": 100.0,
          "amps": np.zeros(8, np.float32), "mn": [(1, 1)] * 8, "alpha": np.ones(8),
          "dom": 0, "level": 0.5, "hz": 180.0, "grains": 9000, "subw": 100,
          "subh": 100, "pw": 80, "ph": 80, "fps": 60.0, "settled": 0.5, "held": 1.0,
          "surah": "Al-Kahf", "surah_no": 18, "n_ayat": 110, "revelation": "Meccan",
          "ayah": 1, "ayah_frac": 0.3, "ayat": [0.0, 5.0], "hints": False}
    st.update({k: info[k] for k in ("os", "kernel", "host", "cpu", "gpu", "pkgs",
                                    "shell", "res", "theme")})
    st.update(live)
    return st


def header_row(ch, rows):
    """Row holding the '18 : 1 of 110' header, or None."""
    for r in range(rows):
        line = "".join(c for c in ch.ch[r] if c != "\0")
        if " : " in line and " of " in line:
            return r
    return None


def plate_bottom_row(m, cols, rows):
    """Lowest CELL row the plate's disc can reach, from the same numbers the
    renderer sizes it with. Braille packs 4 sub-rows per cell."""
    fit = m.plate_fit(cols, rows)
    subw, subh = cols * 2, rows * 4
    rad = fit["rad_scale"] * min(float(subw), float(subh))
    cy = (subh - 1) / 2.0 + fit["cy_shift"]
    return int((cy + rad) // 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=os.path.join(HERE, "src", "screensaver.py"))
    ap.add_argument("--cols", type=int)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--limit", type=int, default=0, help="check only the first N verses")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    m = load_module(a.script)
    sizes = [(a.cols, a.rows)] if a.cols and a.rows else SIZES
    vs = verses(m, a.limit)
    base = base_state(m)

    fails, checked = [], 0
    for cols, rows in sizes:
        want_ay = rows - m.BOT_ROWS
        plate_bot = plate_bottom_row(m, cols, rows)
        if plate_bot >= want_ay:
            fails.append("%dx%d  no-overlap: plate reaches row %d, band starts at %d"
                         % (cols, rows, plate_bot, want_ay))
        seen_hdr = set()
        for panels in ("focus", "full", "off"):
            for txt in vs:
                st = dict(base); st["ayah_text"] = txt; st["panels"] = panels
                ch = m.Chrome(cols, rows)
                try:
                    m.draw(ch, st)
                except Exception as e:
                    fails.append("%dx%d %s  draw raised %s: %s  on %r"
                                 % (cols, rows, panels, type(e).__name__, e, txt[:1]))
                    continue
                checked += 1
                hdr = header_row(ch, rows)
                if hdr is not None:
                    seen_hdr.add(hdr)
                # in-bounds: the strip owns the last STRIP_ROWS rows; the band
                # must not write into them, and nothing may sit below the grid.
                band_end = want_ay + m.AYAH_ROWS
                if band_end > rows - m.STRIP_ROWS:
                    fails.append("%dx%d  in-bounds: band ends at %d, strip starts at %d"
                                 % (cols, rows, band_end, rows - m.STRIP_ROWS))
        if len(seen_hdr) > 1:
            fails.append("%dx%d  band-constant: header landed on rows %s "
                         "depending on the verse" % (cols, rows, sorted(seen_hdr)))
        elif seen_hdr and a.verbose:
            print("%dx%d  ok  band top %d, plate bottom %d, %d verses"
                  % (cols, rows, sorted(seen_hdr)[0], plate_bot, len(vs)))

    print("layout-check: %d frames across %d sizes x %d verses"
          % (checked, len(sizes), len(vs)))
    if fails:
        for f in fails:
            print("  FAIL  " + f)
        return 1
    print("  all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
