#!/usr/bin/env python3
"""Render one --dump frame to a PNG, with no terminal, window or compositor.

WHY THIS EXISTS. Judging the picture meant launching a terminal, waiting, and
screenshotting it -- and a tiling window manager decides that window's size, so
two runs meant to be compared came back at 2508x1550 and 384x227. A comparison
between frames of different sizes measures the layout engine, not the change.
This runs the renderer inside a pseudo-terminal of a size chosen here, parses
the truecolour ANSI it prints, and draws Braille sub-dots as pixels directly, so
the same arguments always produce the same image.

It also fixes a portfolio problem: README screenshots taken of a real desktop
carry that desktop -- hostname, other windows, a cursor. These carry nothing.

    tools/render-dump.py out.png --cols 320 --rows 96 -- --start 3000 --grains 90000
"""
import argparse, fcntl, os, pty, re, select, struct, subprocess, sys, termios

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SGR = re.compile(r"\x1b\[([0-9;]*)m")
# glyph -> (how to draw, stroke thickness as a fraction of the cell)
BOXES = {"─": ("h", 0.07), "━": ("h", 0.17), "│": ("v", 0.09),
         "┃": ("v", 0.22), "█": ("f", 1.0), "▮": ("t", 0),
         "▯": ("o", 0)}
OTHER = re.compile(r"\x1b(\[[0-9;?]*[A-Za-z]|\][^\x07]*\x07)")


def capture(cols, rows, extra, python, script):
    """Run --dump in a pty of exactly cols x rows and return its output."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(python, [python, script, "--dump", "--aspect", "1.0", *extra])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    out = bytearray()
    while True:
        r, _, _ = select.select([fd], [], [], 120)
        if not r:
            break
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    os.waitpid(pid, 0)
    return out.decode("utf-8", "replace")


def paint(text, cols, rows, cw, chh, font):
    img = Image.new("RGB", (cols * cw, rows * chh), (0x15, 0x1a, 0x1f))
    d = ImageDraw.Draw(img)
    fg, bg = (200, 200, 200), None
    r = c = 0
    i = 0
    dw, dh = cw / 2.0, chh / 4.0
    # braille dot bit -> (column, row) within the 2x4 cell
    BITS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 3)]
    for line in text.split("\n"):
        line = line.rstrip("\r")
        c = 0
        pos = 0
        while pos < len(line):
            m = SGR.match(line, pos)
            if m:
                p = [int(x) if x else 0 for x in m.group(1).split(";")]
                k = 0
                while k < len(p):
                    if p[k] == 38 and k + 4 < len(p) and p[k + 1] == 2:
                        fg = tuple(p[k + 2:k + 5]); k += 5
                    elif p[k] == 48 and k + 4 < len(p) and p[k + 1] == 2:
                        bg = tuple(p[k + 2:k + 5]); k += 5
                    elif p[k] in (0, 49):
                        bg = None; k += 1
                    else:
                        k += 1
                pos = m.end()
                continue
            m = OTHER.match(line, pos)
            if m:
                pos = m.end()
                continue
            ch = line[pos]
            pos += 1
            if c >= cols or r >= rows:
                c += 1
                continue
            x0, y0 = c * cw, r * chh
            if bg is not None:
                d.rectangle([x0, y0, x0 + cw - 1, y0 + chh - 1], fill=bg)
            o = ord(ch)
            if ch in BOXES:
                # DRAW BOX AND BLOCK GLYPHS GEOMETRICALLY, NOT FROM THE FONT.
                # PIL places each glyph at its own advance width, which for
                # these is narrower than the cell -- so a run of U+2501 came out
                # as a dashed line and I nearly "fixed" a bar that is solid in
                # every terminal. A terminal tiles these edge to edge; so does
                # this now.
                kind, frac = BOXES[ch]
                if kind == "h":
                    t = max(1.0, chh * frac)
                    d.rectangle([x0, y0 + (chh - t) / 2, x0 + cw, y0 + (chh + t) / 2], fill=fg)
                elif kind == "v":
                    t = max(1.0, cw * frac)
                    d.rectangle([x0 + (cw - t) / 2, y0, x0 + (cw + t) / 2, y0 + chh], fill=fg)
                elif kind == "f":
                    d.rectangle([x0, y0, x0 + cw, y0 + chh], fill=fg)
                elif kind == "t":      # tally: filled tall block, inset
                    d.rectangle([x0 + cw * 0.18, y0 + chh * 0.22,
                                 x0 + cw * 0.82, y0 + chh * 0.78], fill=fg)
                else:                  # "o": hollow tally
                    d.rectangle([x0 + cw * 0.18, y0 + chh * 0.22,
                                 x0 + cw * 0.82, y0 + chh * 0.78], outline=fg)
                c += 1
                continue
            if 0x2800 < o <= 0x28FF:
                bits = o - 0x2800
                for b, (bx, by) in enumerate(BITS):
                    if bits >> b & 1:
                        cx, cy = x0 + (bx + 0.5) * dw, y0 + (by + 0.5) * dh
                        rad = max(1.0, min(dw, dh) * 0.42)
                        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=fg)
            elif ch not in (" ", "⠀"):
                d.text((x0, y0), ch, fill=fg, font=font)
            c += 1
        r += 1
        if r >= rows:
            break
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cols", type=int, default=300)
    ap.add_argument("--rows", type=int, default=92)
    ap.add_argument("--cell", type=int, default=8, help="cell width in px; height is 2x")
    ap.add_argument("--python", default=sys.executable)
    # THE RENDERER READS timeline.npz FROM THE DIRECTORY IT LIVES IN. Pointed at
    # this repo's src/, where there is no analysed library, it silently falls back
    # to a synthetic timeline -- and every frame rendered would be of a recitation
    # that does not exist. Default to the installed copy, which has the data.
    inst = os.path.expanduser("~/.local/share/omarchy-cymatics-screensaver/cyscreen.py")
    ap.add_argument("--script", default=inst if os.path.exists(inst)
                    else os.path.join(HERE, "src", "screensaver.py"))
    ap.add_argument("--font", default="/usr/share/fonts/TTF/IosevkaNerdFontMono-Regular.ttf")
    # SPLIT ON "--" BY HAND. argparse.REMAINDER as a positional after `out`
    # swallowed every option that followed it, including this tool's own
    # --python, and handed them to the screensaver -- which printed a usage
    # error that was then rendered into a PNG and reported as "wrote". A
    # REMAINDER positional does not stop at "--"; it starts at the first thing
    # it can claim.
    argv = sys.argv[1:]
    if "--" in argv:
        k = argv.index("--")
        mine, extra = argv[:k], argv[k + 1:]
    else:
        mine, extra = argv, []
    a = ap.parse_args(mine)
    tl = os.path.join(os.path.dirname(a.script), "timeline.npz")
    if not os.path.exists(tl):
        print("warning: no %s -- this frame is from a SYNTHETIC timeline" % tl, file=sys.stderr)
    text = capture(a.cols, a.rows, extra, a.python, a.script)
    # REFUSE TO SAVE A FRAME THAT HAS NO PLATE IN IT. The first run of this tool
    # wrote a clean-looking PNG of an argparse error and reported success, which
    # is a renderer certifying its own failure. A real frame is mostly Braille.
    nbraille = sum(1 for ch in text if 0x2800 < ord(ch) <= 0x28FF)
    if nbraille < 200:
        sys.stderr.write(text[-1500:] + "\n")
        sys.exit("render-dump: output held %d Braille cells, which is not a frame "
                 "-- refusing to write %s" % (nbraille, a.out))
    try:
        font = ImageFont.truetype(a.font, int(a.cell * 1.6))
    except OSError:
        font = ImageFont.load_default()
    paint(text, a.cols, a.rows, a.cell, a.cell * 2, font).save(a.out)
    print("wrote %s  (%dx%d cells)" % (a.out, a.cols, a.rows))


if __name__ == "__main__":
    main()
