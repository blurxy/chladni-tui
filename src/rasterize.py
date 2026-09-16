#!/usr/bin/env python3
"""Render the screensaver's own terminal output to a PNG.

Iterating on the look by launching a real terminal means putting a window on the
user's live desktop, and a screensaver window exits the moment he touches the
keyboard. Since we emit the escape sequences ourselves we can just draw them:
same font, same cell metrics, same palette. Offline, repeatable, invisible.
"""
import sys, re, argparse
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/TTF/IosevkaNerdFontMono-Regular.ttf"
GROUND = (0x15, 0x1a, 0x1f)
SGR = re.compile(r"\x1b\[((?:\d+;?)*)m")
CUP = re.compile(r"\x1b\[(\d+);(\d+)H")
OSC = re.compile(r"\x1b\][^\x07]*\x07")


def parse(text, cols, rows):
    """Replay the escape stream into a (char, colour) grid."""
    grid = [[(" ", GROUND, GROUND) for _ in range(cols)] for _ in range(rows)]
    text = OSC.sub("", text)
    text = re.sub(r"\x1b\[\?25[lh]|\x1b\[2J|\x1b\[H", "\x00", text)
    r = c = 0
    col = (0xed, 0xf1, 0xf5)
    bgc = GROUND
    i = 0
    while i < len(text):
        m = CUP.match(text, i)
        if m:
            r = int(m.group(1)) - 1; c = int(m.group(2)) - 1; i = m.end(); continue
        m = SGR.match(text, i)
        if m:
            parts = [p for p in m.group(1).split(";") if p != ""]
            k = 0
            while k < len(parts):
                v = parts[k]
                if v == "38" and k + 4 < len(parts) and parts[k + 1] == "2":
                    col = tuple(int(x) for x in parts[k + 2:k + 5]); k += 5
                elif v == "48" and k + 4 < len(parts) and parts[k + 1] == "2":
                    bgc = tuple(int(x) for x in parts[k + 2:k + 5]); k += 5
                elif v == "49":
                    bgc = GROUND; k += 1
                elif v == "0":
                    col = (0xed, 0xf1, 0xf5); bgc = GROUND; k += 1
                else:
                    k += 1
            i = m.end(); continue
        ch = text[i]; i += 1
        if ch == "\x00":
            continue
        if ch == "\n":
            r += 1; c = 0; continue
        if ch == "\r":
            c = 0; continue
        if ch == "\x1b":                      # unknown escape: skip to a letter
            j = i
            while j < len(text) and not text[j].isalpha():
                j += 1
            i = j + 1; continue
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = (ch, col, bgc)
        c += 1
        # Do NOT auto-wrap. A dumped line is exactly `cols` wide and is followed
        # by a newline; advancing the row on both the wrap and the newline
        # leaves every second row blank and shreds the figure into stripes.
        if c > cols:
            c = cols
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("out")
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cell-w", type=float, default=12.03)
    ap.add_argument("--cell-h", type=float, default=30.19)
    args = ap.parse_args()

    with open(args.infile, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    grid = parse(text, args.cols, args.rows)
    cw, chh = args.cell_w, args.cell_h
    cols, rows = args.cols, args.rows
    W, H = int(cw * cols), int(chh * rows)
    img = Image.new("RGB", (W, H), GROUND)
    d = ImageDraw.Draw(img)
    # Size the face so its advance matches the cell width, the way a terminal does.
    px = 8
    while px < 200:
        f = ImageFont.truetype(FONT, px + 1)
        if f.getlength("M") > cw:
            break
        px += 1
    font = ImageFont.truetype(FONT, px)
    asc, _desc = font.getmetrics()
    ybase = (chh - asc) / 2.0
    # background first, as filled rectangles, then the glyphs on top
    for r, row in enumerate(grid):
        startc, runbg = 0, row[0][2]
        for c in range(cols + 1):
            b = row[c][2] if c < cols else None
            if b != runbg:
                if runbg != GROUND:
                    d.rectangle([startc * cw, r * chh, c * cw, (r + 1) * chh], fill=runbg)
                startc, runbg = c, b
    for r, row in enumerate(grid):
        run, runcol, startc = [], None, 0
        for c, cell in enumerate(row + [(None, None, None)]):
            ch, col = cell[0], cell[1]
            if col != runcol or ch is None:
                if run and runcol is not None and any(x != " " for x in run):
                    d.text((startc * cw, r * chh + ybase), "".join(run), font=font, fill=runcol)
                run, runcol, startc = [], col, c
            if ch is not None:
                run.append(ch)
    img.save(args.out)
    print("%s  %dx%d px  (cell %.2fx%.2f, font %dpx)" % (args.out, W, H, cw, chh, px))


if __name__ == "__main__":
    main()
