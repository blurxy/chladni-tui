#!/usr/bin/env python3
"""chladni-tui: a Chladni sand plate for the terminal.

A circular membrane driven by a real recitation, drawn in braille sub-dots.

The animation is not a slideshow of figures. build_timeline.py analysed each
recitation once and stored, per frame, how hard the voice was driving each
membrane mode. This plays those amplitudes back through the same sand physics
the live tool uses, so the figure moves the way the voice moved -- it holds
while a note is held, and reorganises when the reciter changes pitch.

Energy is the INCOHERENT sum  E = sum_i a_i^2 * U_i^2, not (sum a_i U_i)^2.
The modes sit at incommensurate frequencies, so squaring a coherent sum would
invent interference nodes no real plate has.
"""
import os
import random, sys, time, math, signal, argparse, json, glob, struct, fcntl, termios
import atexit, select, tty
import shutil, subprocess

# BLAS THREAD COUNT TRADES HEAT FOR SMOOTHNESS. Measured on a 240Hz panel at
# 1920x1080: 1 thread 43fps at 48C with no throttling; 2 threads 58fps at 78C;
# 4 threads 115fps and ~700 throttle events in 40s. A headless benchmark of the
# physics alone said 4 threads were no faster than 1 -- true of what it timed
# and wrong about the live loop, which also builds and writes the frame.
# So the default follows the power source: 2 plugged in, 1 on battery, where a
# screensaver's heat is also charge. Override with OPENBLAS_NUM_THREADS.
def _on_external_power(root="/sys/class/power_supply"):
    supplies = glob.glob(os.path.join(root, "*"))
    has_battery = False
    for d in supplies:
        try:
            with open(os.path.join(d, "type")) as fh:
                typ = fh.read().strip()
            if typ == "Battery":
                has_battery = True
            elif typ in ("Mains", "USB"):          # USB-C power delivery reports as USB
                with open(os.path.join(d, "online")) as fh:
                    if fh.read().strip() == "1":
                        return True
        except OSError:
            continue
    return not has_battery                          # no battery at all: a desktop


# ---------------------------------------------------------------- settings ---
# Edited from inside the running program (press `s`); the file only persists
# the choices between runs. Loaded before numpy, because the BLAS thread count
# must be in the environment before the library reads it -- which is why
# "performance" is the one setting that takes effect at the next launch.
SETTINGS_PATH = os.path.join(
    os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
    "chladni-tui", "settings.json")

DEFAULTS = {
    "fps": "auto",          # auto = the focused monitor's refresh rate
    "threads": "auto",      # auto = 2 plugged in, 1 on battery
    "detail": 8,            # terminal font size: smaller cell, finer plate
    "grains": "auto",
    "volume": 0.55,
    "subtitles": True,
    "panels": "focus",   # focus | full | off
    "tint": 0.22,
    "gap": 1.5,
    "membrane": "dots",  # off | soft | full | dots -- the plate under the sand
    "framing": "offset",  # offset | centred
    "hold": 6.0,          # seconds a figure must hold before the next one
}


def load_settings(path=SETTINGS_PATH):
    """Saved choices, falling back to defaults for anything missing or broken.

    Never raises: a corrupt settings file must not stop a screensaver from
    starting. Unknown keys are dropped rather than carried, so an older file
    cannot smuggle a setting this version no longer understands.
    """
    out = dict(DEFAULTS)
    try:
        with open(path) as fh:
            saved = json.load(fh)
        if isinstance(saved, dict):
            for k, v in saved.items():
                if k in DEFAULTS and type(v) is type(DEFAULTS[k]) or k in ("fps", "threads", "grains"):
                    out[k] = v
    except (OSError, ValueError):
        pass
    return out


def save_settings(cfg, path=SETTINGS_PATH):
    """Persist atomically; a half-written settings file is worse than none."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = "%s.%d" % (path, os.getpid())
        with open(tmp, "w") as fh:
            json.dump({k: cfg[k] for k in DEFAULTS if k in cfg}, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


SETTINGS = load_settings()

_want = SETTINGS.get("threads", "auto")
if _want in (1, 2, 4, "1", "2", "4"):
    _THREADS = str(_want)
else:
    _THREADS = "2" if _on_external_power() else "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)
import numpy as np
from scipy.special import jv

HERE = os.path.dirname(os.path.abspath(__file__))
TIMELINE = os.path.join(HERE, "timeline.npz")
AUDIO_DIR = os.path.join(HERE, "audio")

# ---------------------------------------------------------------- palette ---
# The steel ramp the cymatics wallpapers were rendered in, so the screensaver
# and the desktop behind it are the same picture.
GROUND = (0x15, 0x1a, 0x1f)
# Sand density runs deep blue -> teal -> cyan -> gold -> warm white, so the
# halo around a nodal line resolves as colour rather than five shades of
# background. The warm stops sit LOW on purpose: a settled line's density
# lands mid-range, so stops placed in the top third left the warm end unused
# and every figure read cold.
SAND_STOPS = [(0.00, (0x15, 0x1a, 0x1f)), (0.06, (0x1b, 0x33, 0x4e)),
              (0.14, (0x22, 0x5c, 0x85)), (0.24, (0x2b, 0x96, 0xae)),
              (0.36, (0x58, 0xc9, 0xc2)), (0.50, (0x9a, 0xdc, 0x9b)),
              (0.64, (0xe8, 0xb5, 0x55)), (0.80, (0xf7, 0xd8, 0x8e)),
              (1.00, (0xff, 0xf8, 0xec))]
NLEV = 14                      # more steps now that the ramp has somewhere to go
CHROME = {
    # raised: at these sizes the old greys (0x2b/0x5d) were unreadable over the wash
    "rule":   (0x3d, 0x4b, 0x57),
    "label":  (0x82, 0x99, 0xab),
    "value":  (0xc4, 0xd4, 0xe2),
    "accent": (0xe0, 0xb2, 0x62),
    "arch":   (0x59, 0x9c, 0xd6),
    "bright": (0xed, 0xf1, 0xf5),
    "dim":    (0x66, 0x78, 0x86),
}
CK = list(CHROME)
C = {k: NLEV + i for i, k in enumerate(CK)}
# The membrane vibrates everywhere the sand is not; drawn as bare background
# it read as a void with a figure floating on it and the disc edge invisible.
# This wash fills the disc, warmed toward the sand ramp's blue so it reads as
# a lit surface -- but always BELOW the lowest sand level, so a quiet cell can
# never compete with a settled one. Index 0 is the terminal's own background,
# so a quiet cell costs 5 bytes, not 19.
BG_STOPS = [(0.00, GROUND), (0.30, (0x1c, 0x26, 0x33)),
            (0.60, (0x24, 0x35, 0x49)), (0.85, (0x2c, 0x44, 0x5f)),
            (1.00, (0x35, 0x53, 0x72))]
NBG = 10                       # see set_field_bg: 5 levels banded into visible blocks
BG0 = NLEV + len(CK)
# How hard the membrane is drawn, and how. A cell BACKGROUND is the coarsest
# thing on screen -- one colour for 2x4 sub-dots -- so at a real window size
# the wash quantises into visible rectangles however finely it is computed.
# "dots" draws the same field in braille instead, at the sand's own
# resolution, which cannot band; it costs a glyph in cells that were empty.
MEMBRANE = {"off": 0.0, "soft": 0.35, "full": 1.0, "dots": 1.0}
# Below this width "dots" falls back to the wash. The braille threshold is an
# 8x8 Bayer tile over SUB-DOTS, which is 4 cells wide -- at 165 columns that is
# 2.4% of the screen and the antinodes render as square patches of the DITHER
# rather than of the field. Measured: invisible at 280 columns, obvious at 165.
DOTS_MIN_COLS = 200


def build_lut(hue_shift=0.0):
    """Sand ramp + chrome in one table. hue_shift rotates only the sand half."""
    import colorsys
    xs = np.linspace(0.0, 1.0, NLEV)
    ps = np.array([st[0] for st in SAND_STOPS])
    cs = np.array([st[1] for st in SAND_STOPS], dtype=np.float64)
    out = np.empty((NLEV + len(CK) + NBG, 3), dtype=np.uint8)
    for c in range(3):
        out[:NLEV, c] = np.clip(np.interp(xs, ps, cs[:, c]), 0, 255).astype(np.uint8)
    if hue_shift:
        for i in range(NLEV):
            r, g, b = (v / 255.0 for v in out[i])
            h, l, sat = colorsys.rgb_to_hls(r, g, b)
            # Leave the near-black and near-white ends alone: rotating them just
            # tints the background and the highlight, which reads as a bug.
            w = min(1.0, max(0.0, 1.0 - abs(l - 0.5) * 2.2))
            r, g, b = colorsys.hls_to_rgb((h + hue_shift * w) % 1.0, l, sat)
            out[i] = [int(round(v * 255)) for v in (r, g, b)]
    for i, k in enumerate(CK):
        out[NLEV + i] = CHROME[k]
    bx = np.linspace(0.0, 1.0, NBG)
    bp = np.array([st[0] for st in BG_STOPS])
    bc = np.array([st[1] for st in BG_STOPS], dtype=np.float64)
    for c in range(3):
        out[BG0:BG0 + NBG, c] = np.clip(np.interp(bx, bp, bc[:, c]), 0, 255).astype(np.uint8)
    return out


LUT = build_lut()
# Each mode gets its own place on the wheel, so a change of figure is also a
# change of colour. Spaced by an irrational step rather than evenly, so adjacent
# modes in the ladder are never adjacent in hue.
MODE_HUE = ((np.arange(64, dtype=np.float64) * 0.381966) % 1.0) - 0.5

BRAILLE = np.array([chr(0x2800 + i) for i in range(256)], dtype="<U1")
DOTW = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], dtype=np.uint16)

# A braille sub-dot is 2 wide x 4 tall inside one cell, so it is square only if
# the cell is exactly 1:2. Iosevka is condensed -- measured here the cell runs
# about 1:2.7, which makes a sub-dot ~1.3x taller than wide and renders every
# circle as a vertical ellipse. Ask the terminal for its real pixel geometry.
SUB_ASPECT = 1.30


def measure_sub_aspect(default=1.30):
    for fd in (1, 0, 2):
        try:
            r, c, xp, yp = struct.unpack('HHHH', fcntl.ioctl(fd, termios.TIOCGWINSZ, b'\0' * 8))
        except (OSError, ValueError):
            continue
        if xp and yp and c and r:
            return (yp / r) / (2.0 * (xp / c))
    return default


# ------------------------------------------------------------ mode fields ---
class ModeBank:
    """Per-mode U^2 and its gradient, summed each frame against live amplitudes.

    Held at a coarser resolution than the display. The picture's sharpness comes
    from where grains land (sub-dot resolution), not from how finely the driving
    field is sampled, so paying for a full-resolution 20-mode basis buys nothing
    and costs both memory and a per-frame sgemv.
    """

    def __init__(self, mn, subw, subh, maxw=448, rim_floor=0.032, rim_strength=0.17,
                 select=6.0, rad_scale=0.48, cy_shift=0.0, cx_shift=0.0):
        self.pw = int(min(subw, maxw))
        self.ph = max(8, int(round(self.pw * (subh * SUB_ASPECT) / float(subw))))
        pw, ph = self.pw, self.ph
        self.sx = pw / float(subw); self.sy = ph / float(subh)
        # The plate has to fit BETWEEN the chrome, not inside the raw viewport:
        # sized to the full height its spokes run off the top and bottom of the
        # screen and the rim -- the thing that tells you it is a plate at all --
        # is never on screen.
        rad = rad_scale * min(float(pw), float(ph))
        cy = (ph - 1) / 2.0 + cy_shift
        yy = (np.arange(ph, dtype=np.float32)[:, None] - cy) / rad
        cx = (pw - 1) / 2.0 + cx_shift
        xx = (np.arange(pw, dtype=np.float32)[None, :] - cx) / rad
        r = np.hypot(xx, yy); th = np.arctan2(yy, xx)
        self.inside = r <= 1.0
        self.r = r
        alphas = []
        E2 = np.zeros((len(mn), ph, pw), dtype=np.float32)
        from scipy.special import jn_zeros
        for i, (m, n) in enumerate(mn):
            a = float(jn_zeros(int(m), int(n))[int(n) - 1])
            alphas.append(a)
            u = (jv(int(m), a * np.clip(r, 0, 1)) * np.cos(int(m) * th)).astype(np.float32)
            u[~self.inside] = 0.0
            e = u * u
            mx = e.max()
            if mx > 0:
                e /= mx
            E2[i] = e
        self.alpha = np.array(alphas, dtype=np.float32)
        # How wide is each mode's dead zone at the middle? A mode with m nodal
        # diameters goes as r^m, so its energy goes as r^(2m): for m=5 the field
        # is below 2% of peak out to ~0.3 of the radius. Grains random-walk into
        # that flat disc and cannot get back out, so it fills with mush. Measure
        # the radius per mode instead of guessing one number for all of them.
        self.r_core = np.zeros(len(mn), dtype=np.float32)
        rad_bins = np.linspace(0.0, 1.0, 120, dtype=np.float32)
        rflat = r.reshape(-1)
        for i in range(len(mn)):
            ei = E2[i].reshape(-1)
            rc = 0.0
            for b in range(1, len(rad_bins)):
                sel_ = (rflat >= rad_bins[b - 1]) & (rflat < rad_bins[b])
                if sel_.any() and ei[sel_].max() > 0.02:
                    rc = float(rad_bins[b - 1]); break
            self.r_core[i] = rc
        # EACH MODE'S ANGULAR FACTOR, cos^2(m*theta), for shaping the centre.
        # J_m fades as r^m toward the middle but cos(m*theta) does not: it is exactly
        # zero on the m nodal diameters at every radius, so the diameters survive
        # inside the dead zone where the full field has flattened to nothing.
        # Its gradient is purely tangential. COMPUTED ANALYTICALLY -- np.gradient on a
        # polar pattern sampled on a Cartesian grid leaks a small RADIAL component,
        # and a small outward drift repeated over hundreds of steps empties the core
        # (measured: no sand inside r=0.10). Here it is true by construction:
        #     grad f = (1/r)(df/dtheta) theta_hat,  df/dtheta = -m sin(2 m theta)
        # in plate units, divided by rad. Faded to zero near the origin, where theta
        # is not resolvable per pixel.
        rpx = r * np.float32(rad)
        fade = np.clip((rpx - 2.0) / 3.0, 0.0, 1.0).astype(np.float32)
        safe_r = np.maximum(r, np.float32(1e-6))
        ang = np.zeros((len(mn), ph, pw), dtype=np.float32)
        agx = np.zeros_like(ang); agy = np.zeros_like(ang)
        for i, (m, _n) in enumerate(mn):
            m = int(m)
            if m > 0:
                ang[i] = np.cos(m * th) ** 2 * fade
                dfdt = -m * np.sin(2 * m * th) / (safe_r * np.float32(rad)) * fade
                agx[i] = dfdt * -np.sin(th)
                agy[i] = dfdt * np.cos(th)
        self.ANG = ang.reshape(len(mn), -1)
        self.AGX = agx.reshape(len(mn), -1).astype(np.float32)
        self.AGY = agy.reshape(len(mn), -1).astype(np.float32)
        gy, gx = np.gradient(E2, axis=(1, 2))
        self.E2 = E2.reshape(len(mn), -1)
        self.GX = gx.reshape(len(mn), -1).astype(np.float32)
        self.GY = gy.reshape(len(mn), -1).astype(np.float32)
        self.select = float(select)
        # Static exterior bowl: linear so its gradient does not vanish at the rim.
        out = ~self.inside
        bowl = np.zeros((ph, pw), dtype=np.float32)
        bowl[out] = (r[out] - 1.0) * np.float32(2.6)
        # The clamped rim is a node of EVERY mode, so without a swept edge band
        # every grain eventually migrates there and the plate empties out into a
        # bright circle with nothing inside. A thin agitation floor near r=1
        # keeps the edge clear so the interior figure is what you see.
        band = np.clip((r - (1.0 - rim_floor)) / rim_floor, 0.0, 1.0).astype(np.float32)
        band[out] = 0.0
        bowl += (band * band) * np.float32(rim_strength)
        self.rnorm = r.reshape(-1).astype(np.float32)
        bgy, bgx = np.gradient(bowl)
        self.bowl = bowl.reshape(-1)
        self.bgx = bgx.reshape(-1).astype(np.float32)
        self.bgy = bgy.reshape(-1).astype(np.float32)
        self.out_flat = out.reshape(-1)
        # cxp/cyp are the plate's centre for EVERYTHING downstream -- the rim
        # ring, escaped-grain reinjection, and the resize remap all read them.
        # Hardcoding cxp to the middle while the field was drawn around cx left
        # the ring orbiting the wrong point and threw recycled grains outside
        # the disc.
        self.cxp = float(cx); self.cyp = float(cy); self.rad = float(rad)
        self._scale = None

    def field_w(self, w):
        """Build the field from explicit per-mode weights (already energies)."""
        a2 = np.asarray(w, dtype=np.float32)
        t = float(a2.sum())
        if t > 1e-9:
            a2 = a2 / t
        return self._assemble(a2)

    def field(self, amps):
        # Sharpen mode selection before squaring. Real audio always bleeds energy
        # into neighbouring modes, and because E is a sum of SQUARES the common
        # zeros of several modes are isolated POINTS, not curves -- leave the
        # bleed in and the sand beads into dots instead of tracing a figure.
        a = np.power(np.clip(amps, 0.0, None), self.select, dtype=np.float32)
        a2 = (a * a).astype(np.float32)
        s = float(a2.sum())
        if s > 1e-9:
            a2 = a2 / s
        return self._assemble(a2)

    def _assemble(self, a2):
        # Agitate exactly as much of the middle as THIS combination of modes
        # leaves dead, so the crossing point stays a tight bright star instead
        # of spreading into a blob (too little) or a donut hole (too much).
        rc = float(a2 @ self.r_core) * 0.80
        E = a2 @ self.E2
        mx = float(E.max())
        if mx > 1e-9:
            E = E / mx
        GX = a2 @ self.GX; GY = a2 @ self.GY
        if mx > 1e-9:
            GX = GX / mx; GY = GY / mx
        E = E + self.bowl
        GX = GX + self.bgx; GY = GY + self.bgy
        if rc > 1e-3:
            # SHAPED BY cos^2(m*theta), AND ONLY ITS TANGENTIAL HALF. A uniform push into
            # the dead zone cannot tell a nodal diameter from the gap between two, so it
            # swept the lines out along with the blob it was meant to clear. Weighting by
            # the angular factor agitates only BETWEEN the diameters and leaves them at
            # zero, driving sand onto them so they cross the centre.
            # The product rule puts a RADIAL term in grad(core*ang) pointing outward
            # wherever ang > 0, which expelled sand from the whole dead zone, diameters
            # included -- so making the bump angular did nothing until this term was
            # dropped. Measured with it: 0 grains inside r=0.05 and a 6.8x density ring
            # exactly where the core ends.
            core = np.clip(1.0 - self.rnorm / np.float32(rc), 0.0, 1.0)
            core *= core
            k = core * np.float32(0.10)
            E = E + k * (a2 @ self.ANG)
            GX = GX + k * (a2 @ self.AGX)
            GY = GY + k * (a2 @ self.AGY)
        np.clip(E, 0.0, 1.0, out=E)
        # Normalise the drift by a high percentile, not the max: a few very steep
        # cells on the u^2 ridges would otherwise scale every gentle long-range
        # slope to nothing and grains far from a nodal line would never converge.
        mag = np.hypot(GX, GY)
        sc = float(np.percentile(mag, 92.0))
        if not np.isfinite(sc) or sc <= 1e-9:
            sc = float(mag.max()) or 1.0
        self._scale = sc if self._scale is None else self._scale * 0.85 + sc * 0.15
        GX = GX / self._scale; GY = GY / self._scale
        m = np.hypot(GX, GY)
        np.maximum(m, np.float32(1e-9), out=m)
        lim = np.minimum(np.float32(1.0) / m, np.float32(1.0))
        return E.astype(np.float32), (GX * lim).astype(np.float32), (GY * lim).astype(np.float32)


class FigurePlayer:
    """Play a precomputed figure schedule, cross-fading on each change.

    Exposes the same .cur/.since the dashboard reads, so it drops in wherever
    the live selector was.
    """

    # FADE 0.35s, IN REAL SECONDS. Two blended modes share zeros only at POINTS, so
    # a crossfade beads the sand into dots instead of lines -- fine for a moment,
    # ruinous as a steady state. Figures change every 1.08s (median), so a 1.0s
    # fade counted in frames at the TARGET rate stretched to 2.1s real and the
    # plate showed a blend 75% of the time. At 0.35s real it is 19%.
    def __init__(self, nm, fps, fade=0.35, min_hold=0.0):
        self.nm = nm; self.fps = float(fps); self.fade = float(fade)
        self.min_hold = float(min_hold)
        self.cur = -1; self.prev = -1; self.left = 0.0; self.since = 0.0

    def update_to(self, idx, dt=None):
        # dt from the caller's clock when there is one; a fixed step otherwise,
        # which is what --dump wants when it settles a figure with no wall clock.
        dt = (1.0 / self.fps) if dt is None else float(dt)
        self.since += dt
        idx = int(idx)
        # A FLOOR ON HOW OFTEN THE PICTURE MAY CHANGE. The schedule is built from
        # the recitation and changes every 1.08s at the median, which is faster
        # than sand settles and faster than anyone can look. Holding the current
        # figure past a change does not desynchronise anything -- the next change
        # the schedule asks for is simply the one taken.
        if idx != self.cur and self.since < self.min_hold:
            idx = self.cur
        if idx != self.cur:
            self.prev = self.cur
            self.cur = idx
            self.left = self.fade if self.prev >= 0 else 0.0
            self.since = 0.0
        w = np.zeros(self.nm, dtype=np.float32)
        if self.left > 0.0 and 0 <= self.prev < self.nm:
            f = 1.0 - (self.left / self.fade)
            w[self.prev] = 1.0 - f
            w[self.cur] = f
            self.left = max(0.0, self.left - dt)
        else:
            w[self.cur] = 1.0
        return w


class FigureSelector:
    """Decide which figure the plate is showing, at a pace sand can follow.

    Measured on these recitations: the instantaneously loudest mode changes
    about every 0.24 s, while sand needs seconds to migrate across the plate.
    Driving the field straight from the spectrum therefore never produces a
    figure at all -- the grains spend the whole time in transit and the plate
    reads as noise. So integrate the spectrum over a couple of seconds, commit
    to a mode, hold it long enough to actually settle, and cross-fade when the
    recitation has clearly moved somewhere else.

    Blending the two ENERGY fields during a change is legitimate: it is what
    driving both modes at once does. It is the amplitudes that must never be
    summed coherently.
    """

    def __init__(self, nm, fps, tau=2.0, hold=3.2, fade=1.1, ratio=1.25):
        self.nm = nm; self.fps = float(fps)
        self.k = 1.0 - math.exp(-(1.0 / self.fps) / tau)
        self.hold = hold; self.fade = fade; self.ratio = ratio
        self.avg = np.zeros(nm, dtype=np.float32)
        self.cur = -1; self.prev = -1
        self.since = 1e9; self.fading = 0.0

    def update(self, amps, select):
        a = np.power(np.clip(amps, 0.0, None), select, dtype=np.float32)
        a2 = (a * a).astype(np.float32)
        t = float(a2.sum())
        if t > 1e-9:
            a2 /= t
        self.avg += (a2 - self.avg) * self.k
        dt = 1.0 / self.fps
        self.since += dt
        cand = int(np.argmax(self.avg))
        if self.cur < 0:
            self.cur = cand; self.since = 0.0
        elif (cand != self.cur and self.since >= self.hold
              and self.avg[cand] > self.avg[self.cur] * self.ratio):
            self.prev = self.cur; self.cur = cand
            self.since = 0.0; self.fading = self.fade
        w = np.zeros(self.nm, dtype=np.float32)
        if self.fading > 0.0 and self.prev >= 0:
            f = 1.0 - (self.fading / self.fade)
            w[self.prev] = (1.0 - f); w[self.cur] = f
            self.fading = max(0.0, self.fading - dt)
        else:
            w[self.cur] = 1.0
        return w


# ------------------------------------------------------------- the grains ---
class Sand:
    """Grains live in the PHYSICS grid, not the display grid.

    The physics grid is already aspect-corrected, so it is isotropic: one unit
    right and one unit down are the same physical distance. Stepping in display
    coordinates instead means the y drift and the y noise each pick up a
    different factor of the cell aspect, which converges the sand harder
    vertically than horizontally -- horizontal and vertical nodal lines come out
    while the diagonals never form.
    """

    EDGE = np.float32(0.972)

    # PHYSICS RUNS AT 24 STEPS/SECOND AND IS RESCALED FOR ANY OTHER RATE. Without
    # this, --fps 120 is not a smoother picture but a different one: five times the
    # steps per second of audio, so the sand travels five times as far (measured:
    # "settled" fell 99% -> 69%). Diffusion scales with sqrt(dt), because a random
    # walk spreads as the root of step count; drift and transport scale with dt.
    REF_FPS = 24.0

    def __init__(self, n, subw, subh, bank, seed=11, fps=REF_FPS):
        self.rng = np.random.default_rng(seed)
        self.subw, self.subh, self.n = subw, subh, n
        self.bank = bank
        self.settled = 0.0
        self.px = self.rng.uniform(0, bank.pw - 1, n).astype(np.float32)
        self.py = self.rng.uniform(0, bank.ph - 1, n).astype(np.float32)
        self.sweep_in()

    def step(self, E, GX, GY, agit, drift=3.2, floor=0.044, transport=np.float32(2.0)):
        # drift and transport RAISED with the faster figure schedule. Grains that
        # need ~3s to reach a nodal line cannot express a figure that changes every
        # 0.9s -- the plate just reads as permanent blur, which is a worse desync
        # than the lock it replaced. Floor lowered slightly so settled lines stay
        # settled rather than being re-agitated out of the figure they just made.
        b = self.bank
        xi = self.px.astype(np.int32); yi = self.py.astype(np.int32)
        np.clip(xi, 0, b.pw - 1, out=xi); np.clip(yi, 0, b.ph - 1, out=yi)
        flat = yi * b.pw + xi
        e = E[flat]
        self.settled = float((e < 0.02).mean())
        dts = getattr(self, 'dts', 1.0)
        sq = np.float32(math.sqrt(dts))
        step = ((np.sqrt(np.maximum(e, 0.0)) * agit + floor) * sq).astype(np.float32)
        dx = (-drift * dts * GX[flat]).astype(np.float32)
        dy = (-drift * dts * GY[flat]).astype(np.float32)
        # Clamp drift by MAGNITUDE: clamping each component separately lets a
        # diagonal step run sqrt(2) longer than a cardinal one, which parks
        # grains in four bright blobs at the compass points. The constant
        # transport term is what lets a grain cross the plate at all -- a pure
        # step*k clamp ties travel speed to agitation, so a calm field never
        # finishes converging.
        lim = step * np.float32(2.5) + transport * np.float32(dts)
        mag = np.hypot(dx, dy)
        np.maximum(mag, np.float32(1e-12), out=mag)
        sc = np.minimum(lim / mag, np.float32(1.0))
        dx *= sc; dy *= sc
        self.px += self.rng.standard_normal(self.n).astype(np.float32) * step + dx
        self.py += self.rng.standard_normal(self.n).astype(np.float32) * step + dy
        self.sweep_in()

    def sweep_in(self):
        """Put escaped grains back on the plate.

        Outside the rim the field saturates, so a grain out there takes maximum
        random steps while the inward drift is comparatively gentle -- the two
        balance and a third of the sand parks off the plate forever. Re-seed
        escapees across the disc rather than pinning them at one radius: a hard
        clamp to the edge lands every one on the same circle and draws a bright
        ring that is not a nodal line.
        """
        b = self.bank
        dx = self.px - b.cxp; dy = self.py - b.cyp
        r = np.hypot(dx, dy) / b.rad
        esc = r > self.EDGE
        k = int(esc.sum())
        if k:
            ang = self.rng.uniform(0.0, 2.0 * np.pi, k)
            rr = self.EDGE * np.sqrt(self.rng.uniform(0.0, 1.0, k))   # uniform over area
            self.px[esc] = (b.cxp + np.cos(ang) * rr * b.rad).astype(np.float32)
            self.py[esc] = (b.cyp + np.sin(ang) * rr * b.rad).astype(np.float32)
        np.clip(self.px, 0, b.pw - 1.001, out=self.px)
        np.clip(self.py, 0, b.ph - 1.001, out=self.py)

    def rebind(self, bank, subw, subh):
        """Carry the sand across a resize instead of reseeding it.

        Positions are stored in physics-grid units, so a new grid means new
        units. Rescale by the normalised position on the plate; reseeding
        instead would blank the figure for several seconds every time the
        window changes size.
        """
        old = self.bank
        nx = (self.px - old.cxp) / old.rad
        ny = (self.py - old.cyp) / old.rad
        self.bank = bank; self.subw, self.subh = subw, subh
        self.px = (bank.cxp + nx * bank.rad).astype(np.float32)
        self.py = (bank.cyp + ny * bank.rad).astype(np.float32)
        self.sweep_in()

    def resize(self, n):
        """Grow or shrink the grain population in place."""
        if n == self.n:
            return
        b = self.bank
        if n < self.n:
            self.px = self.px[:n].copy(); self.py = self.py[:n].copy()
        else:
            k = n - self.n
            ang = self.rng.uniform(0.0, 2.0 * np.pi, k)
            rr = self.EDGE * np.sqrt(self.rng.uniform(0.0, 1.0, k))
            self.px = np.concatenate([self.px, (b.cxp + np.cos(ang) * rr * b.rad).astype(np.float32)])
            self.py = np.concatenate([self.py, (b.cyp + np.sin(ang) * rr * b.rad).astype(np.float32)])
        self.n = n

    def display_xy(self):
        b = self.bank
        return (self.px / b.sx).astype(np.float32), (self.py / b.sy).astype(np.float32)


# ------------------------------------------------------------- the screen ---
class Screen:
    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.subw, self.subh = cols * 2, rows * 4
        self.prev_ch = np.full((rows, cols), "\0", dtype="<U1")
        self.prev_lv = np.full((rows, cols), 255, dtype=np.uint8)
        self.out = []
        self.under = None      # static braille underlay (plate rim, ticks)
        self.under_lv = None
        self.dref = None       # running density reference, see compose()
        self.accum = None      # decaying grain-visit buffer, see compose()
        self.bg = np.zeros((rows, cols), dtype=np.uint8)
        self.prev_bg = np.full((rows, cols), 255, dtype=np.uint8)
        self.memb = None
        self.memb_lv = None

    def set_field_bg(self, E, bank, strength=1.0, dots=False):
        """Wash the plate with the energy driving it.

        Recomputed only when the field changes -- during a held figure E is
        bit-identical frame to frame, so this costs nothing in steady state and
        only shows up in the byte budget during a cross-fade.
        """
        rows, cols = self.rows, self.cols
        # SAMPLE THE CELL, DO NOT POKE IT. This read one plate pixel per cell
        # and called it the cell's energy. A cell spans 2x4 sub-dots, so at a
        # real window size that point missed most of what it stood for --
        # neighbouring cells could sample opposite sides of a nodal line and
        # land two levels apart, which is the mottling that reads as blocks.
        # Four samples per cell, averaged, is still cheap and actually
        # describes the area it fills.
        E2 = E.reshape(bank.ph, bank.pw)
        yq = (np.arange(rows) * 4)[:, None] + np.array([1, 3])[None, :]
        xq = (np.arange(cols) * 2)[:, None] + np.array([0, 1])[None, :]
        ys = np.clip(yq.ravel() * (bank.ph / float(self.subh)), 0, bank.ph - 1).astype(np.int32)
        xs = np.clip(xq.ravel() * (bank.pw / float(self.subw)), 0, bank.pw - 1).astype(np.int32)
        blk = E2[ys[:, None], xs[None, :]].reshape(rows, 2, cols, 2)
        cell = blk.mean(axis=(1, 3))
        inside = bank.inside[ys[:, None], xs[None, :]].reshape(rows, 2, cols, 2).any(axis=(1, 3))
        # QUANTISED AND DITHERED. Background colour is a per-CELL attribute while sand
        # is drawn at 2x4 per cell, so the wash has an eighth of the resolution however
        # it is computed. With few levels the steps land on cell boundaries and the
        # plate reads as a grid of blocks -- worst at a real window size, where the
        # disc is only ~38 cells tall. More levels shrink each step and an ordered
        # Bayer threshold scatters the boundary. The pattern depends only on cell
        # position, so it is static and costs nothing in the frame diff.
        v = np.clip(np.sqrt(np.clip(cell, 0.0, 1.0)) * strength, 0.0, 1.0)
        bay = (np.array([[0, 8, 2, 10], [12, 4, 14, 6],
                         [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32) + 0.5) / 16.0
        dith = np.tile(bay, (rows // 4 + 1, cols // 4 + 1))[:rows, :cols]
        lvl = np.clip(np.floor(v * (NBG - 1) + dith), 0, NBG - 1).astype(np.uint8)
        lvl[~inside] = 0
        if not dots:
            self.bg = lvl
            self.memb = None
            return
        # THE SAME FIELD, AT THE SAND'S RESOLUTION. Sampled per sub-dot and
        # thresholded against an 8x8 Bayer matrix tiled over the SUB-DOT grid --
        # deliberately not 2x4, so the threshold pattern does not repeat on cell
        # boundaries and cannot reassemble into the blocks this exists to avoid.
        # Cells keep a colour each (one foreground per cell is a terminal
        # limit); it is the dot DENSITY that carries the shape.
        self.bg = np.zeros_like(lvl)
        ys4 = np.clip(np.arange(rows * 4) * (bank.ph / float(self.subh)),
                      0, bank.ph - 1).astype(np.int32)
        xs2 = np.clip(np.arange(cols * 2) * (bank.pw / float(self.subw)),
                      0, bank.pw - 1).astype(np.int32)
        vs = np.sqrt(np.clip(E2[ys4[:, None], xs2[None, :]], 0.0, 1.0)) * strength
        b8 = (np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
                        [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
                        [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                        [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]],
                       dtype=np.float32) + 0.5) / 64.0
        thr = np.tile(b8, (rows * 4 // 8 + 1, cols * 2 // 8 + 1))[:rows * 4, :cols * 2]
        on = (vs > thr) & bank.inside[ys4[:, None], xs2[None, :]]
        self.memb = (on.reshape(rows, 4, cols, 2) * DOTW[None, :, None, :]
                     ).sum(axis=(1, 3)).astype(np.uint16)
        self.memb_lv = (BG0 + lvl).astype(np.uint8)

    def set_underlay(self, dots, level):
        """A static sub-dot layer drawn beneath the sand.

        The clamped rim is a node of every mode, so sand genuinely wants to cake
        there -- but the escape sweep re-seeds anything past 0.972 of the radius,
        so a rim ring can never form from grains no matter how the physics is
        tuned. Drawing the plate boundary is both cheaper and steadier: it frames
        the figure instead of flickering with the sand.
        """
        g = dots.reshape(self.rows, 4, self.cols, 2)
        self.under = (g * DOTW[None, :, None, :]).sum(axis=(1, 3)).astype(np.uint16)
        self.under_lv = np.uint8(level)

    def compose(self, sand, gain):
        w, h, cols, rows = self.subw, self.subh, self.cols, self.rows
        dx, dy = sand.display_xy()
        xi = np.clip(dx, 0, w - 1).astype(np.int32)
        yi = np.clip(dy, 0, h - 1).astype(np.int32)
        cnt = np.bincount(yi * w + xi, minlength=w * h).astype(np.float32)
        # GIVE THE SAND A MEMORY, MEASURED IN SECONDS. Counting grains fresh each
        # frame makes a settled figure only as bright as the grains standing on it
        # this instant, which is why the plate read as empty. Decaying the accumulated
        # density keeps about a third of a second of history, so a nodal line builds
        # into a solid stroke and a grain in transit leaves a trail. The decay uses
        # the physics' clock-measured timestep, not a frame count: "8 frames" is a
        # third of a second at 24fps and 0.07s at 115.
        # A DOT LIGHTS ABOVE 0.35, NOT ABOVE ZERO. A float times 0.88 does not reach
        # zero for hundreds of frames, so "> 0" lit every sub-dot any grain had
        # crossed in the last ten seconds -- a 19% ghost haze that buried the live
        # figure. --dump never showed it: it composes once from a fresh buffer.
        decay = np.float32(0.88 ** float(getattr(sand, "dts", 1.0)))
        if self.accum is None or self.accum.shape != cnt.shape:
            self.accum = cnt.copy()
        else:
            self.accum *= decay
            self.accum += cnt
        grid = self.accum.reshape(rows, 4, cols, 2)
        code = ((grid >= 0.35) * DOTW[None, :, None, :]).sum(axis=(1, 3)).astype(np.uint16)
        dens = grid.sum(axis=(1, 3))
        # Spread a little of each cell's density into its neighbours. Without
        # this every cell holding sand saturates to the top of the ramp and the
        # figure is a uniform white line on black -- measured, the two commonest
        # colours in a frame were the background and the panel rule. The halo
        # gives every line an edge that falls through the palette, which both
        # exercises the ramp and reads as a thicker, softer object. Four shifted
        # adds over ~11k cells: well under 0.1 ms.
        halo = np.zeros_like(dens)
        halo[1:, :] += dens[:-1, :]; halo[:-1, :] += dens[1:, :]
        halo[:, 1:] += dens[:, :-1]; halo[:, :-1] += dens[:, 1:]
        dens = dens + halo * np.float32(0.30)
        if self.under is not None:
            code = code | self.under
        # Map density to the ramp LOGARITHMICALLY, against a reference measured per
        # frame. Settled sand is bimodal: cells on a nodal line held a median of 58
        # grains and up to 1055, everything else none. A linear gain tuned for either
        # end pins the other -- the old one put 816 cells on pure white with almost
        # nothing between. A log curve against the 99th percentile spreads that
        # 2-decade range across the ramp at any grain count, size or figure.
        lit = dens[dens > 0]
        if lit.size > 32:
            ref = float(np.percentile(lit, 99.0))
            self.dref = ref if self.dref is None else self.dref * 0.88 + ref * 0.12
        ref = max(12.0, self.dref if self.dref else 12.0)
        # THE CURVE, NOT THE REFERENCE. Line cells averaged level 5.6 of 13 -- the
        # teal/gold boundary -- so figures read cold. Three ways of moving the
        # REFERENCE all measured DIMMER, because the faint cells they excluded were
        # holding the percentile down. Widening the log's span raises the middle
        # without spending the top: line cells 5.6 -> 7.2, pure white still 2.6%.
        K = 40.0
        a = K / ref
        norm = (np.log1p(np.maximum(dens, 0.0) * a) / math.log1p(K)) ** 0.8
        np.clip(norm, 0.0, 1.0, out=norm)
        lv = (norm * (NLEV - 1) + 0.5).astype(np.uint8)
        if self.under is not None:
            # where only the underlay is lit, use its own (dim) level
            lv = np.where((dens <= 0) & (self.under != 0), self.under_lv, lv)
        # The membrane only fills cells the sand has left empty: one foreground
        # colour per cell means a cell holding sand cannot also show the plate,
        # and the sand is what the picture is of.
        if self.memb is not None:
            fill = (code == 0) & (self.memb != 0)
            code = np.where(fill, self.memb, code)
            lv = np.where(fill, self.memb_lv, lv)
        ch = BRAILLE[np.asarray(code, dtype=np.uint8)]
        blank = code == 0
        ch[blank] = " "
        # A blank cell paints nothing, so its colour is free: let it inherit the
        # colour already in effect rather than force an SGR change.
        idx = np.where(~blank, np.arange(cols, dtype=np.int32)[None, :], 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        lv = lv[np.arange(rows, dtype=np.int32)[:, None], idx]
        return ch, lv, self.bg

    def flush(self, ch, lv, bg=None, gap=8):
        """Repaint only what moved, as real runs, in both colour channels.

        Emitting one span per row from its first changed cell to its last is
        simple and wrong: drifting sand puts a change near each end of most rows,
        so a 20%-changed frame repaints 46% of the grid. Bridge gaps shorter than
        a cursor move (~8 bytes) because overwriting a few unchanged cells is
        cheaper than repositioning.

        Foreground and background are tracked separately and emitted in ONE
        escape when both change, which is both fewer bytes than two escapes and
        free of the one-cell window where only half the pair has been updated.
        """
        out = self.out; out.clear()
        if bg is None:
            bg = np.zeros_like(lv)
        changed = (ch != self.prev_ch) | (lv != self.prev_lv) | (bg != self.prev_bg)
        last_fg = -1
        last_bg = -1
        for r in np.flatnonzero(changed.any(axis=1)):
            m = changed[r]
            d = np.diff(np.concatenate(([0], m.view(np.int8), [0])).astype(np.int8))
            st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
            if st.size > 1:
                keep = np.concatenate(([True], (st[1:] - en[:-1]) >= gap))
                st = st[keep]; en = en[np.concatenate((keep[1:], [True]))]
            for a_, b_ in zip(st, en):
                out.append("\033[%d;%dH" % (r + 1, a_ + 1))
                sc_ = ch[r, a_:b_]; sl = lv[r, a_:b_]; sb = bg[r, a_:b_]
                pair = sl.astype(np.int32) * 256 + sb.astype(np.int32)
                bnd = np.flatnonzero(np.diff(pair)) + 1
                ss = np.concatenate(([0], bnd)); ee = np.concatenate((bnd, [pair.size]))
                for u, v in zip(ss, ee):
                    f = int(sl[u]); g = int(sb[u])
                    parts = []
                    if f != last_fg:
                        c = LUT[f]
                        parts.append("38;2;%d;%d;%d" % (c[0], c[1], c[2]))
                        last_fg = f
                    if g != last_bg:
                        if g == 0:
                            parts.append("49")          # the terminal's own ground
                        else:
                            c = LUT[BG0 + g]
                            parts.append("48;2;%d;%d;%d" % (c[0], c[1], c[2]))
                        last_bg = g
                    if parts:
                        out.append("\033[" + ";".join(parts) + "m")
                    out.append("".join(sc_[u:v].tolist()))
        if out:
            try:
                sys.stdout.write("".join(out)); sys.stdout.flush()
            except (BrokenPipeError, ValueError, OSError):
                RUN["go"] = False
                return
        self.prev_ch = ch; self.prev_lv = lv; self.prev_bg = bg


# -------------------------------------------------------------- the chrome ---
class Chrome:
    """Text layer composited over the sand. '\\0' means transparent."""

    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.ch = np.full((rows, cols), "\0", dtype="<U1")
        self.lv = np.zeros((rows, cols), dtype=np.uint8)

    def clear(self):
        self.ch[:] = "\0"

    def put(self, r, c, s, lv):
        if not (0 <= r < self.rows) or not s:
            return
        c = max(0, c)
        s = s[:max(0, self.cols - c)]
        if not s:
            return
        n = len(s)
        self.ch[r, c:c + n] = list(s)
        self.lv[r, c:c + n] = lv

    def rule(self, r, c, w, lv):
        self.put(r, c, "─" * max(0, w), lv)

    def panel(self, r, c, w, h, title, lv_frame, lv_title):
        """A rounded box with its interior cleared so sand does not show through."""
        if r < 0 or c < 0 or r + h > self.rows or c + w > self.cols:
            return
        self.ch[r:r + h, c:c + w] = " "
        self.lv[r:r + h, c:c + w] = lv_frame
        self.put(r, c, "╭" + "─" * (w - 2) + "╮", lv_frame)
        self.put(r + h - 1, c, "╰" + "─" * (w - 2) + "╯", lv_frame)
        for i in range(1, h - 1):
            self.put(r + i, c, "│", lv_frame)
            self.put(r + i, c + w - 1, "│", lv_frame)
        if title:
            self.put(r, c + 2, " " + title + " ", lv_title)

    def kv(self, r, c, w, key, val, lv_k, lv_v):
        self.put(r, c, key, lv_k)
        v = str(val)[:max(0, w - len(key) - 1)]
        self.put(r, c + w - len(v), v, lv_v)

    def bar(self, r, c, w, frac, lv_on, lv_off, glyph="█", empty="░"):
        w = max(0, w)
        k = int(round(np.clip(frac, 0.0, 1.0) * w))
        self.put(r, c, glyph * k, lv_on)
        self.put(r, c + k, empty * (w - k), lv_off)

    def composite(self, ch, lv, bg=None):
        m = self.ch != "\0"
        ch = np.where(m, self.ch, ch)
        lv = np.where(m, self.lv, lv)
        if bg is not None:
            bg = np.where(m, np.uint8(0), bg)
        return ch, lv, bg


# ---------------------------------------------------------------- sysinfo ---
def _read(path, default=""):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return default


def _run(cmd, default=""):
    import subprocess
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=4).stdout.strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        return default


def sysinfo():
    """Everything that cannot change while the screensaver runs. Gathered once."""
    info = {"os": "Arch Linux", "kernel": "", "cpu": "", "cores": "", "host": "",
            "theme": "steel", "gpu": "", "shell": "", "pkgs": "", "res": "", "mem_total": 0}
    for line in _read("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            info["os"] = line.split("=", 1)[1].strip().strip('"')
    try:
        u = os.uname(); info["kernel"] = u.release; info["host"] = u.nodename
    except OSError:
        pass
    n = 0
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name") and not info["cpu"]:
            info["cpu"] = line.split(":", 1)[1].strip()
        if line.startswith("processor"):
            n += 1
    info["cores"] = str(n)
    for bad in ("(R)", "(TM)", "CPU ", "Processor"):
        info["cpu"] = info["cpu"].replace(bad, "")
    info["cpu"] = " ".join(info["cpu"].split())
    gpu = _run(["bash", "-lc", "lspci -mm 2>/dev/null | grep -iE 'vga|3d' | head -1"])
    if gpu:
        parts = [x.strip('"') for x in gpu.split('"') if x.strip() and x.strip() != " "]
        info["gpu"] = (parts[2] if len(parts) > 2 else gpu)[:28]
    info["shell"] = os.path.basename(os.environ.get("SHELL", "")) or "bash"
    pk = _run(["bash", "-lc", "pacman -Qq 2>/dev/null | wc -l"])
    info["pkgs"] = pk if pk.isdigit() else ""
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            info["mem_total"] = int(line.split()[1]) // 1024
    info["res"] = term_pixels()
    for pth in (os.path.expanduser("~/.config/omarchy/theme"),
                os.path.expanduser("~/.config/omarchy/current/theme")):
        try:
            if os.path.islink(pth):
                info["theme"] = os.path.basename(os.path.realpath(pth)); break
        except OSError:
            pass
    return info


TOP_ROWS = 2      # title + rule
PLATE_OFFSET = 0.055   # fraction of the width the plate sits right of centre
# A FIXED GRID. Every band below the plate has a constant height, so the
# layout is identical no matter which ayah is playing. It used to derive the
# band's top edge from the wrapped line count -- a long verse grew the block
# upward INTO the plate, a short one did not, so the picture rearranged itself
# every few seconds and the plate overlapped the words on the long ones.
AYAH_ROWS = 1 + 1 + 2 + 2 + 3   # header, gap, translit x2, gap x2, translation x3
STRIP_ROWS = 2                  # bar-rule + readout
BOT_ROWS = AYAH_ROWS + STRIP_ROWS + 1    # + one clear row above the bar


def plate_fit(cols, rows):
    """Radius and vertical offset so the whole plate sits between the chrome."""
    usable = max(6, rows - TOP_ROWS - BOT_ROWS)
    # OFF-CENTRE ON PURPOSE. A Chladni figure is already radially symmetric;
    # centring it in a rectangle doubles the symmetry and the whole screen reads
    # as a specimen in a display case. Pushed right, away from the DRIVEN MODES
    # ladder, so the two stop competing for the left margin and the negative
    # space either side is unequal.
    off = PLATE_OFFSET if SETTINGS.get("framing", "offset") == "offset" else 0.0
    return {"rad_scale": 0.435 * (usable / float(rows)),
            "cy_shift": (TOP_ROWS - BOT_ROWS) / 2.0 * 4.0,
            "cx_shift": off * cols * 2.0}


def plate_ring(bank, subw, subh, ticks=72, tick_len=0.035, ring=True):
    """Sub-dot mask for the plate rim and its dial ticks, in display coordinates."""
    dots = np.zeros((subh, subw), dtype=bool)
    ys = (np.arange(subh, dtype=np.float32) * bank.sy - bank.cyp)[:, None]
    xs = (np.arange(subw, dtype=np.float32) * bank.sx - bank.cxp)[None, :]
    r = np.hypot(xs, ys) / bank.rad
    if ring:
        w = 0.6 * max(bank.sx, bank.sy) / bank.rad     # hairline
        dots |= np.abs(r - 1.0) < w
    if ticks:
        th = np.arctan2(ys, xs)
        k = np.abs(np.cos(th * (ticks / 2.0)))
        band = (r > 1.0 + tick_len * 0.35) & (r < 1.0 + tick_len)
        dots |= band & (k > 0.995)
    return dots


def term_pixels():
    try:
        _r, _c, xp, yp = struct.unpack('HHHH', fcntl.ioctl(1, termios.TIOCGWINSZ, b'\0' * 8))
        if xp and yp:
            return "%dx%d" % (xp, yp)
    except (OSError, ValueError):
        pass
    return ""


def livestats(info):
    """The handful of numbers that do move. Cheap enough to reread each second."""
    out = {"mem_used": 0, "mem_pct": 0.0, "temp": None, "load": 0.0, "uptime": "--"}
    avail = total = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) // 1024
        elif line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) // 1024
    if total:
        out["mem_used"] = total - avail
        out["mem_pct"] = (total - avail) / float(total)
    best = None
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        t = _read(os.path.join(zone, "type"))
        v = _read(os.path.join(zone, "temp"))
        if v.isdigit():
            c = int(v) / 1000.0
            if 10 < c < 130 and (best is None or "x86_pkg" in t or "coretemp" in t):
                best = c
                if "x86_pkg" in t or "coretemp" in t:
                    break
    out["temp"] = best
    try:
        out["load"] = os.getloadavg()[0]
    except OSError:
        pass
    try:
        sec = float(_read("/proc/uptime", "0").split()[0])
        d, sec = divmod(int(sec), 86400); h, sec = divmod(sec, 3600); m = sec // 60
        out["uptime"] = ("%dd %dh %dm" % (d, h, m)) if d else ("%dh %dm" % (h, m))
    except (ValueError, IndexError):
        pass
    return out


def mmss(sec):
    sec = max(0, int(sec))
    return "%d:%02d" % (sec // 60, sec % 60)


_LONG_VOWELS = (("aa", "\u0101"), ("Aa", "\u0100"), ("AA", "\u0100"),
                ("ee", "\u012b"), ("Ee", "\u012a"), ("EE", "\u012a"),
                ("oo", "\u016b"), ("Oo", "\u016a"), ("OO", "\u016a"),
                ("uu", "\u016b"), ("Uu", "\u016a"),
                ("ii", "\u012b"), ("Ii", "\u012a"))


def accent(s):
    """Mark the long vowels in the transliteration, for reading aloud.

    alquran.cloud publishes exactly one English transliteration and it is the
    plain kind -- checked, the only transliteration editions are en, tr and ru,
    none of them diacritic -- so "laahu" rather than "lahu" with a macron. That
    romanisation is systematic about length: a doubled vowel IS a long vowel, so
    the macrons can be put back mechanically. "Qul huwal laahu ahad" becomes
    "Qul huwal lahu ahad" with the a long, which is what tells you where to hold
    the syllable.

    WHAT THIS CANNOT RECOVER, because the source never encoded it: the emphatic
    consonants (s d t z with a dot beneath), and the difference between hamza and
    ayn, which this edition writes with the same apostrophe. So it is a guide to
    vowel length, not a scholarly romanisation, and the rest of the word is left
    exactly as the source wrote it.
    """
    if not s:
        return s
    for a, b in _LONG_VOWELS:
        s = s.replace(a, b)
    return s


def wrap(text, width, lines):
    """Greedy wrap to a fixed number of lines; last line gets an ellipsis."""
    words = text.split(); out = []; cur = ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) <= width:
            cur = (cur + " " + w) if cur else w
        else:
            out.append(cur); cur = w
            if len(out) == lines:
                break
    if cur and len(out) < lines:
        out.append(cur)
    if len(out) == lines and len(" ".join(out)) < len(text):
        out[-1] = out[-1][:max(0, width - 1)] + "\u2026"
    while len(out) < lines:
        out.append("")
    return out[:lines]


def draw(chrome, st):
    ch = chrome
    cols, rows = ch.cols, ch.rows
    ch.clear()
    pw = int(np.clip(cols // 5, 28, 38))
    lx, rx = 2, cols - 2 - pw
    F, L, V, A, B, D = (C["rule"], C["label"], C["value"], C["accent"],
                        C["bright"], C["dim"])

    # --- top bar -------------------------------------------------------------
    ch.rule(1, 2, cols - 4, F)
    ch.put(0, 2, "OMARCHY", L)
    ch.put(0, 14, "\u00b7  CYMATICS", B)
    ch.put(0, 27, "\u00b7  a recitation, resolved into standing waves", L)
    right = "%s  \u00b7  %s  \u00b7  %s" % (st["theme"], st["res"] or "", time.strftime("%H:%M:%S"))
    ch.put(0, cols - 2 - len(right), right, L)

    # --- left column ---------------------------------------------------------
    y = 3
    # FOCUS vs FULL. Every panel here was judged against one question: does it
    # explain the picture, or is it telemetry about the program drawing it?
    # Focus keeps the first kind and moves the rest behind `i`.
    full = st.get("panels", "focus") == "full"
    if full:
        ch.panel(y, lx, pw, 7, "RECITATION", F, A)
        ch.put(y + 1, lx + 2, st["name"][:pw - 4], B)
        ch.put(y + 2, lx + 2, st["sub"][:pw - 4], L)
        ch.kv(y + 4, lx + 2, pw - 4, "surah", "%d \u00b7 %s" % (st["surah_no"], st["surah"]), L, V)
        ch.kv(y + 5, lx + 2, pw - 4, "f\u2080 \u00b7 elapsed",
              "%.0fHz \u00b7 %s" % (st["f0"], mmss(st["t"])), L, V)
        y += 8
    else:
        ch.put(y, lx + 2, st["name"][:pw - 4], B)
        ch.put(y + 1, lx + 2, ("%d \u00b7 %s" % (st["surah_no"], st["surah"]))[:pw - 4], L)
        y += 4

    # --- which modes the voice is driving ------------------------------------
    nm = len(st["amps"])
    room = rows - y - 6
    show = nm if room >= nm + 2 else max(4, room - 2)
    if not full:
        show = min(show, 6)
    order = np.arange(nm) if show >= nm else np.argsort(-st["amps"])[:show]
    if full:
        ch.panel(y, lx, pw, show + 2, "MODE LADDER", F, A)
    else:
        ch.put(y, lx + 2, "DRIVEN MODES", A)
    bw = pw - 12
    for i, mi in enumerate(order):
        m, n = st["mn"][mi]
        av = float(st["amps"][mi])
        lit = B if mi == st["dom"] else (V if av > 0.25 else D)
        ch.put(y + 1 + i, lx + 2, "%-7s" % ("(%d,%d)" % (m, n)), lit)
        k = int(round(np.clip(av, 0, 1) * bw))
        ch.put(y + 1 + i, lx + 9, "\u2501" * k, A if mi == st["dom"] else V)
        ch.put(y + 1 + i, lx + 9 + k, "\u00b7" * (bw - k), D)
    y += show + 3

    # --- right column --------------------------------------------------------
    y = 3
    m, n = st["mn"][st["dom"]]
    lab = "J%s(\u03b1r)" % chr(0x2080 + min(m, 9))
    lab = lab if m == 0 else lab + "\u00b7cos(%d\u03b8)" % m
    if full:
        ch.panel(y, rx, pw, 10, "FIGURE", F, A)
        ch.put(y + 1, rx + 2, "(%d,%d)" % (m, n), B)
        ch.put(y + 1, rx + 9, lab, L)
        ch.kv(y + 3, rx + 2, pw - 4, "\u03b1", "%.3f" % st["alpha"][st["dom"]], L, V)
        ch.kv(y + 4, rx + 2, pw - 4, "freq", "%.0f Hz" % (st["f0"] * st["alpha"][st["dom"]] / st["alpha"][0]), L, V)
        ch.kv(y + 5, rx + 2, pw - 4, "voice pitch", "%.0f Hz" % st["hz"], L, V)
        ch.kv(y + 6, rx + 2, pw - 4, "nodal diameters", str(m), L, V)
        ch.kv(y + 7, rx + 2, pw - 4, "spokes", str(2 * m), L, V)
        ch.kv(y + 8, rx + 2, pw - 4, "interior rings", str(n - 1), L, V)
        y += 11
    else:
        ch.put(y, rx + 2, "(%d,%d)" % (m, n), B)
        ch.put(y, rx + 9, lab, L)
        ch.put(y + 1, rx + 2, ("%d spokes \u00b7 %d ring%s" % (2 * m, n - 1, "" if n == 2 else "s"))
               if m else ("%d ring%s" % (n - 1, "" if n == 2 else "s")), L)
        y += 4

    if full:
        ch.panel(y, rx, pw, 8, "PLATE", F, A)
        ch.kv(y + 1, rx + 2, pw - 4, "grains", format(st["grains"], ","), L, V)
        ch.kv(y + 2, rx + 2, pw - 4, "canvas", "%d\u00d7%d dots" % (st["subw"], st["subh"]), L, V)
        ch.kv(y + 3, rx + 2, pw - 4, "field", "%d\u00d7%d" % (st["pw"], st["ph"]), L, V)
    # A frame rate nobody measured is printed as a dash. In --dump the loop never
    # runs, so this would otherwise show the TARGET -- 240 on a 240Hz monitor --
    # in a still image that looks like a benchmark, for a renderer that measures
    # 67-89fps live. A dashboard number should be a measurement or absent.
        fps_txt = "\u2014" if not st["fps"] else "%.0f" % st["fps"]
        ch.kv(y + 4, rx + 2, pw - 4, "modes \u00b7 fps", "%d \u00b7 %s" % (len(st["amps"]), fps_txt), L, V)
        ch.kv(y + 5, rx + 2, pw - 4, "settled", "%.0f%%" % (100 * st["settled"]), L, V)
        ch.kv(y + 6, rx + 2, pw - 4, "figure held", "%.1fs" % st["held"], L, V)
        y += 9

    # --- the ayah, across the full width ------------------------------------
    # The words are the reason any of this is on screen, so they get the full
    # width rather than the ~30-column left panel that ellipsised them.
    # GAP=1 between a label and what it labels, GAP*2 between distinct groups:
    # distance is what says these are different things.
    # PAD, DO NOT UNPACK. A track whose metadata carries no translation (or
    # only the transliteration) gave this a 0- or 1-tuple and draw() died with
    # "not enough values to unpack" -- mid-frame, on a real recitation, with
    # the terminal still in cbreak.
    _txt = tuple(st["ayah_text"]) + ("", "")
    tl_, tr_ = _txt[0], _txt[1]
    GAP = 1
    # MEASURE, not full width. Lines ran to 150 columns -- about 130 characters --
    # where running text is comfortable near 65-75. The words are the reason any of
    # this is on screen, so they get a readable column rather than the whole screen.
    wide = min(cols - 8, 110)
    # THE TRANSLATION GETS THREE LINES; THE TRANSLITERATION TWO. Measured across
    # all 760 ayat in the library: translations run to a median of 72 characters
    # but a 99th percentile of 397 and a maximum of 583. At two lines there is no
    # width that fits them -- even at 150 columns, 2.4% were cut off with an
    # ellipsis, which is the "translation doesn't fit" this band kept showing.
    # Three lines at 110 columns leaves about 2% still clipped, and 110 is close
    # enough to a readable measure to be worth the trade.
    tl_lines = [l for l in wrap(tl_, wide, 2) if l] if tl_ else []
    tr_lines = [l for l in wrap(tr_, wide, 3) if l] if tr_ else []
    # THE BAND'S TOP IS A CONSTANT, NOT A FUNCTION OF THE VERSE. Anchored to the
    # same row every frame and sized for the longest case, so a one-line verse
    # and a three-line one put their header in exactly the same place and
    # neither can reach the plate. Short verses leave the tail of the band
    # blank, which is what keeps the rhythm still.
    ay = rows - BOT_ROWS
    if tl_lines and ay > 4:
        hdr = "%d : %d   of %d   \u00b7   %s   \u00b7   %s" % (
            st["surah_no"], st["ayah"], st["n_ayat"], st["surah"], st["revelation"])
        ch.put(ay, max(2, (cols - len(hdr)) // 2), hdr, D)
        r = ay + 1 + GAP
        for ln in tl_lines:
            ch.put(r, max(2, (cols - len(ln)) // 2), ln, B)
            r += 1
        # TWO blank rows before the translation, one after the header. The
        # previous version wrote GAP*2 - 1 here, which is a single blank row --
        # the same as the header gap -- so the three parts read as one block
        # instead of a label, a line to read aloud, and its meaning. Proximity is
        # the whole point: distance is what says these are different things.
        r += GAP * 2
        for ln in tr_lines:
            ch.put(r, max(2, (cols - len(ln)) // 2), ln, V)
            r += 1

    # --- bottom ---------------------------------------------------------------
    # THE RULE IS THE BAR. The separator under the plate and a full-width
    # progress bar draw the same horizontal stroke, so they share one row --
    # which is how the strip went from three rows to two. Gold is everything
    # recited, bright is the ayah playing now, hairline is what is left.
    #
    # The old per-ayah ticks are gone because they could not survive a long
    # surah: Yaseen's 83 boundaries across 80 columns is a dotted line, not
    # information. The bright segment carries position WITHIN the ayah instead,
    # which is the fact the ticks were there to give and holds at any length.
    br = rows - 2
    x0, wid = 2, max(10, cols - 4)
    done = int(round(float(np.clip(st["t"] / max(1e-6, st["dur"]), 0.0, 1.0)) * wid))
    a_t = st["ayat"][st["ayah"] - 1] if st["ayat"] else 0.0
    head = min(done, int(round(float(np.clip(a_t / max(1e-6, st["dur"]), 0.0, 1.0)) * wid)))
    ch.put(br, x0, "\u2501" * head, A)
    ch.put(br, x0 + head, "\u2501" * (done - head), B)
    ch.put(br, x0 + done, "\u2500" * (wid - done), F)

    # One readout line, three zones. The surah and reciter are NOT repeated
    # here -- they are already top-left; the strip carries only what nothing
    # else on screen carries. Drive has no label: that lives in `i`.
    lit = int(round(float(np.clip(st["level"] / 1.6, 0.0, 1.0)) * 7))
    ch.put(br + 1, 2, "\u25ae" * lit, A)
    ch.put(br + 1, 2 + lit, "\u25af" * (7 - lit), F)
    pos = "ayah %d of %d" % (st["ayah"], st["n_ayat"])
    ch.put(br + 1, max(11, (cols - len(pos)) // 2), pos, L)
    tail = "%s / %s" % (mmss(st["t"]), mmss(st["dur"]))
    ch.put(br + 1, cols - 2 - len(tail), tail, L)

    # The hints are the only route to `i` and `s`, so they cannot simply be
    # deleted -- but a screensaver that keeps announcing its own keybindings is
    # not at rest. They ride over the bar for 8s after launch and after any
    # key, then get out of the way.
    if st.get("hints"):
        # Right-aligned, not centred: the right end of the bar is the part that
        # has not been played yet, so the hints cover the stretch carrying the
        # least information. Centred, they sat exactly where the fill ends.
        hint = " i what am I looking at   \u00b7   s settings   \u00b7   any key exits "
        ch.put(br, max(2, cols - 2 - len(hint)), hint, D)


# ---------------------------------------------------------------- runtime ---
RUN = {"go": True, "resized": False}
# Deadline for the transient key hints over the progress bar; bumped on
# every keypress so they come back whenever someone is actually at the
# keyboard, and stay gone the rest of the time.
HINTS = {"until": 0.0}


def membrane_mode(cols):
    """(strength, dots) for however the membrane is set, at this width."""
    m = SETTINGS.get("membrane", "dots")
    if m == "dots" and cols < DOTS_MIN_COLS:
        return MEMBRANE["soft"], False
    return MEMBRANE.get(m, 0.35), m == "dots"


def _stop(*_a):
    RUN["go"] = False


def _winch(*_a):
    RUN["resized"] = True


_SYNC_DIR = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
_AUDIO_LOCK = os.path.join(_SYNC_DIR, "chladni-audio.lock")
_SYNC_FILE = os.path.join(_SYNC_DIR, "chladni-sync.json")
_lock_fd = None


def claim_audio():
    """True for exactly one running instance: the one that plays sound.

    Omarchy starts one screensaver per monitor. Every instance used to play its
    own audio from its own random track, so two monitors meant two DIFFERENT
    recitations over each other -- measured: Al-Mulk and Al-Qadr at once. An
    exclusive, non-blocking flock held for the life of the process picks one
    owner, and the kernel releases it however the owner exits.
    """
    global _lock_fd
    try:
        fd = os.open(_AUDIO_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    _lock_fd = fd                     # keep it open: closing it releases the lock
    return True


def publish_start(track, epoch):
    """The owner tells the others which track it began and exactly when."""
    tmp = _SYNC_FILE + ".%d" % os.getpid()
    with open(tmp, "w") as fh:
        json.dump({"pid": os.getpid(), "track": int(track), "epoch": float(epoch)}, fh)
    os.replace(tmp, _SYNC_FILE)       # atomic, so a reader never sees half a file


def await_start(since, timeout=8.0):
    """A silent instance waits for the owner's start, then follows it.

    Only accepts a record written by a process that is still alive and that
    appeared after this instance started, so a stale file from an earlier
    session cannot steer a new one. Returns None on timeout, and the instance
    then runs on its own clock, silently -- a picture out of step beats no picture.
    """
    end = time.time() + timeout
    while time.time() < end:
        try:
            if os.path.getmtime(_SYNC_FILE) >= since - 1.0:
                with open(_SYNC_FILE) as fh:
                    rec = json.load(fh)
                os.kill(int(rec["pid"]), 0)
                return rec
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.05)
    return None


class Audio:
    """Play the recitation that is driving the figure, in step with it.

    The analysis is precomputed, so nothing here reads the audio -- it only
    plays it. One process per track, replaced when the track changes, killed on
    exit. If there is no player or no device the screensaver carries on silently
    rather than failing; the picture is the point, the sound is the company.
    """

    def __init__(self, volume=0.55, enabled=True):
        self.volume = max(0.0, min(1.0, float(volume)))
        self.enabled = enabled and self.volume > 0.0
        self.proc = None
        self.player = None
        for cand in ("pw-play", "paplay", "ffplay"):
            if shutil.which(cand):
                self.player = cand
                break
        if self.player is None:
            self.enabled = False

    def _cmd(self, path):
        if self.player == "pw-play":
            return ["pw-play", "--volume=%.3f" % self.volume, path]
        if self.player == "paplay":
            return ["paplay", "--volume=%d" % int(self.volume * 65536), path]
        return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                "-volume", str(int(self.volume * 100)), path]

    def play(self, key):
        self.stop()
        if not self.enabled:
            return
        path = os.path.join(AUDIO_DIR, key + ".ogg")
        if not os.path.exists(path):
            return
        try:
            self.proc = subprocess.Popen(
                self._cmd(path), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
        except OSError:
            self.proc = None
            self.enabled = False

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=0.4)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            except OSError:
                pass
            self.proc = None


def die_with_parent():
    """Ask the kernel to kill us if the launcher goes away.

    numpy spends most of each frame inside C code, so a TERM sent by the wrapper
    is not acted on until that returns; if the wrapper exits first the renderer
    is orphaned and keeps drawing to a terminal nobody is watching. Relying on
    the parent to reap us is the fragile half of this -- PR_SET_PDEATHSIG makes
    it the kernel's job instead.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0)
    except (OSError, AttributeError, ValueError):
        pass


def term_size(default=(150, 40)):
    try:
        ts = os.get_terminal_size()
        return max(50, ts.columns), max(14, ts.lines)
    except OSError:
        return default


def detect_refresh(default=60.0):
    """The display's actual refresh rate, so animation is not pinned to 24.

    A fixed 24 was leaving a 120Hz panel running at a fifth of what it can show.
    The physics is rescaled against Sand.REF_FPS, so raising this changes only
    how smooth the motion is -- not how the sand behaves.

    Asks the compositor first; falls back to DRM sysfs, then to `default`. Never
    raises: a screensaver that refuses to start because it could not identify the
    monitor is worse than one running at 60.
    """
    try:
        r = subprocess.run(["hyprctl", "monitors", "-j"], capture_output=True,
                           text=True, timeout=3)
        if r.returncode == 0:
            mons = json.loads(r.stdout)
            # THE MONITOR IT IS SHOWING ON, NOT THE FASTEST ONE ATTACHED. This
            # took the maximum across all outputs, which was the same answer
            # until an external 239.757Hz display was plugged in next to the
            # 120Hz panel -- then the screensaver targeted 240 on a panel that
            # cannot show it, and the loop stopped sleeping between frames to
            # chase a rate no one would see. The screensaver opens on the
            # focused output, so that is the one whose rate matters.
            focused = [m for m in mons if m.get("focused")]
            pick = focused or mons
            best = max((float(m.get("refreshRate") or 0) for m in pick), default=0.0)
            if 20.0 <= best <= 480.0:
                return best
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        for d in glob.glob("/sys/class/drm/card*/modes"):
            with open(d) as fh:
                head = fh.readline().strip()
            if "@" in head:
                hz = float(head.split("@")[1].rstrip("iA-Za-z"))
                if 20.0 <= hz <= 480.0:
                    return hz
    except (OSError, ValueError, IndexError):
        pass
    return default


def load_timeline(path):
    """The analysed recitation, or None to fall back to a synthetic timeline.

    A MISSING file was always handled; a DAMAGED one was not, and a truncated
    or bit-flipped npz raised BadZipFile straight out of main() -- before the
    try/finally, so it crashed AND left the terminal in cbreak. Same outcome
    for the user either way, so treat unreadable as absent and say so on stderr.
    """
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
    except Exception as e:
        sys.stderr.write("cyscreen: %s is unreadable (%s: %s) -- "
                         "falling back to a synthetic timeline\n"
                         % (path, type(e).__name__, e))
        return None
    if isinstance(meta, list):                       # timeline from before ayat
        meta = {"tracks": meta}
    out = {"amps": z["amps"].astype(np.float32), "lvl": z["lvl"].astype(np.float32),
           "hz": z["hz"].astype(np.float32), "offsets": z["offsets"],
           "alpha": z["alpha"], "mn": z["mn"],
           "meta": meta["tracks"], "fatiha": meta.get("fatiha", [])}
    # Precomputed figure schedule. Deciding this live meant the picture trailed
    # the sound by several seconds; offline the decision can be centred on the
    # moment and shifted early so the sand is settled when you hear why.
    out["fig"] = z["fig"].astype(np.int32) if "fig" in z.files else None
    return out


def synth_timeline(nm, fps=24.0, secs=90.0):
    """Fallback when no recitation has been analysed: sweep the modes."""
    n = int(fps * secs)
    amps = np.zeros((n, nm), dtype=np.float32)
    for i in range(n):
        k = (i / n) * nm
        j = int(k) % nm
        amps[i, j] = 1.0
        amps[i, (j + 1) % nm] = max(0.0, 1.0 - abs(k - int(k) - 0.5) * 2.0) * 0.4
    from scipy.special import jn_zeros
    mn = np.array(sorted(((m, n2) for m in range(8) for n2 in range(1, 9)),
                         key=lambda t: jn_zeros(t[0], t[1])[t[1] - 1])[:nm], dtype=np.int16)
    alpha = np.array([jn_zeros(int(m), int(n2))[int(n2) - 1] for m, n2 in mn], dtype=np.float32)
    return {"amps": amps, "lvl": np.full(n, 1.0, np.float32), "hz": np.zeros(n, np.float32),
            "offsets": np.array([0, n], np.int32), "alpha": alpha, "mn": mn,
            "meta": [{"key": "synthetic", "name": "mode sweep", "sub": "no recitation analysed",
                      "f0": 110.0, "frames": n, "fps": fps, "ayat": [], "surah": "\u2014",
                      "surah_no": 0, "revelation": "\u2014", "n_ayat": 0}], "fatiha": []}


# ------------------------------------------------------------------- menu ---
DEFAULT_VOLUME = 0.55
DEFAULT_GAP = 2.5

MENU_ITEMS = [
    ("fps",       "frame rate",   ["auto", 30, 60, 120, 240],  "live"),
    ("threads",   "performance",  ["auto", 1, 2, 4],           "next launch"),
    ("detail",    "detail",       [6, 8, 11, 14],              "next launch"),
    ("grains",    "sand",         ["auto", 0.5, 1.0, 2.0],     "live"),
    ("volume",    "volume",       [0.0, 0.25, 0.4, 0.55, 0.75, 1.0], "next recitation"),
    ("tint",      "tint",         [0.0, 0.12, 0.22, 0.4],      "live"),
    ("gap",       "gap",          [0.0, 1.5, 3.0, 6.0],        "live"),
    ("subtitles", "subtitles",    [True, False],               "live"),
    ("panels",    "panels",       ["focus", "full", "off"],    "live"),
    ("membrane",  "membrane",     ["off", "soft", "full", "dots"], "live"),
    ("framing",   "framing",      ["offset", "centred"],       "next launch"),
    ("hold",      "stillness",    [0.0, 1.5, 3.0, 6.0],        "live"),
]

MENU_HELP = {
    "fps":       "frames per second. auto follows this monitor's refresh rate",
    "threads":   "more threads: smoother, much hotter. auto = 2 on mains, 1 on battery",
    "detail":    "terminal font size. smaller cell = finer plate, more work per frame",
    "grains":    "how much sand. auto scales with the plate",
    "volume":    "recitation volume",
    "tint":      "how far the palette turns with each reciter. 0 keeps one palette",
    "gap":       "seconds of quiet between recitations",
    "subtitles": "the ayah, its transliteration and its meaning",
    "panels":    "focus keeps what explains the art; full adds the telemetry",
    "membrane":  "the vibrating plate under the sand. dots draws it at 8x the\n                  resolution of a cell background, which cannot show blocks",
    "framing":   "offset puts the plate right of centre; a symmetric figure centred\n                  in a rectangle reads as a specimen in a case",
    "hold":      "seconds a figure must stay before the schedule may change it.\n                  0 follows the recitation exactly; higher trades response for calm",
}


def fmt_val(v):
    if v is True:
        return "on"
    if v is False:
        return "off"
    if isinstance(v, float):
        return ("%.2f" % v).rstrip("0").rstrip(".")
    return str(v)


class Menu:
    """The settings overlay. Every adjustable thing lives here.

    The point is that nothing needs a config file or a rebuild: this is the only
    place the program is configured, it writes its own settings, and it says
    per-row whether a change is live or waits for the next launch -- because a
    couple genuinely cannot change in a running process (the BLAS thread count is
    read once at import, and the terminal's font size belongs to the terminal).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.i = 0
        self.open = False
        self.note = ""

    def cur_key(self):
        return MENU_ITEMS[self.i][0]

    def move(self, d):
        self.i = (self.i + d) % len(MENU_ITEMS)

    def cycle(self, d):
        key, _label, values, scope = MENU_ITEMS[self.i]
        cur = self.cfg.get(key, DEFAULTS[key])
        try:
            j = values.index(cur)
        except ValueError:
            j = 0
        self.cfg[key] = values[(j + d) % len(values)]
        ok = save_settings(self.cfg)
        self.note = ("%s = %s  (%s)" % (_label, fmt_val(self.cfg[key]), scope)
                     if ok else "could not write %s" % SETTINGS_PATH)
        return key, self.cfg[key], scope


def draw_menu(ch, menu, cols, rows, C):
    """Draw the overlay. Sized to its content, centred, over the plate."""
    w = min(cols - 6, 72)
    h = len(MENU_ITEMS) + 7
    r0 = max(1, (rows - h) // 2)
    c0 = max(2, (cols - w) // 2)
    ch.panel(r0, c0, w, h, "SETTINGS", C["rule"], C["accent"])
    for n, (key, label, values, scope) in enumerate(MENU_ITEMS):
        r = r0 + 2 + n
        sel = (n == menu.i)
        val = fmt_val(menu.cfg.get(key, DEFAULTS[key]))
        ch.put(r, c0 + 2, ("\u25b8 " if sel else "  ") + label,
               C["bright"] if sel else C["label"])
        # value right-aligned in its own column; scope sits left of it, never over it
        vw = max(len(fmt_val(v)) for _k, _l, vs, _s in MENU_ITEMS for v in vs)
        ch.put(r, c0 + w - 3 - len(val), val, C["accent"] if sel else C["value"])
        if scope != "live":
            ch.put(r, c0 + w - 5 - vw - len(scope), scope, C["dim"])
    ch.put(r0 + h - 4, c0 + 2, MENU_HELP.get(menu.cur_key(), "")[:w - 4], C["dim"])
    if menu.note:
        ch.put(r0 + h - 3, c0 + 2, menu.note[:w - 4], C["value"])
    ch.put(r0 + h - 2, c0 + 2,
           "up/down choose \u00b7 left/right change \u00b7 s or esc close \u00b7 q quit"[:w - 4],
           C["label"])


INFO = [
    ("What this is", [
        "A recitation of the Qur'an -- the scripture of Islam -- driving a physics simulation.",
        "Sand on a vibrating plate gathers where the plate is still. Pitch decides the pattern;",
        "these are the real standing waves of a circular drum skin, not an animation.",
    ]),
    ("Top left", [
        "The reciter, and which of the Qur'an's 114 chapters (a surah) is being read.",
        "Meccan or Medinan says which period of the revelation it belongs to.",
    ]),
    ("Under the plate", [
        "89 : 8   of 30   is chapter 89, verse 8, of that chapter's 30 verses. A verse is an ayah.",
        "Middle line: the Arabic sounds written in Latin letters. Read it aloud --",
        "a, i and u with a bar over them are held long.",
        "Bottom line: what it means, in Muhammad Asad's English translation.",
    ]),
    ("Under the plate, the line", [
        "The rule beneath the plate is also the timeline: gold is what has been recited,",
        "the bright tip is the verse being read now, the thin part is what is left.",
        "The seven marks bottom-left are drive -- how hard the voice is hitting the plate.",
    ]),
    ("Why the sand lands where it does", [
        "E = \u03a3 a\u1d62\u00b2 U\u1d62\u00b2 -- an INCOHERENT sum, not (\u03a3 a\u1d62U\u1d62)\u00b2.",
        "Sand rests only where every driven mode is quiet at once, so one clean tone gives",
        "you lines and a chord gives you isolated dots. This is a clamped membrane, not a",
        "free-edge metal plate, and says so: a real plate's modes are not Bessel functions.",
    ]),
    ("Right", [
        "FIGURE names the standing wave: (2,2) is 2 nodal diameters -- so 4 spokes -- and 1 ring.",
        "J is a Bessel function, the mathematics of how a circular membrane vibrates.",
        "MODE LADDER is how strongly the voice is exciting each shape right now.",
        "PLATE is the simulation itself; drive, bottom left, is how loud the voice is.",
    ]),
]


def draw_info(ch, cols, rows, C):
    """Plain-English guide to the screen, for a viewer who knows none of this."""
    # DROP WHOLE SECTIONS RATHER THAN OVERFLOW. Chrome.panel() refuses to draw a
    # box taller than the grid and returns silently, so a window too short for
    # the full guide used to open the overlay onto nothing at all -- a help
    # screen that disappears exactly on the small terminals that need it most.
    rendered, used = [], 5
    for title, lines in INFO:
        if used + len(lines) + 2 > rows - 2:
            break
        rendered.append((title, lines))
        used += len(lines) + 2
    if not rendered:
        return
    body = [ln for _t, lines in rendered for ln in ([""] + lines)]
    w = min(cols - 6, max(len(l) for l in body) + 6)
    h = len(body) + len(rendered) + 5
    r0 = max(1, (rows - h) // 2)
    c0 = max(2, (cols - w) // 2)
    ch.panel(r0, c0, w, h, "WHAT AM I LOOKING AT", C["rule"], C["accent"])
    r = r0 + 2
    for title, lines in rendered:
        ch.put(r, c0 + 2, title, C["accent"]); r += 1
        for ln in lines:
            ch.put(r, c0 + 2, ln[:w - 4], C["value"]); r += 1
        r += 1
    foot = "i closes this \u00b7 s settings \u00b7 any other key exits"
    if len(rendered) < len(INFO):
        foot = "%d more section%s need a taller window \u00b7 " % (
            len(INFO) - len(rendered), "" if len(INFO) - len(rendered) == 1 else "s") + foot
    ch.put(r0 + h - 2, c0 + 2, foot[:w - 4], C["label"])


_tty_saved = None


def keyboard(on=True):
    """Own the keyboard, so the settings menu can exist at all.

    A screensaver exits on any key, and that check used to live in the launcher
    script -- which meant the renderer never saw a keystroke and could not offer
    anything but exiting. Now the renderer reads the keys and keeps the same
    contract: any key still quits, except the ones that open and drive the menu.
    A terminal left in cbreak mode is a broken shell, so main() also registers
    keyboard(False) with atexit -- the finally block alone did not cover the
    early-return paths.
    """
    global _tty_saved
    try:
        fd = sys.stdin.fileno()
        if on:
            if _tty_saved is None and os.isatty(fd):
                _tty_saved = termios.tcgetattr(fd)
                tty.setcbreak(fd, termios.TCSANOW)   # TCSAFLUSH would discard keys typed during startup
        elif _tty_saved is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, _tty_saved)
            _tty_saved = None
    except (OSError, termios.error, ValueError):
        _tty_saved = None


def read_keys():
    """Whatever is waiting on stdin, as tokens. Never blocks."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return []
        if not select.select([fd], [], [], 0)[0]:
            return []
        data = os.read(fd, 64).decode("utf-8", "replace")
    except (OSError, ValueError):
        return []
    keys, i = [], 0
    while i < len(data):
        c = data[i]
        if c == "\x1b" and data[i + 1:i + 2] == "[":
            code = data[i + 2:i + 3]
            keys.append({"A": "up", "B": "down", "C": "right", "D": "left"}.get(code, "esc"))
            i += 3
        elif c == "\x1b":
            keys.append("esc"); i += 1
        else:
            keys.append(c); i += 1
    return keys


def main():
    launched_at = time.time()
    HINTS["until"] = launched_at + 8.0
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=0.0,
                    help="target frame rate (default: the monitor's refresh rate)")
    ap.add_argument("--grains", type=int, default=0)
    ap.add_argument("--size", default="")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--start", type=float, default=None,
                    help="seconds into the timeline (default: a random track)")
    ap.add_argument("--no-chrome", action="store_true")
    ap.add_argument("--anonymise", "--anonymize", dest="anon", action="store_true",
                    help="redact machine identifiers, for screenshots")
    ap.add_argument("--dump", action="store_true", help="settle, print one frame, exit")
    ap.add_argument("--aspect", type=float, default=0.0, help="override sub-dot h/w")
    ap.add_argument("--rim", type=float, default=0.17, help="how hard the clamped rim is swept")
    ap.add_argument("--select", type=float, default=6.0, help="mode selectivity")
    ap.add_argument("--volume", type=float, default=DEFAULT_VOLUME, help="recitation volume 0-1")
    ap.add_argument("--silent", action="store_true", help="do not play the recitation")
    ap.add_argument("--gap", type=float, default=DEFAULT_GAP, help="seconds of quiet between reciters")
    args = ap.parse_args()
    keyboard(True)                 # before the timeline load, so early keys are not lost
    # RESTORE ON *EVERY* PATH, NOT JUST THE ONE WITH THE finally. keyboard(True)
    # runs here, but the try/finally that undid it starts ~180 lines below --
    # so --dump's `return 0`, a corrupt timeline.npz raising out of
    # load_timeline, and any other early exit all left the terminal in cbreak
    # and handed the user a broken shell. atexit covers returns, sys.exit and
    # uncaught exceptions alike; keyboard(False) is idempotent, so the existing
    # finally running first is harmless.
    atexit.register(keyboard, False)
    # settings file supplies anything not given on the command line
    if args.fps <= 0:
        want_fps = SETTINGS.get("fps", "auto")
        args.fps = detect_refresh() if want_fps == "auto" else float(want_fps)
    if SETTINGS.get("panels") == "off":
        args.no_chrome = True
    if args.gap == DEFAULT_GAP:
        args.gap = float(SETTINGS.get("gap", DEFAULT_GAP))
    if args.volume == DEFAULT_VOLUME:
        args.volume = float(SETTINGS.get("volume", DEFAULT_VOLUME))
    menu = Menu(SETTINGS)
    info_open = False

    if args.size:
        cols, rows = (int(v) for v in args.size.lower().split("x"))
    else:
        cols, rows = term_size()
    cols = max(50, cols); rows = max(14, rows)
    globals()["SUB_ASPECT"] = args.aspect if args.aspect > 0 else measure_sub_aspect()

    tl = load_timeline(TIMELINE)
    if tl is None:
        tl = synth_timeline(20)
    nm = tl["amps"].shape[1]

    scr = Screen(cols, rows)
    subw, subh = scr.subw, scr.subh
    bank = ModeBank(tl["mn"], subw, subh, rim_strength=args.rim,
                    select=args.select, **plate_fit(cols, rows))
    ngrain = args.grains or int(np.clip(subw * subh * 0.55, 45000, 300000))
    sand = Sand(ngrain, subw, subh, bank, fps=args.fps)
    sand.dts = Sand.REF_FPS / max(args.fps, 1.0)
    _last_step_t = 0.0
    _sel_t = 0.0
    was_fading = False
    gain = 1.0 / max(3.0, ngrain / float(cols * rows) * 4.5)
    chrome = Chrome(cols, rows)
    scr.set_underlay(plate_ring(bank, subw, subh), C["rule"])
    info = sysinfo()
    if args.anon:
        # The panel prints the hostname, and on a single-user box that is the
        # username. Screenshots outlive the terminal they were taken in.
        info["host"] = "localhost"
        info["kernel"] = info["kernel"].split("-")[0] if info["kernel"] else ""


    sel = (FigurePlayer(nm, args.fps, min_hold=float(SETTINGS.get("hold", 0.0))) if tl.get("fig") is not None
           else FigureSelector(nm, args.fps))
    fps_meas = args.fps
    pending_size, pending_at = (0, 0), 0.0
    offs = tl["offsets"]; total = int(offs[-1])
    if args.start is None:
        # START ON A RANDOM TRACK BOUNDARY. A screensaver runs for minutes and dies on
        # a keypress, so a fixed start means only the first track or two is ever seen;
        # with Ya-Sin at position 16 of a 55-minute playlist the long recitations were
        # unreachable. Snapping to a boundary opens on a first ayah, not mid-word.
        starts = [int(o) for o in offs[:-1]] or [0]
        gi = starts[random.randrange(len(starts))]
    else:
        # SECONDS OF TIMELINE, CONVERTED AT THE TIMELINE'S RATE. This used args.fps,
        # which was the same number (24) until the display's refresh rate became
        # the default. On a 240Hz monitor --start 3000 then asked for frame
        # 720,000 of a 170,220-frame timeline, was clamped to the last frame, and
        # every --start rendered the same final moment. The analysis runs at its
        # own fixed rate; the display rate is for pacing only.
        tl_fps = float(tl["meta"][0].get("fps", 24.0)) if tl.get("meta") else 24.0
        gi = int(np.clip(args.start * tl_fps, 0, total - 1))

    stats = livestats(info)

    def state(gi, amps, held=(0,)):
        ti = int(np.clip(np.searchsorted(offs, gi, side="right") - 1, 0, len(tl["meta"]) - 1))
        md = tl["meta"][ti]
        tfps = md.get("fps", 24.0)
        local = (gi - int(offs[ti])) / tfps
        dur = md["frames"] / tfps
        ayat = md.get("ayat", []) or []
        ai = 0
        for k, a_t in enumerate(ayat):
            if local >= a_t:
                ai = k
        nxt = ayat[ai + 1] if ai + 1 < len(ayat) else dur
        span = max(1e-6, nxt - (ayat[ai] if ayat else 0.0))
        frac = np.clip((local - (ayat[ai] if ayat else 0.0)) / span, 0.0, 1.0)
        words = md.get("text") or tl.get("fatiha") or []
        txt = tuple(words[ai]) if ai < len(words) else ("", "")
        if txt and txt[0]:
            txt = (accent(txt[0]),) + tuple(txt[1:])
        return {"name": md["name"], "sub": md["sub"], "f0": md["f0"],
                "t": local, "dur": dur, "amps": amps, "mn": tl["mn"], "alpha": tl["alpha"],
                "dom": int(held[0]), "level": float(tl["lvl"][gi]), "hz": float(tl["hz"][gi]),
                "grains": ngrain, "subw": subw, "subh": subh, "pw": bank.pw, "ph": bank.ph,
                "fps": fps_meas, "settled": sand.settled, "held": sel.since,
                "surah": md.get("surah", "\u2014"), "surah_no": md.get("surah_no", 0),
                "n_ayat": md.get("n_ayat", len(ayat)), "revelation": md.get("revelation", ""),
                "ayah": ai + 1, "ayah_text": txt, "ayah_frac": float(frac), "ayat": ayat,
                "panels": SETTINGS.get("panels", "focus"),
                "hints": time.time() < HINTS["until"],
                "theme": info["theme"], "os": info["os"], "kernel": info["kernel"],
                "host": info["host"], "cpu": info["cpu"], "gpu": info["gpu"],
                "pkgs": info["pkgs"], "shell": info["shell"], "res": info["res"],
                "mem_used": stats["mem_used"], "mem_pct": stats["mem_pct"],
                "temp": stats["temp"], "load": stats["load"], "uptime": stats["uptime"]}

    if args.dump:
        amps = tl["amps"][gi]
        sel = (FigurePlayer(nm, args.fps, min_hold=float(SETTINGS.get("hold", 0.0))) if tl.get("fig") is not None
               else FigureSelector(nm, args.fps))
        for _ in range(int(args.fps * 4)):
            w = (sel.update_to(tl["fig"][gi]) if tl.get("fig") is not None
                 else sel.update(amps, bank.select))
        E, GX, GY = bank.field_w(w)
        # NO WALL CLOCK HERE, so run at the REFERENCE step. sand.dts is set from
        # the target frame rate for the live loop; left at that value, 700 steps
        # on a 120Hz display count as 140 reference steps and --dump prints a
        # figure one fifth of the way to settled. The frame-rate fix silently
        # changed this path's meaning, which is the kind of regression a flag
        # that only runs on request is best at hiding.
        sand.dts = 1.0
        fps_meas = 0.0          # nothing was timed; the dashboard shows a dash
        for i in range(700):
            sand.step(E, GX, GY, 0.85 if i < 60 else 0.30)
        scr.set_field_bg(E, bank, *membrane_mode(cols))
        ch, lv, bgv = scr.compose(sand, gain)
        if not args.no_chrome:
            draw(chrome, state(gi, amps, (sel.cur,)))
            ch, lv, bgv = chrome.composite(ch, lv, bgv)
        buf = []
        for r in range(rows):
            line = []; last = -1
            lastb = -1
            for c in range(cols):
                v = int(lv[r, c]); g = int(bgv[r, c])
                if v != last:
                    rgb = LUT[v]; line.append("\033[38;2;%d;%d;%dm" % tuple(int(x) for x in rgb)); last = v
                if g != lastb:
                    if g == 0:
                        line.append("\033[49m")
                    else:
                        rgb = LUT[BG0 + g]; line.append("\033[48;2;%d;%d;%dm" % tuple(int(x) for x in rgb))
                    lastb = g
                line.append(ch[r, c])
            buf.append("".join(line))
        sys.stdout.write("\033[0m\n".join(buf) + "\033[0m\n")
        return 0

    die_with_parent()
    for sgn in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(sgn, _stop)
    try:
        signal.signal(signal.SIGWINCH, _winch)
    except (AttributeError, ValueError):
        pass

    sys.stdout.write("\033]11;rgb:%02x/%02x/%02x\007\033[?25l\033[2J" % GROUND)
    sys.stdout.flush()

    # The timeline is driven by the WALL CLOCK, not by a frame counter, so the
    # picture cannot drift away from the recitation playing beside it. A dropped
    # frame costs a frame, not sync.
    owner = claim_audio()
    audio = Audio(volume=args.volume, enabled=(not args.silent) and owner)
    cur_hue = None
    last_fig = -2
    ntracks = len(tl["meta"])
    track = int(np.clip(np.searchsorted(offs, gi, side="right") - 1, 0, ntracks - 1))
    track_started = time.time()
    if owner:
        publish_start(track, track_started)
    else:
        rec = await_start(launched_at)
        if rec is not None:
            track = int(rec["track"]) % ntracks
            track_started = float(rec["epoch"])
    gap_until = 0.0
    audio.play(tl["meta"][track]["key"])

    started = time.time(); t0 = time.time(); frame = 0
    try:
        while RUN["go"]:
            tmeta = tl["meta"][track]
            tfps = tmeta.get("fps", 24.0)
            tframes = int(tmeta["frames"])
            now = time.time()
            if gap_until:
                # A beat of quiet between reciters: the sand keeps settling on
                # the last figure while the next voice is cued up.
                if now >= gap_until:
                    # The next track starts at a COMPUTED time, not at whatever
                    # moment this loop noticed. Two instances noticing on
                    # different frames would otherwise drift a frame apart per
                    # track, and a screensaver runs for hours.
                    track_started = gap_until
                    gap_until = 0.0
                    track = (track + 1) % ntracks
                    audio.play(tl["meta"][track]["key"])
                    continue
                local = tframes - 1
            else:
                local = int((now - track_started) * tfps)
                if local >= tframes:
                    audio.stop()
                    gap_until = track_started + tframes / tfps + max(0.0, args.gap)
                    local = tframes - 1
            gi = int(offs[track]) + min(local, tframes - 1)
            amps = tl["amps"][gi % total]
            if tl.get("fig") is not None:
                _fdt = min(0.25, now - _sel_t) if _sel_t else (1.0 / max(args.fps, 1.0))
                _sel_t = now
                w = sel.update_to(tl["fig"][gi % total], _fdt)
            else:
                w = sel.update(amps, bank.select)
            E, GX, GY = bank.field_w(w)
            # REDRAW THE WASH THROUGH THE WHOLE CROSSFADE. This fired only when sel.cur
            # changed -- the first frame of the fade, when E is still almost entirely the
            # outgoing figure -- so the wash froze on the previous figure and the new
            # figure's nodal lines were drawn across its bright antinodes.
            fading = float(getattr(sel, "left", 0.0)) > 0.0
            if sel.cur != last_fig or fading or was_fading:
                last_fig = sel.cur
                scr.set_field_bg(E, bank, *membrane_mode(cols))
            was_fading = fading
            # Loudness drives how hard the plate is hit; the spectrum decides
            # which modes. Keeping them separate stops quiet passages freezing
            # the sand and loud ones blowing the figure apart.
            lv_drive = float(tl["lvl"][gi % total])
            # The figure changes on a multi-second schedule, so this is the only channel
            # that answers the voice instantly. Slope rescaled once lvl stopped saturating
            # (median 1.599 -> 0.768); the old value, tuned against a signal pinned at its
            # ceiling, would have halved the plate's energy.
            agit = 0.10 + 1.05 * float(np.clip(lv_drive, 0.0, 1.6))
            # dt FROM THE CLOCK, NOT FROM THE FLAG. Setting this once from
            # --fps was correct about the number it was given and silent about
            # what the machine did: asked for 120 the renderer sustains ~67, so
            # a startup-computed dts of 24/120=0.20 ran the physics at 55% of
            # real speed and figures took twice as long to form. Measuring the
            # actual frame interval makes the sand behave identically whether
            # the machine hits its target, misses it, or is interrupted.
            _tn = time.time()
            _dt = _tn - _last_step_t if _last_step_t else (1.0 / max(args.fps, 1.0))
            _last_step_t = _tn
            sand.dts = float(np.clip(_dt * Sand.REF_FPS, 0.10, 2.0))
            sand.step(E, GX, GY, agit)
            ch, lvv, bgv = scr.compose(sand, gain)
            for k in read_keys():
                HINTS["until"] = time.time() + 8.0
                if info_open and k != "i":
                    info_open = False
                    if k in ("s", ","):
                        menu.open = True
                    elif k not in ("esc",):
                        RUN["go"] = False
                elif menu.open:
                    if k in ("up", "down"):
                        menu.move(-1 if k == "up" else 1)
                    elif k in ("left", "right"):
                        key, val, scope = menu.cycle(-1 if k == "left" else 1)
                        if key == "fps":
                            args.fps = detect_refresh() if val == "auto" else float(val)
                        elif key == "gap":
                            args.gap = float(val)
                        elif key == "panels":
                            args.no_chrome = (val == "off")
                        elif key == "volume":
                            audio.volume = float(val)
                        elif key == "grains":
                            n = (args.grains or int(np.clip(subw * subh * 0.55, 45000, 300000))
                                 if val == "auto" else int(np.clip(subw * subh * 0.55 * float(val),
                                                                   5000, 400000)))
                            sand.resize(n); ngrain = n
                            gain = 1.0 / max(3.0, ngrain / float(cols * rows) * 4.5)
                    elif k in ("s", "esc", ","):
                        menu.open = False
                    elif k == "q":
                        RUN["go"] = False
                elif k in ("s", ","):
                    menu.open = True; info_open = False
                elif k == "i":
                    info_open = not info_open
                else:
                    RUN["go"] = False          # screensaver contract: any other key exits
            if not args.no_chrome or menu.open or info_open:
                if not args.no_chrome:
                    draw(chrome, state(gi % total, amps, (sel.cur,)))
                if menu.open:
                    draw_menu(chrome, menu, cols, rows, C)
                elif info_open:
                    draw_info(chrome, cols, rows, C)
                ch, lvv, bgv = chrome.composite(ch, lvv, bgv)
            # TINT PER RECITATION, NOT PER FIGURE. Keyed to the figure it rotated the
            # palette every 1.08s (median), strobing through hues -- and every retint
            # invalidates the shared lookup table, forcing a full repaint about once a
            # second against a 0.12 MB/s byte budget. At half amplitude it also reached
            # magenta and acid green, not colours this plate has. The forced repaint
            # below is still required: the table is shared by every cell.
            want = float(MODE_HUE[track % len(MODE_HUE)]) * float(SETTINGS.get("tint", 0.22))
            if cur_hue is None or abs(want - cur_hue) > 1e-6:
                cur_hue = want
                globals()["LUT"] = build_lut(cur_hue)
                scr.prev_ch[:] = "\0"; scr.prev_lv[:] = 255; scr.prev_bg[:] = 255
            scr.flush(ch, lvv, bgv)
            frame += 1

            # Re-layout on resize. SIGWINCH is the fast path; the poll is the
            # honest one -- the signal is missed often enough (and never arrives
            # at all if the size changed before the handler was installed) that
            # relying on it alone is how the window ends up rendering at its
            # startup size forever, which is exactly what fullscreen did.
            if RUN["resized"] or frame % 6 == 0:
                RUN["resized"] = False
                nc, nr = term_size((cols, rows))
                # DEBOUNCE. A compositor ANIMATES a fullscreen toggle, so one
                # super+F emits a whole sequence of intermediate sizes. Rebuilding
                # on each one means ~20 ModeBank constructions and grain
                # reallocations during a single keypress, each clearing the screen
                # -- which is the glitching, not a drawing bug. Wait until the size
                # has stopped changing before paying for a rebuild.
                if (nc, nr) != (cols, rows):
                    # DEBOUNCE ON TIME, NOT ON POLL COUNT. A poll count is a
                    # different amount of real time at every frame rate -- two
                    # polls is 0.08s at 24fps and 0.03s at 67, so the same code
                    # debounced five times harder on a slow machine. Ctrl+scroll
                    # zoom emits a size change per notch and a compositor
                    # ANIMATES fullscreen, so both arrive as bursts; wait for
                    # the burst to stop before paying for a rebuild.
                    now_t = time.time()
                    if (nc, nr) != pending_size:
                        pending_size = (nc, nr)
                        pending_at = now_t
                        continue
                    if now_t - pending_at < 0.22:
                        continue
                if (nc, nr) != (cols, rows):
                    shrank = (nc < cols) or (nr < rows)
                    cols, rows = nc, nr
                    globals()["SUB_ASPECT"] = args.aspect if args.aspect > 0 else measure_sub_aspect()
                    scr = Screen(cols, rows)
                    subw, subh = scr.subw, scr.subh
                    newbank = ModeBank(tl["mn"], subw, subh, rim_strength=args.rim,
                    select=args.select, **plate_fit(cols, rows))
                    sand.rebind(newbank, subw, subh)
                    bank = newbank
                    ngrain = args.grains or int(np.clip(subw * subh * 0.55, 45000, 300000))
                    sand.resize(ngrain)
                    gain = 1.0 / max(3.0, ngrain / float(cols * rows) * 4.5)
                    chrome = Chrome(cols, rows)
                    last_fig = -2
                    scr.set_underlay(plate_ring(bank, subw, subh), C["rule"])
                    info["res"] = term_pixels() or info["res"]
                    # 3J, NOT JUST 2J, AND ONLY WHEN THE GRID SHRANK. The terminal reflows on
                    # resize, so content can land outside the viewport 2J clears; 3J drops the
                    # scrollback and H parks the cursor at origin. But a fresh Screen repaints
                    # every cell anyway, so clearing when the grid GREW only inserts a black
                    # flash -- on a zoom burst that flash IS the glitching.
                    if shrank:
                        sys.stdout.write("\033[3J\033[2J\033[H"); sys.stdout.flush()
                    pending_size, pending_at = (0, 0), 0.0
                    continue
            if frame % 24 == 0:
                stats = livestats(info)
            if frame % 12 == 0:
                el = time.time() - t0
                if el > 0:
                    fps_meas = fps_meas * 0.7 + (12.0 / el) * 0.3
                t0 = time.time()
            if args.seconds and time.time() - started > args.seconds:
                break
            lag = started + frame / args.fps - time.time()
            if lag > 0:
                time.sleep(lag)
            elif lag < -2.0:
                started = time.time(); frame = 0
    finally:
        audio.stop()
        keyboard(False)
        sys.stdout.write("\033[0m\033[?25h\033[2J\033[H"); sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
