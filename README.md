# chladni-tui

**A Chladni sand plate, simulated from first principles and drawn in your terminal — driven by the frequency content of real audio.**

![the plate](docs/screensaver.png)

<sub>Husary reciting Ya-Sin 36:14. The figure is (1,3) — one nodal diameter and two
interior rings, at 0.377 and 0.690 of the radius where J₁'s zeros put them. Its
frequency is **550 Hz**; the peak in the voice at that moment is **551 Hz**. That is
one well-chosen frame: across the whole library the correlation between recited
pitch and the figure shown is a median of 0.50 per track. Rendered headlessly by
`tools/render-dump.py`, so it carries no desktop — and note what `--dump` is: it
holds one figure for 700 physics steps before drawing, so this is a fully settled
figure. Live, figures change about once a second and their lines render dimmer.</sub>

Sand on a vibrating plate collects wherever the plate is still. This simulates
that — 45,000 to 300,000 grains depending on plate size, on a clamped circular
membrane with real Bessel eigenmodes — and
renders it in braille sub-dots, so one terminal cell becomes a 2×4 block of
pixels. Which figure appears is decided by analysing an actual recording.

No GPU, no graphics library, no image output. Escape sequences and arithmetic.

---

## The interesting part is the pipeline, not the simulator

Plenty of Chladni demos exist. The part with engineering in it is getting a
**physical simulation to follow real audio** and still look like anything.

The naive version does not work, and the reason is measurable. The
instantaneously loudest mode in these recordings changes about every **0.24
seconds**. Sand needs **seconds** to migrate across the plate. Drive the field
straight from the spectrum and the grains spend the entire time in transit — the
plate never forms a figure at all, it just looks like noise.

So the spectrum is smoothed, a mode is committed to, held long enough to settle,
and cross-faded when the recording has clearly moved on. Because the analysis is
offline, the schedule does not have to be *causal*: it is smoothed **forward and
backward** so the decision is centred on the moment rather than lagging it, then
shifted early so a transition begins before the audio does.

The first tuning of that overcorrected into a different desync. Smoothing over
1.6 s and holding for 3 s removed the lag and replaced it with a **lock**: the
figure settled and then sat there while the recitation moved on. A captured frame
showed it directly — the figure panel read (1,1) while the live mode ladder's
loudest modes were (5,1) and (0,3), with "figure held 3.9s" printed beside it.
Recitation changes pitch roughly every half-second to second, so it is now
smoothed over 0.55 s, held 0.9 s, and led by 0.45 s.

```
figure schedule: zero-phase smoothing (tau 0.55s), hold 0.9s, lead 0.45s
measured: all 1,438 transitions fire 0.46s early against the same schedule
          with no lead (0.45s, snapped to whole frames at 24 fps)
```

## Things that were wrong, and how they were found

Most of the work here was measurement, not cleverness.

**The palette never appeared.** Settled sand is bimodal — cells on a nodal line
held a median of 58 grains and up to 1055; every other cell held none. The linear
density→colour gain was **16× too hot**, pinning 816 cells on pure white with
almost nothing in between. No linear ramp covers two decades. It is now
logarithmic against a reference the picture measures on itself each frame, so it
stays correct at any grain count or terminal size.

**The plate read as empty.** `compose()` counted grain positions fresh every
frame, so a sub-dot went black the instant a grain left it — a settled figure was
only ever as bright as the grains standing on it at that exact instant. A decaying
accumulation buffer (~8 frames of memory) is what makes a nodal line read as a
stroke. It also made output *cheaper*: **0.71 → 0.12 MB/s** of escape sequences,
because smoother density means far fewer cells to repaint.

**Circles came out as ellipses.** A braille sub-dot is only square if the cell is
exactly 1:2. Iosevka is condensed — measured, its cell is about 1:2.7, making a
sub-dot 1.3× taller than wide. The real pixel geometry is read from the terminal
via `TIOCGWINSZ` instead of assumed.

**A third of the sand sat outside the plate.** Beyond the rim the field
saturates, so those grains took maximum random steps while the inward drift stayed
gentle; the two balanced and the sand parked there permanently. Re-seeding
escapees across the disc — rather than clamping them to the edge, which draws a
bright ring that is not a nodal line — took grains on the nodal set from **57% to
99%**.

**The centre filled with mush.** A mode with `m` nodal diameters goes as `rᵐ`, so
its *energy* goes as `r²ᵐ`. For (5,1) that is below 2% of peak out to 0.36 of the
radius — grains random-walk in and cannot get out. The dead zone is now measured
per mode (0.00 for (0,1), 0.36 for (5,1)); one fixed radius cannot be right for
all of them.

**…and the fix for the mush dug a hole.** The dead zone was agitated uniformly,
which stops a blob forming but cannot tell a nodal diameter from the space between
two, so it swept the diameters out as well: a (2,1) cross rendered as four
separate spokes ending in a ring of piled sand. Histogramming grain radius on a
pure (2,1) field found **0 grains inside r = 0.05** and a ring at **6.76× uniform
density** exactly where the treatment ended. Two causes, and fixing the obvious one
alone changed nothing. The agitation is now shaped by `cos²(mθ)`, which is exactly
zero on every diameter at every radius. But the drift had used the full gradient
of `core(r)·cos²(mθ)`, and by the product rule that includes a **radial, outward**
term — measured at +0.68 outward *on the nodal diagonal itself*. The drift now uses
only the tangential term, computed analytically, because `np.gradient` of a polar
pattern on a Cartesian grid leaks small radial components that still emptied
high-m cores over hundreds of steps. After: density at r = 0.05–0.10 rose from
0.24× to **5.98×** uniform and the ring flattened to 3.38×. High-m figures still
keep a small clear centre — at r = 0.07 a (5,1) figure's ten spokes are ~4.6
sub-dots apart and cannot be drawn as separate lines.

**The figure did not follow the voice — because it was told not to.** Each
recording's `f0` sets the whole mode ladder, and the search that picked it weighted
`variety` — the fraction of the 20 modes used — as its largest term. So it chose
whichever `f0` spread figures most evenly, and got exactly that: occupancy was
**9–12% per mode**, near-uniform, a plate cycling through shapes on a schedule of
its own. Variety should be a consequence of a good `f0`, never the target.
Underneath it was a second defect the first one hid: the search grid ran 70–130 Hz
and the ladder spans ~5.5× its `f0`, topping out near 720 Hz — while the recited
pitch has a median of 410 Hz and a 90th percentile of **1,131 Hz**. No candidate
could have answered most of the voice, and the winner still looked like a winner.
The objective now scores **coverage** (does the ladder span this voice) and
**tracking** (does the figure climb when the pitch climbs), over a grid reaching
350 Hz. Now: occupancy concentrates (27%, 16%, 12%, 11%, 5% for the top five), and
tracking — the correlation between recited pitch and the frequency of the figure
shown — has a per-track median of **0.50** (0.24–0.75), **0.375** across the whole
library. There is a ceiling on that number worth naming: pitch here is the FFT's
spectral peak, often a harmonic rather than the fundamental.

> A correction to this repo's own history: commit `76eafcc` cites a correlation of
> 0.258 as the "before" figure. That number was computed with the first track's
> `f0` applied to every track, which mislabels mode frequencies wherever `f0`
> differs; the same method understates the current library as 0.293 against a
> correct 0.375. The old timeline was overwritten, so no valid "before"
> correlation exists — the occupancy and grid-ceiling evidence above does not
> depend on it.

**The loudness channel was a flat line.** Frame-by-frame loudness is the only thing
that answers the voice instantly, and it was computed against a hardcoded
reference that these recordings saturated: median 1.599, 90th percentile 1.600,
maximum 1.600 — over half of every track pinned at the clip ceiling, so the plate
was struck with identical force through loud passages and silence. Referenced to
each track's own 90th-percentile loudness, no frame sits at the ceiling and the
swing in agitation went from 1.55× to 2.18×.

**A faster frame rate made a different picture, not a smoother one.** Raising the
frame rate ran the physics more times per second of audio, so at 120 fps sand
travelled five times as far: "settled" fell from 99% to 69% and figures held for
0.2 s. Diffusion now scales with `√dt` and drift with `dt`, and `dt` is read from
the clock rather than the requested rate, because asked for 120 the renderer
sustains about 67–89 — a timestep derived from the flag would have run the physics
at 55% speed. The default rate is the refresh rate of the monitor it opens on.

**The live screensaver drew dots while `--dump` drew lines** — three defects,
found only after three plausible explanations were each ruled out by measurement.
Not convergence: a static figure puts 81–85% of its grains on its nodal lines
within 1.8 s at every frame rate. Not the schedule: replaying real excerpts
headlessly, sand sits on the *current* figure's lines 74% of the time. The physics
was right; the drawing was wrong.
*First*, the crossfade was counted in frames at the target rate — 240 on a 240 Hz
monitor — while the renderer ran at 115, stretching a one-second fade to two.
Two blended modes vanish together only at points, and with figures changing every
1.08 s the plate was mid-blend **75%** of the time. It is now timed by the clock
and lasts 0.35 s: 13%. *Second*, the background wash froze on the previous figure:
its docstring said it was redrawn during a crossfade, but its call site fired only
on the first frame of one, so each new figure's lines ran across the old figure's
bright regions (mean wash level under on-line sand 0.47, against 0.03 from the
current field). *Third*, ghost dots: a sub-dot was lit when its decaying history
was above zero, and a float multiplied by 0.88 takes hundreds of frames to reach
zero, so **20.7%** of the canvas stayed lit with trails of sand that had long since
left. `--dump` never showed any of this because it draws once, from a fresh
buffer. Lit area over real playback is now 6.1%.

**Figure lines read cold, and the obvious fix made them colder.** A cell on a
nodal line averaged level 5.6 of 13 — exactly where the ramp turns from teal to
gold — so live figures looked washed out beside the settled stills. The suspect
was the brightness reference: it is the 99th percentile of every cell holding any
density, so the caked rim sets the exposure for the whole plate. Measured over 15 s
of real playback, all three ways of narrowing that reference made lines *dimmer*
(6.4 → 5.8 → 5.5), because the faint cells being excluded were the ones holding
the percentile down. The curve was the lever, not the reference: widening the
log's span and easing it lifts line cells to 7.2 and doubles the share of lit
cells in the warm half, while pure white stays at 2.6% — a lower percentile
reached 8.9 but put 11% of lit cells on white, and white is the only level that
says *crossing* rather than *bright*.

**Every visual check was of the wrong program.** The screenshots were real,
correctly captured, and of a stale window: `setsid foot &` returns a PID that is
not the window's process, so killing it never closed the previous instance, the
new code crashed on launch (a variable used four lines before it was assigned),
and the old window stayed up to be photographed. The crash shipped in two commits
reported as visually verified. `tools/render-dump.py` replaces that method: it runs
`--dump` in a pseudo-terminal of a chosen size and draws the output straight to a
PNG — no terminal, window or compositor, so the same arguments give the same image
and there is no stale process to photograph. It refuses to write a frame with no
plate in it, which is how it found the crash on its second run; its *first* run had
rendered an argparse error into a PNG and reported success. Every image in this
README is now produced by it, with `--anonymise`.

## The physics

A circular membrane clamped at its rim has eigenmodes

```
U_mn(r, θ) = J_m(α_mn · r) · cos(m·θ)
```

`α_mn` is the n-th positive zero of the Bessel function `J_m`. **m** gives nodal
*diameters* (so `2m` spokes), **n** gives `n−1` interior nodal *circles*.
Frequencies scale as `f_mn = f₀ · α_mn / α_01`.

Real audio excites several modes at once at **incommensurate** frequencies, so
the time-averaged agitation is the *incoherent* sum

```
E(x,y) = Σ aᵢ² · Uᵢ(x,y)²
```

not `(Σ aᵢUᵢ)²` — squaring a coherent sum invents interference nodes no real plate
has. This has a visible consequence: sand rests only where **every** active mode
is quiet, and the common zeros of several independent functions are *points*, not
curves. One clean mode traces lines; a chord beads into dots.

Each grain random-walks with a step proportional to `√E` and drifts down `∇E`, so
it stalls where `E ≈ 0` — the nodal set. The drift is clamped by **magnitude, not
per component**: clamping each axis separately lets a diagonal step run √2 longer
than a cardinal one, which parks grains in four bright blobs at the compass
points.

**Honest about the model:** a real metal Chladni plate has a *free* edge and obeys
the biharmonic plate equation; its modes are not Bessel functions. This is the
clamped-membrane model. And nothing about a particular recording produces these
shapes — a sustained tone at the same pitch gives the same figure. What sustained
recitation provides is a pitch stable long enough for sand to reach equilibrium.

![a different surah](docs/al-ikhlaas.png)

![the figure at width](docs/figure.png)

## Library

```
chladni-add                                  # search, pick reciters, add
chladni-add add --surah 36 --reciter Husary_128kbps
chladni-add add --url <any link yt-dlp can reach>
```

114 surahs, 79 reciters. Search folds transliteration variants, so `rahman` finds
*Ar-Rahmaan*, `fatiha` finds *Al-Faatiha*, `the cow` finds *Al-Baqara*, `36` finds
*Yaseen* — doubled vowels and the assimilated definite article are normalised away.

Play order is not insertion order. Several recordings of the same surah back to
back is the same words several times, so tracks are grouped and the order always
takes from the largest group that is *not* the one just played — which avoids
adjacent repeats whenever it is possible at all (iff no surah holds more than
⌈n/2⌉ of the tracks). When it is not possible, it says so rather than pretending.

**No audio or scripture text is in this repository.** Recitations come from
[everyayah.com](https://everyayah.com); surah metadata and ayah text from
[alquran.cloud](https://alquran.cloud). Those are third-party works under their
own terms — translations in particular are often under active copyright. This
ships the client; you fetch the content.

## Install

```
git clone https://github.com/blurxy/chladni-tui && cd chladni-tui
./install.sh
chladni-add            # add something to play
chladni-screensaver    # run it
```

Python 3.11+, numpy, scipy; `ffmpeg`, `curl`, `fzf`; optionally `yt-dlp`, `gum`.
A Nerd Font with braille coverage — built against Iosevka.

Resizes live from a 60×16 terminal to fullscreen. Any key exits.
`--silent` for no audio, `--anonymise` to redact machine identifiers in screenshots.

`tools/render-dump.py` runs the renderer inside a pseudo-terminal of a size you
choose and draws its output to a PNG — same font, same cell metrics, no window
and no compositor, so two frames meant to be compared come back the same size.
It refuses to write a frame holding fewer than 200 braille cells, because its
first run rendered an argparse error to a clean-looking PNG and reported
success. Every screenshot here was taken with it.

`tools/layout-check.py` draws every verse in the library across seven terminal
sizes and asserts the layout does not move: the ayah header lands on the same
row whatever is playing, the plate never reaches it, nothing writes into the
bottom strip.

## License

MIT. See [LICENSE](LICENSE).
