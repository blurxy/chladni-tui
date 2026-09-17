#!/usr/bin/env python3
"""The chladni-tui library: what the screensaver plays, and in what order.

Audio comes in three ways -- a surah by a named reciter from everyayah.com, any
URL yt-dlp can reach, or a local file. Whatever the source, the same thing is
stored: the audio itself, and a per-frame record of which membrane modes it
drives. The screensaver never analyses anything at runtime.

Ordering is not the order things were added. Al-Fatiha by six reciters back to
back is the same words six times, which is a strange thing to sit through, so
the play order deliberately separates repeats of the same surah.
"""
import os, sys, json, re, time, argparse, subprocess, shutil, tempfile, unicodedata
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "audio")
META = os.path.join(HERE, "meta")
LIB = os.path.join(HERE, "library.json")
TIMELINE = os.path.join(HERE, "timeline.npz")
EVERYAYAH = "https://everyayah.com/data"
QURAN_API = "https://api.alquran.cloud/v1"
UA = {"User-Agent": "cymatics-screensaver/1.0"}

SR = 48000
FFT_N = 8192
FPS = 24.0
NM = 20


# ----------------------------------------------------------------- storage ---
def load_library():
    if not os.path.exists(LIB):
        return {"tracks": []}
    try:
        with open(LIB, encoding="utf-8") as fh:
            d = json.load(fh)
        d.setdefault("tracks", [])
        return d
    except (OSError, ValueError):
        return {"tracks": []}


def save_library(d):
    os.makedirs(HERE, exist_ok=True)
    tmp = LIB + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    os.replace(tmp, LIB)


def ensure_meta():
    """Fetch the surah and reciter catalogues if they are not here yet.

    These are deliberately NOT in the repository -- they are third-party data,
    and this project ships the client rather than the content. But that means a
    fresh clone has no meta/ at all, and the first command a new user runs is a
    search. Fetch on demand instead of crashing with a traceback about a missing
    file the README never mentions.
    """
    os.makedirs(META, exist_ok=True)
    need = [f for f in ("surahs.json", "reciters.json", "ayah_counts.json")
            if not os.path.exists(os.path.join(META, f))]
    if not need:
        return
    sys.stderr.write("fetching surah and reciter catalogues (first run)...\n")
    try:
        data = json.loads(_get("%s/surah" % QURAN_API))["data"]
        surahs = [{"no": x["number"], "name": x["englishName"],
                   "meaning": x["englishNameTranslation"],
                   "revelation": x["revelationType"], "ayat": x["numberOfAyahs"]}
                  for x in data]
        rec_raw = json.loads(_get("%s/recitations.js" % EVERYAYAH))
        recs = sorted(({"id": v["subfolder"], "name": v["name"], "bitrate": v["bitrate"]}
                       for k, v in rec_raw.items() if k != "ayahCount"),
                      key=lambda r: (r["name"], r["bitrate"]))
        with open(os.path.join(META, "surahs.json"), "w", encoding="utf-8") as fh:
            json.dump(surahs, fh, indent=0)
        with open(os.path.join(META, "reciters.json"), "w", encoding="utf-8") as fh:
            json.dump(recs, fh, indent=0)
        with open(os.path.join(META, "ayah_counts.json"), "w", encoding="utf-8") as fh:
            json.dump(rec_raw["ayahCount"], fh)
        sys.stderr.write("  %d surahs, %d reciters\n" % (len(surahs), len(recs)))
    except (urllib.error.URLError, ValueError, KeyError, OSError) as e:
        raise SystemExit(
            "could not fetch the catalogues (%s).\n"
            "They come from alquran.cloud and everyayah.com -- check your network." % e)


def _json(path):
    with open(os.path.join(META, path), encoding="utf-8") as fh:
        return json.load(fh)


def surahs():
    return _json("surahs.json")


def reciters():
    """Reciters, with display names tidied.

    5 of the 79 upstream names carry the filename's underscores straight into
    the display string -- "Yasser_Ad-Dussary", "Nasser_Alqatami". Cleaning here
    rather than at each call site means the search bar, the library and the
    on-screen credit all agree; a name fixed in only one of those is how the
    same reciter ends up looking like two.
    """
    out = []
    for r in _json("reciters.json"):
        r = dict(r)
        r["name"] = re.sub(r"\s+", " ", r["name"].replace("_", " ")).strip()
        out.append(r)
    return out


def ayah_counts():
    return _json("ayah_counts.json")


# ------------------------------------------------------------------ search ---
def _norm(s):
    """Fold transliteration variants together.

    These names are romanised inconsistently -- Ar-Rahmaan / Ar-Rahman,
    Al-Faatiha / Al-Fatiha, Yaseen / Yasin. Collapsing runs of the same letter
    makes doubled vowels stop mattering, which is what breaks a plain substring
    search: "rahman" is not a substring of "arrahmaan", but both fold to the
    same thing.
    """
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "", s.lower())
    return re.sub(r"(.)\1+", r"\1", s)


# Arabic definite article, assimilated to the following letter. Someone typing
# "baqara" should find "Al-Baqara" without knowing which form it takes.
_ARTICLES = ("al", "ar", "an", "as", "at", "ad", "az", "ash", "a")


def _variants(name):
    n = _norm(name)
    out = {n}
    for pre in _ARTICLES:
        if n.startswith(pre) and len(n) > len(pre) + 2:
            out.add(n[len(pre):])
    return out


def search_surahs(q, limit=200):
    """Match on number, transliterated name, or English meaning."""
    if not q:
        return surahs()[:limit]
    qn = _norm(q)
    out = []
    for s in surahs():
        score = None
        if q.strip().isdigit() and int(q.strip()) == s["no"]:
            score = 0
        else:
            vs = _variants(s["name"])
            if qn in vs:
                score = 1
            elif any(v.startswith(qn) for v in vs):
                score = 2
            elif qn and any(qn in v for v in vs):
                score = 3
            elif qn and qn in _norm(s["meaning"]):
                score = 4
        if score is not None:
            out.append((score, s["no"], s))
    out.sort(key=lambda t: (t[0], t[1]))
    return [s for _, _, s in out[:limit]]


def search_reciters(q, limit=100):
    if not q:
        return reciters()[:limit]
    qn = _norm(q)
    out = [r for r in reciters() if qn in _norm(r["name"]) or qn in _norm(r["id"])]
    return out[:limit]


# ---------------------------------------------------------------- download ---
def _get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_ayah_text(surah_no):
    """Transliteration + English for every ayah, so the screensaver can show it."""
    out = []
    try:
        tr = json.loads(_get("%s/surah/%d/en.transliteration" % (QURAN_API, surah_no)))
        en = json.loads(_get("%s/surah/%d/en.asad" % (QURAN_API, surah_no)))
        ta = tr["data"]["ayahs"]; ea = en["data"]["ayahs"]
        for i in range(min(len(ta), len(ea))):
            out.append([" ".join(ta[i]["text"].split()), " ".join(ea[i]["text"].split())])
    except (urllib.error.URLError, ValueError, KeyError, OSError):
        pass
    return out


def download_surah(surah_no, reciter_id, progress=True):
    """Pull every ayah of a surah and concatenate them into one ogg.

    Returns (path, ayah_start_times). Per-ayah files are what everyayah serves,
    and keeping their durations gives exact ayah boundaries for free -- far
    better than guessing where one ends inside a single long file.
    """
    n = ayah_counts()[surah_no - 1]
    os.makedirs(AUDIO, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="cym-")
    try:
        def one(i):
            url = "%s/%s/%03d%03d.mp3" % (EVERYAYAH, reciter_id, surah_no, i)
            dst = os.path.join(tmp, "%03d.mp3" % i)
            for attempt in range(3):
                try:
                    data = _get(url, timeout=40)
                    if len(data) < 500:
                        raise ValueError("short file")
                    with open(dst, "wb") as fh:
                        fh.write(data)
                    return i, None
                except (urllib.error.URLError, ValueError, OSError) as e:
                    if attempt == 2:
                        return i, str(e)
                    time.sleep(0.6 * (attempt + 1))
            return i, "failed"

        errs = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            for k, (i, err) in enumerate(ex.map(one, range(1, n + 1)), 1):
                if err:
                    errs.append((i, err))
                if progress and (k % 10 == 0 or k == n):
                    sys.stderr.write("\r  downloading %d/%d ayat" % (k, n)); sys.stderr.flush()
        if progress:
            sys.stderr.write("\n")
        if errs:
            raise RuntimeError("could not fetch %d ayat (first: ayah %d, %s)"
                               % (len(errs), errs[0][0], errs[0][1]))

        starts, t = [], 0.0
        listing = os.path.join(tmp, "list.txt")
        with open(listing, "w", encoding="utf-8") as fh:
            for i in range(1, n + 1):
                f = os.path.join(tmp, "%03d.mp3" % i)
                starts.append(round(t, 3))
                t += _duration(f)
                fh.write("file '%s'\n" % f)
        out = os.path.join(AUDIO, "%03d-%s.ogg" % (surah_no, reciter_id))
        _run(["ffmpeg", "-nostdin", "-v", "error", "-f", "concat", "-safe", "0",
              "-i", listing, "-c:a", "libvorbis", "-q:a", "4", "-ar", "48000",
              "-ac", "1", out, "-y"])
        return out, starts
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def download_url(url, title=None):
    """Any link yt-dlp understands, plus plain audio URLs and local files."""
    os.makedirs(AUDIO, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="cym-url-")
    try:
        if os.path.exists(url):
            src = url
            name = title or os.path.splitext(os.path.basename(url))[0]
        else:
            stem = os.path.join(tmp, "dl")
            _run(["yt-dlp", "--no-playlist", "-f", "bestaudio/best",
                  "-x", "--audio-format", "vorbis", "-o", stem + ".%(ext)s",
                  "--no-progress", "--quiet", url])
            got = [f for f in os.listdir(tmp) if f.startswith("dl.")]
            if not got:
                raise RuntimeError("yt-dlp produced nothing for %s" % url)
            src = os.path.join(tmp, got[0])
            if not title:
                try:
                    title = _run(["yt-dlp", "--no-playlist", "--get-title",
                                  "--quiet", url]).strip()
                except (OSError, RuntimeError):
                    title = None
            name = title or "audio"
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name)[:48].strip("-") or "audio"
        out = os.path.join(AUDIO, "url-%s.ogg" % slug)
        k = 1
        while os.path.exists(out):
            k += 1
            out = os.path.join(AUDIO, "url-%s-%d.ogg" % (slug, k))
        _run(["ffmpeg", "-nostdin", "-v", "error", "-i", src, "-c:a", "libvorbis",
              "-q:a", "4", "-ar", "48000", "-ac", "1", out, "-y"])
        return out, name
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("%s failed: %s" % (cmd[0], (p.stderr or p.stdout)[-400:]))
    return p.stdout


def _duration(path):
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", path]).strip()
    return float(out) if out else 0.0


# ---------------------------------------------------------------- analysis ---
def _np():
    import numpy as np
    return np


def mode_table(n_modes=NM):
    np = _np()
    from scipy.special import jn_zeros
    modes = sorted(((m, n, float(a)) for m in range(8)
                    for n, a in enumerate(jn_zeros(m, 8), 1)), key=lambda t: t[2])[:n_modes]
    alpha = np.array([a for _, _, a in modes], dtype=np.float32)
    return modes, alpha, alpha / modes[0][2]


def load_mono(path):
    np = _np()
    from scipy.io import wavfile
    wav = os.path.splitext(path)[0] + ".__tmp.wav"
    _run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "wav",
          "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1", wav, "-y"])
    try:
        _sr, x = wavfile.read(wav)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    finally:
        if os.path.exists(wav):
            os.remove(wav)


def analyse(x, f0, ratio, gamma=4.5, sigma=0.055):
    # sigma WIDENED and gamma RAISED together, because they trade against each
    # other and changing either alone makes the other worse.
    #
    # sigma=0.030 is +/-3% in log-frequency -- THINNER THAN THE GAP BETWEEN
    # ADJACENT BESSEL MODES, so a harmonic could fall between two bins and be
    # heard by neither. That is a selector reading a spectrum through slits: what
    # it reports depends on whether a partial happens to land on a bin centre,
    # which is close to arbitrary and is why the figure did not follow the voice.
    # 0.055 lets adjacent modes overlap and cross-fade, which is what the
    # docstring below always claimed happened.
    #
    # Widening alone would cost crispness -- more modes lit at once. Measured
    # before this change, purity was 0.31, i.e. the winning mode held under a
    # third of the energy and the plate drew a blur of three figures rather than
    # one. gamma 3.0 -> 4.5 sharpens the contrast between the leader and its
    # neighbours after the cross-fade, restoring a single readable figure.
    """Per-frame mode amplitudes, loudness and dominant pitch.

    A Gaussian in LOG frequency around each mode's centre, so a voice sliding in
    pitch cross-fades modes instead of snapping between them.
    """
    np = _np()
    win = np.hanning(FFT_N).astype(np.float32)
    freqs = np.fft.rfftfreq(FFT_N, 1.0 / SR).astype(np.float32)
    lo = int(np.searchsorted(freqs, 55.0)); hi = int(np.searchsorted(freqs, 3000.0))
    fb = freqs[lo:hi]
    lf = np.log(np.maximum(fb, 1e-3))[None, :]
    M = np.exp(-0.5 * ((lf - np.log(f0 * ratio)[:, None]) / sigma) ** 2).astype(np.float32)
    M /= M.sum(axis=1, keepdims=True) + 1e-9
    freqs_all = np.fft.rfftfreq(FFT_N, 1.0 / SR).astype(np.float32)
    fb_pitch = freqs_all
    plo = int(np.searchsorted(freqs_all, 70.0))     # a reciting voice lives here
    phi = min(int(np.searchsorted(freqs_all, 500.0)), len(freqs_all) // 4)
    hop = int(SR / FPS)
    nfr = max(0, (len(x) - FFT_N) // hop)
    amps = np.zeros((nfr, len(ratio)), dtype=np.float32)
    lvl = np.zeros(nfr, dtype=np.float32); hz = np.zeros(nfr, dtype=np.float32)
    cur = np.zeros(len(ratio), dtype=np.float32); lev = 0.0
    dt = 1.0 / FPS
    rise = 1.0 - np.exp(-dt / 0.06); fall = 1.0 - np.exp(-dt / 0.45)
    k = 1.0 - np.exp(-dt / 0.10)
    # PER-TRACK LOUDNESS REFERENCE, measured before the loop rather than assumed.
    #
    # The fixed 0.035 divisor saturated. Measured across the built timeline:
    # median lvl 1.599, p90 1.600, max 1.600 -- over half of all frames pinned at
    # the clip ceiling, so the "loudness" channel carried no information and the
    # plate was struck at constant force no matter what the voice did. That is
    # the desync: not the figure being late, but the only channel that answers
    # the voice instantly being a flat line.
    #
    # A recitation normalised to -14 LUFS and one mastered 12 dB quieter cannot
    # share a constant. Reference each track to its own 90th-percentile frame
    # RMS, so the loud passages of THIS recording land near the top of the range
    # and the quiet ones actually read as quiet.
    _rms = np.sqrt(np.maximum([np.mean(x[i*hop: i*hop+FFT_N]**2) for i in range(nfr)], 1e-12))
    _ref = float(np.percentile(_rms, 90)) or 0.035
    _ref = max(_ref, 1e-4)

    for i in range(nfr):
        seg = x[i * hop: i * hop + FFT_N]
        rms = float(np.sqrt(np.mean(seg * seg)))
        lev += (float(np.clip((rms / _ref) ** 0.5, 0.0, 1.6)) - lev) * k
        spec = np.abs(np.fft.rfft(seg * win)).astype(np.float32)
        power = (spec[lo:hi] ** 2)
        tot = power.sum()
        if tot > 1e-12:
            # PITCH, NOT THE LOUDEST BIN. The spectral peak of a reciting voice is
            # usually a harmonic: measured over 900 frames, its median was 398Hz
            # against a true pitch of 146Hz -- a ratio of 2.04, the second
            # harmonic -- and it jumped between frames 12x more than the pitch did.
            # Steering the plate by it meant the figure answered an overtone.
            # Harmonic product spectrum: multiply decimated copies so the harmonics
            # of the true fundamental land on top of each other.
            h = spec[:len(spec) // 4].copy()
            for d in (2, 3, 4):
                h *= spec[::d][:len(h)]
            hz[i] = fb_pitch[int(np.argmax(h[plo:phi])) + plo] if phi > plo else 0.0
            power = power / tot
        target = M @ power
        pk = target.max()
        if pk > 1e-9:
            target = (target / pk) ** gamma
        d = target - cur
        cur += d * np.where(d > 0, rise, fall).astype(np.float32)
        amps[i] = cur; lvl[i] = lev
    return amps, lvl, hz


def _score(amps, lvl, hz=None, f0=None, ratio=None):
    """How well does this f0 make the PLATE ANSWER THE VOICE?

    THE PREVIOUS OBJECTIVE OPTIMISED FOR THE WRONG THING, and it is worth being
    precise about how: it weighted `variety` -- the fraction of the 20 modes that
    got used -- at 1.2, the largest term. So the search preferred an f0 that
    spread the figures evenly over the mode ladder, and that is exactly what it
    produced. Measured on the built timeline: figure occupancy was 9-12% for each
    of modes 2..7, essentially uniform, and the correlation between the recited
    pitch and the frequency of the figure on screen was 0.26. The plate was
    cycling through shapes on a schedule of its own, next to audio it was not
    listening to. A viewer reads that instantly as "not in sync", which is the
    correct reading -- it was not.

    Variety is a CONSEQUENCE of a good f0, never a target. A recitation that
    stays in one register should hold one figure; forcing twenty is the bug.

    So score correspondence instead:
      coverage -- what fraction of the voice's pitch actually lands inside the
                  mode ladder's frequency range at all. An f0 whose modes top out
                  at 443Hz cannot answer a voice whose p90 is 1131Hz, and the old
                  objective had no term that noticed.
      tracking -- correlation between log(voice pitch) and log(frequency of the
                  mode being selected). This is the thing a viewer perceives as
                  sync: pitch goes up, figure climbs the ladder.
      purity   -- one mode clearly dominant, so the figure is a figure and not a
                  blur of three.
    """
    np = _np()
    live = lvl > 0.25
    if live.sum() < 30:
        return -1.0, {}
    a = amps[live]
    purity = float(np.mean(a.max(axis=1) / (a.sum(axis=1) + 1e-9)))
    activity = float(np.mean(lvl[live]))
    dom = a.argmax(axis=1)
    det = {"purity": round(purity, 4), "activity": round(activity, 4),
           "distinct_modes": int(len(np.unique(dom)))}

    coverage = tracking = 0.0
    if hz is not None and f0 is not None and ratio is not None:
        mode_hz = np.asarray(f0) * np.asarray(ratio)
        h = np.asarray(hz)[live]
        ok = h > 0
        if ok.sum() >= 30:
            lo, hi = float(mode_hz.min()), float(mode_hz.max())
            coverage = float(((h[ok] >= lo) & (h[ok] <= hi)).mean())
            sel = mode_hz[dom[ok]]
            lv_, ls_ = np.log(np.maximum(h[ok], 1.0)), np.log(np.maximum(sel, 1.0))
            if lv_.std() > 1e-6 and ls_.std() > 1e-6:
                tracking = float(np.corrcoef(lv_, ls_)[0, 1])
            det["coverage"] = round(coverage, 4)
            det["tracking"] = round(tracking, 4)

    return coverage * 1.3 + max(tracking, 0.0) * 1.0 + purity * 0.5 + activity * 0.2, det


def analyse_track(track, progress=True):
    """Pick the f0 that makes this recording drive the widest range of figures."""
    np = _np()
    _modes, _alpha, ratio = mode_table()
    x = load_mono(track["audio"])
    best = None
    # THE GRID COULD NOT REACH THE VOICE. Mode frequencies run f0*alpha/alpha0,
    # so the 20-mode ladder spans about 5.5x its f0 -- meaning the old grid's
    # best case topped out near 720Hz. Measured against a real recitation, the
    # dominant pitch has median 410Hz and p90 1131Hz, so most of the voice sat
    # ABOVE the highest mode and no f0 in the grid could have answered it. The
    # search was choosing the least-bad option from a set that excluded every
    # good one, which is invisible in the score: the winner still looks like a
    # winner. Extended upward so a ladder that actually spans the voice exists
    # to be chosen.
    for f0 in (70.0, 95.0, 130.0, 170.0, 220.0, 280.0, 350.0):
        amps, lvl, hz = analyse(x, f0, ratio)
        sc, det = _score(amps, lvl, hz, f0, ratio)
        if best is None or sc > best[0]:
            best = (sc, f0, amps, lvl, hz, det)
    sc, f0, amps, lvl, hz, det = best
    fig = schedule_figures(amps, hz)
    npz = os.path.join(AUDIO, track["id"] + ".npz")
    np.savez_compressed(npz, amps=amps.astype(np.float16),
                        lvl=lvl.astype(np.float16), hz=hz.astype(np.float16),
                        fig=fig.astype(np.int16))
    track["analysis"] = os.path.basename(npz)
    track["f0"] = f0
    track["frames"] = int(len(amps))
    track["fps"] = FPS
    track["score"] = round(sc, 4)
    track.update(det)
    if progress:
        print("  analysed %-40s f0=%5.1f  %d frames (%.0fs)  %d distinct modes"
              % (track["title"][:40], f0, len(amps), len(amps) / FPS,
                 det.get("distinct_modes", 0)))
    return track


def _schedule_from_pitch(np, hz, nmodes, fps, tau, hold, lead):
    """Figure follows the recited PITCH: higher note, higher mode.

    Selecting by per-mode energy answered whichever overtone happened to be loud,
    which is why the plate looked unrelated to the voice. Pitch is the thing a
    listener hears change, so it is the thing the figure should answer.

    Mapped by RANK within the track, not by frequency: a reciting voice moves over
    roughly 129-205Hz while the mode ladder spans 5.5x its f0, so a literal
    frequency match would sit on the lowest two modes forever. Rank keeps the
    relationship monotonic -- the figure climbs exactly when the voice does -- and
    uses the whole ladder. The plate is driven by a frequency derived from the
    voice, not equal to it; the README says so.
    """
    voiced = hz > 0
    if voiced.sum() < max(60, 0.05 * len(hz)):
        return None                       # too little pitch to steer with
    lp = np.log(np.maximum(hz, 1.0)).astype(np.float32)
    idx = np.arange(len(lp))
    lp = np.interp(idx, idx[voiced], lp[voiced]).astype(np.float32)   # bridge unvoiced gaps

    k = 1.0 - np.exp(-(1.0 / fps) / max(tau, 1e-3))                   # zero-phase smoothing
    for pas in (1, -1):
        acc = lp[0] if pas == 1 else lp[-1]
        rng = range(len(lp)) if pas == 1 else range(len(lp) - 1, -1, -1)
        for i in rng:
            acc += (lp[i] - acc) * k
            lp[i] = acc

    lo, hi = np.percentile(lp[voiced], 3.0), np.percentile(lp[voiced], 97.0)
    rank = np.clip((lp - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    want = np.clip((rank * (nmodes - 1) + 0.5).astype(np.int32), 0, nmodes - 1)

    out = np.empty(len(want), dtype=np.int32)                         # minimum dwell
    cur, since, dwell = int(want[0]), 0, max(1, int(hold * fps))
    for i, w in enumerate(want):
        if w != cur and since >= dwell:
            cur, since = int(w), 0
        else:
            since += 1
        out[i] = cur

    shift = int(round(lead * fps))                                    # start before the audio
    if shift > 0:
        out = np.concatenate([out[shift:], np.full(shift, out[-1], dtype=np.int32)])
    return out


def schedule_figures(amps, hz=None, select=6.0, fps=FPS, tau=0.55, hold=0.9,
                     ratio=1.12, lead=0.45):
    """Decide offline which figure is on the plate at every frame.

    Doing this live meant the picture trailed the sound. The selector integrated
    the spectrum over a couple of seconds, held, then cross-faded -- so the
    figure on screen answered audio from several seconds ago, and the sand then
    took seconds more to settle into it. That reads as out of sync because it is.

    Offline there is no reason to be causal. Smooth the spectrum FORWARD AND
    BACKWARD so the decision is centred on the moment rather than lagging it,
    then shift the whole schedule EARLIER by `lead` seconds so the transition
    starts before the audio does and the sand has finished moving by the time
    you hear why.

    THE FIRST VERSION OF THIS OVERCORRECTED INTO A DIFFERENT DESYNC. tau=1.6 and
    hold=3.0 removed the lag and replaced it with a lock: the figure settled and
    then sat there for seconds while the recitation moved on. Caught in a real
    captured frame -- the FIGURE panel read (1,1) while the live mode ladder had
    (5,1) and (0,3) as the loudest modes, with "figure held 3.9s" printed beside
    it. Nothing was trailing; the picture had simply stopped answering.

    A cymatics reel changes pattern when the PITCH changes, which in recitation
    is roughly every half-second to second, not every three. So: smooth over
    ~0.5s instead of 1.6, hold ~0.9s instead of 3.0, and switch on a 12% margin
    instead of 25%. `lead` drops with them -- a large pre-shift made sense when a
    transition took seconds to complete and overshoots once it does not.
    """
    np = _np()
    if hz is not None:
        sched = _schedule_from_pitch(np, np.asarray(hz), amps.shape[1], fps, tau, hold, lead)
        if sched is not None:
            return sched
    a = np.power(np.clip(amps, 0.0, None), select, dtype=np.float32)
    a2 = (a * a).astype(np.float32)
    tot = a2.sum(axis=1, keepdims=True)
    tot[tot <= 0] = 1.0
    a2 /= tot

    k = 1.0 - np.exp(-(1.0 / fps) / tau)
    fwd = np.empty_like(a2); acc = a2[0].copy()
    for i in range(len(a2)):
        acc += (a2[i] - acc) * k
        fwd[i] = acc
    bwd = np.empty_like(a2); acc = fwd[-1].copy()
    for i in range(len(a2) - 1, -1, -1):
        acc += (fwd[i] - acc) * k
        bwd[i] = acc                      # zero-phase: no net delay

    nfr = len(a2)
    fig = np.zeros(nfr, dtype=np.int16)
    cur = int(np.argmax(bwd[0])); since = 1e9
    hold_fr = hold * fps
    for i in range(nfr):
        since += 1.0
        cand = int(np.argmax(bwd[i]))
        if cand != cur and since >= hold_fr and bwd[i][cand] > bwd[i][cur] * ratio:
            cur = cand; since = 0.0
        fig[i] = cur
    if lead > 0:
        shift = int(round(lead * fps))
        if 0 < shift < nfr:
            fig = np.concatenate([fig[shift:], np.full(shift, fig[-1], dtype=np.int16)])
    return fig


# -------------------------------------------------------------- sequencing ---
def order_tracks(tracks):
    """Play order: never the same surah twice in a row.

    Six recordings of Al-Fatiha in a row is the same words six times. Group by
    surah, then always take from the largest group that is not the one just
    played -- the standard rearrangement that guarantees no adjacent repeat
    whenever one is possible at all (it is possible iff no group holds more than
    ceil(n/2) of the tracks). Within a group, prefer a reciter who has not just
    been heard, so two different recordings of the same voice also separate.
    """
    import heapq
    from collections import defaultdict
    if len(tracks) < 2:
        return list(tracks), []
    groups = defaultdict(list)
    for t in tracks:
        groups[t.get("surah_no") or ("url:" + t["id"])].append(dict(t))
    heap = [(-len(v), str(k)) for k, v in groups.items()]
    heapq.heapify(heap)
    keymap = {str(k): k for k in groups}
    out, warn = [], []
    prev_key = None
    last_reciter = None
    biggest = max(len(v) for v in groups.values())
    if biggest > (len(tracks) + 1) // 2:
        warn.append("%d of %d tracks are the same surah, so some repeats are "
                    "unavoidable" % (biggest, len(tracks)))
    while heap:
        first = heapq.heappop(heap)
        pick = first
        stash = None
        if prev_key is not None and first[1] == prev_key and heap:
            stash = first
            pick = heapq.heappop(heap)
        cnt, skey = pick
        grp = groups[keymap[skey]]
        idx = 0
        for i, t in enumerate(grp):
            if t.get("reciter_id") != last_reciter:
                idx = i
                break
        chosen = grp.pop(idx)
        out.append(chosen)
        prev_key = skey
        last_reciter = chosen.get("reciter_id")
        if grp:
            heapq.heappush(heap, (-len(grp), skey))
        if stash is not None:
            heapq.heappush(heap, stash)
    return out, warn


# ------------------------------------------------------------------- build ---
def build(verbose=True):
    """Analyse anything new, order it, and write the timeline the screensaver reads."""
    np = _np()
    lib = load_library()
    tracks = lib["tracks"]
    if not tracks:
        raise RuntimeError("library is empty -- add something first")
    for t in tracks:
        if not t.get("analysis") or not os.path.exists(os.path.join(AUDIO, t["analysis"])):
            analyse_track(t, progress=verbose)
    save_library(lib)

    ordered, warn = order_tracks(tracks)
    for w in warn:
        print("  note: " + w)

    modes, alpha, _ratio = mode_table()
    A, L, H, F, offs, meta = [], [], [], [], [0], []
    for t in ordered:
        z = np.load(os.path.join(AUDIO, t["analysis"]))
        A.append(z["amps"].astype(np.float16))
        L.append(z["lvl"].astype(np.float16))
        H.append(z["hz"].astype(np.float16))
        F.append(z["fig"].astype(np.int16) if "fig" in z
                 else schedule_figures(z["amps"].astype(np.float32),
                                       z["hz"].astype(np.float32)))
        offs.append(offs[-1] + len(z["amps"]))
        meta.append({
            "key": os.path.splitext(os.path.basename(t["audio"]))[0],
            "name": t.get("reciter") or t.get("title", "audio"),
            "sub": t.get("subtitle", ""),
            "f0": t.get("f0", 110.0),
            "frames": int(len(z["amps"])), "fps": t.get("fps", FPS),
            "ayat": t.get("ayat", []), "text": t.get("text", []),
            "surah": t.get("surah", "—"), "surah_no": t.get("surah_no", 0),
            "revelation": t.get("revelation", ""), "n_ayat": t.get("n_ayat", 0),
        })
    np.savez_compressed(
        TIMELINE,
        amps=np.concatenate(A, axis=0), lvl=np.concatenate(L),
        hz=np.concatenate(H), fig=np.concatenate(F),
        offsets=np.array(offs, dtype=np.int32),
        alpha=alpha.astype(np.float32),
        mn=np.array([[m, n] for m, n, _ in modes], dtype=np.int16),
        meta=np.array(json.dumps({"tracks": meta})))
    if verbose:
        print("\nplay order (%d tracks, %.1f min):" % (len(meta), offs[-1] / FPS / 60))
        prev = None
        for m in meta:
            flag = "  <-- REPEAT" if prev == m["surah_no"] and m["surah_no"] else ""
            print("  %-26s %-28s %5.0fs%s"
                  % (m["surah"], m["name"], m["frames"] / m["fps"], flag))
            prev = m["surah_no"]
        print("\nwrote %s (%.0f KB)" % (TIMELINE, os.path.getsize(TIMELINE) / 1024))
    return meta


# --------------------------------------------------------------------- add ---
def add_surah(surah_no, reciter_id, verbose=True):
    lib = load_library()
    tid = "%03d-%s" % (surah_no, reciter_id)
    if any(t["id"] == tid for t in lib["tracks"]):
        print("  already have %s" % tid)
        return None
    s = next((x for x in surahs() if x["no"] == surah_no), None)
    if s is None:
        raise RuntimeError("no surah %d" % surah_no)
    rec = next((r for r in reciters() if r["id"] == reciter_id), None)
    if rec is None:
        raise RuntimeError("unknown reciter %r" % reciter_id)
    if verbose:
        print("adding %s (%s) - %s" % (s["name"], s["meaning"], rec["name"]))
    path, starts = download_surah(surah_no, reciter_id, progress=verbose)
    track = {
        "id": tid, "kind": "surah", "audio": path,
        "title": "%s · %s" % (s["name"], rec["name"]),
        "surah_no": s["no"], "surah": s["name"], "meaning": s["meaning"],
        "revelation": s["revelation"], "n_ayat": s["ayat"],
        "reciter_id": reciter_id, "reciter": rec["name"],
        "subtitle": "%s · %s" % (s["meaning"], rec["bitrate"]),
        "ayat": starts, "text": fetch_ayah_text(surah_no),
    }
    analyse_track(track, progress=verbose)
    lib["tracks"].append(track)
    save_library(lib)
    return track


def add_url(url, title=None, verbose=True):
    lib = load_library()
    if verbose:
        print("fetching %s" % url)
    path, name = download_url(url, title)
    tid = os.path.splitext(os.path.basename(path))[0]
    track = {
        "id": tid, "kind": "url", "audio": path, "title": name,
        "surah_no": 0, "surah": name[:40], "revelation": "", "n_ayat": 0,
        "reciter_id": "", "reciter": name[:40],
        "subtitle": "imported audio", "ayat": [], "text": [], "source": url,
    }
    analyse_track(track, progress=verbose)
    lib["tracks"].append(track)
    save_library(lib)
    return track


def migrate_existing():
    """Adopt the six Al-Fatiha files that predate the library."""
    lib = load_library()
    have = {t["id"] for t in lib["tracks"]}
    s1 = next(x for x in surahs() if x["no"] == 1)
    text = fetch_ayah_text(1)
    found = 0
    for f in sorted(os.listdir(AUDIO)):
        if not f.endswith(".ogg") or f.startswith("url-") or f[:3].isdigit():
            continue
        rid = f[:-4]
        rec = next((r for r in reciters() if r["id"] == rid), None)
        if rec is None:
            continue
        tid = "001-%s" % rid
        if tid in have:
            continue
        src = os.path.join(AUDIO, f)
        dst = os.path.join(AUDIO, "%s.ogg" % tid)
        os.rename(src, dst)
        # exact ayah bounds are lost with the source mp3s; leave empty rather
        # than invent them, the renderer already handles a track without them
        lib["tracks"].append({
            "id": tid, "kind": "surah", "audio": dst,
            "title": "%s · %s" % (s1["name"], rec["name"]),
            "surah_no": 1, "surah": s1["name"], "meaning": s1["meaning"],
            "revelation": s1["revelation"], "n_ayat": s1["ayat"],
            "reciter_id": rid, "reciter": rec["name"],
            "subtitle": "%s · %s" % (s1["meaning"], rec["bitrate"]),
            "ayat": [], "text": text,
        })
        found += 1
    save_library(lib)
    return found


# --------------------------------------------------------------------- cli ---
def main():
    ap = argparse.ArgumentParser(prog="chladni-add",
                                 description="what the cymatics screensaver plays")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("search", help="find a surah"); p.add_argument("query", nargs="*")
    p = sub.add_parser("reciters", help="list/find reciters"); p.add_argument("query", nargs="*")
    p = sub.add_parser("add", help="add a surah, or any URL")
    p.add_argument("--surah", type=int); p.add_argument("--reciter", action="append", default=[])
    p.add_argument("--url"); p.add_argument("--title")
    sub.add_parser("list", help="what is in the library")
    p = sub.add_parser("remove", help="drop a track"); p.add_argument("id")
    sub.add_parser("sync", help="refresh the surah/reciter catalogues")
    sub.add_parser("build", help="re-order and write the timeline")
    sub.add_parser("migrate", help="adopt pre-library audio")
    a = ap.parse_args()

    if a.cmd in ("search", "reciters", "add", "migrate"):
        ensure_meta()

    if a.cmd == "search":
        for s in search_surahs(" ".join(a.query)):
            print("%3d  %-24s %-30s %-8s %3d ayat"
                  % (s["no"], s["name"], s["meaning"], s["revelation"], s["ayat"]))
    elif a.cmd == "reciters":
        for r in search_reciters(" ".join(a.query)):
            print("%-36s %-30s %s" % (r["id"], r["name"], r["bitrate"]))
    elif a.cmd == "add":
        if a.url:
            add_url(a.url, a.title)
        elif a.surah:
            if not a.reciter:
                ap.error("--surah needs at least one --reciter")
            for rid in a.reciter:
                add_surah(a.surah, rid)
        else:
            ap.error("give --surah/--reciter or --url")
        build()
    elif a.cmd == "list":
        lib = load_library()
        for t in lib["tracks"]:
            print("%-30s %-40s %6.0fs" % (t["id"], t["title"][:40],
                                          t.get("frames", 0) / t.get("fps", FPS)))
        print("%d tracks" % len(lib["tracks"]))
    elif a.cmd == "remove":
        lib = load_library()
        keep = [t for t in lib["tracks"] if t["id"] != a.id]
        if len(keep) == len(lib["tracks"]):
            print("no track %r" % a.id); return 1
        lib["tracks"] = keep; save_library(lib); build()
    elif a.cmd == "migrate":
        print("adopted %d existing files" % migrate_existing()); build()
    elif a.cmd == "sync":
        for f in ("surahs.json", "reciters.json", "ayah_counts.json"):
            fp = os.path.join(META, f)
            if os.path.exists(fp):
                os.remove(fp)
        ensure_meta()
    elif a.cmd == "build":
        build()
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
