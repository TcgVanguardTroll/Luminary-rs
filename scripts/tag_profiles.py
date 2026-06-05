#!/usr/bin/env python3
"""#tier3-4: per-performer ATTRIBUTE tag profiles for attribute search (not just
frame filtering). Two sources:

  --source footage : re-seek each KEPT footage frame (clip+sample#) and tag it.
  --source stills  : download a sample of each performer's STILL images (by URL
                     from the `images` table) and tag them — covers the whole
                     roster (1008), not just the 9 footage performers.

Keeps tags that recur across a performer's frames (stable traits: hair/breast
size/tattoos/piercings) and drops transient act/pose/scene tags + a denylist.
Writes a `performer_tags(performer,tag,freq)` table. Resumable: with --source
stills, performers already present are skipped unless --force.

Usage: tag_profiles.py [--source footage|stills] [--write] [--sample N]
                       [--min-freq F] [--force] [performer ...]
"""
import argparse
import os
import sqlite3
import sys
import tempfile
import urllib.request
from collections import Counter

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refine_frames as rf  # Tagger, load_tagger, sample_times, DB, ROOT, REJECT_TAGS

DENY = rf.REJECT_TAGS | {
    "1girl", "solo", "1boy", "realistic", "photorealistic", "blurry",
    "motion_blur", "depth_of_field", "looking_at_viewer", "looking_back",
    "looking_to_the_side", "parted_lips", "open_mouth", "closed_eyes",
    "standing", "sitting", "kneeling", "squatting", "bent_over", "all_fours",
    "from_behind", "from_side", "from_above", "from_below", "arched_back",
    "on_bed", "on_couch", "leg_up", "knees_up", "indoors", "outdoors",
    "bed", "couch", "sofa", "pillow", "bedroom", "wall", "curtains", "window",
    "bathroom", "mirror", "chair", "table", "floor", "carpet", "kitchen",
    "nude", "completely_nude", "topless", "bottomless", "barefoot",
    "english_text", "watermark", "web_address", "censored", "uncensored",
    "dark-skinned_male", "male_focus", "yaoi", "2boys", "multiple_boys",
    "hetero",
    "dark_skin", "dark-skinned_female", "pale_skin", "light_skin", "tan",
    "very_dark_skin", "sun_tattoo", "tanlines", "tan_lines",
    "breasts", "nipples", "lips", "nose", "navel", "cleavage", "mouth", "teeth",
    "tongue", "collarbone", "stomach", "thighs", "ass", "pussy", "anus",
    "armpits", "feet", "toes", "fingernails", "knees", "hand", "ribs",
    "underwear", "bra", "panties", "pants", "shirt", "dress", "hat", "shorts",
    "skirt", "swimsuit", "bikini", "lingerie", "thighhighs", "socks", "shoes",
    "gloves", "clothes_lift", "clothing_aside", "shirt_lift",
    # pose / framing / expression (not durable attributes)
    "full_body", "upper_body", "lower_body", "cowboy_shot", "legs", "arms",
    "arms_up", "arm_up", "hands_up", "arms_behind_back", "smile", "blush",
    "tongue_out", "kneeling", "lying_down", "contrapposto", "feet_up",
    # footwear / common transient clothing
    "high_heels", "sandals", "boots", "shoes_removed", "miniskirt", "jeans",
    "denim", "crop_top", "tank_top", "jacket", "sweater", "sunglasses",
}


def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Luminary)"})
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    with urllib.request.urlopen(req, timeout=15) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def footage_frames(con, perf, tagger, thresh):
    """Yield tags for each kept footage frame of `perf` (re-seek)."""
    rows = con.execute(
        "SELECT url FROM images WHERE performer=? AND source='footage'", (perf,)).fetchall()
    byclip = {}
    for (url,) in rows:
        clip = url[len("footage://"):url.index("#")]
        byclip.setdefault(clip, []).append(int(url[url.index("#") + 1:]))
    for clip, sidxs in byclip.items():
        path = os.path.join(rf.ROOT, perf, clip)
        if len(path) > 255:
            path = "\\\\?\\" + os.path.abspath(path)
        times, cap = rf.sample_times(path)
        if times is None:
            cap.release()
            continue
        for sidx in sidxs:
            cap.set(cv2.CAP_PROP_POS_MSEC, times[min(max(sidx - 1, 0), len(times) - 1)])
            ok, frame = cap.read()
            if ok:
                yield tagger.predict(frame, thresh)
        cap.release()


def still_frames(con, perf, tagger, thresh, sample):
    """Yield tags for up to `sample` of `perf`'s best still images (download)."""
    urls = [u for (u,) in con.execute(
        "SELECT url FROM images WHERE performer=? AND source IN ('pornpics','pichunter') "
        "ORDER BY quality DESC LIMIT ?", (perf, sample))]
    for url in urls:
        tmp = None
        try:
            tmp = download(url)
            img = cv2.imread(tmp)
            if img is not None:
                yield tagger.predict(img, thresh)
        except Exception:  # noqa: BLE001
            pass
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("footage", "stills"), default="footage")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--sample", type=int, default=6, help="stills per performer (stills mode)")
    ap.add_argument("--min-freq", type=float, default=0.40)
    ap.add_argument("--tag-thresh", type=float, default=0.35)
    ap.add_argument("--force", action="store_true", help="re-tag performers already in the table")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()

    tagger = rf.load_tagger()
    if tagger is None:
        print("WD14 tagger model not found at", rf.MODEL_DIR)
        return
    con = sqlite3.connect(rf.DB, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    if args.write:
        con.execute("CREATE TABLE IF NOT EXISTS performer_tags "
                    "(performer TEXT, tag TEXT, freq REAL, PRIMARY KEY(performer,tag))")

    if args.names:
        performers = args.names
    elif args.source == "footage":
        performers = [r[0] for r in con.execute(
            "SELECT DISTINCT performer FROM images WHERE source='footage' ORDER BY performer")]
    else:
        performers = [r[0] for r in con.execute("SELECT name FROM body_index ORDER BY name")]

    done = set()
    if args.source == "stills" and not args.force:
        done = {r[0] for r in con.execute("SELECT DISTINCT performer FROM performer_tags")}

    total = len(performers)
    tagged = skipped = 0
    for i, perf in enumerate(performers, 1):
        if perf in done:
            skipped += 1
            continue
        counts, n = Counter(), 0
        gen = (footage_frames(con, perf, tagger, args.tag_thresh) if args.source == "footage"
               else still_frames(con, perf, tagger, args.tag_thresh, args.sample))
        for tags in gen:
            n += 1
            counts.update(tags)
        if not n:
            continue
        profile = [(t, c / n) for t, c in counts.items()
                   if c / n >= args.min_freq and t not in DENY]
        profile.sort(key=lambda z: -z[1])
        tagged += 1
        if args.source == "footage" or args.names:
            print(f"  {perf:22} ({n} frames) {', '.join(f'{t} {f*100:.0f}%' for t, f in profile[:12])}")
        if args.write:
            con.execute("DELETE FROM performer_tags WHERE performer=?", (perf,))
            con.executemany("INSERT OR REPLACE INTO performer_tags VALUES (?,?,?)",
                            [(perf, t, round(f, 3)) for t, f in profile])
            con.commit()
        if args.source == "stills" and i % 50 == 0:
            print(f"  ...{i}/{total} ({tagged} tagged, {skipped} skipped)", flush=True)

    con.close()
    print(f"\nDONE: {tagged} performers tagged, {skipped} skipped. "
          f"{'wrote performer_tags' if args.write else 'DRY-RUN'}.")


if __name__ == "__main__":
    main()
