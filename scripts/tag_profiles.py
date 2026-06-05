#!/usr/bin/env python3
"""#tier3-4: per-performer ATTRIBUTE tag profiles from footage frames, for
attribute search (not just frame filtering). Re-seeks each KEPT footage frame
(via stored clip+sample#), runs the WD14 tagger, and keeps tags that recur across
a performer's frames (stable traits: hair/skin/tattoos/body type) while dropping
transient act/pose/scene tags. Writes a `performer_tags(performer,tag,freq)` table.

Reuses refine_frames.Tagger + sample_times so tagging matches the filter pass.
Default: the footage performers. DRY by default prints profiles; --write stores them.

Usage: tag_profiles.py [--write] [--min-freq F] [performer ...]
"""
import argparse
import os
import sqlite3
import sys
from collections import Counter

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refine_frames as rf  # Tagger, load_tagger, sample_times, DB, ROOT, REJECT_TAGS

# High-frequency tags that are NOT durable physical attributes (so the frequency
# filter alone wouldn't drop them): people-count/generic, pose, scene/objects,
# transient clothing-state. The act tags come from rf.REJECT_TAGS.
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
    # skin-tone tags are anime-relative (tan + warm lighting), NOT ethnicity —
    # unreliable on real photos; use StashDB `ethnicity` instead.
    "dark_skin", "dark-skinned_female", "pale_skin", "light_skin", "tan",
    "very_dark_skin", "sun_tattoo", "tanlines", "tan_lines",
    # near-universal body parts (present for ~everyone -> not discriminative)
    "breasts", "nipples", "lips", "nose", "navel", "cleavage", "mouth", "teeth",
    "tongue", "collarbone", "stomach", "thighs", "ass", "pussy", "anus",
    "armpits", "feet", "toes", "fingernails", "knees", "hand", "ribs",
    # transient clothing
    "underwear", "bra", "panties", "pants", "shirt", "dress", "hat", "shorts",
    "skirt", "swimsuit", "bikini", "lingerie", "thighhighs", "socks", "shoes",
    "gloves", "clothes_lift", "clothing_aside", "bottomless", "shirt_lift",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="store profiles in performer_tags table")
    ap.add_argument("--min-freq", type=float, default=0.40, help="keep tags in >= this fraction of frames")
    ap.add_argument("--tag-thresh", type=float, default=0.35)
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()

    tagger = rf.load_tagger()
    if tagger is None:
        print("WD14 tagger model not found at", rf.MODEL_DIR)
        return
    con = sqlite3.connect(rf.DB, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    if args.names:
        q = ("SELECT DISTINCT performer FROM images WHERE source='footage' AND performer IN (%s)"
             % ",".join("?" * len(args.names)))
        performers = [r[0] for r in con.execute(q, args.names)]
    else:
        performers = [r[0] for r in con.execute(
            "SELECT DISTINCT performer FROM images WHERE source='footage' ORDER BY performer")]
    if args.write:
        con.execute("CREATE TABLE IF NOT EXISTS performer_tags "
                    "(performer TEXT, tag TEXT, freq REAL, PRIMARY KEY(performer,tag))")

    for perf in performers:
        rows = con.execute(
            "SELECT url FROM images WHERE performer=? AND source='footage'", (perf,)).fetchall()
        byclip = {}
        for (url,) in rows:
            clip = url[len("footage://"):url.index("#")]
            byclip.setdefault(clip, []).append(int(url[url.index("#") + 1:]))

        counts = Counter()
        nframes = 0
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
                if not ok:
                    continue
                nframes += 1
                for t in tagger.predict(frame, args.tag_thresh):
                    counts[t] += 1
            cap.release()

        if not nframes:
            print(f"  {perf}: no readable frames")
            continue
        profile = [(t, c / nframes) for t, c in counts.items()
                   if c / nframes >= args.min_freq and t not in DENY]
        profile.sort(key=lambda z: -z[1])
        print(f"\n=== {perf}  ({nframes} frames) ===")
        print("  " + ", ".join(f"{t} {f*100:.0f}%" for t, f in profile[:18]))
        if args.write:
            con.execute("DELETE FROM performer_tags WHERE performer=?", (perf,))
            con.executemany("INSERT OR REPLACE INTO performer_tags VALUES (?,?,?)",
                            [(perf, t, round(f, 3)) for t, f in profile])
            con.commit()

    con.close()
    print(f"\n{'WROTE profiles to performer_tags' if args.write else 'DRY-RUN (use --write to store)'}.")


if __name__ == "__main__":
    main()
