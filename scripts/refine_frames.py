#!/usr/bin/env python3
"""Refine the footage corpus (#24): re-judge each kept footage frame and drop the
ones that aren't a clean, single-subject BODY-DISPLAY shot, so only good frames
feed the body centroids. The corpus stores vectors, not pixels, so we re-seek each
kept frame via its stored `clip + sample#` and re-evaluate it.

Two filters (geometry alone proved insufficient — it kept partnered/action frames
and dropped non-standing ones, because landmark-visibility can't tell a clean body
display from a sex act):
  1. LIGHT geometry (MediaPipe): a body must be detectable (pose + torso visible).
     `--geom full` re-enables the stricter legs+height-span gate.
  2. SEMANTIC tagger (WD14 ONNX, the real filter): reject frames tagged with sex
     acts / multiple people / close-up / lying-extreme poses. Keeps single-subject
     standing/posing body-display frames. Auto-on if the model is at MODEL_DIR.

DRY-RUN by default (reports keep/drop + drop reasons, saves sample KEEP/DROP frames
to --out for review). `--apply` DELETEs rejected rows; then re-run `luminary
aggregate` to rebuild centroids from the cleaned set.

Usage: refine_frames.py [--apply] [--geom light|full] [--tag-thresh F]
                        [--out DIR] [--min-body-frac F] [--samples N] [names...]
Model: WD14 ViT v3 at %LOCALAPPDATA%\\wd14-tagger\\{model.onnx,selected_tags.csv}.
"""
import argparse
import csv
import os
import sqlite3
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import body_embed as be  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

from _paths import db_path  # cross-platform DB location
DB = db_path()
ROOT = r"D:\Gooniverse\Milfs"
MODEL_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ""), "wd14-tagger")

EVERY_S, MIN_SAMPLES, MAX_SAMPLES = 8.0, 24, 2000
PROC_WIDTH = 640
TORSO_IDX = (11, 12, 23, 24)
LEG_IDX = (25, 26, 27, 28)
MIN_VIS = 0.5
DEFAULT_BODY_FRAC = 0.55
DEFAULT_TAG_THRESH = 0.35

# Booru-vocab tags (WD14) that mean "not a clean single-subject body display".
REJECT_TAGS = {
    # sex acts / positions
    "sex", "vaginal", "anal", "oral", "fellatio", "cunnilingus", "irrumatio",
    "paizuri", "handjob", "doggystyle", "cowgirl_position",
    "reverse_cowgirl_position", "missionary", "girl_on_top", "prone_bone",
    "mating_press", "sex_from_behind", "group_sex", "gangbang", "threesome",
    "double_penetration", "deepthroat", "grabbing_another's_breast",
    # explicit partner / fluids
    "penis", "erection", "cum", "cumshot", "ejaculation", "cum_in_pussy",
    "cum_on_body", "testicles", "fingering",
    # multiple women
    "2girls", "3girls", "multiple_girls",
    # close-up / framing
    "close-up", "portrait", "pov",
    # lying / extreme pose
    "lying", "on_back", "on_stomach", "spread_legs", "legs_up", "spread_pussy",
    "split", "m_legs",
}


def load_pose():
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=be._cached(be.POSE_MODEL_URL)),
            running_mode=vision.RunningMode.IMAGE, num_poses=1))


class Tagger:
    """WD14 ViT v3 ONNX image->tags. Single model, no torch/tokenizer."""

    def __init__(self, model_path, csv_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = next((d for d in inp.shape if isinstance(d, int) and d > 3), 448)
        self.names, self.cats = [], []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.names.append(row["name"])
                self.cats.append(int(row["category"]))
        self.n = len(self.names)

    def predict(self, bgr, thresh):
        h, w = bgr.shape[:2]
        s = max(h, w)
        canvas = np.full((s, s, 3), 255, dtype=np.uint8)
        y0, x0 = (s - h) // 2, (s - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = bgr
        img = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = np.expand_dims(img.astype(np.float32), 0)  # BGR, 0-255, NHWC
        probs = self.sess.run(None, {self.input_name: x})[0][0]
        return {self.names[i] for i in range(self.n)
                if self.cats[i] == 0 and probs[i] >= thresh}


def load_tagger():
    m = os.path.join(MODEL_DIR, "model.onnx")
    c = os.path.join(MODEL_DIR, "selected_tags.csv")
    if os.path.exists(m) and os.path.exists(c):
        try:
            return Tagger(m, c)
        except Exception as e:  # noqa: BLE001
            print(f"tagger load failed ({e}); falling back to geometry only")
    return None


def sample_times(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, cap
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    if fps > 0 and nfr > 0:
        dur = nfr / fps
        n = max(MIN_SAMPLES, min(MAX_SAMPLES, int(dur // EVERY_S)))
        return [dur * (k + 0.5) / n * 1000.0 for k in range(n)], cap
    return None, cap


def geom_verdict(lm, geom, body_frac):
    if be._min_visibility(lm, TORSO_IDX) < MIN_VIS:
        return False, "torso-occluded"
    if geom == "full":
        if be._min_visibility(lm, LEG_IDX) < MIN_VIS:
            return False, "legs-not-visible"
        if (max(lm[27].y, lm[28].y) - min(lm[11].y, lm[12].y)) < body_frac:
            return False, "body-too-small"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--geom", choices=("light", "full"), default="light")
    ap.add_argument("--tag-thresh", type=float, default=DEFAULT_TAG_THRESH)
    ap.add_argument("--out", default=os.path.join(ROOT, "_refine_samples"))
    ap.add_argument("--min-body-frac", type=float, default=DEFAULT_BODY_FRAC)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--reject-out", default=None, help="TSV of rejects written during dry-run")
    ap.add_argument("--delete-from", help="apply a reviewed dry-run: delete the perf<TAB>url footage rows in this TSV (no tagger pass)")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()

    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    if args.delete_from:
        n = 0
        with open(args.delete_from, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    con.execute("DELETE FROM images WHERE performer=? AND url=? AND source='footage'",
                                (parts[0], parts[1]))
                    n += 1
        con.commit()
        con.close()
        print(f"deleted {n} footage rows from {args.delete_from}. Now run `luminary aggregate`.")
        return
    if args.names:
        q = ("SELECT DISTINCT performer FROM images WHERE source='footage' AND performer IN (%s)"
             % ",".join("?" * len(args.names)))
        performers = [r[0] for r in con.execute(q, args.names)]
    else:
        performers = [r[0] for r in con.execute(
            "SELECT DISTINCT performer FROM images WHERE source='footage' ORDER BY performer")]
    if not performers:
        print("no footage performers found")
        return

    pose = load_pose()
    tagger = load_tagger()
    print(f"mode: {'APPLY (deletes)' if args.apply else 'DRY-RUN'} ; "
          f"geom={args.geom} ; tagger={'ON' if tagger else 'OFF'} ; "
          f"tag_thresh={args.tag_thresh}")
    reject_out = args.reject_out or os.path.join(args.out, "reject_list.tsv")
    rej_f = None
    if not args.apply:
        os.makedirs(args.out, exist_ok=True)
        rej_f = open(reject_out, "w", encoding="utf-8")

    gk = gd = 0
    for perf in performers:
        rows = con.execute(
            "SELECT url, view FROM images WHERE performer=? AND source='footage'", (perf,)).fetchall()
        byclip = {}
        for url, view in rows:
            clip = url[len("footage://"):url.index("#")]
            byclip.setdefault(clip, []).append((int(url[url.index("#") + 1:]), url, view))

        keep = drop = sk = sd = 0
        reasons = {}
        kviews = {}
        for clip, items in byclip.items():
            path = os.path.join(ROOT, perf, clip)
            if len(path) > 255:
                path = "\\\\?\\" + os.path.abspath(path)
            times, cap = sample_times(path)
            if times is None:
                cap.release()
                continue
            for sidx, url, view in items:
                cap.set(cv2.CAP_PROP_POS_MSEC, times[min(max(sidx - 1, 0), len(times) - 1)])
                ok, frame = cap.read()
                if not ok:
                    continue
                h0, w0 = frame.shape[:2]
                fr = cv2.resize(frame, (PROC_WIDTH, max(1, int(h0 * PROC_WIDTH / w0))))
                res = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                if not res.pose_landmarks:
                    good, reason = False, "no-pose"
                else:
                    good, reason = geom_verdict(res.pose_landmarks[0], args.geom, args.min_body_frac)
                if good and tagger:
                    bad = tagger.predict(frame, args.tag_thresh) & REJECT_TAGS
                    if bad:
                        good, reason = False, "tag:" + ",".join(sorted(bad)[:3])
                if good:
                    keep += 1
                    kviews[view] = kviews.get(view, 0) + 1
                    if not args.apply and sk < args.samples:
                        cv2.imwrite(os.path.join(args.out, f"{perf}__KEEP_{view}_{sk}.jpg"), frame)
                        sk += 1
                else:
                    drop += 1
                    rk = reason.split(":")[0]
                    reasons[rk] = reasons.get(rk, 0) + 1
                    if args.apply:
                        con.execute("DELETE FROM images WHERE performer=? AND url=?", (perf, url))
                    else:
                        rej_f.write(f"{perf}\t{url}\t{reason}\n")
                        if sd < args.samples:
                            safe = reason.replace(":", "_").replace(",", "-").replace("'", "")
                            cv2.imwrite(os.path.join(args.out, f"{perf}__DROP_{safe}_{sd}.jpg"), frame)
                            sd += 1
            cap.release()
        if args.apply:
            con.commit()
        total = keep + drop
        print(f"  {perf:20} keep {keep:5} / drop {drop:5}  "
              f"({(100 * keep / total) if total else 0:.0f}% kept)   "
              f"kept_views={kviews}   {reasons}")
        gk += keep
        gd += drop

    con.close()
    if rej_f:
        rej_f.close()
    print(f"\n{'DELETED' if args.apply else 'WOULD DROP'} {gd}; kept {gk}.")
    if not args.apply:
        print(f"Samples under {args.out} ; reject list: {reject_out}")
        print(f"To apply this exact review: refine_frames.py --delete-from \"{reject_out}\"")
    else:
        print("Run `luminary aggregate` to rebuild centroids from the cleaned frames.")


if __name__ == "__main__":
    main()
