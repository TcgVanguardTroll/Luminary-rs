#!/usr/bin/env python3
"""Validate instance_seg on a clip: sample frames, report the per-frame
people-count distribution (does YOLOv8-seg actually find 2 separate bodies in
partnered frames?), and save a few color-tinted overlays (one tint per person)
for eyeballing mask quality. Run under the Python 3.13 instance-seg env.

Usage: seg_validate.py <clip.mp4> [n_frames]
"""
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_seg as iseg

TINTS = [(0, 180, 0), (180, 0, 0), (0, 0, 200), (0, 180, 180)]


def main():
    clip = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    yolo = iseg.load_models()[0]
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = total / fps if fps else 0
    dist, saved = Counter(), 0
    outdir = os.path.join(os.path.dirname(clip) or ".", "_seg_overlays")
    os.makedirs(outdir, exist_ok=True)
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, dur * (k + 0.5) / n * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        ppl = iseg.person_instances(yolo, fr)
        dist[len(ppl)] += 1
        sizes = sorted((int(m.sum()) for m, _, _ in ppl), reverse=True)
        print(f"  t={dur*(k+0.5)/n/60:4.1f}min  people={len(ppl)}  mask_px={sizes}", flush=True)
        if len(ppl) >= 2 and saved < 4:
            ov = fr.copy()
            for i, (m, _, _) in enumerate(ppl):
                ov[m] = (0.45 * ov[m] + 0.55 * np.array(TINTS[i % 4])).astype(np.uint8)
            out = os.path.join(outdir, f"seg_{saved}.png")
            cv2.imwrite(out, ov)
            print(f"     overlay -> {out}", flush=True)
            saved += 1
    cap.release()
    print(f"\npeople-count distribution: {dict(dist)}")
    print(f"overlays in {outdir} (eyeball whether each tint is one clean body)")


if __name__ == "__main__":
    main()
