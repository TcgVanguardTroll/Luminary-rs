#!/usr/bin/env python3
"""Join motion_dataset.json with milfs_stats.json and test the core hypothesis:
do the per-performer MEAN motion descriptors track body type (cup size, weight,
hips)? Per-clip motion is noisy; averaging over clips is where signal should
appear. Pure analysis — no DB, no network.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
motion = json.load(open(os.path.join(HERE, "motion_dataset.json"), encoding="utf-8"))
stats = json.load(open(os.path.join(HERE, "milfs_stats.json"), encoding="utf-8"))

CUP = {"A": 1, "B": 2, "C": 3, "D": 4, "DD": 5, "E": 5, "DDD": 6, "F": 6,
       "G": 7, "H": 8}


def parse_meas(m):
    """'40DD-25-37' -> (cup_num, waist, hip)."""
    if not m:
        return (None, None, None)
    mm = re.match(r"(\d+)([A-Z]+)-(\d+)-(\d+)", m)
    if not mm:
        return (None, None, None)
    return (CUP.get(mm.group(2)), int(mm.group(3)), int(mm.group(4)))


def num(s):
    if not s:
        return None
    d = "".join(c for c in str(s) if c.isdigit() or c == ".")
    return float(d) if d else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return None
    sx = sum(p[0] for p in pairs); sy = sum(p[1] for p in pairs)
    mx, my = sx / n, sy / n
    cov = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    vx = sum((p[0] - mx) ** 2 for p in pairs)
    vy = sum((p[1] - my) ** 2 for p in pairs)
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


rows = []
for name, rec in motion.items():
    ok = [c for c in rec.get("clips", []) if c.get("ok")]
    if not ok:
        continue
    st = stats.get(name) or {}
    cup, waist, hip = parse_meas(st.get("measurements"))
    rows.append({
        "name": name,
        "n": len(ok),
        "cup": cup,
        "weight": num(st.get("weight")),
        "hip": hip,
        "bust_flow": mean([c.get("bust_flow") for c in ok]),
        "glute_flow": mean([c.get("glute_flow") for c in ok]),
        "jiggle_bust": mean([c.get("jiggle_bust") for c in ok]),
        "jiggle_glute": mean([c.get("jiggle_glute") for c in ok]),
        "bust_freq": mean([c.get("bust_freq") for c in ok]),
    })

rows.sort(key=lambda r: (r["cup"] or 0))
print(f"{'performer':20} {'n':>2} {'cup':>3} {'wt':>4} {'hip':>3} "
      f"{'bustF':>7} {'gluteF':>7} {'jigB':>6} {'jigG':>6} {'bFreq':>6}")
for r in rows:
    print(f"{r['name'][:20]:20} {r['n']:>2} {str(r['cup'] or '?'):>3} "
          f"{str(int(r['weight']) if r['weight'] else '?'):>4} "
          f"{str(r['hip'] or '?'):>3} "
          f"{(r['bust_flow'] or 0):>7.4f} {(r['glute_flow'] or 0):>7.4f} "
          f"{(r['jiggle_bust'] or 0):>6.2f} {(r['jiggle_glute'] or 0):>6.2f} "
          f"{(r['bust_freq'] or 0):>6.2f}")

print("\nCorrelations (Pearson) over per-performer means:")
for label, xk, yk in [
    ("cup  vs bust_flow", "cup", "bust_flow"),
    ("cup  vs jiggle_bust", "cup", "jiggle_bust"),
    ("weight vs bust_flow", "weight", "bust_flow"),
    ("hip  vs glute_flow", "hip", "glute_flow"),
    ("hip  vs jiggle_glute", "hip", "jiggle_glute"),
]:
    r = pearson([row[xk] for row in rows], [row[yk] for row in rows])
    print(f"  {label:22}: {('%.2f' % r) if r is not None else 'n/a'}")
print(f"\n({len(rows)} performers with motion+stats)")
