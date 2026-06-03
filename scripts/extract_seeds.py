#!/usr/bin/env python3
"""Extract seed face embeddings for the footage (Milfs) performers from the
library, for #24's clip-level identity gate (so a co-star's side silhouette
can't pollute a performer's proj). Read-only on the DB; writes scripts/seeds.json
{name: [512 floats]}. The mediapipe/insightface python can't open the container
DB, so this runs under plain python and hands the seeds to apply_footage.py.
"""
import json
import os
import sqlite3
import struct

DB = r"C:\Users\TCGVANGUARDTROLL\AppData\Local\luminary\luminary.db"
ROOT = r"D:\Gooniverse\Milfs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds.json")


def decode(blob):
    """performers.embedding is JSON text on most rows, raw LE-f32 on some."""
    if blob is None:
        return None
    if isinstance(blob, str):
        blob = blob.encode()
    try:
        v = json.loads(blob.decode("utf-8", "ignore"))
        if isinstance(v, list) and v:
            return [float(x) for x in v]
    except Exception:  # noqa: BLE001
        pass
    if len(blob) % 4 == 0:
        n = len(blob) // 4
        return list(struct.unpack(f"<{n}f", blob))
    return None


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c = con.cursor()
    names = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    seeds = {}
    for name in names:
        row = c.execute(
            "select embedding from performers where lower(name)=? and embedding is not null",
            (name.lower(),),
        ).fetchone()
        v = decode(row[0]) if row else None
        if v and len(v) >= 128:
            seeds[name] = v
            print(f"  {name:22} -> seed dim {len(v)}")
        else:
            print(f"  {name:22} -> NO seed (footage-self-identity fallback)")
    con.close()
    json.dump(seeds, open(OUT, "w", encoding="utf-8"))
    print(f"\nwrote {OUT}: {len(seeds)}/{len(names)} seeds")


if __name__ == "__main__":
    main()
