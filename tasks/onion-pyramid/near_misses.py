"""Plain-Python checks of score() and success() in task.py; no simulator needed.

    python near_misses.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import task  # noqa: E402
from labkit import PartState  # noqa: E402

Z = np.zeros(3)
R = task.R
B0 = task.FLOOR_Z + R  # a bottom onion's centre height


def onion(p, moving=False):
    p = np.asarray(p, float)
    return PartState(pos=p, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.array([0.05, 0, 0]) if moving else Z,
                     ang_vel=Z, low=p - R, high=p + R)


def start():
    o = {n: onion((x, y, B0)) for n, (x, y) in zip(task.BASE, task.BASE_SPOTS)}
    o.update({n: onion((x, y, task.TOP + task.FLOOR_T + R)) for n, (x, y) in zip(task.LOOSE, task.CRATE_SPOTS)})
    return o


def stacked(n_second, top=False, **moves):
    o = start()
    for k in range(n_second):
        hx, hy = task.HOLLOWS[k]
        o[task.LOOSE[k]] = onion((hx, hy, B0 + task.RISE))
    if top:
        o[task.LOOSE[4]] = onion((task.DX, task.DY, B0 + 2 * task.RISE))
    for n, p in moves.items():
        o[n] = onion(p) if not isinstance(p, PartState) else p
    return o


CASES = [
    ("starting layout", start(), 0.0, False),
    ("one onion in a hollow", stacked(1), 0.2, False),
    ("four in the hollows, none on top", stacked(4), 0.8, False),
    ("full pyramid", stacked(4, top=True), 1.0, True),
    ("full pyramid, last onion still rolling",
     stacked(4, top=True, onion_4=onion((task.DX, task.DY, B0 + 2 * task.RISE), moving=True)), 0.8, False),
    ("top onion on three (one hollow empty): rests lower, touches three",
     stacked(3, onion_4=(task.DX - 0.01, task.DY - 0.01, B0 + task.RISE + 0.035)), 0.6, False),
    ("onion balanced on a bottom onion's crown (touches one)",
     stacked(0, onion_0=(task.BASE_SPOTS[0][0], task.BASE_SPOTS[0][1], B0 + 2 * R)), 0.0, False),
    ("full pyramid but one onion fell to the table",
     stacked(4, top=True, onion_5=(0.20, 0.05, task.TOP + R)), 0.8, False),
    ("two onions on the floor beside the table, nothing stacked",
     stacked(0, onion_0=(0.9, 0.0, R), onion_1=(0.9, 0.1, R)), 0.0, False),
    ("full pyramid, the leftover onion still in the crate", stacked(4, top=True), 1.0, True),
    ("a bottom onion pulled out of the display, an onion in the gap it leaves",
     stacked(1, base_0=(0.20, 0.10, task.TOP + R), onion_0=(*task.BASE_SPOTS[0], B0)), 0.0, False),
    ("onion held in the air over a hollow", stacked(0, onion_0=(*task.HOLLOWS[0], B0 + task.RISE + 0.05)), 0.0, False),
    ("onion in a hollow but 4 mm too far from one support (gap in the bottom layer)",
     stacked(1, base_0=(task.BASE_SPOTS[0][0] - 0.008, task.BASE_SPOTS[0][1] - 0.008, B0)), 0.0, False),
]


def main():
    bad = 0
    for label, objs, want_score, want_success in CASES:
        s, ok = task.score(objs), task.success(objs)
        good = abs(s - want_score) < 1e-9 and ok == want_success
        bad += not good
        print(f"{'ok ' if good else 'BAD'} score={s:.2f} (want {want_score:.2f}) success={ok}  {label}")
    # the fallen test is only about table level: a held onion over the table is not fallen
    o = start()
    o["onion_0"] = onion((0.20, 0.05, task.TOP + 0.20))
    assert not task.fallen(o)
    # clearances of the demo's grasp, along the diagonal between neighbours in the crate and over the stack
    pad_out = 0.047 + 0.005  # pads open 94 mm apart, ~10 mm thick
    diag_gap = np.hypot(task.CRATE_PITCH, task.CRATE_PITCH / 2 * 2) - R
    assert diag_gap > pad_out, diag_gap
    print(f"diagonal neighbour in the crate {diag_gap * 1000:.0f} mm from an onion's centre; pads reach {pad_out * 1000:.0f}")
    print("all ok" if not bad else f"{bad} BAD")
    return bad


if __name__ == "__main__":
    sys.exit(main())
