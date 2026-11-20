"""On the station table: a low wooden produce display bin straight ahead of the robots, its floor covered by a bottom
layer of nine onions packed three by three, and to the robots' right a green plastic crate holding six more onions,
two rows of three.

Round produce only stays up where it sits in a hollow: an onion resting in the dip between four onions below it is
held there, but one set down on top of a single onion, or at the edge of the stack, rolls off and falls.

The task: stack onions from the crate on the display as a pyramid, one in each hollow of the bottom layer and then
one on top.

Run it:  inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \\
           -E control_hz=20 --scorer task_score --grader none --max-steps 2400 --seed 1 \\
           --instruction "<see task.json>"
"""

import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import on_reset, on_step, part  # noqa: E402
import yam_arena.demo as _demo  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

ONION_C = (0.72, 0.50, 0.24)  # yellow onion skin
BIN_C = (0.62, 0.46, 0.30)  # pale wood
CRATE_C = (0.18, 0.45, 0.26)  # green polypropylene

# == the onions ==========================================================================================================
R = 0.035  # 70 mm across: a medium yellow onion
ONION_KG = 0.170
PITCH = 0.0702  # the bottom layer's spacing: touching, with 0.2 mm left so they don't start pressed together

# == the display bin (fixed) =============================================================================================
DX, DY = 0.360, -0.030  # its middle
IN_HALF = 0.107  # inside 214 x 214 mm: three onions and 4 mm to spare each way
FLOOR_T = 0.006
WALL_T = 0.008
WALL_H = 0.035  # walls 35 mm above the floor: up to the bottom onions' middles
FLOOR_Z = TOP + FLOOR_T
OUT_HALF = IN_HALF + WALL_T


def _tray(name, cx, cy, half_x, half_y, wall_h, color):
    hx, hy, wz = half_x + WALL_T, half_y + WALL_T, TOP + FLOOR_T + wall_h / 2
    return part(
        name, pos=(0.0, 0.0, 0.0), fixed=True, friction=0.6,  # fixed: shapes in world coordinates
        boxes=[
            ((cx, cy, TOP + FLOOR_T / 2), (2 * hx, 2 * hy, FLOOR_T), color),
            *[((cx + s * (half_x + WALL_T / 2), cy, wz), (WALL_T, 2 * hy, wall_h), color) for s in (-1, 1)],
            *[((cx, cy + s * (half_y + WALL_T / 2), wz), (2 * hx, WALL_T, wall_h), color) for s in (-1, 1)],
        ],
    )


DISPLAY = _tray("display", DX, DY, IN_HALF, IN_HALF, WALL_H, BIN_C)

# == the crate (fixed), six onions in it ==================================================================================
CX, CY = 0.250, -0.250
CRATE_HALF = (0.109, 0.073)  # inside 218 x 146 mm: two rows of three onions, 72 mm apart
CRATE_WALL_H = 0.025  # low walls, so the fingers clear them when they close round an onion
CRATE = _tray("crate", CX, CY, *CRATE_HALF, CRATE_WALL_H, CRATE_C)
CRATE_PITCH = 0.072

BASE_SPOTS = [(DX + i * PITCH, DY + j * PITCH) for i in (-1, 0, 1) for j in (-1, 0, 1)]
CRATE_SPOTS = [(CX + i * CRATE_PITCH, CY + j * CRATE_PITCH / 2) for i in (-1, 0, 1) for j in (-1, 1)]


def _onion(name, x, y, z):
    return part(name, pos=(x, y, z), mass=ONION_KG, friction=0.6, spheres=[((0, 0, 0), R, ONION_C)])


BASE = [f"base_{k}" for k in range(9)]
LOOSE = [f"onion_{k}" for k in range(6)]
ONIONS = BASE + LOOSE
for n in ONIONS:
    labkit.TORSIONAL_PATCH[n] = 0.010
OBJECTS = [DISPLAY, CRATE,
           *[_onion(n, x, y, FLOOR_Z + R) for n, (x, y) in zip(BASE, BASE_SPOTS)],
           *[_onion(n, x, y, TOP + FLOOR_T + R) for n, (x, y) in zip(LOOSE, CRATE_SPOTS)]]

# The hollows: between four touching onions spaced PITCH apart, an onion sits RISE above them.
RISE = math.sqrt((2 * R) ** 2 - 2 * (PITCH / 2) ** 2)  # 49.4 mm
HOLLOWS = [(DX + sx * PITCH / 2, DY + sy * PITCH / 2) for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1))]

# == scoring =============================================================================================================
TOUCH = 0.003  # touching: centres within 2R + 3 mm
BELOW = 0.030  # a support's centre at least 30 mm lower
FLAT_TOL = 0.005  # bottom layer: on the display's floor, centre within 5 mm of R above it


def _inside(p, cx, cy, hx, hy):
    return abs(p[0] - cx) <= hx and abs(p[1] - cy) <= hy


def layers(objs):
    """(bottom, second, top): the onions on the display's floor, those resting on four touching bottom-layer onions,
    and those resting on four touching second-layer onions."""
    c = {n: np.asarray(objs[n].pos, float) for n in ONIONS}
    bottom = [n for n in ONIONS if _inside(c[n], DX, DY, IN_HALF, IN_HALF) and abs(c[n][2] - FLOOR_Z - R) <= FLAT_TOL]

    def seated(n, supports):
        under = [m for m in supports if m != n and np.linalg.norm(c[n] - c[m]) <= 2 * R + TOUCH
                 and c[n][2] - c[m][2] >= BELOW]
        return len(under) >= 4 and objs[n].resting()

    second = [n for n in ONIONS if n not in bottom and seated(n, bottom)]
    top = [n for n in ONIONS if n not in bottom and n not in second and seated(n, second)]
    return bottom, second, top


def fallen(objs):
    """Onions down at table level or below (not held up), outside the display and the crate."""
    out = []
    for n in ONIONS:
        p = objs[n].pos
        if p[2] > TOP + 2 * R:
            continue
        if _inside(p, DX, DY, OUT_HALF, OUT_HALF) or _inside(p, CX, CY, CRATE_HALF[0] + WALL_T, CRATE_HALF[1] + WALL_T):
            continue
        out.append(n)
    return out


def score(objs) -> float:
    """0.2 for each onion seated in the second layer (four hollows) and 0.2 for one on top of those, less 0.2 for each
    onion that has fallen to the table or floor; from 0 to 1."""
    _, second, top = layers(objs)
    s = 0.2 * (min(4, len(second)) + min(1, len(top))) - 0.2 * len(fallen(objs))
    return float(min(1.0, max(0.0, s)))


SUCCESS_HOLD_S = 2.0


def success(objs):
    return score(objs) >= 1.0 - 1e-9 and all(objs[n].resting() for n in ONIONS)


# == live state ==========================================================================================================
_trial = {"t": 0.0, "objs": None}
_DEBUG = bool(os.environ.get("ONION_DEBUG"))


@on_reset
def _reset(env):
    _trial.update(t=0.0, objs=None)
    return False


@on_step
def _watch(env):
    objs = {n: labkit._part_state(env, labkit._SPECS[n]) for n in ONIONS}
    _trial["objs"] = objs
    _trial["t"] += float(getattr(env.unwrapped, "step_dt", 0.05))
    if _DEBUG and int(round(_trial["t"] / 0.05)) % 10 == 0:
        b, s2, t3 = layers(objs)
        loose = {n: np.round(objs[n].pos * 1000).astype(int).tolist() for n in LOOSE}
        print(f"[onion] t={_trial['t']:5.1f} bottom={len(b)} second={s2} top={t3} fallen={fallen(objs)} "
              f"score={score(objs):.2f} loose={loose}", file=sys.stderr, flush=True)


# == scripted demo =======================================================================================================
# The right hand takes five onions from the crate one at a time, fingers closing across the onion along the diagonal
# (between its neighbours in the crate, and over the gaps of the stack), and sets each down 10 mm above its hollow:
# the four hollows of the bottom layer, then the one on top of those. Each hollow is aimed at from where the four
# onions under it are seen.


class Orient(float):
    """A Waypoint ``yaw`` that carries a full TCP rotation."""

    def __new__(cls, rot):
        obj = float.__new__(cls, 0.0)
        obj.rot = np.asarray(rot, float)
        return obj


_down_grip = _demo.grip_rotation
if not getattr(_down_grip, "_orient", False):
    def _grip_rotation(yaw=0.0):
        return yaw.rot if isinstance(yaw, Orient) else _down_grip(yaw)

    _grip_rotation._orient = True
    _demo.grip_rotation = _grip_rotation


def _down(axis) -> Orient:
    """Pointing straight down, the fingers closing along ``axis`` (a unit vector in the table plane)."""
    z, x = np.array([0.0, 0.0, -1.0]), np.asarray(axis, float) / np.linalg.norm(axis)
    return Orient(np.column_stack([x, np.cross(z, x), z]))


GRIP = _down([float(v) for v in os.environ.get("ONION_AXIS", "1,1,0").split(",")])
GRIP_DZ = float(os.environ.get("ONION_GRIP_DZ", "0.008"))  # the TCP 8 mm above the onion's middle
RELEASE = -0.045  # fingers 90 mm apart to let go
DROP = 0.010  # let go 10 mm above the seat
CARRY_Z = TOP + 0.24  # a carried onion's middle: its underside 65 mm above the top of the finished pyramid
LEFT_PARK = (0.12, 0.30, TOP + 0.18)
RIGHT_PARK = (0.14, -0.30, TOP + 0.22)
PICKS = [int(v) for v in os.environ.get("ONION_PICKS", "1,3,5,0,2").split(",")]


class _Seen:
    """A position read from what the cameras see, evaluated every control step."""

    def __init__(self, fn):
        self.fn = fn

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.fn(), dtype=dtype or float)


def _pos(name, fallback):
    o = _trial["objs"]
    return np.asarray(o[name].pos, float) if o is not None else np.asarray(fallback, float)


def _hollow(k):
    """Where the k-th onion goes: over the middle of the four onions under that hollow, as seen, RISE above them."""
    o = _trial["objs"]
    if k < 4:
        hx, hy = HOLLOWS[k]
        if o is None:
            return np.array([hx, hy, FLOOR_Z + R + RISE])
        b = [n for n in ONIONS if _inside(o[n].pos, DX, DY, IN_HALF, IN_HALF) and o[n].pos[2] < FLOOR_Z + R + 0.02]
        under = sorted(b, key=lambda n: np.hypot(o[n].pos[0] - hx, o[n].pos[1] - hy))[:4]
    else:
        if o is None:
            return np.array([DX, DY, FLOOR_Z + R + 2 * RISE])
        mid = [n for n in ONIONS if o[n].pos[2] > FLOOR_Z + R + 0.03 and o[n].pos[2] < FLOOR_Z + R + RISE + 0.02
               and _inside(o[n].pos, DX, DY, IN_HALF, IN_HALF)]
        under = sorted(mid, key=lambda n: np.hypot(o[n].pos[0] - DX, o[n].pos[1] - DY))[:4]
    if len(under) < 4:
        return np.array([*(HOLLOWS[k] if k < 4 else (DX, DY)), FLOOR_Z + R + (RISE if k < 4 else 2 * RISE)])
    m = np.mean([o[n].pos for n in under], axis=0)
    return m + [0.0, 0.0, RISE]


def _place(k, onion, spot, where=None):
    """Take ``onion`` from the crate and let it go DROP above its seat: hollow ``k`` (or ``where()``), aimed once
    as the hand arrives over the display."""
    where = where or (lambda: _hollow(k))
    src = lambda dz: _Seen(lambda: _pos(onion, (*spot, TOP + FLOOR_T + R)) + [0, 0, dz])  # noqa: E731
    tgt = {}

    def at(z):
        def f():
            if "p" not in tgt:
                tgt["p"] = np.asarray(where(), float)
            p = tgt["p"]
            return np.array([p[0], p[1], z(p)])
        return _Seen(f)

    return [
        Waypoint(1.6, src(GRIP_DZ + 0.07), FINGER_OPEN, f"above {onion}", GRIP),
        Waypoint(1.0, src(GRIP_DZ), FINGER_OPEN, "fingers either side of it", GRIP),
        Waypoint(0.6, src(GRIP_DZ), FINGER_CLOSED, "grip it", GRIP),
        Waypoint(1.2, (*spot, CARRY_Z + GRIP_DZ), FINGER_CLOSED, "lift it out", GRIP),
        Waypoint(1.8, at(lambda p: CARRY_Z + GRIP_DZ), FINGER_CLOSED, f"over its seat ({k})", GRIP),
        Waypoint(1.6, at(lambda p: p[2] + DROP + GRIP_DZ), FINGER_CLOSED, "lower it to just above its seat", GRIP),
        Waypoint(0.5, at(lambda p: p[2] + DROP + GRIP_DZ), RELEASE, "let go", GRIP),
        Waypoint(0.8, at(lambda p: p[2] + 0.09 + GRIP_DZ), FINGER_OPEN, "up", GRIP),
    ]


CHECK = os.environ.get("ONION_CHECK", "")


def demo():
    right = [Waypoint(1.5, RIGHT_PARK, FINGER_OPEN, "start")]
    for k, i in enumerate(PICKS):
        where = None
        if CHECK.startswith("on_crown") and k == 0:  # check: set the first onion down over the front-right corner
            # bottom onion, CHECK_OUT mm outward of its crown (along the diagonal, away from the display's middle)
            out = float(os.environ.get("ONION_CHECK_OUT", "15")) / 1000 / math.sqrt(2)
            where = lambda: _pos(BASE[0], (*BASE_SPOTS[0], FLOOR_Z + R)) + [-out, -out, 2 * R]  # noqa: E731
        right += _place(k, LOOSE[i], CRATE_SPOTS[i], where)
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"))
    t = sum(w.duration for w in right)
    left = [Waypoint(1.5, LEFT_PARK, FINGER_OPEN, "park"), Waypoint(t - 1.5, LEFT_PARK, FINGER_OPEN, "rest")]
    return {"left": left, "right": right}
