"""Plain-Python checks of score(), success() and the touch test in task.py; no simulator needed.

    python near_misses.py
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import task  # noqa: E402
from labkit import PartState  # noqa: E402

Z = np.zeros(3)
CONTACT = task.JAW_OPEN_R - task.BAR_R  # jaw travel at which a jaw meets a centred bar


def quat_about(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[a * math.sin(angle / 2), math.cos(angle / 2)]


def bar(stick_mm=60.0, dx_mm=0.0, dz_mm=0.0, tilt_deg=0.0):
    """The bar lying along y with its free end stick_mm out of the jaws, offset dx/dz from the spindle axis, tipped
    tilt_deg about x."""
    end_y = task.CY - stick_mm / 1000
    q = quat_about((1, 0, 0), math.radians(tilt_deg))
    c = np.array([task.AX + dx_mm / 1000, end_y + task.BAR_L / 2, task.TOP + task.H + dz_mm / 1000])
    return PartState(pos=c, quat_xyzw=q, lin_vel=Z, ang_vel=Z, low=c - 0.07, high=c + 0.07)


def lathe(key=-1.3, jaws=None, carriage=0.0, cross=0.0):
    jaws = [CONTACT] * 3 if jaws is None else jaws
    j = {"key": key, "carriage": carriage, "cross": cross, **{f"jaw{i}": v for i, v in enumerate(jaws)}}
    return PartState(pos=np.array([task.AX, 0, task.TOP]), quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=Z,
                     ang_vel=Z, low=Z, high=Z, joints=j)


def tool_at(gap_mm, radial_mm, stick_mm=60.0):
    """Slide settings that put the tip gap_mm from the end face and radial_mm outside the diameter."""
    end_y = task.CY - stick_mm / 1000
    carriage = end_y - gap_mm / 1000 - task.TIP_Y0
    cross = task.AX - task.BAR_R - radial_mm / 1000 - task.TIP_X0
    return dict(carriage=carriage, cross=cross)


def objs(lathe_kw=None, bar_kw=None):
    return {"lathe": lathe(**(lathe_kw or {})), "bar": bar(**(bar_kw or {}))}


def done(**over):
    """The finished state, with any overrides."""
    lk = dict(key=-1.3, **tool_at(2.0, 1.5))
    bk = dict(stick_mm=60.0)
    for k, v in over.items():
        (bk if k in ("stick_mm", "dx_mm", "dz_mm", "tilt_deg") else lk)[k] = v
    return objs(lk, bk)


cases = []
# start: bar in its cradle, jaws open, tool back
start = {"lathe": lathe(key=0.0, jaws=[0.0] * 3),
         "bar": PartState(pos=np.array([*task.BAR_XY, task.TOP + task.BAR_R]), quat_xyzw=np.array([0, 0, 0, 1.0]),
                          lin_vel=Z, ang_vel=Z, low=Z, high=Z)}
cases.append(("untouched", start, None, 0.0))
cases.append(("all done", done(), False, 1.0))
cases.append(("all done, but the tool touched the bar on the way", done(), True, 0.5))
# clamping
cases.append(("bar in the jaws, key not turned", done(key=0.0, jaws=[0.0, 0.0, 0.0]), False, 0.0))
cases.append(("key turned only to first contact (no grip)", done(key=-0.73), False, 0.0))
cases.append(("key turned to 3.9 N a jaw (just under 4 N)", done(key=-(CONTACT + 0.0013) / task.JAW_TRAVEL * task.KEY_TRAVEL),
              False, 0.0))
cases.append(("key turned to 4.2 N a jaw", done(key=-(CONTACT + 0.0014) / task.JAW_TRAVEL * task.KEY_TRAVEL), False, 1.0))
cases.append(("jaws closed on nothing (bar not in them)", done(jaws=[task.JAW_TRAVEL * 1.3 / 1.6] * 3), False, 0.0))
cases.append(("one jaw held off the bar by 1 mm (bar off-centre)", done(jaws=[CONTACT - 0.001, CONTACT, CONTACT]),
              False, 0.0))
cases.append(("bar 1.2 mm off the spindle axis", done(dx_mm=1.2), False, 0.0))
cases.append(("bar 0.8 mm off the spindle axis", done(dx_mm=0.8), False, 1.0))
cases.append(("bar tipped 1.5 deg in the jaws", done(tilt_deg=1.5), False, 0.0))
# stick-out (the tool follows the bar's end, so only the stick-out is under test)
for s, want in [(54.0, 0.25), (55.5, 1.0), (64.5, 1.0), (66.0, 0.25), (40.0, 0.25)]:
    cases.append((f"stick-out {s} mm", done(stick_mm=s, **tool_at(2.0, 1.5, s)), False, want))
# the tool
for g, r, want, label in [
    (2.0, 1.5, 1.0, "tool 2 mm off the face, 1.5 mm outside"),
    (0.8, 1.5, 0.5, "tool 0.8 mm off the face"),
    (3.3, 1.5, 0.5, "tool 3.3 mm off the face"),
    (2.0, 3.4, 0.5, "tool 3.4 mm outside the diameter"),
    (2.0, -0.5, 0.5, "tool tip 0.5 mm inside the diameter"),
    (-1.0, 1.5, 0.5, "tool past the end face, beside the bar"),
    (-1.0, -2.0, 0.5, "tool pushed into the bar"),
    (40.0, 30.0, 0.5, "tool left where it started"),
]:
    cases.append((label, done(**tool_at(g, r)), False, want))

ok = True
for label, o, touched, want in cases:
    got = task.score(o, touched=bool(touched))
    succ = got == 1.0
    good = abs(got - want) < 1e-9 and succ == (want == 1.0)
    ok &= good
    print(f"{'ok ' if good else 'BAD'}  {got:.2f} (want {want:.2f})  {label}")

# the touch test
for g, r, want in [(2.0, 1.5, False), (0.2, -1.0, True), (-5.0, 0.1, True), (-5.0, 0.5, False), (1.0, -12.0, False),
                   (0.1, -25.0, True)]:
    got = task.tool_touches(done(**tool_at(g, r)))
    good = got == want
    ok &= good
    print(f"{'ok ' if good else 'BAD'}  touch={got} (want {want})  tip {g} mm off the face, {r} mm outside")

# the key-to-jaw scroll
for key, want in [(0.0, 0.0), (-0.8, 0.0055), (-1.6, task.JAW_TRAVEL), (-2.0, task.JAW_TRAVEL), (0.3, 0.0)]:
    got = task.jaw_setting(key)
    good = abs(got - want) < 1e-9
    ok &= good
    print(f"{'ok ' if good else 'BAD'}  key {key} rad -> jaws wound in {got * 1000:.2f} mm")

print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
