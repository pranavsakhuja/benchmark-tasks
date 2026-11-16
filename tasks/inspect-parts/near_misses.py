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


def bar_at(name, x, y, z_base=task.TOP + task.BOX_FLOOR, along_y=False):
    """A bar lying flat with its base at z_base, centred at (x, y), long axis along x (or y)."""
    l = task.LENGTH_MM[name] / 1000
    sx, sy = (task.BAR_W, l) if along_y else (l, task.BAR_W)
    lo = np.array([x - sx / 2, y - sy / 2, z_base])
    hi = np.array([x + sx / 2, y + sy / 2, z_base + task.BAR_H])
    return PartState(pos=np.array([x, y, z_base]), quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=Z, ang_vel=Z,
                     low=lo, high=hi)


def start():
    return {n: bar_at(n, x, y, task.TOP) for n, x, y, _ in task.BAR_LAYOUT}


def sort(assign):
    """assign: name -> "go" / "nogo" / None (left where it started)."""
    objs, slots = start(), {"go": 0, "nogo": 0}
    for n, where in assign.items():
        if where is None:
            continue
        cx, cy = task.GO_XY if where == "go" else task.NOGO_XY
        objs[n] = bar_at(n, cx, cy - 0.06 + 0.03 * slots[where])  # up to five bars, 3 cm apart
        slots[where] += 1
    return objs


truth = {n: ("go" if task.is_go(n) else "nogo") for n in task.BAR_NAMES}
GO = [n for n in task.BAR_NAMES if task.is_go(n)]
NOGO = [n for n in task.BAR_NAMES if not task.is_go(n)]
assert GO == ["bar1"] and NOGO == ["bar2", "bar3"], (GO, NOGO)

cases = []
cases.append(("untouched", start(), 0.0))
cases.append(("all correct", sort(truth), 1.0))
cases.append(("the GO bar in the red box (all three in red, unmeasured)", sort({**truth, "bar1": "nogo"}), 0.0))
cases.append(("one NO-GO bar in the green box", sort({**truth, "bar3": "go"}), 0.0))
cases.append(("one left on the table", sort({**truth, "bar2": None}), 0.0))
cases.append(("two swapped", sort({**truth, "bar1": "nogo", "bar2": "go"}), 0.0))
cases.append(("all in green", sort({n: "go" for n in task.BAR_NAMES}), 0.0))
cases.append(("all in red", sort({n: "nogo" for n in task.BAR_NAMES}), 0.0))
cases.append(("nearest misses swapped (40.3 GO, 40.8 NO-GO)", sort({**truth, "bar1": "nogo", "bar2": "go"}), 0.0))
held = sort(truth)
held["bar3"] = bar_at("bar3", *task.GO_XY, z_base=task.TOP + task.BOX_WALL_H + 0.01)  # held over the green box
cases.append(("one held above its box", held, 0.0))
rim = sort(truth)
rim["bar1"] = bar_at("bar1", task.GO_XY[0] + task.BOX_IN[0] / 2, task.GO_XY[1])  # straddling the wall
cases.append(("one straddling the box wall", rim, 0.0))
upright = sort(truth)  # a bar dropped on end inside its box still counts: it is inside, below the rim
l = task.LENGTH_MM["bar1"] / 1000
upright["bar1"] = PartState(pos=np.array([*task.GO_XY, task.TOP + task.BOX_FLOOR]), quat_xyzw=np.array([0, 0, 0, 1.0]),
                            lin_vel=Z, ang_vel=Z, low=np.array([task.GO_XY[0] - 0.013, task.GO_XY[1] - 0.01, 0]),
                            high=np.array([task.GO_XY[0] + 0.013, task.GO_XY[1] + 0.01, task.TOP + task.BOX_FLOOR + l]))
cases.append(("one standing on end in its box", upright, 1.0))
across = sort(truth)
across["bar1"] = bar_at("bar1", *task.GO_XY, along_y=True)
cases.append(("one turned across its box", across, 1.0))

ok = True
for label, objs, want in cases:
    got = task.score(objs)
    good = abs(got - want) < 1e-9 and task.success(objs) == (want == 1.0)
    ok &= good
    print(f"{'ok ' if good else 'BAD'}  {got:.1f} (want {want:.1f})  {label}")
# the display
for gap, text in [(55.0, "55.0"), (40.0, "40.0"), (39.84, "39.8"), (40.26, "40.3"), (9.5, " 9.5")]:
    shown = task.display_text(gap)
    good = shown[:3] == text.replace(".", "")[:3] and shown[3] == " "
    ok &= good
    print(f"{'ok ' if good else 'BAD'}  display {gap} mm -> '{shown[:2]}.{shown[2]}'")
print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
