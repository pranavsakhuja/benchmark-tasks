"""On the station table: a bench-mounted digital caliper straight ahead of the robots, three identical-looking
aluminium bars on the table to its left, and two open boxes, one green (GO) and one red (NO-GO), beyond the bars.

The caliper: a long flat beam lying along the table's x axis (away from the robots). Its fixed jaw stands at the far
end of the beam; a sliding jaw, with a tall thumb tab behind it, runs along the beam on a light spring that keeps the
jaws wide open (55 mm). A small digital display on the far end of the beam shows the gap between the jaws in
millimetres, to 0.1 mm, live (the tops of its digits point away from the robots). Pushing the thumb tab away from the
robots closes the jaws; let go and the spring opens them again. A bar standing between the jaws, with the jaws closed
on its two ends, is measured: the display reads the bar's length.

The bars: three aluminium bars, each 20 mm wide and 26 mm tall, lying on the table with their lengths along x. They look
the same; they differ in length by at most 1.6 mm (39.2 to 40.8 mm). The drawing calls for 40.0 mm, plus or minus 0.5 mm: a
bar whose length is within that is GO, any other bar is NO-GO.

The task: measure each bar's length with the caliper and sort the three bars, the GO bars into the green box and the NO-GO
bars into the red box.

Run it:  inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \\
           -E control_hz=20 --scorer task_score --grader none --max-steps 3600 \\
           --instruction "<see task.json>"
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

STEEL = (0.74, 0.75, 0.78)  # the bars (bare aluminium)
CASE = (0.93, 0.93, 0.91)
DARK = (0.16, 0.17, 0.19)
SCREEN = (0.04, 0.05, 0.05)
LIT = (0.55, 1.00, 0.70)  # the display's lit segments
GREEN = (0.10, 0.55, 0.30)
RED = (0.72, 0.16, 0.14)
TAB = (0.95, 0.55, 0.10)  # the slider's thumb tab

# ---- the caliper (fixed to the table). Part frame: origin on the table at the fixed jaw's inner face, on the beam's
# ---- long axis; x away from the robots (the beam runs from -0.17 to +0.047), y to the robots' left.
CAL_X, CAL_Y = 0.30, -0.01  # world position of that origin
JAW_T, JAW_Y0, JAW_Y1, JAW_H = 0.008, 0.010, 0.050, 0.012  # jaws: 8 mm thick, 40 mm wide, 12 mm tall, beside the beam
BEAM_X0, BEAM_X1, BEAM_W, BEAM_H = -0.17, 0.047, 0.020, 0.006
OPEN_GAP = 0.055  # the slider rests with the jaws 55 mm apart
SPRING = 15.0  # N/m: 0.8 N with the jaws closed
CARRIAGE = (0.030, 0.026, 0.008)  # rides on the beam, behind the sliding jaw
TAB_SIZE = (0.010, 0.030, 0.026)  # the thumb tab behind the carriage: its back face is 48 mm behind the jaw face
TAB_BACK = JAW_T + CARRIAGE[0] + TAB_SIZE[0]  # 0.048
SLIDE_CLEAR = 0.003  # the sliding jaw's underside is 3 mm up: clear of the platen (2 mm) and the table
PLATEN_T = 0.002  # a polished platen between the jaws for the bar to slide on

# The slider's zero position is fully open (the spring's rest, and where every reset starts it); it travels up to 55 mm
# towards the fixed jaw, so the gap between the jaw faces is 55 mm minus its position.
SLIDER = Joint(
    "slider", kind="slide", axis="X", anchor=(-OPEN_GAP - 0.02, 0.0, 0.006), lower=0.0, upper=OPEN_GAP, initial=0.0,
    stiffness=SPRING, damping=2.0, target=0.0, mass=0.03,
    boxes=[
        ((-OPEN_GAP - JAW_T / 2, (JAW_Y0 + JAW_Y1) / 2, (JAW_H + SLIDE_CLEAR) / 2),
         (JAW_T, JAW_Y1 - JAW_Y0, JAW_H - SLIDE_CLEAR), DARK),  # sliding jaw, clear of the platen: it drags on nothing
        ((-OPEN_GAP - JAW_T - CARRIAGE[0] / 2, 0.0, BEAM_H + CARRIAGE[2] / 2), CARRIAGE, DARK),
        ((-OPEN_GAP - JAW_T - CARRIAGE[0] - TAB_SIZE[0] / 2, 0.0, BEAM_H + TAB_SIZE[2] / 2), TAB_SIZE, TAB),  # thumb tab
    ],
)

# Live display: four 7-segment digits 2 cm tall, tops towards +x (away from the robots), the decimal point after the
# second. Each segment is a box on its own sprung slide, driven up flush with the screen when lit and down out of
# sight into the housing when not. Reads the jaw gap in mm to 0.1 mm; the last digit is always blank.
HOUSING = (0.036, 0.076, 0.012)
DISP_X, DISP_Y = 0.065, 0.0  # screen centre
SCREEN_SIZE = (0.030, 0.068, 0.001)
FACE_Z = HOUSING[2] + SCREEN_SIZE[2]
DIG_H, DIG_W, SEG_T, DIG_PITCH = 0.020, 0.011, 0.0022, 0.0155
SEG_SUNK = -0.003
SEG_Z = FACE_Z + 0.0004 - 0.0005  # centre of a lit segment: its top 0.4 mm proud of the screen
SEGS = "abcdefg"
GLYPHS = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg", "6": "acdefg",
          "7": "abc", "8": "abcdefg", "9": "abcdfg", "-": "g", " ": ""}


def _digit_y(i: int) -> float:
    return DISP_Y + (1.5 - i) * DIG_PITCH  # digit 0 (tens) on the left


def _segment(i: int, s: str):
    """(centre xy, size xy) of segment s of digit i, in the caliper frame."""
    xc, yc = DISP_X, _digit_y(i)
    hl, vl = DIG_W - SEG_T - 0.0006, DIG_H / 2 - SEG_T / 2 - 0.0006  # horizontal and vertical segment lengths
    side = DIG_W / 2 - SEG_T / 2
    return {
        "a": ((xc + DIG_H / 2 - SEG_T / 2, yc), (SEG_T, hl)),
        "b": ((xc + DIG_H / 4, yc - side), (vl, SEG_T)),
        "c": ((xc - DIG_H / 4, yc - side), (vl, SEG_T)),
        "d": ((xc - DIG_H / 2 + SEG_T / 2, yc), (SEG_T, hl)),
        "e": ((xc - DIG_H / 4, yc + side), (vl, SEG_T)),
        "f": ((xc + DIG_H / 4, yc + side), (vl, SEG_T)),
        "g": ((xc, yc), (SEG_T, hl)),
    }[s]


def display_text(gap_mm: float) -> str:
    """The four characters the display shows for a jaw gap (mm): tens (blank if zero), units, tenths, blank."""
    v = min(max(round(float(gap_mm), 1), 0.0), 99.9)
    s = f"{v:04.1f}"
    return (" " if s[0] == "0" else s[0]) + s[1] + s[3] + " "


def _lit(text: str) -> list[float]:
    return [0.0 if s in GLYPHS[ch] else SEG_SUNK for ch in text for s in SEGS]


# Each segment's joint zero is its state in the starting reading (55.0), so the display is right from the first frame
# after a reset (the simulator starts every joint at zero). A segment lit at the start slides down from 0 to SEG_SUNK
# to go dark; one dark at the start sits SEG_SUNK down at zero and slides up by -SEG_SUNK to light.
_START = _lit(display_text(OPEN_GAP * 1000))


def _targets(text: str) -> list[float]:
    """Joint targets that show ``text``: each segment's lit/dark height less its height at joint zero."""
    return [q - q0 for q, q0 in zip(_lit(text), _START)]


SEGMENT_JOINTS = []
for _i in range(4):
    for _k, _s in enumerate(SEGS):
        (_cx, _cy), (_sx, _sy) = _segment(_i, _s)
        _z0 = SEG_Z + _START[7 * _i + _k]  # the segment's height at joint zero
        _lo, _hi = (SEG_SUNK, 0.0) if _START[7 * _i + _k] == 0.0 else (0.0, -SEG_SUNK)
        SEGMENT_JOINTS.append(Joint(
            f"seg{_i}{_s}", kind="slide", axis="Z", anchor=(_cx, _cy, _z0), lower=_lo, upper=_hi, initial=0.0,
            target=0.0, stiffness=200.0, damping=2.0, mass=0.002,
            boxes=[((_cx, _cy, _z0), (_sx, _sy, 0.001), LIT)],
        ))
_dot = (DISP_X - DIG_H / 2 + SEG_T / 2, (_digit_y(1) + _digit_y(2)) / 2)

CALIPER = part(
    "caliper", pos=(CAL_X, CAL_Y, TOP), fixed=True, friction=0.6,
    boxes=[
        (((BEAM_X0 + BEAM_X1) / 2, 0.0, BEAM_H / 2), (BEAM_X1 - BEAM_X0, BEAM_W, BEAM_H), CASE),  # beam
        ((JAW_T / 2, (JAW_Y0 + JAW_Y1) / 2, JAW_H / 2), (JAW_T, JAW_Y1 - JAW_Y0, JAW_H), DARK),  # fixed jaw
        ((DISP_X, DISP_Y, HOUSING[2] / 2), HOUSING, CASE),  # display housing
        ((DISP_X, DISP_Y, HOUSING[2] + SCREEN_SIZE[2] / 2), SCREEN_SIZE, SCREEN),
        ((_dot[0], _dot[1], SEG_Z), (SEG_T, SEG_T, 0.001), LIT),  # decimal point (always lit)
    ],
    joints=[SLIDER, *SEGMENT_JOINTS],
)

# The platen: a 2 mm polished steel plate on the table between the jaws, 75 mm long (up to the fixed jaw) and as wide as the
# jaws. A bar set on it slides almost freely, so the sliding jaw can push it up against the fixed jaw.
PLATEN = part(
    "platen", pos=(CAL_X, CAL_Y, TOP), fixed=True, friction=0.05,
    boxes=[((-0.0375, (JAW_Y0 + JAW_Y1) / 2, PLATEN_T / 2), (0.075, JAW_Y1 - JAW_Y0, PLATEN_T), (0.62, 0.64, 0.68))],
)

# ---- the bars (loose). Lengths along x, in mm; the drawing calls for 40.0 +- 0.5.
NOMINAL_MM, TOL_MM = 40.0, 0.5
BAR_W, BAR_H, BAR_KG = 0.020, 0.026, 0.056  # 20 x 26 mm section: 40 x 20 x 26 mm of aluminium is 56 g
BAR_LAYOUT = [  # name, centre x, centre y (m), length (mm)
    ("bar1", 0.23, 0.090, 40.3),  # GO, 0.2 mm inside the upper limit
    ("bar2", 0.23, 0.145, 40.8),  # NO-GO, 0.3 mm over
    ("bar3", 0.23, 0.200, 39.2),  # NO-GO, 0.3 mm under
]
BAR_NAMES = [b[0] for b in BAR_LAYOUT]
LENGTH_MM = {n: l for n, _, _, l in BAR_LAYOUT}
BARS = [
    part(n, pos=(x, y, TOP), mass=BAR_KG, friction=0.3,
         boxes=[((0.0, 0.0, BAR_H / 2), (l / 1000, BAR_W, BAR_H), STEEL)])
    for n, x, y, l in BAR_LAYOUT
]

# ---- the two boxes (fixed): an open tray each, 4 mm floor, 6 mm walls 30 mm tall, inside 9 cm (x) by 17 cm (y)
BOX_IN = (0.090, 0.170)
BOX_WALL, BOX_FLOOR, BOX_WALL_H = 0.006, 0.004, 0.030
GO_XY, NOGO_XY = (0.22, 0.335), (0.22, 0.530)


def _box(name, xy, color):
    wx, wy = BOX_IN[0] / 2 + BOX_WALL / 2, BOX_IN[1] / 2 + BOX_WALL / 2
    return part(
        name, pos=(*xy, TOP), fixed=True, friction=0.6,
        boxes=[
            ((0, 0, BOX_FLOOR / 2), (BOX_IN[0] + 2 * BOX_WALL, BOX_IN[1] + 2 * BOX_WALL, BOX_FLOOR), color),
            ((wx, 0, BOX_WALL_H / 2), (BOX_WALL, BOX_IN[1] + 2 * BOX_WALL, BOX_WALL_H), color),
            ((-wx, 0, BOX_WALL_H / 2), (BOX_WALL, BOX_IN[1] + 2 * BOX_WALL, BOX_WALL_H), color),
            ((0, wy, BOX_WALL_H / 2), (BOX_IN[0], BOX_WALL, BOX_WALL_H), color),
            ((0, -wy, BOX_WALL_H / 2), (BOX_IN[0], BOX_WALL, BOX_WALL_H), color),
        ],
    )


GO_BOX = _box("go_box", GO_XY, GREEN)
NOGO_BOX = _box("nogo_box", NOGO_XY, RED)

OBJECTS = [CALIPER, PLATEN, GO_BOX, NOGO_BOX, *BARS]

# ---- the live reading -----------------------------------------------------------------------------------------------
_trial = {"text": display_text(OPEN_GAP * 1000)}
_ids: dict = {}


@on_reset
def _clear(env):
    _trial.update(text=display_text(OPEN_GAP * 1000))
    _ids.pop("set", None)  # the segments restart at their defaults: write the display again


@on_step
def _watch(env):
    import torch

    s = labkit._part_state(env, labkit._SPECS["caliper"])
    text = display_text(jaw_gap_mm(s))
    asset = env.unwrapped.scene["caliper"]
    if "segments" not in _ids:
        names = list(asset.joint_names)
        _ids["segments"] = [names.index(j.joint_name) for j in SEGMENT_JOINTS]
    if text != _trial["text"] or "set" not in _ids:
        targets = torch.tensor([_targets(text)], device=asset.device)
        asset.set_joint_position_target(targets, joint_ids=_ids["segments"])
        _trial["text"], _ids["set"] = text, True


def jaw_gap_mm(caliper) -> float:
    """The gap between the jaw faces (mm): 55 mm open, less the slider's travel."""
    return (OPEN_GAP - caliper.joints["slider"]) * 1000.0


def reading_mm(objects) -> float:
    """What the display shows (mm): the jaw gap, to 0.1 mm."""
    return round(jaw_gap_mm(objects["caliper"]), 1)


# ---- scoring ----------------------------------------------------------------------------------------------------------
def is_go(name: str) -> bool:
    """True length within 40.0 +- 0.5 mm."""
    return abs(LENGTH_MM[name] - NOMINAL_MM) <= TOL_MM + 1e-9


def _inside(bar, xy, tol=0.001) -> bool:
    """The bar lies wholly inside the box's walls (world-aligned bounds), down in it, not held above it."""
    half = np.array(BOX_IN) / 2
    lo, hi = np.array(xy) - half - tol, np.array(xy) + half + tol
    return bool(np.all(bar.low[:2] >= lo) and np.all(bar.high[:2] <= hi) and bar.pos[2] <= TOP + BOX_WALL_H)


def sorted_into(objects, name: str) -> str | None:
    """"go", "nogo", or None (not in either box)."""
    if _inside(objects[name], GO_XY):
        return "go"
    if _inside(objects[name], NOGO_XY):
        return "nogo"
    return None


def wrong(objects) -> int:
    """Bars not in the box that matches their length: in the wrong box, or in neither."""
    return sum(sorted_into(objects, n) != ("go" if is_go(n) else "nogo") for n in BAR_NAMES)


def score(objects) -> float:
    """1.0 with all three bars in the right box; 0 with any bar wrong (in the wrong box, or on the table or anywhere
    else). No partial credit: with three bars, two share a box, so putting all three in that box would leave only one
    wrong without measuring anything."""
    w = wrong(objects)
    return 1.0 if w == 0 else 0.0


SUCCESS_HOLD_S = 3.0  # end a trial early once every bar has been in the right box for 3 s


def success(objects):
    return wrong(objects) == 0


# ---- scripted demo ----------------------------------------------------------------------------------------------------
# The left arm carries each bar to the caliper and sets it between the jaws, with its far end 4 mm short of the fixed
# jaw, and backs off beyond the caliper to watch the display; the right arm, fingers closed, pushes the thumb tab until the jaws close on the bar (the display reads its
# length), holds for 1.5 s, lets the spring open the jaws and backs off to its park pose; the left arm then takes the bar out and drops it in the
# box for its length. The demo knows the layout (which bar is where), as a scripted demo may.
TIP = 0.013  # pad tips below the TCP
GRIP_Z = TOP + TIP + 0.014  # pad tips 14 mm above the table: the pads hold the top 12 mm of a bar (the jaws stand 12 mm)
HOVER_Z = TOP + 0.16
CARRY_Z = TOP + 0.09  # a carried bar's underside is 63 mm up, well over the jaws, display and box walls
WIDE = -0.016  # 32 mm between the pads round the 20 mm bar
HOLD = -0.007
FINGER_FACE = 0.011  # the closed fingers' +x face past the TCP, measured here: the slider first moves at TCP x 0.186
PUSH_Z = TOP + 0.035  # TCP height for pushing the tab: pad tips 22 mm up the 26 mm tab, clear of the carriage
SLOT_DX = 0.004  # a bar is set 4 mm short of the fixed jaw
BAR_CY = CAL_Y + (JAW_Y0 + JAW_Y1) / 2 - 0.006  # the bar's centre line, 24 mm from the beam's, between the jaws
PUSH_CMD = 0.040  # the push is commanded to this slider travel; the shortest bar (39.2 mm) stops it at 15.8 mm, and the
#   arm trails its command by about 6.6 mm, so every bar stops the slider first
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
LEFT_WATCH = (0.34, 0.10, TOP + 0.15)  # beyond the caliper and to its left: clear of the right arm, display in view
TAB_X_OPEN = CAL_X - OPEN_GAP - TAB_BACK - FINGER_FACE - 0.006  # TCP x with the tab fully back, 6 mm clear
TAB_X_PUSH = CAL_X - OPEN_GAP - TAB_BACK - FINGER_FACE + PUSH_CMD


def _dur(wps):
    return sum(w.duration for w in wps)


def _hold_until(wps, t, label):
    """Keep the arm where it is until time t."""
    last = wps[-1]
    if t > _dur(wps):
        wps.append(Waypoint(t - _dur(wps), last.pos, last.finger, label, last.yaw))


def _pick_place(wps, pick_xy, place_xy, place_z, speed=1.0, pick_z=None):
    k = 1.0 / speed
    pick_z = GRIP_Z if pick_z is None else pick_z
    (px, py), (qx, qy) = pick_xy, place_xy
    wps += [
        Waypoint(1.6 * k, (px, py, HOVER_Z), WIDE, "above the bar"),
        Waypoint(1.4 * k, (px, py, pick_z), WIDE, "down beside the bar"),
        Waypoint(0.8 * k, (px, py, pick_z), HOLD, "grip it"),
        Waypoint(1.2 * k, (px, py, CARRY_Z), HOLD, "lift"),
        Waypoint(1.8 * k, (qx, qy, CARRY_Z), HOLD, "carry"),
        Waypoint(1.2 * k, (qx, qy, place_z), HOLD, "lower"),
        Waypoint(0.6 * k, (qx, qy, place_z), WIDE, "let go"),
        Waypoint(1.0 * k, (qx, qy, HOVER_Z), WIDE, "up"),
    ]


def demo():
    left = []
    right = [Waypoint(1.0, RIGHT_PARK, FINGER_OPEN, "start")]
    c = FINGER_CLOSED
    go_slot = nogo_slot = 0
    for name, bx, by, _ in BAR_LAYOUT:
        half = LENGTH_MM[name] / 2000.0
        set_xy = (CAL_X - SLOT_DX - half, BAR_CY)  # where the bar is set between the jaws
        clamped_xy = (CAL_X - half, BAR_CY)  # where it ends up once the jaws push it up to the fixed jaw
        _pick_place(left, (bx, by), set_xy, GRIP_Z + PLATEN_T)
        # The arms take turns over the caliper, so they never meet there: the left arm backs off beyond the caliper
        # (its wrist camera on the display) before the right arm comes in to work the tab, and the right arm backs off
        # to its park pose before the left comes back.
        left.append(Waypoint(1.4, LEFT_WATCH, WIDE, "back clear of the caliper, watching the display"))
        t_clear = _dur(left)
        # right arm: push the tab shut, read, let go
        _hold_until(right, t_clear, "wait for the left arm to clear")
        right += [
            Waypoint(1.8, (TAB_X_OPEN, CAL_Y, PUSH_Z + 0.045), c, "above the thumb tab"),
            Waypoint(1.0, (TAB_X_OPEN, CAL_Y, PUSH_Z), c, "beside the tab"),
            Waypoint(2.8, (TAB_X_PUSH, CAL_Y, PUSH_Z), c, "push the jaws shut"),
            Waypoint(1.5, (TAB_X_PUSH, CAL_Y, PUSH_Z), c, "hold while the display settles"),
            Waypoint(1.8, (TAB_X_OPEN, CAL_Y, PUSH_Z), c, "ease off: the spring opens the jaws"),
            Waypoint(0.8, (TAB_X_OPEN, CAL_Y, PUSH_Z + 0.045), c, "lift"),
            Waypoint(1.4, RIGHT_PARK, c, "back clear of the caliper"),
        ]
        t_free = _dur(right)
        # left arm: take the bar out and drop it in its box
        _hold_until(left, t_free, "wait for the right arm to clear")
        if is_go(name):
            box_x, box_y = GO_XY[0], GO_XY[1] + (go_slot - 1) * 0.05
            go_slot += 1
        else:
            box_x, box_y = NOGO_XY[0], NOGO_XY[1] + (nogo_slot - 1) * 0.05
            nogo_slot += 1
        _pick_place(left, clamped_xy, (box_x, box_y), TOP + BOX_FLOOR + TIP + 0.014 + 0.002, pick_z=GRIP_Z + PLATEN_T)
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park"))
    left.append(Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "park"))
    t_end = max(_dur(left), _dur(right)) + 1.0
    _hold_until(left, t_end, "rest")
    _hold_until(right, t_end, "rest")
    return {"left": left, "right": right}
