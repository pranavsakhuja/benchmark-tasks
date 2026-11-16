"""On the station table: a small bench lathe straight ahead of the robots, powered off, with its spindle axis running
left to right (along y) 60 mm above the table and its headstock on the robots' left. A 20 mm round aluminium bar
140 mm long lies in a cradle on the table to the robots' right, its length along y.

The chuck: an 80 mm three-jaw chuck on the spindle, its face towards the robots' right. Its jaws are open, 30 mm
apart, wider than the bar; behind them the spindle bore takes up to 100 mm of bar. A chuck key stands in the
chuck's top socket with its T-handle on top: turning the handle clockwise (seen from above) closes the three jaws
together, anticlockwise opens them. The key stays where it is left.

The carriage and tool: the carriage rides on the lathe bed along the spindle axis and carries a cross-slide that
moves towards and away from the axis (along x). A square turning tool in the tool post on the cross-slide points
away from the robots, its tip at spindle height. Each slide has a grab knob (the carriage's on its front, nearest
the robots; the cross-slide's at its back end) and stays where it is left. At the start the tool is well back
from the axis and 180 mm to the right of the chuck.

The task: clamp the bar in the chuck with 55 to 65 mm of it sticking out of the jaws, then move the tool so its tip
is 1 to 3 mm from the bar's end face and 0 to 3 mm outside the bar's diameter, ready to face the end, without the
tool touching the bar.

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
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

CAST = (0.30, 0.42, 0.36)  # the lathe's painted castings
STEEL = (0.62, 0.63, 0.66)  # chuck body and slides
JAW = (0.40, 0.41, 0.44)
KEY = (0.15, 0.15, 0.17)
KNOB = (0.08, 0.08, 0.09)
TOOL = (0.85, 0.70, 0.25)  # the tool bit (gold, like a coated insert)
ALU = (0.80, 0.81, 0.83)  # the bar
WOOD = (0.55, 0.40, 0.25)  # the cradle

# ---- the lathe (fixed). Part frame: origin on the table under the spindle axis, at world x AX; part y = world y.
AX = 0.30  # world x of the spindle axis
H = 0.060  # spindle axis height above the table
CY = 0.060  # world y of the jaws' front faces (the chuck face looks towards -y, the robots' right)

# Chuck body: an 80 mm disc 40 mm deep behind the jaws, with a 32 mm bore; drawn as a ring of 12 boxes.
CHUCK_R, BORE_R = 0.040, 0.016
BODY_Y0, BODY_Y1 = CY + 0.012, CY + 0.052
RING_N = 12

# Jaws: three, 120 deg apart (at 6, 2 and 10 o'clock seen from the robots' right), each 12 mm deep radially, 10 mm wide
# and 25 mm long along the axis. Each slides radially inward from fully open (inner faces 15 mm from the axis).
JAW_ANGLES = (-math.pi / 2, math.pi / 6, 5 * math.pi / 6)  # in the x-z plane, from +x towards +z
JAW_OPEN_R, JAW_DEPTH, JAW_W, JAW_LEN = 0.015, 0.012, 0.010, 0.025
JAW_TRAVEL = 0.011  # fully closed: inner faces 4 mm from the axis
JAW_K, JAW_DAMP = 3000.0, 40.0  # N/m: the jaws (geared together) are sprung towards the position the key sets

# Chuck key: in a socket on top of the chuck, between the 10 and 2 o'clock jaws. Hinge about the vertical; turning
# clockwise seen from above (negative about z) closes the jaws, by JAW_TRAVEL over KEY_TRAVEL.
KEY_Y = CY + 0.035
KEY_Z0 = H + CHUCK_R
KEY_SHAFT_H = 0.028
KEY_T_Z = KEY_Z0 + KEY_SHAFT_H + 0.005  # centre of the T-handle
KEY_T_LEN, KEY_T_SEC = 0.060, 0.010
KEY_YAW0 = 0.5  # the handle's direction at the start (rad from +x, anticlockwise from above)
KEY_TRAVEL = 1.6  # rad of key turn from fully open to fully closed

# Headstock behind the chuck: a block with the spindle bore through it, closed by a stop 100 mm behind the jaw faces.
HEAD_Y0, HEAD_Y1, HEAD_HALF_X, HEAD_TOP = BODY_Y1, CY + 0.170, 0.050, H + 0.035
HEAD_BORE = 0.018  # half-width of the square bore through the headstock
STOP_Y = CY + 0.100
BED_Y0, BED_HALF_X, BED_H = -0.25, 0.035, 0.012

# Carriage (slides along y on the bed) and cross-slide (slides along x on the carriage), with the tool. Positions are
# given for the tool tip: the tip is the tool bit's corner nearest the chuck (+y) and the axis (+x), at axis height.
TIP_X0, TIP_Y0 = AX - 0.045, -0.120  # at the start: 45 mm out from the axis, 180 mm from the jaw faces
CARR_MAX, CROSS_MAX = CY - 0.010 - TIP_Y0, AX + 0.005 - TIP_X0  # travel limits: never into the jaws, 5 mm past the axis
TOOL_SEC, TOOL_LEN = 0.008, 0.040
SADDLE_Z0, SADDLE_Z1 = BED_H, BED_H + 0.014
SLIDE_Z1 = SADDLE_Z1 + 0.010
POST_Z1 = H + 0.010
CARR_KNOB = (0.180, TIP_Y0 + 0.030)  # world xy of the carriage knob at the start
CROSS_KNOB = (TIP_X0 - 0.075, TIP_Y0 - 0.020)  # world xy of the cross-slide knob at the start
KNOB_R = 0.006
CARR_KNOB_TOP, CROSS_KNOB_TOP = SADDLE_Z1 + 0.035, SLIDE_Z1 + 0.034


def _lx(x):  # world x -> lathe-part x
    return x - AX


def _ring():
    """The chuck body: 12 boxes round the bore, each turned about the axis (part y) by pitch -theta."""
    rm, t = (CHUCK_R + BORE_R) / 2, CHUCK_R - BORE_R
    w = 2 * CHUCK_R * math.tan(math.pi / RING_N) + 0.001
    out = []
    for k in range(RING_N):
        th = 2 * math.pi * k / RING_N + math.pi / RING_N  # no segment straight up: the key socket sits there
        out.append(((rm * math.cos(th), (BODY_Y0 + BODY_Y1) / 2, H + rm * math.sin(th)), (t, BODY_Y1 - BODY_Y0, w),
                    STEEL, (0.0, -th)))
    return out


def _jaw(i, th):  # jaws 1 and 2 are geared to jaw 0, as a scroll gears them
    rc = JAW_OPEN_R + JAW_DEPTH / 2  # jaw centre radius, fully open
    c = (rc * math.cos(th), CY + JAW_LEN / 2, H + rc * math.sin(th))
    return Joint(
        f"jaw{i}", kind="slide", axis="X", pitch=-(th + math.pi), anchor=c, lower=0.0, upper=JAW_TRAVEL,
        initial=0.0, target=0.0, stiffness=JAW_K, damping=JAW_DAMP, mass=0.03, mimic="jaw0" if i else "",
        boxes=[(c, (JAW_DEPTH, JAW_LEN, JAW_W), JAW, (0.0, -th))],
    )


JAWS = [_jaw(i, th) for i, th in enumerate(JAW_ANGLES)]

KEY_JOINT = Joint(
    "key", kind="hinge", axis="Z", anchor=(0.0, KEY_Y, KEY_Z0), lower=-KEY_TRAVEL, upper=0.0, initial=0.0,
    target=0.0, stiffness=0.0, damping=0.3, mass=0.04,
    boxes=[((0.0, KEY_Y, KEY_T_Z), (KEY_T_LEN, KEY_T_SEC, KEY_T_SEC), KEY, KEY_YAW0)],
    cylinders=[((0.0, KEY_Y, KEY_Z0 + KEY_SHAFT_H / 2), 0.004, KEY_SHAFT_H, KEY, "Z")],
)

CARRIAGE = Joint(
    "carriage", kind="slide", axis="Y", anchor=(_lx(AX - 0.04), TIP_Y0, SADDLE_Z1), lower=0.0, upper=CARR_MAX,
    initial=0.0, target=0.0, stiffness=0.0, damping=40.0, mass=0.30,
    boxes=[((_lx((0.165 + 0.345) / 2), TIP_Y0 + 0.0025, (SADDLE_Z0 + SADDLE_Z1) / 2),
            (0.345 - 0.165, 0.095, SADDLE_Z1 - SADDLE_Z0), STEEL)],
    cylinders=[((_lx(CARR_KNOB[0]), CARR_KNOB[1], (SADDLE_Z1 + CARR_KNOB_TOP) / 2), KNOB_R,
                CARR_KNOB_TOP - SADDLE_Z1, KNOB, "Z")],
)

CROSS = Joint(
    "cross", kind="slide", axis="X", parent="carriage", anchor=(_lx(TIP_X0 - 0.03), TIP_Y0, SLIDE_Z1),
    lower=0.0, upper=CROSS_MAX, initial=0.0, target=0.0, stiffness=0.0, damping=30.0, mass=0.15,
    boxes=[
        ((_lx((TIP_X0 - 0.085 + TIP_X0 + 0.010) / 2), TIP_Y0 - 0.010, (SADDLE_Z1 + SLIDE_Z1) / 2),
         (0.095, 0.040, SLIDE_Z1 - SADDLE_Z1), STEEL),  # the cross-slide
        ((_lx(TIP_X0 - 0.031), TIP_Y0 - 0.005, (SLIDE_Z1 + POST_Z1) / 2),
         (0.024, 0.018, POST_Z1 - SLIDE_Z1), KNOB),  # the tool post
        ((_lx(TIP_X0 - TOOL_LEN / 2), TIP_Y0 - TOOL_SEC / 2, H), (TOOL_LEN, TOOL_SEC, TOOL_SEC), TOOL),  # the tool bit
    ],
    cylinders=[((_lx(CROSS_KNOB[0]), CROSS_KNOB[1], (SLIDE_Z1 + CROSS_KNOB_TOP) / 2), KNOB_R,
                CROSS_KNOB_TOP - SLIDE_Z1, KNOB, "Z")],
)


def _headstock():
    yc, ly = (HEAD_Y0 + HEAD_Y1) / 2, HEAD_Y1 - HEAD_Y0
    zb0, zb1 = H - HEAD_BORE, H + HEAD_BORE
    return [
        ((0.0, yc, zb0 / 2), (2 * HEAD_HALF_X, ly, zb0), CAST),  # below the bore
        ((0.0, yc, (zb1 + HEAD_TOP) / 2), (2 * HEAD_HALF_X, ly, HEAD_TOP - zb1), CAST),  # above it
        ((-(HEAD_BORE + HEAD_HALF_X) / 2, yc, H), (HEAD_HALF_X - HEAD_BORE, ly, 2 * HEAD_BORE), CAST),  # beside it
        ((+(HEAD_BORE + HEAD_HALF_X) / 2, yc, H), (HEAD_HALF_X - HEAD_BORE, ly, 2 * HEAD_BORE), CAST),
        ((0.0, (STOP_Y + HEAD_Y1) / 2, H), (2 * HEAD_BORE, HEAD_Y1 - STOP_Y, 2 * HEAD_BORE), CAST),  # the bore stop
    ]


# Friction 0.15 on every lathe surface, and "min" so it holds against the bar's 0.9: the jaws slide a bar that is
# released off-centre to the middle instead of jamming it against two of them.
LATHE = part(
    "lathe", pos=(AX, 0.0, TOP), fixed=True, friction=0.15, friction_combine="min",
    boxes=[
        ((0.0, (BED_Y0 + HEAD_Y1) / 2, BED_H / 2), (2 * BED_HALF_X, HEAD_Y1 - BED_Y0, BED_H), CAST),  # the bed
        *_headstock(),
        *_ring(),
        ((0.0, KEY_Y, KEY_Z0 - 0.002), (0.012, 0.012, 0.006), STEEL),  # the key socket's boss
    ],
    joints=[*JAWS, KEY_JOINT, CARRIAGE, CROSS],
)

# ---- the bar (loose): 20 mm round aluminium, 140 mm long, lying along y in a cradle
BAR_R, BAR_L = 0.010, 0.140
BAR_KG = 2700 * math.pi * BAR_R**2 * BAR_L  # 0.119 kg
BAR_XY = (0.22, -0.29)  # centre at the start
# Simulated as a 12-sided prism, 20 mm across flats (its corners 0.35 mm proud of a 20 mm circle): six boxes, one per
# pair of opposite flats. Flats face the gripper pads (along x), the table and each jaw, so the pads hold it face to
# face; a true cylinder touches each flat pad along one line and pivots between them.
BAR_SIDE = 2 * BAR_R * math.tan(math.pi / 12)
BAR = part("bar", pos=(*BAR_XY, TOP + BAR_R), mass=BAR_KG, friction=0.9,
           boxes=[((0.0, 0.0, 0.0), (2 * BAR_R, BAR_L, BAR_SIDE + 0.0002), ALU, (0.0, -k * math.pi / 6))
                  for k in range(6)])

# The cradle: two wooden rails 5 mm tall, 64 mm apart (inside), either side of the bar, so it can't roll away.
RAIL_GAP, RAIL_W, RAIL_H, RAIL_L = 0.064, 0.010, 0.005, 0.150
CRADLE = part(
    "cradle", pos=(*BAR_XY, TOP), fixed=True, friction=0.6,
    boxes=[((s * (RAIL_GAP + RAIL_W) / 2, 0.0, RAIL_H / 2), (RAIL_W, RAIL_L, RAIL_H), WOOD) for s in (-1, 1)],
)

OBJECTS = [LATHE, CRADLE, BAR]


# ---- the chuck's scroll: the key sets where the jaws go --------------------------------------------------------------
def jaw_setting(key_angle: float) -> float:
    """How far (m) the key has wound the jaws in from fully open."""
    return JAW_TRAVEL * min(max(-key_angle, 0.0), KEY_TRAVEL) / KEY_TRAVEL


_trial = {"touched": False}
_ids: dict = {}


@on_reset
def _reset(env):
    _trial["touched"] = False
    _ids.pop("set", None)


@on_step
def _step(env):
    import torch

    states = {
        "lathe": labkit._part_state(env, labkit._SPECS["lathe"]),
        "bar": labkit._part_state(env, labkit._SPECS["bar"]),
    }
    asset = env.unwrapped.scene["lathe"]
    if "jaws" not in _ids:
        names = list(asset.joint_names)
        _ids["jaws"] = [names.index(j.joint_name) for j in JAWS]
    q = jaw_setting(states["lathe"].joints["key"])
    asset.set_joint_position_target(torch.tensor([[q] * len(JAWS)], device=asset.device), joint_ids=_ids["jaws"])
    if tool_touches(states):
        _trial["touched"] = True


# ---- measuring --------------------------------------------------------------------------------------------------------
def tool_tip(lathe) -> np.ndarray:
    """World position of the tool tip."""
    j = lathe.joints
    return np.array([TIP_X0 + j["cross"], TIP_Y0 + j["carriage"], TOP + H])


def bar_axis(bar):
    """(centre, unit direction along the bar)."""
    return np.asarray(bar.pos, float), bar.rotation() @ np.array([0.0, 1.0, 0.0])


def bar_free_end(bar) -> np.ndarray:
    """Centre of the bar's end face nearer the robots' right (-y): the end that sticks out of the chuck."""
    c, d = bar_axis(bar)
    a, b = c + d * BAR_L / 2, c - d * BAR_L / 2
    return a if a[1] < b[1] else b


def stick_out_mm(objects) -> float:
    """How far the bar's free end stands out from the jaws' front faces, along the spindle axis (mm)."""
    return (CY - bar_free_end(objects["bar"])[1]) * 1000.0


def off_axis_mm(objects) -> tuple[float, float]:
    """(how far the bar's axis is from the spindle axis at the jaw faces, mm; its tilt from the spindle axis, deg)."""
    c, d = bar_axis(objects["bar"])
    if abs(d[1]) < 1e-6:
        return 1e9, 90.0
    p = c + d * (CY - c[1]) / d[1]  # where the bar's axis crosses the jaw-face plane
    off = math.hypot(p[0] - AX, p[2] - (TOP + H)) * 1000.0
    tilt = math.degrees(math.acos(min(1.0, abs(d[1]))))
    return off, tilt


def jaw_forces(objects) -> list[float]:
    """Each jaw's grip on whatever stops it (N): its spring's pull past where it stands."""
    j = objects["lathe"].joints
    q = jaw_setting(j["key"])
    return [JAW_K * max(q - j[f"jaw{i}"], 0.0) for i in range(len(JAWS))]


CLAMP_N = 4.0  # each jaw must bear on the bar with at least this force
JAW_ON_BAR_MM = 0.7  # each jaw within this of the bar's surface
CENTRE_MM, TILT_DEG = 1.0, 1.0


def clamped(objects) -> bool:
    """The bar is held by all three jaws, tightened, and running true: every jaw stands within 0.7 mm of the bar's
    surface and bears on it with at least 4 N; the bar's axis is within 1 mm of the spindle axis at the jaw faces and
    within 1 deg of it."""
    j = objects["lathe"].joints
    contact = JAW_OPEN_R - BAR_R  # jaw travel at which a jaw meets a centred bar
    on_bar = all(abs(j[f"jaw{i}"] - contact) * 1000.0 <= JAW_ON_BAR_MM for i in range(len(JAWS)))
    tight = all(f >= CLAMP_N for f in jaw_forces(objects))
    off, tilt = off_axis_mm(objects)
    return on_bar and tight and off <= CENTRE_MM and tilt <= TILT_DEG


STICK_OUT_MM = (55.0, 65.0)


def stick_out_ok(objects) -> bool:
    lo, hi = STICK_OUT_MM
    return lo <= stick_out_mm(objects) <= hi


def tool_offsets_mm(objects) -> tuple[float, float]:
    """(axial gap from the tool tip to the bar's end face, along the bar; radial distance of the tip outside the bar's
    diameter), mm. Negative axial gap: the tip is past the end face."""
    tip = tool_tip(objects["lathe"])
    c, d = bar_axis(objects["bar"])
    end = bar_free_end(objects["bar"])
    out = (end - c) / np.linalg.norm(end - c)  # unit vector out of the free end
    axial = float(np.dot(tip - end, out))
    radial = float(np.linalg.norm((tip - end) - axial * out)) - BAR_R
    return axial * 1000.0, radial * 1000.0


GAP_MM = (1.0, 3.0)
RADIAL_MM = (0.0, 3.0)


def tool_ready(objects) -> bool:
    axial, radial = tool_offsets_mm(objects)
    return GAP_MM[0] <= axial <= GAP_MM[1] and RADIAL_MM[0] <= radial <= RADIAL_MM[1]


TOUCH_MM = 0.3
# Points over the tool bit's surface, relative to its tip: every 1 mm along its length, on its four long edges and
# its four face centre lines.
_BIT = np.array([(-dx, dy, dz) for dx in np.arange(0.0, TOOL_LEN + 1e-9, 0.001)
                 for dy, dz in [(0.0, -0.004), (0.0, 0.0), (0.0, 0.004), (-0.004, -0.004), (-0.004, 0.004),
                                (-0.008, -0.004), (-0.008, 0.0), (-0.008, 0.004)]])


def tool_touches(objects) -> bool:
    """Any point of the tool bit within 0.3 mm of the bar (inside its diameter plus 0.3 mm, and between its end faces
    with 0.3 mm to spare)."""
    pts = tool_tip(objects["lathe"]) + _BIT
    c, d = bar_axis(objects["bar"])
    rel = pts - c
    t = rel @ d
    radial = np.linalg.norm(rel - np.outer(t, d), axis=1)
    tol = TOUCH_MM / 1000.0
    return bool(np.any((np.abs(t) <= BAR_L / 2 + tol) & (radial <= BAR_R + tol)))


# ---- scoring ----------------------------------------------------------------------------------------------------------
def score(objects, touched: bool | None = None) -> float:
    """0.25 clamped (all three jaws tight on the bar, running true); 0.5 clamped with 55 to 65 mm sticking out;
    1.0 also the tool tip 1 to 3 mm from the end face and 0 to 3 mm outside the diameter, the tool never having touched
    the bar. Otherwise 0."""
    touched = _trial["touched"] if touched is None else touched
    if not clamped(objects):
        return 0.0
    if not stick_out_ok(objects):
        return 0.25
    if touched or tool_touches(objects) or not tool_ready(objects):
        return 0.5
    return 1.0


SUCCESS_HOLD_S = 2.0


def success(objects):
    return score(objects) == 1.0


# ---- scripted demo ----------------------------------------------------------------------------------------------------
# The right arm picks the bar out of its cradle 20 mm from one end, carries it to the chuck, lines it up on the spindle
# axis and slides it into the open jaws until 60 mm sticks out, lets go (the bar rests on the bottom jaw) and backs off.
# The left arm grips the key's T-handle and turns it 75 deg clockwise, closing the jaws on the bar, lets go and parks.
# The right arm then drags the carriage knob to bring the tool to 2 mm from the bar's end, and the cross-slide knob to
# bring the tip to 1.5 mm outside the diameter. The demo knows the layout, as a scripted demo may.
TIP = 0.013  # pad tips below the TCP
HOVER = TOP + 0.16
STICK_OUT = 0.060
GRIP_FROM_END = 0.020
BAR_GRIP_SLIP = 0.004  # the bar slides 4 mm along its length, away from the chuck, as the pads close and lift it
BAR_AIM_DX = 0.0024  # held out over the chuck, the arm puts the bar 2.4 mm short of its command in x
BAR_HOLD_DZ = TIP - 0.004  # TCP above the bar's centre: the pads close on it 4 mm below its centre line
BAR_WIDE, BAR_HOLD = -0.020, -0.004  # 40 mm open; 8 mm commanded on the 20 mm bar: 30 N a pad, so the bar,
#   held 50 mm from its centre of mass, does not pivot between the pads
SMALL_WIDE, SMALL_HOLD = -0.014, -0.002  # for the key's 10 mm handle and the 12 mm knobs
ALONG_X = math.pi / 2  # grip yaw closing the fingers along x
KEY_TURN = 1.30  # rad: the jaws meet the bar at 0.73 rad, and every jaw bears with 12 N at 1.30
GAP_TARGET, RADIAL_TARGET = 0.002, 0.0015
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
RIGHT_CLEAR = (0.18, -0.16, TOP + 0.16)  # out of the left arm's way while it turns the key


def _dur(wps):
    return sum(w.duration for w in wps)


def _hold_until(wps, t, label):
    last = wps[-1]
    if t > _dur(wps):
        wps.append(Waypoint(t - _dur(wps), last.pos, last.finger, label, last.yaw))


def _drag_knob(wps, xy, top, delta, yaw):
    """Grip a knob from above (pad tips 15 mm below its top) and drag it by delta (dx, dy)."""
    z = top - 0.015 + TIP
    x, y = xy
    tx, ty = x + delta[0], y + delta[1]
    wps += [
        Waypoint(1.8, (x, y, HOVER), SMALL_WIDE, "above the knob", yaw),
        Waypoint(1.4, (x, y, z), SMALL_WIDE, "down round the knob", yaw),
        Waypoint(0.8, (x, y, z), SMALL_HOLD, "grip the knob", yaw),
        Waypoint(4.0, (tx, ty, z), SMALL_HOLD, "drag the slide", yaw),
        Waypoint(1.0, (tx, ty, z), SMALL_HOLD, "settle", yaw),
        Waypoint(0.8, (tx, ty, z), SMALL_WIDE, "let go", yaw),
        Waypoint(1.2, (tx, ty, HOVER), SMALL_WIDE, "up", yaw),
    ]


def demo():
    right, left = [], [Waypoint(1.0, LEFT_PARK, FINGER_OPEN, "start")]
    # -- right arm: load the bar
    bx, by = BAR_XY
    grip_y0 = by - BAR_L / 2 + GRIP_FROM_END  # 20 mm from the bar's -y end
    pick_z = TOP + BAR_R + BAR_HOLD_DZ
    axis_z = TOP + H + BAR_HOLD_DZ + 0.0005  # the bar's centre 0.5 mm over the axis: its far end droops a little
    aim_x = AX + BAR_AIM_DX
    grip_in = CY - STICK_OUT + GRIP_FROM_END + BAR_GRIP_SLIP  # grip point with the bar fully in
    grip_pre = grip_in - 0.095  # the bar's inner end 15 mm short of the jaws
    right += [
        Waypoint(1.8, (bx, grip_y0, HOVER), BAR_WIDE, "above the bar", ALONG_X),
        Waypoint(1.6, (bx, grip_y0, pick_z), BAR_WIDE, "down beside the bar", ALONG_X),
        Waypoint(0.8, (bx, grip_y0, pick_z), BAR_HOLD, "grip it", ALONG_X),
        Waypoint(1.4, (bx, grip_y0, HOVER - 0.03), BAR_HOLD, "lift", ALONG_X),
        Waypoint(2.4, (aim_x, grip_pre, HOVER - 0.03), BAR_HOLD, "carry to the chuck", ALONG_X),
        Waypoint(1.8, (aim_x, grip_pre, axis_z), BAR_HOLD, "down onto the spindle axis", ALONG_X),
        Waypoint(0.6, (aim_x, grip_pre, axis_z), BAR_HOLD, "steady", ALONG_X),
        Waypoint(3.5, (aim_x, grip_in, axis_z), BAR_HOLD, "slide it into the jaws", ALONG_X),
        Waypoint(0.6, (aim_x, grip_in, axis_z), BAR_HOLD, "steady", ALONG_X),
        Waypoint(0.8, (aim_x, grip_in, axis_z), BAR_WIDE, "let go", ALONG_X),
        Waypoint(1.2, (AX, grip_in, TOP + 0.12), BAR_WIDE, "up", ALONG_X),  # higher here is past joint4's limit
        Waypoint(1.0, (AX - 0.06, grip_in - 0.08, TOP + 0.13), BAR_WIDE, "back off", ALONG_X),
        Waypoint(1.6, RIGHT_CLEAR, BAR_WIDE, "back clear of the chuck", 0.0),
    ]
    # -- left arm: tighten the chuck
    _hold_until(left, _dur(right), "wait for the bar")
    key_z = TOP + KEY_T_Z + TIP - 0.005  # pad tips 5 mm below the handle's centre line
    kx, ky = AX, KEY_Y
    left += [
        Waypoint(1.8, (kx, ky, HOVER), SMALL_WIDE, "above the key", KEY_YAW0),
        Waypoint(1.4, (kx, ky, key_z), SMALL_WIDE, "down round the handle", KEY_YAW0),
        Waypoint(0.8, (kx, ky, key_z), SMALL_HOLD, "grip the handle", KEY_YAW0),
        Waypoint(3.5, (kx, ky, key_z), SMALL_HOLD, "turn it clockwise: the jaws close", KEY_YAW0 - KEY_TURN),
        Waypoint(1.0, (kx, ky, key_z), SMALL_HOLD, "hold", KEY_YAW0 - KEY_TURN),
        Waypoint(0.8, (kx, ky, key_z), SMALL_WIDE, "let go", KEY_YAW0 - KEY_TURN),
        Waypoint(1.2, (kx, ky, HOVER), SMALL_WIDE, "up", KEY_YAW0 - KEY_TURN),
        Waypoint(1.8, LEFT_PARK, FINGER_OPEN, "park", 0.0),
    ]
    # -- right arm: bring the tool up
    _hold_until(right, _dur(left) - 1.8, "wait for the key")
    end_y = CY - STICK_OUT  # the bar's free end
    carr = end_y - GAP_TARGET - TIP_Y0
    cross = AX - BAR_R - RADIAL_TARGET - TIP_X0
    _drag_knob(right, CARR_KNOB, TOP + CARR_KNOB_TOP, (0.0, carr), 0.0)
    _drag_knob(right, (CROSS_KNOB[0], CROSS_KNOB[1] + carr), TOP + CROSS_KNOB_TOP, (cross, 0.0), ALONG_X)
    right.append(Waypoint(2.0, RIGHT_PARK, FINGER_OPEN, "park", 0.0))
    t_end = max(_dur(left), _dur(right)) + 1.0
    _hold_until(left, t_end, "rest")
    _hold_until(right, t_end, "rest")
    return {"left": left, "right": right}
