# Clamp a bar in a lathe's three-jaw chuck and bring the tool up to its end face

A 20 mm aluminium bar lies in a cradle beside a small bench lathe, which is off with its spindle locked. The robot
loads the bar into the open three-jaw chuck with 55 to 65 mm sticking out, then tightens the chuck with the key in
its top socket. Next it moves the carriage and the cross-slide until the turning tool's tip is 1 to 3 mm from the
bar's end face and just outside its diameter, ready to face the end, without touching the bar. Two things are
scored: the stick-out, which decides whether the part can be turned without chatter, and whether the bar is truly
clamped. A bar left sitting in the open jaws looks almost the same as a clamped one.

The task is `task.py` for the `yam_arena` embodiment of yam-isaac-lab-arena (station scene; Isaac Lab Arena 0.3.1,
Isaac Sim 6.1, Isaac Lab 3.0, inspect-robots 0.60.0). The folder also has:
- `labkit.py`, a kit for multi-shape parts with sliding and hinged joints, and `on_step` / `on_reset` callbacks.
  Its slides can run along slanted lines, which the chuck's jaws need.
- `arena_assets_fix.py`, imported by `labkit.py`. It points Arena's object library at NVIDIA's production bucket.
- `near_misses.py`, plain-Python checks of `score()`, `success()` and the tool-touch test that need no simulator
  (`python near_misses.py`).

The helper files patch the stock `yam_arena` package in the task's own process only. Nothing is changed on disk.

## Run a trial

From this folder, on a machine set up by the yam-isaac-lab-arena launchable:

```bash
I="Clamp the aluminium bar in the lathe chuck with 55 to 65 mm sticking out of the jaws, then move the tool so its tip is 1 to 3 mm from the end of the bar and 0 to 3 mm outside its diameter, without touching the bar."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --grader none --max-steps 3600 \
  --store-frames --log-dir ~/yam-eval/lathe-chuck-setup-dream

# Scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  --instruction "$I" --scorer task_score --grader none --max-steps 3600 \
  --store-frames --log-dir ~/yam-eval/lathe-chuck-setup-demo
```

- **Run limit:** 3,600 control steps (3 minutes of simulated time at 20 Hz; `-E control_hz=20` is needed, since the
  embodiment's default is 10 Hz). The scripted demo's plan takes 54.6 s.
- **Runtime per trial:** at most 3 minutes of simulated time. The scripted demo's run (50 s of simulated time, ended
  early on success at step 1,003) took 3 minutes 8 s of wall time on an L40S at 640 x 360 with the Isaac Sim window
  open, plus Isaac Sim's start-up.
- **Seeds:** none needed. Runs use `-E object_jitter=0`, so every object starts where `task.py` puts it.

## The lathe

Coordinates are from the robot: x away from the arm bases, y to the left, heights above the table.

| part | what it is | where |
|---|---|---|
| spindle | axis along y, locked | x 0.300, 60 mm up |
| chuck | 80 mm three-jaw, 40 mm deep, 32 mm bore | y 0.072 to 0.112 |
| jaws | 25 x 10 x 12 mm; one below the axis, two 30 deg above horizontal | front faces at y 0.060; open to 30 mm |
| key | T-handle 60 mm, in the chuck's top socket | y 0.095, handle centre 133 mm up |
| bore stop | closes the spindle bore | 100 mm behind the jaw faces |
| carriage knob | 12 mm, on the saddle | (0.180, -0.090), top 61 mm up |
| cross-slide knob | 12 mm, on the cross-slide | (0.180, -0.140), top 70 mm up |
| tool tip | 8 mm square HSS bit, at spindle height | (0.255, -0.120) at the start |
| bar | 6061 aluminium, 20 mm x 140 mm, 119 g | centre (0.220, -0.290), along y, in a cradle |

**The chuck.** Turning the key clockwise, seen from above, winds all three jaws in together, 11 mm over 92 deg. The
jaws meet the 20 mm bar 42 deg into the turn. The key has no spring and stays where it is left. In the simulation
the jaws are geared together, like a scroll, and sprung (3,000 N/m each) towards the position the key sets. The
force a jaw puts on the bar is that spring's stretch, so turning the key past first contact tightens the grip: 4 N a
jaw at 11 deg past contact, 12 N at the demo's 75 deg turn. The lathe's surfaces have a friction of 0.15, so the jaws
slide a bar released off-centre to the middle. The bar is simulated as a 12-sided prism (within 0.35 mm of round),
so the gripper pads and the jaws bear on its flats.

**The slides.** The carriage runs along the bed (y) and the cross-slide across it (x). Both are damped, with no
spring, so they stay where they are left. Each has a grab knob in place of a handwheel.

## Scoring

`score()` in `task.py` reads the bar's pose, the key and jaw positions and the two slides at the end of the trial.

- **Clamped:** every jaw within 0.7 mm of the bar's surface and pressing on it with at least 4 N. The bar's axis must
  be within 1 mm of the spindle axis where it crosses the jaw faces, and within 1 deg of it.
- **Stick-out:** from the jaw faces to the bar's free end, along the spindle axis: 55 to 65 mm.
- **Tool lined up:** the tip 1 to 3 mm from the end face along the bar's axis, and 0 to 3 mm outside its diameter.
- **Touched:** checked every control step. The tool coming within 0.3 mm of the bar at any time in the trial costs
  the tool's half of the score.

| outcome | score |
|---|---|
| bar not clamped (loose, in the open jaws, held by the robot, elsewhere) | 0 |
| clamped, stick-out outside 55 to 65 mm | 0.25 |
| clamped with the right stick-out; tool not lined up, or it touched the bar | 0.5 |
| clamped with the right stick-out, tool lined up, never touched | 1.0 |

The trial ends early at 1.0 once that has held for 2 s.

## The scripted demo

The right arm picks the bar up 20 mm from one end, lines it up on the spindle axis and slides it into the open jaws
until 60 mm sticks out. The plan allows for two repeatable offsets measured in the simulation: the bar slips 4 mm along
its length as the pads close on it, and held out over the chuck it sits 2.4 mm short of the commanded x. It lets go, so the bar rests on the bottom jaw, and backs off. The left arm grips the key's
T-handle, turns it 75 deg clockwise to close the jaws on the bar, lets go and parks. The right arm then drags the
carriage knob 118 mm, bringing the tool to 2 mm from the bar's end, and drags the cross-slide knob 33.5 mm, bringing
the tip to 1.5 mm outside the diameter.

## Reset

Turn the key anticlockwise to its start, with the handle 29 deg anticlockwise from x and the jaws open to 30 mm.
Take the bar out and lay it back in its cradle. Wind the carriage and cross-slide back until the tool tip is at
(0.255, -0.120).
