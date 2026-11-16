# Measure three look-alike bars on a bench caliper and sort them GO / NO-GO

Three aluminium bars lie on the table. They look the same, but their lengths are 40.3, 40.8 and 39.2 mm. The drawing
calls for 40.0 ± 0.5 mm. The robot measures each bar on a bench-mounted digital caliper and puts the in-tolerance
bar in the green (GO) tray and the other two in the red (NO-GO) tray. One hand sets a bar between the jaws, and the
other pushes the caliper's thumb tab to close the jaws on it, so the display reads the bar's length. One reject is
0.3 mm over the upper limit and the other 0.3 mm under the lower one. The GO bar and the long reject differ by
0.5 mm, so nobody can tell them apart by eye. The
reading is the only way to sort them.

The task is `task.py` for the `yam_arena` embodiment of yam-isaac-lab-arena (station scene; Isaac Lab Arena 0.3.1,
Isaac Sim 6.1, Isaac Lab 3.0, inspect-robots 0.60.0). The folder also has:
- `labkit.py`, a kit for multi-shape parts with sliding and hinged joints, and `on_step` / `on_reset` callbacks.
  The caliper's slider and its display are joints on one part.
- `arena_assets_fix.py`, imported by `labkit.py`. It points Arena's object library at NVIDIA's production bucket.
- `near_misses.py`, plain-Python checks of `score()`, `success()` and the display that need no simulator
  (`python near_misses.py`).

The helper files patch the stock `yam_arena` package in the task's own process only. Nothing is changed on disk.

## Run a trial

From this folder, on a machine set up by the yam-isaac-lab-arena launchable:

```bash
I="Measure each bar with the caliper, then put the bars that are 39.5 to 40.5 mm long in the green box and the others in the red box."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --grader none --max-steps 3600 \
  --store-frames --log-dir ~/yam-eval/inspect-parts-dream

# Scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  --instruction "$I" --scorer task_score --grader none --max-steps 3600 \
  --store-frames --log-dir ~/yam-eval/inspect-parts-demo
```

- **Run limit:** 3,600 control steps (3 minutes of simulated time at 20 Hz; `-E control_hz=20` is needed, since the
  embodiment's default is 10 Hz). The scripted demo takes about 96 s.
- **Runtime per trial:** at most 3 minutes of simulated time. The scripted demo's run (96 s of simulated time, ended
  early on success) took about 5 minutes of wall time on an L40S at 640 x 360, including Isaac Sim's start-up.
- **Seeds:** none needed. Runs use `-E object_jitter=0`, so every object starts where `task.py` puts it.
  The bars lie in a row, their centres 5.5 cm apart in y.

## Objects

| bar | length | GO? |
|---|---|---|
| bar1 | 40.3 mm | GO (0.2 mm inside the upper limit) |
| bar2 | 40.8 mm | NO-GO (0.3 mm over) |
| bar3 | 39.2 mm | NO-GO (0.3 mm under) |

Every bar is 20 mm wide and 26 mm tall, aluminium, 56 g. They lie in a row at x 0.23 m (bar centres): bar1 at
y 0.090, bar2 at y 0.145, bar3 at y 0.200 (y to the robot's left).

**Caliper.** A fixed beam with a fixed jaw at its far end (inner face at x 0.300 m). The sliding jaw rides on a
carriage with an orange thumb tab behind it, and a 15 N/m spring holds the jaws 55 mm open (0.2 N at a 40 mm
gap). Pushing the tab away from the robot closes the jaws. A 2 mm polished steel platen (friction 0.05) lies between
the jaws, so the sliding jaw can push a bar along it up against the fixed jaw. The sliding jaw clears the platen by
1 mm. The display at the beam's far end shows the jaw gap in mm to 0.1 mm, live. Its four 7-segment digits are 20 mm
tall and upright as seen from the robot. Each segment is a small box on a sprung slide that rises flush with the
screen when lit and sinks into the housing when not.

**Trays.** Fixed, open, 90 x 170 mm inside, 4 mm floor, 6 mm walls 30 mm tall. Green (GO) centred at
(0.22, 0.335) m, red (NO-GO) at (0.22, 0.530) m.

## How the simulator scores it

- **GO:** `abs(length - 40.0) <= 0.5 mm` (with a 1e-9 margin). The lengths are fixed in `task.py`; the score never
  uses the display.
- **"In a tray":** the bar's world-aligned bounding box lies within the tray's inner walls in x and y, with a 1 mm
  allowance, and its body origin (the centre of its underside at rest) is no higher than the walls' tops (30 mm).
  A bar lying, turned or standing on end inside the tray counts.
- **Wrong:** a bar in the other tray or in neither: on the table, on the caliper or platen, straddling a wall,
  dropped off the table, or held above a tray.
- **Score:** 1.0 with no bar wrong, 0 otherwise. `task_score` takes the score at the last step. `success()` (all
  three right) held for 3 s ends a trial early.
- **No partial credit:** with three bars, two of them share a tray. Partial credit for one wrong bar would pay a robot
  that put all three in that tray without measuring anything.

## Scripted demo

`demo()` knows the layout (which bar is where), as a scripted demo may. For each bar in turn:
1. The left arm picks the bar from above, with the pads on the top 12 mm of its sides. It carries the bar to the
   caliper, sets it on the platen between the jaws, 4 mm short of the fixed jaw, and backs off beyond the caliper,
   where its wrist camera watches the display.
2. The right arm, fingers closed, pushes the thumb tab away from the robot until the sliding jaw closes on the bar
   and pushes it against the fixed jaw. The display then reads the bar's length. The arm holds for 1.5 s, then eases
   back, the spring opens the jaws, and the arm backs off to its park pose.
3. The left arm comes back, takes the bar out and puts it in its tray, 5 cm apart.

The arms take turns over the caliper, so they never work there at the same time. Both arms park at the end.

It scores 1.0 and ends early on success at step 1,919 (96 s). The display reads 40.3, 40.8 and 39.2 mm in turn,
which the left wrist camera shows in the demo video.

## Shortcuts considered

All are scored offline in `near_misses.py`.
- Doing nothing, or putting all three bars in one tray: 0. All in red leaves only one bar wrong, which is why there
  is no partial credit.
- Sorting by start position: the layout is fixed, so a policy trained on this exact scene could memorise which
  bar goes where. The bars look the same, so a policy seeing the task fresh can only tell them apart by the reading.
- Swapping the GO bar with the long reject (40.3 and 40.8 mm): 0.
- Leaving a bar out, on the caliper or held above a tray: 0.
- Guessing: a random tray for each bar is all right 1 time in 8. One guess at which bar is GO, the other two in red,
  is right 1 time in 3.
- Sorting by eye: the GO bar and the long reject differ by 0.5 mm, about 1% of their length.

## Limits

- **Caliper:** a sprung slider pushed by a thumb tab stands in for a caliper's thumb wheel. It reads to 0.1 mm;
  real digital calipers read to 0.01 mm. The tolerance band (±0.5 mm) and the bars' lengths keep every bar at least
  0.2 mm from a limit, so the 0.1 mm reading is always decisive. No measuring force or jaw wear is modelled.
- **Bars:** they differ only in length and all weigh 56 g, and they are rigid primitives.
- **Third-party assets:** none; every shape is a primitive.
