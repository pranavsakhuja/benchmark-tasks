# Stack loose onions on a produce display into a pyramid

A low wooden produce display bin stands on the table with nine onions packed three by three on its floor. A crate to
the right holds six more. The robot builds the display up into a pyramid: one onion in each of the four hollows
between the bottom onions, then one on top of those four. An onion only stays up where it sits in a hollow, touching
four onions below it. Set down on one onion's crown, or near the edge of the stack, it rolls off and falls to the
table, and a careless hand can knock the bottom layer out of its square. The top onion has a seat only once all four
second-layer onions are in their hollows.

The task is `task.py` for the `yam_arena` embodiment of yam-isaac-lab-arena (station scene; Isaac Lab Arena 0.3.1,
Isaac Sim 6.1, Isaac Lab 3.0, inspect-robots 0.60.0). The folder also has:
- `labkit.py`, a kit for multi-shape parts (here: the display bin, the crate and the onions, which are spheres) with
  finger-pad contact patches and `on_step` / `on_reset` callbacks.
- `arena_assets_fix.py`, imported by `labkit.py`. It points Arena's object library at NVIDIA's production bucket.
  This task uses no library assets.
- `near_misses.py`, plain-Python checks of `score()` and `success()` that need no simulator
  (`python near_misses.py`).

The helper files patch the stock `yam_arena` package in the task's own process only. Nothing is changed on disk.

## Run a trial

From this folder, on a machine set up by the yam-isaac-lab-arena launchable:

```bash
I="Stack onions from the crate on the display as a pyramid: one in each hollow of the bottom layer, then one on top."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$I" --scorer task_score --grader none --max-steps 2400 --seed 1 \
  --store-frames --log-dir ~/yam-eval/onion-pyramid-dream

# Scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=$PWD/task.py -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=360 \
  --instruction "$I" --scorer task_score --grader none --max-steps 2400 --seed 1 \
  --store-frames --log-dir ~/yam-eval/onion-pyramid-demo
```

- **Run limit:** 2,400 control steps (2 minutes of simulated time at 20 Hz; `-E control_hz=20` is needed, since the
  embodiment's default is 10 Hz). The scripted demo ends early on success at 49 s.
- **Runtime per trial:** at most 2 minutes of simulated time. The scripted demo's run took about 4 minutes of wall
  time on an L40S at 320 x 176, including Isaac Sim's start-up.
- **Seeds:** none. `-E object_jitter=0` keeps every object exactly where `task.py` puts it, so every trial starts
  the same.

## Objects

Coordinates are from the robot: x away from the arm bases, y to the left, heights above the table.

| object | what it is | where |
|---|---|---|
| onions (15) | yellow onions, 70 mm across, 170 g | 9 on the display's floor, 6 in the crate |
| display bin | pale wood, inside 214 x 214 mm, 6 mm floor, walls 8 mm thick and 35 mm tall | fixed, centred at (0.360, -0.030) |
| bottom layer | 9 onions in a 3 x 3 square, 70.2 mm apart | centred in the display bin |
| crate | green polypropylene, inside 218 (x) x 146 (y) mm, 6 mm floor, walls 8 mm thick and 25 mm tall | fixed, centred at (0.250, -0.250) |
| crate onions | 6 onions in two rows of three along x, 72 mm apart both ways | centred in the crate |

The bin's walls come up to the bottom onions' middles, so they hold the bottom layer in its square. An onion seated in
a hollow rests 49 mm above the four under it, and the top onion 49 mm above the second layer. The finished pyramid
has 14 onions; one onion is left over in the crate.

For a real trial: any low display tray or produce bin with these inside sizes (or a box with the same inside sizes,
taped to the table), a plastic produce crate, and 15 onions picked to the same size (70 mm, within a few mm).

## How the simulator scores it

- **Bottom layer:** onions on the display's floor (their centres within 5 mm of 35 mm above it).
- **Seated:** an onion at rest touching four onions of the layer below (centres within 73 mm, that is 2 x 35 mm plus
  3 mm), each of the four at least 30 mm lower than it.
- **Second layer:** onions seated on four bottom-layer onions. **Top:** an onion seated on four second-layer onions.
- **Fallen:** an onion at table level or below (its centre under 70 mm above the table top), outside the display and
  the crate: on the table, or on the floor.
- **Score:** 0.2 per second-layer onion (up to 4) plus 0.2 for an onion on top, minus 0.2 per fallen onion; from 0
  to 1. `task_score` takes the score at the last step. A full score held for 2 s, with every onion at rest, ends a
  trial early.
- In a real trial: a photo from above and one from the side. Seated: the onion touches each of the four below it and
  stays put. Fallen: any onion on the table outside the display and the crate, or on the floor.

## Scripted demo

`demo()` reads what a camera would see: where each onion is.
1. The right hand takes an onion from the crate, its fingers closing across the onion along the table's diagonal.
   That puts the pads in the wide gaps between diagonal neighbours in the crate, and over gaps in the stack.
2. It carries the onion high over the display and aims at the middle of the four onions under the chosen hollow, as
   seen, 49 mm above them.
3. It lowers the onion to 10 mm above its seat and lets go. The onion drops into the hollow.
4. It does this for the four hollows of the bottom layer, nearest the robot first, then sets a fifth onion on top of
   the four.

The demo scores 1.0 and ends early on success at 49 s.

**Dream-1** (`--seed 1`, the Dream-1 command above) scores 0. Its right arm reaches over the crate again and again
and hovers above the onions, nudging two of them a few millimetres, but never closes on one or lifts it; at the end it
hovers over the display. No onion leaves the crate and none falls.

**Misplacement check.** `ONION_CHECK=on_crown` runs the demo but sets the first onion down over the bottom layer's
front-right corner onion instead of in a hollow, `ONION_CHECK_OUT` mm (default 15) outward of its crown along the
diagonal. At 15 mm the onion rolls off the stack, across the table and onto the floor; the rest of the demo fills the
four hollows with its other four onions, leaving none for the top, and the trial scores 0.6 (0.8, less 0.2 for the
fallen onion). Set down dead on the crown (`ONION_CHECK_OUT=0`), the onion rolled inward into the nearest hollow
instead: a crown is a tipping point, and which way an onion leaves it depends on millimetres.

## Limits

- Rigid primitives at true size. Each onion is a rigid sphere, all one size. Real onions vary by a few millimetres,
  are slightly flattened, and have papery skins that grip and dent a little, so a real pyramid is a little less
  regular. A rigid sphere also rolls much farther on the table than a real onion, which stops within a few
  centimetres; a fallen onion is scored the same either way.
- The display bin and crate are fixed to the table.
- Third-party assets: none; every shape is a primitive.
