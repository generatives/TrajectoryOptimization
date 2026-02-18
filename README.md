# Trajectory Optimization Sandbox

A lightweight simulator for path-finding and trajectory-optimization experiments with a differential-drive (two-wheeled) robot. The simulation, UI, and "autonav" algorithm were written by AI. The "trajoptim.py" file was written by me. Most of the README is AI written but I have made some edits.

## Features

- Grid-map environment with static obstacles.
- Differential-drive robot kinematics (`v`, `omega`).
- Collision checking against occupied cells.
- PyGame visualizer with:
  - teleoperation mode,
  - click-to-set-goal,
  - simple A* global path,
  - waypoint follower for autonomous mode,
  - trajectory trail rendering.

## Setup

```bash
uv sync
```

## Run

```bash
uv run main.py
```

## Controls

- `Up/Down`: forward/reverse velocity
- `Left/Right`: angular velocity
- `Left click`: set navigation goal (pixel position)
- `Space`: toggle `teleop` / `autonav`
- `R`: clear trail and current path
