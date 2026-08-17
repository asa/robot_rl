"""Verify ramp terrain -> gait assignment (tinh-lpa-ramp.5).

Each env must be pinned to the reference matching ITS terrain column,
and the slope observation must agree. Checks the plumbing BEFORE any
long training run.

  python scripts/rsl_rl/ramp_assign_probe.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch

import robot_rl.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robot_rl.tasks.manager_based.robot_rl.lpa.lpa_walking_clf_env_cfg import (
    LpaWalkingCLFRampEnvCfg,
)
from robot_rl.tasks.manager_based.robot_rl.lpa.lpa_ramp_terrain import (
    RAMP_COLUMNS,
)

env_cfg = LpaWalkingCLFRampEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env = gym.make("LPA-walking-clf-ramp", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
uenv = env.unwrapped
cmd = uenv.command_manager.get_term("traj_ref")
lib = cmd.manager

expect = torch.tensor([lib.ref_id_of(r) for _, r, _ in RAMP_COLUMNS],
                      device="cuda:0")
print("column -> ref_id:", {n: int(expect[i])
                            for i, (n, _, _) in enumerate(RAMP_COLUMNS)})

env.get_observations()
zero = torch.zeros(args.num_envs, uenv.action_manager.total_action_dim,
                   device="cuda:0")
types = uenv.scene.terrain.terrain_types.long()
want = expect[types.clamp(max=len(RAMP_COLUMNS) - 1)]

# SPAWN AUDIT: absolute root height vs the env origin's terrain
# height, per column, plus which term fires in the first steps.
# (ramp2: mean episode length 2.59 steps = terminated at reset.)
robot = uenv.scene["robot"]
org = uenv.scene.env_origins
print("--- spawn audit ---")
for i, (n, _, d) in enumerate(RAMP_COLUMNS):
    m = types == i
    if int(m.sum()):
        print(f"  col {i} {n:5s} origin_z={float(org[m][:, 2].median()):+.3f}"
              f" root_z={float(robot.data.root_pos_w[m][:, 2].median()):+.3f}"
              f" clearance={float((robot.data.root_pos_w[m][:, 2] - org[m][:, 2]).median()):+.3f}")
tm0 = uenv.termination_manager
for k in range(4):
    with torch.inference_mode():
        env.step(zero)
    fired = {nm: int(tm0.get_term(nm).sum()) for nm in tm0.active_terms
             if int(tm0.get_term(nm).sum())}
    print(f"  step {k}: terms={fired}"
          f" root_z_med={float(robot.data.root_pos_w[:, 2].median()):+.3f}")
print("--- end spawn audit ---")
# Cumulative: envs on steep slopes fall in LOCKSTEP under zero
# actions, so an end-of-run snapshot can land in the post-reset
# assignment gap for a whole column at once.
ever_ok = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda:0")
steps_assigned = torch.zeros(args.num_envs, device="cuda:0")
wrong = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda:0")
for step in range(args.steps):
    with torch.inference_mode():
        env.step(zero)
    a = cmd.active_ref_id
    on = a >= 0
    steps_assigned += on.float()
    ever_ok |= on & (a == want)
    wrong |= on & (a != want)

got = cmd.active_ref_id
assigned = ever_ok
match = ever_ok & ~wrong
print(f"assigned-fraction median:"
      f" {float((steps_assigned / args.steps).median()):.3f}")
print(f"envs assigned: {int(assigned.sum())}/{args.num_envs}")
print(f"correct gait:  {int(match.sum())}/{int(assigned.sum())}")
for i, (n, _, d) in enumerate(RAMP_COLUMNS):
    m = types == i
    if int(m.sum()):
        sl = getattr(cmd, "_env_slope_deg", torch.zeros_like(got).float())
        print(f"  col {i} {n:5s} envs={int(m.sum()):3d}"
              f" ever_ok={int(ever_ok[m].sum()):3d}"
              f" wrong={int(wrong[m].sum()):3d}"
              f" frac={float((steps_assigned[m] / args.steps).median()):.2f}"
              f" slope_obs={float(sl[m].median()):+.2f} (want {d:+.2f})")
ok = int(assigned.sum()) == args.num_envs and int(match.sum()) == args.num_envs
print("RAMP-ASSIGN", "PASS" if ok else "FAIL")
app.close()
