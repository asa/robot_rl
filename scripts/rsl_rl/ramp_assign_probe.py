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
for step in range(args.steps):
    with torch.inference_mode():
        env.step(zero)

types = uenv.scene.terrain.terrain_types.long()
want = expect[types.clamp(max=len(RAMP_COLUMNS) - 1)]
got = cmd.active_ref_id
assigned = got >= 0
match = (got == want) & assigned
print(f"envs assigned: {int(assigned.sum())}/{args.num_envs}")
print(f"correct gait:  {int(match.sum())}/{int(assigned.sum())}")
for i, (n, _, d) in enumerate(RAMP_COLUMNS):
    m = types == i
    if int(m.sum()):
        sl = getattr(cmd, "_env_slope_deg", torch.zeros_like(got).float())
        print(f"  col {i} {n:5s} envs={int(m.sum()):3d}"
              f" ref_ok={int((got[m] == want[m]).sum()):3d}"
              f" slope_obs={float(sl[m].median()):+.2f} (want {d:+.2f})")
ok = int(assigned.sum()) == args.num_envs and int(match.sum()) == args.num_envs
print("RAMP-ASSIGN", "PASS" if ok else "FAIL")
app.close()
