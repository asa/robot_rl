"""Anchored-yaw validation (tinh-lpa-clfrl.7.7.v3).

A ZERO-ACTION robot forced into a turn reference must show a GROWING
CORE-yaw tracking error (~0.13 rad per turn cycle) — under the old
heading-relative alignment the error reset at every domain switch,
which is exactly how gt15 turned 17 deg instead of 74.

  python scripts/rsl_rl/validate_anchored_yaw.py
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch

import robot_rl.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from robot_rl.tasks.manager_based.robot_rl.lpa.lpa_walking_clf_env_cfg import (
    LpaWalkingCLFGraphTurnEnvCfg,
)

env_cfg = LpaWalkingCLFGraphTurnEnvCfg()
env_cfg.scene.num_envs = 2
# No sampler interference: drive the handoff manually.
env_cfg.events.graph_skills.params["p_turn"] = 0.0
env = gym.make("LPA-walking-clf-graphturn", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
uenv = env.unwrapped
cmd = uenv.command_manager.get_term("traj_ref")

# Find a turn reference: nonzero conditioner yaw rate.
yaw_rates = cmd.manager.conditioning_vars[:, 2]
turn_ids = (yaw_rates.abs() > 0.1).nonzero().flatten()
assert len(turn_ids) > 0, "no turn reference in the library"
ref = int(turn_ids[0])
print(f"turn ref id {ref} yaw rate {float(yaw_rates[ref]):+.3f}")

env.get_observations()
zero = torch.zeros(2, uenv.action_manager.total_action_dim,
                   device="cuda:0")
ids = torch.tensor([0, 1], device="cuda:0")
cmd.set_next_ref(ids, ref, entry_time=0.0)
errs = []
for step in range(args.steps):
    with torch.inference_mode():
        env.step(zero)
    if step % 50 == 49:
        y_err = cmd.clf.compute_y_err(cmd.y_act, cmd.y_des)
        # CORE ori tangent rows: the box_minus block for CORE.
        rows = cmd.clf.ori_body_indices_tangent.get("CORE")
        e = y_err[:, rows].norm(dim=1) if rows else y_err.norm(dim=1)
        errs.append(float(e.mean()))
        print(f"step {step+1}: active={cmd.active_ref_id.tolist()}"
              f" CORE-ori err {float(e.mean()):.4f}")
grew = len(errs) >= 3 and errs[-1] > errs[0] + 0.3
print("ANCHOR-VALIDATE", "PASS" if grew else "FAIL",
      f"(first {errs[0]:.3f} -> last {errs[-1]:.3f})")
app.close()
