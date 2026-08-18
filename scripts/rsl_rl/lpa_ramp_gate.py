"""Climb gate: does the robot actually gain altitude on each ramp?
(tinh-lpa-ramp.6)

Per terrain column, measures REALIZED climb against the plane:

    dz_actual  vs  tan(theta) * dx_travelled

A robot that walks on the flat portion, or slides back down, fails
even if it never falls. Also reports fall rate and lateral drift per
column, because "didn't fall" is not the same as "climbed".

  python -u scripts/rsl_rl/lpa_ramp_gate.py <run_dir> [--steps 800]
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--min-ratio", type=float, default=0.6,
                    help="fraction of the plane's climb that counts")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import glob
import math
import os

import gymnasium as gym
import torch
import yaml

import robot_rl.tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
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

ckpts = sorted(glob.glob(os.path.join(args.run, "model_*.pt")),
               key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
agent_cfg = yaml.safe_load(open(os.path.join(args.run, "params/agent.yaml")))
runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
runner.load(ckpts[-1], load_optimizer=False)
policy = runner.get_inference_policy(device="cuda:0")
print(f"ramp gate on {ckpts[-1]}")

uenv = env.unwrapped
robot = uenv.scene["robot"]
types = uenv.scene.terrain.terrain_types.long()
N = args.num_envs

ret = env.get_observations()
obs = ret[0] if isinstance(ret, tuple) else ret

# Accumulate TRUE per-step motion; a reset teleports the robot, so
# steps spanning a reset contribute nothing (episode_length_buf <= 1).
prev = robot.data.root_pos_w.clone()
sum_dz = torch.zeros(N, device="cuda:0")
sum_dr = torch.zeros(N, device="cuda:0")
falls = torch.zeros(N, device="cuda:0")
tm = uenv.termination_manager

for step in range(args.steps):
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    if "base_height" in tm.active_terms:
        falls += tm.get_term("base_height").float()
    cur = robot.data.root_pos_w
    moved = cur - prev
    fresh = uenv.episode_length_buf <= 1
    z = torch.where(fresh, torch.zeros_like(moved[:, 2]), moved[:, 2])
    r = torch.where(fresh, torch.zeros_like(z), moved[:, :2].norm(dim=1))
    sum_dz += z
    sum_dr += r
    prev = cur.clone()

dz = sum_dz
dr = sum_dr

print(f"{'col':<6} {'envs':>4} {'dz':>7} {'want':>7} {'ratio':>6} "
      f"{'dist':>6} {'falls':>6}")
ok_cols = 0
for i, (n, _, deg) in enumerate(RAMP_COLUMNS):
    m = types == i
    if not int(m.sum()):
        continue
    mdz = float(dz[m].median())
    mdr = float(dr[m].median())
    want = math.tan(math.radians(deg)) * mdr
    ratio = (mdz / want) if abs(want) > 1e-3 else float("nan")
    f = float(falls[m].sum())
    good = (abs(want) < 1e-3) or (ratio >= args.min_ratio)
    ok_cols += int(good)
    print(f"{n:<6} {int(m.sum()):>4} {mdz:>+7.3f} {want:>+7.3f} "
          f"{ratio:>6.2f} {mdr:>6.2f} {f:>6.0f}  "
          f"{'ok' if good else 'LOW'}")

print(f"RAMP-CLIMB columns_ok={ok_cols}/{len(RAMP_COLUMNS)}"
      f" total_falls={int(falls.sum())}")
print("RAMP-GATE", "PASS" if ok_cols == len(RAMP_COLUMNS) else "FAIL")
app.close()
