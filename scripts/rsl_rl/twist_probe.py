"""Waist-twist exploit probe (graphturn14 diagnosis).

Rolls the graphturn env under a checkpoint and reports the WAIST_YAW
joint excursion plus pelvis-vs-chest yaw divergence. Hypothesis: the
heading-relative output alignment zeroes the base-yaw position error,
so turn tracking is velocity-level only and the policy fakes turn
rate by twisting the torso instead of stepping.

  python scripts/rsl_rl/twist_probe.py <run_dir> [--steps 600]
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import robot_rl.tasks  # noqa: F401
from robot_rl.tasks.manager_based.robot_rl.lpa.lpa_walking_clf_env_cfg import (
    LpaWalkingCLFGraphTurnEnvCfg,
)

env_cfg = LpaWalkingCLFGraphTurnEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env = gym.make("LPA-walking-clf-graphturn", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

import yaml
agent_cfg = yaml.safe_load(
    open(os.path.join(args_cli.run, "params/agent.yaml")))
runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
ckpts = sorted(glob.glob(os.path.join(args_cli.run, "model_*.pt")),
               key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
runner.load(ckpts[-1], load_optimizer=False)
policy = runner.get_inference_policy(device="cuda:0")
print(f"probe on {ckpts[-1]}")

uenv: ManagerBasedRLEnv = env.unwrapped
robot = uenv.scene["robot"]
jn = robot.data.joint_names
waist_idx = next(i for i, n in enumerate(jn) if "WAIST" in n.upper())
print(f"waist joint: {jn[waist_idx]}")

obs, _ = env.get_observations()
wmax = torch.zeros(args_cli.num_envs, device="cuda:0")
for step in range(args_cli.steps):
    with torch.inference_mode():
        act = policy(obs)
        obs, _, _, _ = env.step(act)
    w = robot.data.joint_pos[:, waist_idx].abs()
    wmax = torch.maximum(wmax, w)
    if step % 100 == 99:
        cmd = uenv.command_manager.get_term("traj_ref")
        st = getattr(cmd, "_graph_state", None)
        n_turn = int((st == 2).sum()) if st is not None else -1
        print(f"step {step+1}: waist now p50={w.median():.3f}"
              f" max={w.max():.3f} | run-max p50={wmax.median():.3f}"
              f" max={wmax.max():.3f} | envs>1.0rad:"
              f" {int((wmax > 1.0).sum())}/{args_cli.num_envs}"
              f" | turning now: {n_turn}")
print(f"TWIST-PROBE waist run-max p50={wmax.median():.3f}"
      f" p90={wmax.quantile(0.9):.3f} max={wmax.max():.3f}"
      f" envs>1.0: {int((wmax > 1.0).sum())}/{args_cli.num_envs}")
simulation_app.close()
