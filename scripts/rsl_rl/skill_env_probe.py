"""Live-env probe for the behavior-graph curriculum (8.5d gate):
step the SKILL env with the trained policy and report how many envs
actually cycle through the laser segments (active_ref_id histogram,
handoff counts, skill-obs activity).

    python scripts/rsl_rl/skill_env_probe.py <run_glob> --steps 600
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_glob")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--num-envs", type=int, default=64)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import glob
import os

import gymnasium as gym
import torch
import yaml
import robot_rl.tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

env_cfg = parse_env_cfg("LPA-walking-clf-skill", device="cuda:0",
                        num_envs=args.num_envs)
# Bias toward standing so holds lock and the sampler fires often.
env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.2)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
env = gym.make("LPA-walking-clf-skill", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
run = sorted(glob.glob(args.run_glob))[-1]
ckpt = sorted(glob.glob(os.path.join(run, "model_*.pt")),
              key=lambda p: int(p.split("_")[-1].split(".")[0]))[-1]
agent_cfg = yaml.safe_load(open(os.path.join(run, "params/agent.yaml")))
runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
try:
    runner.load(ckpt)
except KeyError:
    runner.load(ckpt, load_optimizer=False)
policy = runner.get_inference_policy(device="cuda:0")

cmd = env.unwrapped.command_manager.get_term("traj_ref")
lib = cmd.manager
names = lib.ref_names
counts = torch.zeros(len(names) + 1, dtype=torch.long)
handoffs = 0
prev = cmd.active_ref_id.clone()
obs, _ = env.get_observations()
for k in range(args.steps):
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    a = cmd.active_ref_id
    handoffs += int((a != prev).sum())
    prev = a.clone()
    counts[-1] += int((a < 0).sum())
    for i in range(len(names)):
        counts[i] += int((a == i).sum())

total = counts.sum().item()
print(f"steps {args.steps} x {args.num_envs} envs, "
      f"handoffs observed: {handoffs}")
for i, n in enumerate(names):
    print(f"  {n:<16} {100.0 * counts[i] / total:5.2f}% env-steps")
print(f"  {'locomotion(-1)':<16} {100.0 * counts[-1] / total:5.2f}%")
print("SKILL-PROBE", "PASS" if handoffs >= 4 and counts[:len(names)].sum() > 0
      else "FAIL")
app.close()
