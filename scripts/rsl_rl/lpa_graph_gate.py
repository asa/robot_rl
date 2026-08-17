"""Graph-traversal turn gate (tinh-lpa-clfrl.7.7 v2).

Turning is a GRAPH TRAVERSAL (user direction 2026-08-14), not a
vel_yaw command — so the gate runs the graphturn env with its
sampler hot (high p_turn) and scores completed traversals directly:

  * every env completes >= 1 stop -> turn -> start sequence
  * per-sequence base-yaw delta within tol of the reference turn
    (turn cycle vel_yaw 0.269 rad/s x turn_periods x period)
  * no waist_twist / runaway terminations during the run

    python -u scripts/rsl_rl/lpa_graph_gate.py <run_dir> --steps 1500
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--yaw-tol", type=float, default=0.45,
                    help="per-sequence |yaw error| bound (rad)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import glob
import os

import gymnasium as gym
import torch
import yaml

import robot_rl.tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from robot_rl.tasks.manager_based.robot_rl.lpa.lpa_walking_clf_env_cfg import (
    LpaWalkingCLFGraphTurnEnvCfg,
)

env_cfg = LpaWalkingCLFGraphTurnEnvCfg()
env_cfg.scene.num_envs = args.num_envs
# Hot sampler: every free-walking env starts a turn sequence quickly.
env_cfg.events.graph_skills.params["p_turn"] = 0.25
env = gym.make("LPA-walking-clf-graphturn", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

ckpts = sorted(glob.glob(os.path.join(args.run, "model_*.pt")),
               key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
agent_cfg = yaml.safe_load(open(os.path.join(args.run, "params/agent.yaml")))
runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device="cuda:0")
runner.load(ckpts[-1], load_optimizer=False)
policy = runner.get_inference_policy(device="cuda:0")
print(f"graph gate on {ckpts[-1]}")

uenv = env.unwrapped
robot = uenv.scene["robot"]
cmd = uenv.command_manager.get_term("traj_ref")

_TURNING = 2


def base_yaw():
    return euler_xyz_from_quat(robot.data.root_quat_w)[2]


ret = env.get_observations()
obs = ret[0] if isinstance(ret, tuple) else ret
N = args.num_envs
prev_turning = torch.zeros(N, dtype=torch.bool, device="cuda:0")
yaw_at_entry = torch.zeros(N, device="cuda:0")
step_at_entry = torch.zeros(N, dtype=torch.long, device="cuda:0")
ep_at_entry = torch.zeros(N, dtype=torch.long, device="cuda:0")
seq_count = torch.zeros(N, dtype=torch.long, device="cuda:0")
deltas = []
durations = []
aborted = 0
bad_terms = 0
for step in range(args.steps):
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    st = getattr(cmd, "_graph_state", None)
    if st is None:
        # The sampler creates its state lazily on the first
        # interval tick (~0.5 s in) — nothing to score yet.
        continue
    turning = st == _TURNING
    entered = turning & ~prev_turning
    exited = ~turning & prev_turning
    if entered.any():
        yaw_at_entry[entered] = base_yaw()[entered]
        step_at_entry[entered] = step
        ep_at_entry[entered] = uenv.episode_length_buf[entered]
    if exited.any():
        # An episode RESET mid-turn also looks like an exit (the
        # reset clears graph state) — those are ABORTS, not
        # completed sequences, and their yaw delta is meaningless.
        ep_now = uenv.episode_length_buf[exited]
        clean = ep_now > ep_at_entry[exited]
        ids_ex = exited.nonzero().flatten()
        d = wrap_to_pi(base_yaw()[exited] - yaw_at_entry[exited])
        dur = (step - step_at_entry[exited]).float()
        for k in range(len(ids_ex)):
            if bool(clean[k]):
                deltas.append(float(d[k]))
                durations.append(float(dur[k]))
            else:
                aborted += 1
        seq_count[ids_ex[clean]] += 1
    prev_turning = turning
    tm = uenv.termination_manager
    for name in ("waist_twist", "runaway"):
        if name in tm.active_terms:
            bad_terms += int(tm.get_term(name).sum())
    if step % 100 == 99:
        # State histogram + per-term forensics (gt16: sequences=0,
        # bad-terms 1558 while training showed those terms at 0).
        hist = torch.bincount(st.clamp(min=0).long(), minlength=5)
        terms = {}
        for name in tm.active_terms:
            v = int(tm.get_term(name).sum())
            if v:
                terms[name] = terms.get(name, 0) + v
        print(f"step {step+1}: sequences {int(seq_count.sum())}"
              f" (envs with >=1: {int((seq_count > 0).sum())}/{N})"
              f" bad-terms {bad_terms}"
              f" states[free,stop,turn,start,x]={hist.tolist()}"
              f" active>=0:{int((cmd.active_ref_id >= 0).sum())}"
              f" pending:{int((cmd.pending_ref_id > -2).sum())}"
              f" terms-now:{terms}")

# Expected magnitude: turn cycle vel_yaw x periods x cycle time.
turn_periods = env_cfg.events.graph_skills.params.get("turn_periods", 2.0)
expect = 0.269 * turn_periods * 2.4
dt = torch.tensor(deltas) if deltas else torch.zeros(0)
ok_cov = int((seq_count > 0).sum()) == N
err = (dt.abs() - expect).abs() if len(dt) else torch.zeros(0)
ok_yaw = len(dt) > 0 and bool((err <= args.yaw_tol).float().mean() >= 0.8)
ok_term = bad_terms == 0
du = torch.tensor(durations) if durations else torch.zeros(0)
# Steps the turn SHOULD last: turn_periods x cycle time / dt.
want_steps = turn_periods * 2.4 / float(uenv.step_dt)
if len(dt):
    # Rate check: yaw actually achieved per second of turning,
    # vs the reference's commanded rate. Separates "cut short"
    # from "refuses to rotate".
    secs = du * float(uenv.step_dt)
    rate = (dt.abs() / secs.clamp(min=1e-3))
    print(f"GRAPH-GATE sequences={len(dt)} aborted={aborted}"
          f" envs-covered={int((seq_count > 0).sum())}/{N}"
          f" |dyaw| median={dt.abs().median():.3f} expect={expect:.3f}"
          f" within-tol={float((err <= args.yaw_tol).float().mean()):.2f}"
          f" dur median={du.median():.0f}/{want_steps:.0f} steps"
          f" rate median={rate.median():.3f} vs 0.269 rad/s"
          f" bad-terms={bad_terms}")
else:
    print(f"GRAPH-GATE sequences=0 aborted={aborted}"
          f" envs-covered=0/{N} bad-terms={bad_terms}")
print("GRAPH-GATE", "PASS" if (ok_cov and ok_yaw and ok_term) else "FAIL")
app.close()
