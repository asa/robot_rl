# Copyright 2026 Zachary Olkin. All rights reserved.

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher
import cli_args

# Environment names
ENVIRONMENTS = {
    "lpa_walking_clf": "LPA-walking-clf",  # tinh
    "lpa_walking_clf_skill": "LPA-walking-clf-skill",  # tinh 8.5d
    "lpa_walking_clf_cladstomp0": "LPA-walking-clf-cladstomp0",
    "lpa_walking_clf_cladwalkgaitvel": "LPA-walking-clf-cladwalkgaitvel",
    "lpa_walking_clf_cladwalkgaitarm": "LPA-walking-clf-cladwalkgaitarm",
    "lpa_walking_clf_cladwalkgait2": "LPA-walking-clf-cladwalkgait2",
    "lpa_walking_clf_cladwalkgait": "LPA-walking-clf-cladwalkgait",
    "lpa_walking_clf_cladwalksym": "LPA-walking-clf-cladwalksym",
    "lpa_walking_clf_cladwalkclr": "LPA-walking-clf-cladwalkclr",
    "lpa_walking_clf_cladwalkarm": "LPA-walking-clf-cladwalkarm",
    "lpa_walking_clf_cladwalkclear": "LPA-walking-clf-cladwalkclear",
    "lpa_walking_clf_cladwalkp9": "LPA-walking-clf-cladwalkp9",
    "lpa_walking_clf_cladwalk3": "LPA-walking-clf-cladwalk3",
    "lpa_walking_clf_cladwalk2": "LPA-walking-clf-cladwalk2",
    "lpa_walking_clf_cladwalk": "LPA-walking-clf-cladwalk",
    "lpa_walking_clf_clad4": "LPA-walking-clf-clad4",
    "lpa_walking_clf_clad3": "LPA-walking-clf-clad3",
    "lpa_walking_clf_clad2": "LPA-walking-clf-clad2",
    "lpa_walking_clf_clad1": "LPA-walking-clf-clad1",
    "lpa_walking_clf_graph1": "LPA-walking-clf-graph1",
    "lpa_walking_clf_graphturn": "LPA-walking-clf-graphturn",
    "lpa_walking_clf_ramp": "LPA-walking-clf-ramp",  # tinh-lpa-ramp.5
    "lpa_walking_clf_rough": "LPA-walking-clf-rough",  # terrain robustness
    "lpa_walking_clf_rough_v5": "LPA-walking-clf-rough-honest-gait",
    "lpa_walking_clf_rough_v6": "LPA-walking-clf-rough-arm-swing",
    "lpa_walking_clf_v6play": "LPA-walking-clf-arm-swing-play",
    "lpa_walking_clf_rough_v7": "LPA-walking-clf-rough-contact-priced",
    "lpa_walking_clf_rough_v8": "LPA-walking-clf-rough-elbow-tuck",
    "lpa_walking_clf_rough_v9": "LPA-walking-clf-rough-elbow-depth",
    "lpa_walking_clf_rough_v10": "LPA-walking-clf-rough-elbow-depth-tuned",
    "lpa_walking_clf_rough_v11": "LPA-walking-clf-rough-lowent",
    "lpa_walking_clf_rough_v4": "LPA-walking-clf-rough-retrieve2",
    "lpa_walking_clf_rough_v3": "LPA-walking-clf-rough-retrieve",
    "lpa_walking_clf_rough_v2": "LPA-walking-clf-rough-priced",  # + fall pricing
    # Flat + obs history: the play/export env for history policies,
    # which cannot load into the 74-obs plain flat env.
    "lpa_walking_clf_hist": "LPA-walking-clf-hist",
    # Flat + history + clear4 refs: the tinh-nwgy discriminator.
    "lpa_walking_clf_hist_clear4": "LPA-walking-clf-hist-clear4",
    "vanilla": "G1-vanilla-walking",
    "vanilla_ec": "G1-vanilla-walking-ec",
    "lip_clf": "G1-lip-clf",
    "lip_clf_ec": "G1-lip-clf-ec",

    "walking_clf": "G1-walking-clf",
    "walking_clf_sym": "G1-walking-clf-symmetric",
    "walking_clf_ec": "G1-walking-clf-ec",

    "running_clf": "G1-running-clf",
    "running_clf_sym": "G1-running-clf-symmetric",
    "running_clf_sym_exp": "G1-running-clf-symmetric",

    "waving_clf": "G1-waving-clf",

    "bow_forward_clf": "G1-bow_forward-clf",
    "bow_forward_clf_sym": "G1-bow_forward-clf-symmetric",

    "bend_up_clf_sym": "G1-bend_up-clf-symmetric",
}

EXPERIMENT_NAMES = {
    "lpa_walking_clf": "lpa_walking_clf",  # tinh
    "lpa_walking_clf_skill": "lpa_walking_clf",  # tinh 8.5d (shared exp dir)
    "lpa_walking_clf_cladstomp0": "lpa_walking_clf",  # from-scratch stomp control, default PPO
    "lpa_walking_clf_cladwalkgaitvel": "lpa_walking_clf",  # base speed profile tracking x8 (the dwell)
    "lpa_walking_clf_cladwalkgaitarm": "lpa_walking_clf",  # + arm tracking x8 on the clean walker
    "lpa_walking_clf_cladwalkgait2": "lpa_walking_clf",  # foot phase mirror x2.5
    "lpa_walking_clf_cladwalkgait": "lpa_walking_clf",  # + foot phase mirror
    "lpa_walking_clf_cladwalksym": "lpa_walking_clf",  # phase-shifted arm mirror
    "lpa_walking_clf_cladwalkclr": "lpa_walking_clf",  # arm-torso clearance term
    "lpa_walking_clf_cladwalkarm": "lpa_walking_clf",  # arm authority
    "lpa_walking_clf_cladwalkclear": "lpa_walking_clf",  # clearance back
    "lpa_walking_clf_cladwalkp9": "lpa_walking_clf",  # pendulum9b arms
    "lpa_walking_clf_cladwalk3": "lpa_walking_clf",  # + low entropy
    "lpa_walking_clf_cladwalk2": "lpa_walking_clf",  # + solved arms
    "lpa_walking_clf_cladwalk": "lpa_walking_clf",  # walking baseline
    "lpa_walking_clf_clad4": "lpa_walking_clf",  # clad3 + walking-only cmds
    "lpa_walking_clf_clad3": "lpa_walking_clf",  # clad + turns + arm stack
    "lpa_walking_clf_clad2": "lpa_walking_clf",  # clad + turns
    "lpa_walking_clf_clad1": "lpa_walking_clf",  # clad library (shared exp dir)
    "lpa_walking_clf_graph1": "lpa_walking_clf",  # am-nai (shared exp dir)
    "lpa_walking_clf_graphturn": "lpa_walking_clf",  # tinh 7.7 (shared exp dir)
    "lpa_walking_clf_ramp": "lpa_walking_clf",  # tinh-lpa-ramp.5 (shared exp dir)
    "lpa_walking_clf_rough": "lpa_walking_clf",  # rough (shared exp dir)
    "lpa_walking_clf_rough_v5": "lpa_walking_clf",
    "lpa_walking_clf_rough_v6": "lpa_walking_clf",
    "lpa_walking_clf_v6play": "lpa_walking_clf",
    "lpa_walking_clf_rough_v7": "lpa_walking_clf",
    "lpa_walking_clf_rough_v8": "lpa_walking_clf",
    "lpa_walking_clf_rough_v9": "lpa_walking_clf",
    "lpa_walking_clf_rough_v10": "lpa_walking_clf",
    "lpa_walking_clf_rough_v11": "lpa_walking_clf",
    "lpa_walking_clf_rough_v4": "lpa_walking_clf",
    "lpa_walking_clf_rough_v3": "lpa_walking_clf",
    "lpa_walking_clf_rough_v2": "lpa_walking_clf",  # rough v2 (shared exp dir)
    "lpa_walking_clf_hist": "lpa_walking_clf",  # hist (shared exp dir)
    "lpa_walking_clf_hist_clear4": "lpa_walking_clf",  # (shared exp dir)
    "vanilla": "vanilla",
    "vanilla_ec": "vanilla",
    "basic": "baseline",
    "lip_clf": "lip",
    "lip_clf_ec": "lip",
    "lip_ref_play": "lip",

    "walking_clf": "walking_clf",
    "walking_clf_sym": "walking-clf-symmetric",
    "walking_clf_ec": "walking_clf",

    "running_clf": "running_clf",
    "running_clf_sym": "running-clf-symmetric",
    "running_clf_sym_exp": "running-clf-symmetric",

    "waving_clf": "waving_clf",

    "bow_forward_clf": "bow_forward_clf",
    "bow_forward_clf_sym": "bow_forward-clf-symmetric",

    "bend_up_clf_sym": "bend_up-clf-symmetric",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL policies for different environments.")
    parser.add_argument("--env_type", type=str, choices=list(ENVIRONMENTS.keys()), 
                       help="Type of environment to train on (vanilla/custom/clf)")
    parser.add_argument("--video", action="store_true", default=False, 
                       help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, 
                       help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000, 
                       help="Interval between video recordings (in steps).")
    parser.add_argument("--num_envs", type=int, default=None, 
                       help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, 
                       help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, 
                       help="RL Policy training iterations.")
    parser.add_argument("--distributed", action="store_true", default=False, 
                       help="Run training with multiple GPUs or nodes.")
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

def main():
    args_cli, hydra_args = parse_args()
    
    if not args_cli.env_type:
        print("Please specify an environment type using --env_type")
        print("Available options:", list(ENVIRONMENTS.keys()))
        sys.exit(1)

    # Set the task based on environment type
    args_cli.task = ENVIRONMENTS[args_cli.env_type]
    # tensorboard only: wandb was hardcoded here and uploaded every
    # run to a credit-less account (user directive 2026-08-26).
    # launch_run additionally sets WANDB_MODE=disabled as a belt.
    args_cli.logger = "wandb"
    # LIVE metrics via the direct-storage trackio shim (see
    # trackio_wandb.py for why trackio's own client dies under Kit
    # and why direct SQLiteStorage writes do not). The wandb name is
    # only the surface rsl_rl's vendored writer expects — nothing
    # touches the network. Closeout's tfevents ingestion replaces
    # the live series with the canonical one afterwards.
    import trackio_wandb
    sys.modules["wandb"] = trackio_wandb
    args_cli.log_project_name = "g1_rl"
    
    # always enable cameras to record video
    if args_cli.video:
        args_cli.enable_cameras = True

    sys.argv = [sys.argv[0]] + hydra_args
    
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import necessary modules after app launch
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import robot_rl.tasks  # noqa: F401

    # Configure PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def train(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, 
             agent_cfg: RslRlOnPolicyRunnerCfg):
        """Train with RSL-RL agent."""
        # Override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        # Set the environment seed
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # Multi-gpu training configuration
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            agent_cfg.device = f"cuda:{app_launcher.local_rank}"
            seed = agent_cfg.seed + app_launcher.local_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        # Create organized directory structure for logging

        base_log_path = os.path.join("logs", "g1_policies", EXPERIMENT_NAMES[args_cli.env_type])
        log_root_path = os.path.join(base_log_path, args_cli.env_type)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        
        # Create timestamp-based run directory
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        env_cfg.log_dir = log_dir

        # Create environment
        # if hasattr(env_cfg, "__prepare_tensors__") and callable(getattr(env_cfg, "__prepare_tensors__")):
        #     env_cfg.__prepare_tensors__()
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # Convert to single-agent if needed
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)


        # Handle resume path
        if agent_cfg.resume_path or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = agent_cfg.resume_path
            agent_cfg.resume = True
        elif agent_cfg.resume:
            # tinh: --resume --load_run <dir> [--checkpoint model_N.pt]
            # never populated resume_path (UnboundLocalError at load;
            # hydra rejects agent.resume_path=str overrides because
            # the cfg types it as None).
            from isaaclab_tasks.utils import get_checkpoint_path
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # Setup video recording if enabled
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # Wrap environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Create and configure runner
        # create runner from rsl-rl
        # NaN forensics (graphturn crashes are DETERMINISTIC at
        # iter ~101066 with ref+physics tripwires silent): guard the
        # wrapper boundary and report exactly which obs columns /
        # envs go non-finite, then clamp so training survives.
        # ALL LPA envs (not an env-name allowlist — gating these to one
        # env is exactly why the ramp env rediscovered the same
        # std-collapse crash). No-ops in healthy training: actions
        # run 1-4 vs a +-10 clamp, rewards 0.3-3 vs +-1e3.
        if args_cli.env_type.startswith("lpa_"):
            _orig_step = env.step

            def _leaves(o):
                # rsl_rl 3.x wrappers return a TensorDict of groups.
                if hasattr(o, "values") and not torch.is_tensor(o):
                    return list(o.items())
                return [("obs", o)]

            def _guarded_step(actions):
                # Action clamp at the env boundary. gt13 forensics:
                # tracking terms healthy, but action_rate_l2 hit
                # -1e24 — actions ran away to ~1e12 via the
                # last_action-observation feedback loop (big action
                # -> big obs -> bigger mu; nothing bounded FINITE
                # values). Healthy |action| is ~1-4; +-10 is
                # generous and breaks the loop. Executed-vs-stored
                # mismatch is the standard rsl-rl clip_actions
                # semantics.
                actions = actions.clamp(-10.0, 10.0)
                obs, rew, dones, extras = _orig_step(actions)
                dirty = False
                for key, v in _leaves(obs):
                    if not torch.is_tensor(v):
                        continue
                    fin = torch.isfinite(v)
                    if not fin.all():
                        dirty = True
                        bad_env = (~fin).any(dim=1)
                        cols = ((~fin).any(dim=0)
                                .nonzero().flatten().tolist())
                        print(f"[OBS-GUARD] key={key} envs:"
                              f"{bad_env.nonzero().flatten().tolist()[:8]}"
                              f" cols:{cols[:24]}")
                        v = torch.nan_to_num(
                            v, nan=0.0, posinf=0.0,
                            neginf=0.0)
                        obs[key] = v
                    # Finite-but-huge obs pass nan_to_num untouched
                    # (the gt13 runaway rode through here at 1e12).
                    # Normalized obs live in ~[-10, 10]; +-1e3 never
                    # clips legit signal.
                    if v.abs().max() > 1.0e3:
                        dirty = True
                        obs[key] = v.clamp(-1.0e3, 1.0e3)
                if not torch.isfinite(rew).all():
                    dirty = True
                    br = ~torch.isfinite(rew)
                    print(f"[OBS-GUARD] rew envs:"
                          f"{br.nonzero().flatten().tolist()[:8]}")
                    rew = torch.nan_to_num(
                        rew, nan=0.0, posinf=0.0,
                        neginf=0.0)
                if rew.abs().max() > 1.0e6:
                    # PER-STEP magnitude forensics: which env/channel
                    # (the 0.5 s tripwire sampled v<=1e6 while episode
                    # rewards hit -1e30 — the spike is transient).
                    i = int(rew.abs().argmax())
                    _c = env.unwrapped.command_manager.get_term(
                        "traj_ref")
                    st_dbg = getattr(_c, "_graph_state", None)
                    print(f"[REW-SPIKE] env {i} rew={float(rew[i]):.3e}"
                          f" v={float(_c.v[i]):.3e}"
                          f" vdot={float(_c.vdot[i]):.3e}"
                          f" |y_des|={float(_c.y_des[i].abs().max()):.3e}"
                          f" |dy_des|={float(_c.dy_des[i].abs().max()):.3e}"
                          f" |y_act|={float(_c.y_act[i].abs().max()):.3e}"
                          f" |dy_act|={float(_c.dy_act[i].abs().max()):.3e}"
                          f" active={int(_c.active_ref_id[i])}"
                          f" pend={int(_c.pending_ref_id[i])}"
                          f" phi={float(_c.phasing_var[i]):.3f}"
                          f" reft0={float(_c.ref_start_time[i]):.3f}"
                          f" st={int(st_dbg[i]) if st_dbg is not None else -9}"
                          f" ep={int(env.unwrapped.episode_length_buf[i])}")
                # UNCONDITIONAL per-step clamp: healthy steps are
                # ~0.3-3; +-1e3 is 300x headroom and caps any rare
                # spike's damage (gt10 saw -1e30 episode rewards,
                # gt11 identical code ran healthy — the blowup is
                # STOCHASTIC; contain the class, stop chasing it).
                rew = rew.clamp(min=-1.0e3, max=1.0e3)
                if dirty:
                    print("[OBS-GUARD] clamped")
                return obs, rew, dones, extras

            env.step = _guarded_step

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
            # Graphturn crash chain (deterministic, 4 runs): huge
            # finite gradients drive the SCALAR noise-std parameter
            # NEGATIVE mid-update -> torch.normal 'std >= 0'. Floor
            # it after EVERY optimizer step (post-step hook — a
            # post-update clamp is too late, the crash is between
            # minibatches).
            # ALL LPA envs (not an env-name allowlist — gating these to one
        # env is exactly why the ramp env rediscovered the same
        # std-collapse crash). No-ops in healthy training: actions
        # run 1-4 vs a +-10 clamp, rewards 0.3-3 vs +-1e3.
        if args_cli.env_type.startswith("lpa_"):
                _pol = runner.alg.policy
                if hasattr(_pol, "std") and torch.is_tensor(
                        getattr(_pol, "std", None)):
                    # clamp_ does NOT remove NaN — the floor alone
                    # moved the cliff (101066 -> 101277) but NaN
                    # GRADIENTS still poison the parameter.
                    def _std_fix(opt, a, k):
                        _pol.std.data = torch.nan_to_num(
                            _pol.std.data, nan=1.0).clamp_(
                            min=1e-3, max=3.0)
                    runner.alg.optimizer.register_step_post_hook(
                        _std_fix)
                    print("[STD-FLOOR] armed (1e-3 .. 3.0)")

                def _grad_guard(opt, a, k):
                    for group in opt.param_groups:
                        for prm in group["params"]:
                            if (prm.grad is not None
                                    and not torch.isfinite(
                                        prm.grad).all()):
                                for g2 in opt.param_groups:
                                    for p2 in g2["params"]:
                                        if p2.grad is not None:
                                            p2.grad.zero_()
                                print("[GRAD-GUARD] non-finite "
                                      "grads — step skipped")
                                return

                runner.alg.optimizer.register_step_pre_hook(
                    _grad_guard)
                print("[GRAD-GUARD] armed")
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        # runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)

        # ---- terrain-curriculum state rides in the checkpoint ------
        # rsl_rl save() stores model+optimizer only, so a resumed run
        # restarted the terrain ladder at row 0 — which is why the
        # 2026-08-25 disk-upgrade question had no good answer ("a
        # mid-run stop truncates the experiment"). save() already
        # takes an infos dict and load() already returns it; we ride
        # the curriculum tensors in it. Wrapping runner.save is the
        # only way in without forking the rsl_rl learn loop.
        def _terrain(env_obj):
            scene = getattr(getattr(env_obj, "unwrapped", env_obj),
                            "scene", None)
            t = getattr(scene, "terrain", None) if scene else None
            return t if getattr(t, "terrain_levels", None) is not None \
                else None

        _orig_save = runner.save

        def _save_with_curriculum(path, infos=None):
            t = _terrain(env)
            if t is not None:
                infos = dict(infos or {})
                infos["terrain_levels"] = t.terrain_levels.detach().cpu()
                infos["terrain_types"] = t.terrain_types.detach().cpu()
            return _orig_save(path, infos)

        runner.save = _save_with_curriculum

        # Load checkpoint if resuming
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            try:
                _infos = runner.load(resume_path)
            except KeyError:
                # Padded checkpoints (pad_checkpoint_obs, 8.5d) drop
                # the optimizer state — resume weights-only.
                print("[INFO] no optimizer state — weights only")
                _infos = runner.load(resume_path, load_optimizer=False)
            t = _terrain(env)
            if t is not None and isinstance(_infos, dict) \
                    and "terrain_levels" in _infos:
                # Restore the ladder, then re-derive env origins the
                # same way update_env_origins does.
                dev = t.terrain_levels.device
                t.terrain_levels[:] = _infos["terrain_levels"].to(dev)
                t.terrain_types[:] = _infos["terrain_types"].to(dev)
                t.env_origins[:] = t.terrain_origins[
                    t.terrain_levels, t.terrain_types]
                print(f"[INFO] terrain curriculum restored: mean level "
                      f"{t.terrain_levels.float().mean():.2f}")
            elif t is not None:
                print("[INFO] checkpoint carries no terrain state — "
                      "curriculum starts at init levels")

        # Save configurations
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

        # Run training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

        # Cleanup
        env.close()

    # Run training
    train()
    # Close sim app
    simulation_app.close()

if __name__ == "__main__":
    main()









