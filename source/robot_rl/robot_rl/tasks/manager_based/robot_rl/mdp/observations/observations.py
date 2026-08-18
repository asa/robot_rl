# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.observations import generated_commands

def skill_onehot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Active-skill one-hot (tinh-lpa-clfrl.8.5d): the slot of each
    env's ACTIVE reference (behavior-graph handoff state, 8.5c) in the
    cfg-declared skill_slots vocabulary. Locomotion (-1) maps to its
    own slot, so pre-graph episodes emit a constant locomotion
    one-hot — checkpoint-paddable."""
    cmd = env.command_manager.get_term(command_name)
    lib = cmd.manager
    slots = torch.zeros(env.num_envs, len(cmd.cfg.skill_slots),
                        device=env.device)
    loco = cmd.cfg.skill_slots.index("locomotion")
    idx = torch.full((env.num_envs,), loco, dtype=torch.long,
                     device=env.device)
    explicit = cmd.active_ref_id >= 0
    if explicit.any():
        idx[explicit] = lib.skill_slot_of_traj[
            cmd.active_ref_id[explicit]]
    slots[torch.arange(env.num_envs, device=env.device), idx] = 1.0
    return slots


def skill_params(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Named conditioner params of the active reference (8.5d) —
    zero for locomotion and for segments without that channel."""
    cmd = env.command_manager.get_term(command_name)
    lib = cmd.manager
    out = torch.zeros(env.num_envs, lib.params_of_traj.shape[1],
                      device=env.device)
    explicit = cmd.active_ref_id >= 0
    if explicit.any():
        out[explicit] = lib.params_of_traj[cmd.active_ref_id[explicit]]
    return out


def base_z(env: ManagerBasedRLEnv) -> torch.Tensor:
    base_z = env.scene["robot"].data.root_pos_w[:,2]
    return base_z.unsqueeze(-1)

def contact_state(env: ManagerBasedRLEnv, sensor_cfg, threshold: float = 50.0) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:,-1,sensor_cfg.body_ids,:]
    contact_flag = (torch.abs(net_forces) > threshold).float()
    #reshape from num_env, num_bodies, 3 to num_env, num_bodies*3
    return contact_flag.reshape(env.num_envs, -1)

def foot_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    left_foot_vel = cmd.robot.data.body_lin_vel_w[:,cmd.feet_bodies_idx[0],:]
    right_foot_vel = cmd.robot.data.body_lin_vel_w[:,cmd.feet_bodies_idx[1],:]

    foot_vel = torch.cat([left_foot_vel, right_foot_vel], dim=-1)

    return foot_vel

def foot_ang_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    left_foot_ang_vel = cmd.robot.data.body_ang_vel_w[:,cmd.feet_bodies_idx[0],:]
    right_foot_ang_vel = cmd.robot.data.body_ang_vel_w[:,cmd.feet_bodies_idx[1],:]

    foot_ang_vel = torch.cat([left_foot_ang_vel, right_foot_ang_vel], dim=-1)

    return foot_ang_vel

def ref_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj = cmd.y_des.clone()
    # ref_traj[:,8] *= 50.0       # TODO: What is this?
    return ref_traj

def act_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj = cmd.y_act.clone()
    # act_traj[:,8] *= 50.0   # TODO: What is this?
    return act_traj

def ref_traj_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj_vel = cmd.dy_des
    return ref_traj_vel

def act_traj_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj_vel = cmd.dy_act
    return act_traj_vel

def traj_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Make an observation using the error to the trajectory.
    """
    cmd = env.command_manager.get_term(command_name)
    ref_traj_vel = cmd.dy_des
    act_traj_vel = cmd.dy_act

    act_traj = cmd.y_act.clone()
    ref_traj = cmd.y_des.clone()

    traj_error = torch.cat([act_traj - ref_traj, act_traj_vel - ref_traj_vel], dim=-1)
    return traj_error


def ref_sin_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)

    phase = 2*torch.pi * cmd.get_phasing_var()

    sphase = torch.sin(phase)
    if sphase.ndim == 1:
        # [B] → [B, 1]
        sphase = sphase.unsqueeze(-1)

    return sphase

def ref_cos_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)

    phase = 2*torch.pi * cmd.get_phasing_var()

    cphase = torch.cos(phase)
    if cphase.ndim == 1:
        # [B] → [B, 1]
        cphase = cphase.unsqueeze(-1)
    return cphase

def sin_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    period = generated_commands(env, command_name).clone()

    phase = 2 * torch.pi * (env.sim.current_time / period)
    sphase = torch.sin(phase).unsqueeze(-1)

    return sphase

def cos_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    period = env.command_manager.get_command(command_name).clone()

    phase = 2 * torch.pi * (env.sim.current_time / period)
    cphase = torch.cos(phase).unsqueeze(-1)

    return cphase

def domain_flag(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return a domain flag based on which hybrid domain the reference trajectory is in.
    0 = standing, 1 = walking, 2 = running
    """
    cmd = env.command_manager.get_term(command_name)
    commanded_velocity = env.command_manager.get_command("base_velocity")

    # Boolean masks
    standing = torch.norm(commanded_velocity, dim=1) < cmd.standing_threshold
    running = commanded_velocity[:, 0] >= 1.05
    walking = (commanded_velocity[:, 0] > 0.0) & (commanded_velocity[:, 0] < 1.05)

    # Default to standing (0)
    domain = torch.zeros_like(commanded_velocity[:, 0], dtype=torch.long)

    # Assign walking and running where applicable
    domain[walking] = 1
    domain[running] = 0

    # Standing should override everything else
    domain[standing] = 2

    return domain.unsqueeze(-1)

def multiskill_phase(env: ManagerBasedRLEnv, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a number of phasing variables at different frequencies to cover a range.
    """

    frequencies = torch.tensor([0.01, 0.1, 1, 10, 100], device=env.device)   # Hz
    num_freq = len(frequencies)

    cmd = env.command_manager.get_term(command_name)

    t = env.episode_length_buf * env.step_dt

    episodic = cmd.is_episodic()
    phasing_var = cmd.get_phasing_var()

    sp = torch.zeros(env.num_envs, num_freq, device=env.device)
    cp = torch.zeros(env.num_envs, num_freq, device=env.device)

    for i in range(num_freq):
        sp[:, i] = torch.sin(2 * torch.pi * frequencies[i] * t)
        cp[:, i] = torch.cos(2 * torch.pi * frequencies[i] * t)

        # Overwrite the episodic envs to go from phase 0 to 1
        sp[episodic, i] = torch.sin(2 * torch.pi * phasing_var[episodic])
        cp[episodic, i] = torch.cos(2 * torch.pi * phasing_var[episodic])

    return sp, cp

def skill_selector(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    Passes an encoding of the skill currently in use.

    For now, we will use a 1 hot encoding.
    """
    # TODO: Check/test function

    encoding = {"locomotion": 0, "bow_forward": 1}

    cmd = env.command_manager.get_term(command_name)

    skill_encoding = torch.zeros(env.num_envs, device=env.device)

    for i in cmd.manager.traj_names:
        skill_encoding[cmd.manager.manager_indices[i]] = encoding[cmd.manager.traj_names[i]]

    return skill_encoding

def terrain_slope(env, command_name: str = "traj_ref"):
    """Signed terrain slope (radians) under each env, from the ramp
    sampler's terrain-column lookup (tinh-lpa-ramp.5). Zeros before
    the first sampler tick / on flat envs."""
    import torch

    cmd = env.command_manager.get_term(command_name)
    v = getattr(cmd, "_env_slope_deg", None)
    if v is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return (v * (torch.pi / 180.0)).unsqueeze(-1)


def ground_climb(env, command_name: str = "traj_ref",
                 cap: float = 0.05):
    """Reward REALIZED climb: per-step gain in stance-foot ground
    height, signed by the terrain slope (tinh-lpa-ramp.6).

    The CLF cannot express this — both sides re-anchor to the stance
    frame at every domain, so altitude error never accumulates (the
    same per-domain-reset leak that broke turning, in z instead of
    yaw). This term pays for exactly what the climb gate measures:
    stance-foot height moving WITH the slope. Zero on flat, zero
    across resets, per-step delta capped against teleports.
    """
    import torch

    robot = env.scene["robot"]
    if not hasattr(env, "_gc_feet_idx"):
        env._gc_feet_idx = [robot.body_names.index(n)
                            for n in ("L_ANKLE", "R_ANKLE")]
        env._gc_prev = robot.data.body_pos_w[
            :, env._gc_feet_idx, 2].min(dim=1).values.clone()
    g = robot.data.body_pos_w[:, env._gc_feet_idx, 2].min(dim=1).values
    d = (g - env._gc_prev).clamp(-cap, cap)
    env._gc_prev = g.clone()
    cmd = env.command_manager.get_term(command_name)
    slope = getattr(cmd, "_env_slope_deg", None)
    if slope is None:
        return torch.zeros(env.num_envs, device=env.device)
    direction = torch.sign(slope)
    fresh = env.episode_length_buf <= 1
    return torch.where(fresh, torch.zeros_like(d), d * direction)
