# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
import re
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv, quat_apply

KAPPA = 0.5

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def vdot_tanh(env: ManagerBasedRLEnv, command_name: str, alpha: float = 1.0) -> torch.Tensor:
    # Retrieve the CLF-related quantities: V and its time derivative
    ref_term = env.command_manager.get_term(command_name)  # [B]
    vdot = ref_term.vdot  # [B]
    v = ref_term.v        # [B]

    # Compute the CLF decay condition violation
    clf_decay_violation = vdot + alpha * v  # [B]

    # Reward is higher when this violation is negative (i.e., condition is satisfied)
    vdot_reward = torch.tanh(-clf_decay_violation)  # [B]

    return vdot_reward


def clf_reward(env: ManagerBasedRLEnv, command_name: str, max_eta_err: float = 0.15, eps: float = 1e-6) -> torch.Tensor:
    """CLF-based reward: r = exp(-V(η) / V_max), clipped to [0, 1]."""

    ref_term = env.command_manager.get_term(command_name)
    v = ref_term.v  # [B] scalar CLF value per env
    max_clf = ref_term.clf.lambda_max * max_eta_err ** 2 + eps # principled normalization; lambda_max(P) * eta**2

    reward = torch.exp(-v / max_clf)
    # reward = torch.exp(-torch.clamp(v, max=5.0 * max_clf) / max_clf)
    # reward = torch.exp(-torch.clamp(v, max=200 * max_clf) / (10*max_clf))    # 200, 100   # NOTE: Used for bend over (10*max_clf is normal)
    return reward

def base_pos_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_pos = cmd_term.clf.v_subgroups["pelvis_pos"]

    return torch.exp(-KAPPA * v_base_pos / sigma)

def base_lin_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_lin_vel = cmd_term.clf.v_subgroups["pelvis_lin_vel"]

    return torch.exp(-KAPPA * v_base_lin_vel / sigma)

def base_ori_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_ori = cmd_term.clf.v_subgroups["pelvis_ori"]

    return torch.exp(-KAPPA * v_base_ori / sigma)

def base_ang_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_ang_vel = cmd_term.clf.v_subgroups["pelvis_ang_vel"]

    return torch.exp(-KAPPA * v_base_ang_vel / sigma)

def joint_pos_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_joint_pos = cmd_term.clf.v_subgroups["joint_pos"]

    return torch.exp(-KAPPA * v_joint_pos / sigma)

def joint_vel_group_reward(env: ManagerBasedRLEnv, command_name: str,
                           sigma: float, group: str) -> torch.Tensor:
    """Velocity tracking for one limb group — the anti-parking term.

    Position error cannot distinguish "swinging with a small lag"
    from "parked at a clever offset"; VELOCITY error can. A parked
    arm has ~zero joint velocity against an oscillating reference,
    so it pays the full swing amplitude continuously, while a swing
    lagging by dphi costs only ~sin(dphi/2) — small lags near-free
    (the phase-tolerance requirement, with no phase-search
    machinery). rough21-23 walks parked arms behind the torso as
    lean ballast; this term makes that the expensive strategy."""
    cmd_term = env.command_manager.get_term(command_name)
    v = cmd_term.clf.v_subgroups[f"joint_vel_{group}"]
    return torch.exp(-KAPPA * v / sigma)


def joint_group_deviation_cap(env: ManagerBasedRLEnv, command_name: str,
                              group: str, theta_max: float) -> torch.Tensor:
    """Hinge on per-joint POSITION deviation beyond theta_max.

    Inside the band deviation is free (slight off-reference motion
    and phase lag are allowed by design); outside it grows
    quadratically. Bounds the parked-arm offset (typically 0.5-1.0
    rad) without taxing in-band swing. Use with NEGATIVE weight."""
    cmd_term = env.command_manager.get_term(command_name)
    idx = cmd_term.clf.subgroup_indices[f"joint_pos_{group}"]
    eta_pos = cmd_term.clf.eta[:, idx]
    excess = (eta_pos.abs() - theta_max).clamp(min=0.0)
    return (excess * excess).sum(dim=-1)


def joint_abs_band(env: ManagerBasedRLEnv, joint_names: list,
                   lo: float, hi: float, asset_cfg=None) -> torch.Tensor:
    """Quadratic hinge on ABSOLUTE joint angle outside [lo, hi].

    Ref-relative bands could not break the elbow riding its -2.2
    stop (rough28: the fold now cycles with the gait but troughs at
    the stop, ~1 rad past the reference). The trough lives at a
    fixed ANGLE, not a fixed deviation — an absolute band names the
    style rule directly: the walking elbow pumps to lo and no
    deeper, whatever the reference is doing. Use with NEGATIVE
    weight."""
    from isaaclab.assets import Articulation
    asset: Articulation = env.scene["robot"]
    key = "_abs_band_idx_" + "_".join(joint_names)
    if not hasattr(env, key):
        ids, _ = asset.find_joints(joint_names, preserve_order=True)
        setattr(env, key, torch.tensor(ids, dtype=torch.long,
                                       device=env.device))
    idx = getattr(env, key)
    q = asset.data.joint_pos[:, idx]
    excess = (q - hi).clamp(min=0.0) + (lo - q).clamp(min=0.0)
    return (excess * excess).sum(dim=-1)


def joint_group_deviation_cap_asym(
        env: ManagerBasedRLEnv, command_name: str, group: str,
        theta_max: float, bands: dict) -> torch.Tensor:
    """Per-DIRECTION hinge on per-joint position deviation.

    v6's symmetric 0.4 rad band let the policy wander to yaw -0.95
    (deeper in than the removed carriage), press roll into the torso
    and park the elbow folded at eta -1.7 — all unpriced until the
    self-collision impact toppled the robot (rough26 rollout curves,
    2026-08-27). bands maps joint name -> (lo, hi) bounds on
    eta = act - ref; unlisted joints keep +-theta_max. Outward
    freedom stays; inward (toward the body) is tight, from the
    collision-bank clearance table. Use with NEGATIVE weight."""
    cmd_term = env.command_manager.get_term(command_name)
    idx = cmd_term.clf.subgroup_indices[f"joint_pos_{group}"]
    key = f"_asym_bands_{group}"
    if not hasattr(cmd_term, key):
        # subgroup indices address the INTERLEAVED eta
        # (pos = 2*i, vel = 2*i + 1); the tangent-channel name
        # index is i // 2 (first launch died on the raw index).
        names = [cmd_term.ordered_vel_output_names[int(i) // 2]
                 .removeprefix("joint:") for i in idx]
        unknown = set(bands) - set(names)
        if unknown:
            raise ValueError(
                f"asym bands name joints not in group {group}: "
                f"{sorted(unknown)} (group has {names})")
        device = cmd_term.clf.eta.device
        lo = torch.full((len(names),), -theta_max, device=device)
        hi = torch.full((len(names),), theta_max, device=device)
        for j, n in enumerate(names):
            if n in bands:
                lo[j], hi[j] = bands[n]
        setattr(cmd_term, key, (lo, hi))
    lo, hi = getattr(cmd_term, key)
    eta_pos = cmd_term.clf.eta[:, idx]
    excess = (eta_pos - hi).clamp(min=0.0) + (lo - eta_pos).clamp(min=0.0)
    return (excess * excess).sum(dim=-1)


def upright_bonus(env: ManagerBasedRLEnv, sigma_deg: float,
                  asset_cfg=None) -> torch.Tensor:
    """Per-step bonus for being upright — duty-cycle uprightness.

    Summed over an episode this rewards MOSTLY-upright strategies:
    a brief lean (recovery, acceleration) forfeits a few steps of
    bonus; sustained lean-as-strategy bleeds continuously. The
    torso_pitch deadband stays wide so transients remain cheap —
    this term shapes the preference, the deadband sets the limit."""
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import quat_apply_inverse
    import math
    asset: Articulation = env.scene["robot"]
    grav = quat_apply_inverse(asset.data.root_quat_w,
                              asset.data.GRAVITY_VEC_W)
    # tilt angle from vertical via the projected gravity xy magnitude
    tilt = torch.asin(grav[:, :2].norm(dim=-1).clamp(max=1.0))
    sigma = math.radians(sigma_deg)
    return torch.exp(-(tilt / sigma) ** 2)


def joint_pos_group_error_l2(env: ManagerBasedRLEnv, command_name: str,
                             group: str) -> torch.Tensor:
    """||eta_group|| — LINEAR tracking distance to the reference.

    The exp form above cannot RETRIEVE a limb once it leaves the
    basin (gradient ~ exp(-KAPPA*v/sigma) vanishes at large v —
    rough21 walked 3400 iters with both arms wrapped at chest level
    and Episode_Reward/joint_pos_arms flat at ~0.13). Stability of
    CLF-RL (arXiv 2605.01978) Lemma 5 sandwiches the exponential by
    LINEAR costs on sublevel sets — the exponential is training
    numerics, not the guarantee — so a linear-in-error penalty
    toward the REFERENCE (not a fixed pose) is the theory-consistent
    retrieval term: constant-magnitude gradient at any distance,
    and it vanishes exactly on-orbit, so it cannot fight the style.

    Use with a NEGATIVE weight."""
    cmd_term = env.command_manager.get_term(command_name)
    v = cmd_term.clf.v_subgroups[f"joint_pos_{group}"]
    return torch.sqrt(v + 1e-8)


def joint_pos_group_reward(env: ManagerBasedRLEnv, command_name: str,
                           sigma: float, group: str) -> torch.Tensor:
    """joint_pos tracking for ONE limb group: legs / arms / torso.

    The combined term gives every joint a single shared exponential,
    so a large leg error -- which any uneven ground forces -- drives
    exp(-KAPPA*V/sigma) toward zero and takes the gradient for ARM
    tracking with it. The arms then drift to whatever posture helps
    balance, which is functional but off-style.

    Splitting restores an independent gradient per group. sigma then
    expresses TOLERANCE (how far a group may deviate before the
    reward stops caring) and weight expresses PRIORITY (how hard it
    is pulled back), which is what lets the arms do balance work and
    still come home. sigma should carry sqrt(n_joints) so the
    per-joint scale matches the combined term.
    """
    cmd_term = env.command_manager.get_term(command_name)
    v = cmd_term.clf.v_subgroups[f"joint_pos_{group}"]

    return torch.exp(-KAPPA * v / sigma)


def joint_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_joint_vel = cmd_term.clf.v_subgroups["joint_vel"]

    return torch.exp(-KAPPA * v_joint_vel / sigma)

def body_pos_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_pos = cmd_term.clf.v_subgroups["other_body_pos"]

    return torch.exp(-KAPPA * v_body_pos / sigma)

def body_lin_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_lin_vel = cmd_term.clf.v_subgroups["other_body_lin_vel"]

    return torch.exp(-KAPPA * v_body_lin_vel / sigma)

def body_ori_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_ori = cmd_term.clf.v_subgroups["other_body_ori"]

    return torch.exp(-KAPPA * v_body_ori / sigma)

def body_ang_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_ang_vel = cmd_term.clf.v_subgroups["other_body_ang_vel"]

    return torch.exp(-KAPPA * v_body_ang_vel / sigma)

def clf_decreasing_condition(
    env: ManagerBasedRLEnv,
    command_name: str,
    alpha: float = 1.0,
    eta_max: float = 0.15,
    eta_dot_max: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalty for violating CLF decrease condition: 𝑟 = clip((ΔV + αV) / max_violation, [0, 1])
    where:
        max_violation ≈ 2‖P‖ η_max η̇_max + α λ_max(P) η_max²
    """

    ref_term = env.command_manager.get_term(command_name)
    v = ref_term.v        # [B]
    vdot = ref_term.vdot  # [B]

    lambda_max = ref_term.clf.lambda_max
    norm_P = ref_term.clf.norm_P

    # Theoretical upper bound on violation
    max_violation = (
        2.0 * norm_P * eta_max * eta_dot_max + alpha * lambda_max * eta_max ** 2 + eps
    )
    # Only penalize when violation is positive
    violation = torch.clamp(vdot + alpha * v, min=0.0)
    penalty = violation / max_violation
    penalty = torch.clamp(penalty, min=0.0, max=1.0)
    return penalty


def v_dot_penalty(env: ManagerBasedRLEnv, command_name: str,eta_max: float = 0.15,
    eta_dot_max: float = 0.5,eps: float = 1e-6) -> torch.Tensor:
    ref_term = env.command_manager.get_term(command_name)                    # [B]
    vdot = ref_term.vdot # [B]

    norm_P = ref_term.clf.norm_P

    max_violation = (
        2.0 * norm_P * eta_max * eta_dot_max + eps
    )

    vdot_penalty = torch.tanh(torch.clamp(vdot, min=0.0) / max_violation) 
    return vdot_penalty


def contact_no_vel(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward feet contact with zero velocity."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids] * contacts.unsqueeze(-1)
    # shape [B, num_feet, 3]
    penalize = torch.square(body_vel[:,:,:3])
    return torch.sum(penalize, dim=(1,2))


def holonomic_constraint_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_vel: float = (0.1)**0.5
) -> torch.Tensor:
    """
    Unified holonomic‐velocity constraint reward:
      r = exp( – ‖[v, ω_z]‖² / σ_vel² )
    where v∈R³ is the foot’s linear velocity and ω_z its yaw rate.
    Using σ_vel=√0.1 matches the original bandwidth (denominator=0.1).
    """
    cmd = env.command_manager.get_term(command_name)

    # Get the velocities
    v = cmd.current_contact_vels

    # # linear velocity [B,3] and yaw rate [B,1]
    # v = cmd.stance_foot_vel  # [vx, vy, vz]
    # wz = cmd.stance_foot_ang_vel[:, 2].unsqueeze(-1)  # [ω_z]
    #
    # # stack into [B,4] error vector
    # e_vel = torch.cat([v, wz], dim=-1)
    #
    # not_flight_mask = cmd.get_not_flight_envs()
    # return not_flight_mask * torch.exp(- (e_vel**2).sum(dim=-1) / sigma_vel**2)

    return torch.exp(-(v.sum(dim=-1).sum(dim=-1)**2) / sigma_vel**2)

def holonomic_constraint(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_pose: float = (5 * 0.01) ** 0.5,
    z_offset: float = 0.036
) -> torch.Tensor:
    """
    Unified holonomic‐pose constraint reward:
        r = exp( – ‖e_pose‖² / σ_pose² )
    where e_pose = [Δx, Δy, Δz, φ, Δψ] and
      • Δx, Δy are planar errors from the recorded foot position,
      • Δz = p_z_cur – z_offset (encourages foot to stay on the floor),
      • φ is roll,
      • Δψ is yaw error wrapped to [–π, π].
    """

    cmd = env.command_manager.get_term(command_name)

    # TODO: Re-write to handle arbitrary contacts

    # Get the current pose
    des_contact_poses = cmd.desired_contact_poses
    contact_poses = cmd.current_contact_poses

    # Compute error
    pose_err = contact_poses - des_contact_poses

    # Wrap yaw error
    pose_err = wrap_to_pi(pose_err[:, -1])

    # # planar position error [B,2]
    # p0_xy = cmd.stance_foot_pos_0[:, :2]
    # p_xy = cmd.stance_foot_pos[:, :2]
    # delta_xy = p_xy - p0_xy
    #
    # # vertical error to the floor plane [B,1]
    # z_cur = cmd.stance_foot_pos[:, 2].unsqueeze(-1)
    # delta_z = z_cur - cmd.stance_foot_pos_0[:, 2].unsqueeze(-1)
    #
    # # roll error [B,1]
    # roll = cmd.stance_foot_ori[:, 0].unsqueeze(-1)
    #
    # # yaw error wrapped to [–π, π] [B,1]
    # psi0 = cmd.stance_foot_ori_0[:, 2]
    # psi = cmd.stance_foot_ori[:, 2]
    # delta_psi = ((psi - psi0 + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)
    #
    # # stack into [B,5] error vector
    # e_pose = torch.cat([delta_xy, delta_z, roll, delta_psi], dim=-1)
    #
    # not_flight_mask = cmd.get_not_flight_envs()
    # return not_flight_mask * torch.exp(- (e_pose ** 2).sum(dim=-1) / sigma_pose ** 2)

    return torch.exp(-(pose_err**2).sum(dim=-1) / sigma_pose ** 2)

def reference_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    term_std: Sequence[float],
    term_weight: Sequence[float],
) -> torch.Tensor:
    """
    Exponential reward per dimension, scaled by weight — ignores zero-weight terms.
    """
    command = env.command_manager.get_term(command_name)
    err = command.y_act - command.y_out  # [B, D]

    weight_vec = torch.as_tensor(term_weight, dtype=err.dtype, device=err.device)  # [D]
    std_vec = torch.as_tensor(term_std, dtype=err.dtype, device=err.device)        # [D]

    # [B, D] scaled squared error per dimension
    err_sq_scaled = (err ** 2) / (std_vec ** 2)

    # Apply element-wise exp(-error²/std²) and weight
    reward_per_dim = weight_vec * torch.exp(-err_sq_scaled)  # [B, D]
    reward = reward_per_dim.sum(dim=1)/torch.sum(weight_vec)  # [B]

    return reward


def reference_vel_tracking(    env: ManagerBasedRLEnv,
    command_name: str,
    term_std: Sequence[float],
    term_weight: Sequence[float],
) -> torch.Tensor:
    """Reference tracking with element-wise term weights."""
    # 1. fetch the command and compute error [B, D]
    command = env.command_manager.get_term(command_name)
    err = command.dy_act - command.dy_out

    weight_vec = torch.as_tensor(term_weight, dtype=err.dtype, device=err.device)  # [D]
    std_vec = torch.as_tensor(term_std, dtype=err.dtype, device=err.device)        # [D]

    # [B, D] scaled squared error per dimension
    err_sq_scaled = (err ** 2) / (std_vec ** 2)

    # Apply element-wise exp(-error²/std²) and weight
    reward_per_dim = weight_vec * torch.exp(-err_sq_scaled)  # [B, D]
    reward = reward_per_dim.sum(dim=1)/torch.sum(weight_vec)  # [B]
    return reward


def foot_clearance(env: ManagerBasedRLEnv,
                   target_height: float,
                   sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
                   height_sensor_cfg: SceneEntityCfg | None = None,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward foot clearance."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get contact state
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    if height_sensor_cfg is not None:
        sensor: RayCaster = env.scene[height_sensor_cfg.name]
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[...,2],dim=1).unsqueeze(-1)
    else:
        adjusted_target_height = target_height

    # Calculate foot heights
    feet_z_err = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - adjusted_target_height
    pos_error = torch.square(feet_z_err) * ~contacts

    return torch.sum(pos_error, dim=(1))

def phase_contact(
    env: ManagerBasedRLEnv,
        period: float = 0.8,
        command_name: str | None = None,
        Tswing: float =0.4,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward foot contact with regards to phase."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get contact state
    res = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    # Contact phase
    tp = (env.sim.current_time % period) / period     # Scaled between 0-1
    phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + Tswing), device=env.device)

    stance_i = int(0.5 - 0.5 * torch.sign(phi_c))


     # check if robot needs to be standing
    if command_name is not None:
        command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
        is_small_command = command_norm < 0.005
        for i in range(2):
            is_stance = stance_i == i
            # set is_stance to be true if the command is small
            is_stance = is_stance | is_small_command
            contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids[i], :].norm(dim=-1).max(dim=1)[0] > 1.0
            res += ~(contact ^ is_stance)
    else:
        for i in range(2):
            is_stance = stance_i == i
            # set is_stance to be true if the command is small
            is_stance = is_stance
            contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids[i], :].norm(dim=-1).max(dim=1)[0] > 1.0
            res += ~(contact ^ is_stance)
    return res

# TODO: Test
def contact_schedule_penalty(env: ManagerBasedRLEnv, command_name: str,
                           sensor_cfg: SceneEntityCfg, weight_scalar: float) -> torch.Tensor:
    """Penalize contacts while in the flight phase."""
    cmd = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Time into the episode
    t = env.episode_length_buf * env.step_dt

    # Get bodies not in contact for each env
    contact_states = cmd.get_contact_state(t)
    contact_body_names = cmd.contact_bodies

    contact_forces = torch.zeros(t.shape[0], dtype=torch.float, device=env.device)
    for i, body_name in enumerate(contact_body_names):
        contact_mask = contact_states[:, i] == 1
        indices = torch.tensor([i for i, v in enumerate(sensor_cfg.body_names) if v == body_name])
        body_id = sensor_cfg.body_ids[indices]
        contact_forces[contact_mask] += contact_sensor.data.net_forces_w[contact_mask, body_id, :].norm(dim=-1)  # Gets the most recent force only

    penalty = weight_scalar * torch.tanh(contact_forces / 0.5)  # TODO: Think about if this is what I want
    return penalty

def track_lin_vel_y_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error =  torch.square(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1])
    return torch.exp(-lin_vel_error / std**2)


def ankle_roll_zero(
    env: ManagerBasedRLEnv, std: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward keeping both ankle roll joints near zero position using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get ankle roll joint indices - these are typically the last joints in each leg
    # Based on the controller.py joint order:
    # Index 19: left_ankle_roll_joint
    # Index 20: right_ankle_roll_joint
    ankle_roll_indices = [19, 20]  # left and right ankle roll joints
    
    # Get current ankle roll joint positions
    ankle_roll_positions = asset.data.joint_pos[:, ankle_roll_indices]  # [B, 2]
    
    # Compute squared error from zero position
    ankle_roll_error = torch.square(ankle_roll_positions)  # [B, 2]
    
    # Sum errors for both ankle roll joints and apply exponential kernel
    total_error = ankle_roll_error.sum(dim=-1)  # [B]
    reward = torch.exp(-total_error / std**2)
    
    return reward

def torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.
    For implicit actuators, we manually compute the PD controller torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Manually compute PD controller torques for implicit actuators
    computed_torque = torch.zeros_like(asset.data.joint_pos)
    
    # Get current joint positions, velocities, and desired positions
    current_pos = asset.data.joint_pos
    current_vel = asset.data.joint_vel
    desired_pos = asset.data.joint_pos_target
    
    # Access actuator configurations from the asset
    actuator_groups = asset.cfg.actuators
    
    for group_name, actuator_cfg in actuator_groups.items():
        # Get joint indices for this actuator group
        joint_indices = asset.find_joints(actuator_cfg.joint_names_expr)[0]
        
        # Get stiffness and damping values for this group
        if isinstance(actuator_cfg.stiffness, dict):
            # Handle per-joint stiffness values
            kp_values = torch.zeros(len(joint_indices), dtype=torch.float32, device=env.device)
            for i, joint_idx in enumerate(joint_indices):
                joint_name = asset.joint_names[joint_idx]
                # Find matching stiffness pattern
                for pattern, value in actuator_cfg.stiffness.items():
                    if re.match(pattern.replace(".*", ".*"), joint_name):
                        kp_values[i] = value
                        break
        else:
            # Single stiffness value for all joints in this group
            kp_values = torch.full((len(joint_indices),), actuator_cfg.stiffness, dtype=torch.float32, device=env.device)
        
        if isinstance(actuator_cfg.damping, dict):
            # Handle per-joint damping values
            kd_values = torch.zeros(len(joint_indices), dtype=torch.float32, device=env.device)
            for i, joint_idx in enumerate(joint_indices):
                joint_name = asset.joint_names[joint_idx]
                # Find matching damping pattern
                for pattern, value in actuator_cfg.damping.items():
                    if re.match(pattern.replace(".*", ".*"), joint_name):
                        kd_values[i] = value
                        break
        else:
            # Single damping value for all joints in this group
            kd_values = torch.full((len(joint_indices),), actuator_cfg.damping, dtype=torch.float32, device=env.device)
        
        # Compute PD torques for this group: tau = kp * (q_des - q) - kd * q_dot
        pos_error = desired_pos[:, joint_indices] - current_pos[:, joint_indices]
        pd_torque = (kp_values[None, :] * pos_error - kd_values[None, :] * current_vel[:, joint_indices])
        
        # Store computed torques
        computed_torque[:, joint_indices] = pd_torque
    
    # Compute torque limit violations
    torque_limits_upper = asset.data.joint_effort_limits[0, asset_cfg.joint_ids]  # Upper limits

    # Get computed torques for the specified joints
    joint_torques = computed_torque[:, asset_cfg.joint_ids]
    
    # Compute violations: how much torques exceed the limits
    violation = torch.clamp(torch.abs(joint_torques) - torque_limits_upper, min=0)

    # Sum all violations
    return torch.sum(violation, dim=1)

def body_upright_roll(env,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                      ) -> torch.Tensor:
    """tinh: L2 penalty on a BODY's world roll tilt (gravity's lateral
    component in the body frame).

    The chest-upright ask is a SUM (base roll + TORSO_ROLL): weighting
    the two CLF channels separately let the policy park the PELVIS
    level and swing the chest anti-phase with the tracked joint
    (user viewer note 2026-08-07). Penalizing the summed quantity is
    assignment-free: pelvis and SONAX split the roll however dynamics
    prefer, as long as the chest stays level. Pitch is left free.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    quat = asset.data.body_quat_w[:, body_id]
    # gravity direction in body frame; y-component = roll tilt
    g_w = torch.tensor([0.0, 0.0, -1.0], device=quat.device).expand(
        quat.shape[0], 3)
    g_b = quat_rotate_inverse(quat, g_w)
    return torch.square(g_b[:, 1])


def forward_progress_heading(env, command_name: str,
                             upright_gate: bool = True,
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """tinh: forward_progress measured in the YAW-ONLY heading frame.

    forward_progress reads root_lin_vel_b[:, 0] -- body-frame X. That
    axis tilts with PITCH, so during a forward faceplant it rotates to
    point partly downward and the fall velocity projects onto it:

        v . x_b = v_fwd cos(theta) - v_down sin(theta)

    with v_down < 0 while falling, BOTH terms are positive. A faceplant
    scores as forward progress, and at weight 10 this was 43% of all
    positive reward -- the single largest incentive in the stack was
    partly satisfied by falling over (measured 2026-08-25, rough17:
    50.66% of steps beyond 60 deg from vertical).

    Projecting onto the yaw-only heading instead makes the reference
    axis horizontal by construction, so vertical motion contributes
    exactly zero however far the torso has pitched.

    upright_gate additionally zeroes the reward past 60 deg of tilt, so
    a robot that is already going down cannot bank progress on the way.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    v_heading = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w),
                                    asset.data.root_lin_vel_w)
    v_x = v_heading[:, 0]

    cmd_x = cmd[:, 0]
    denom = torch.clamp(torch.abs(cmd_x), min=0.1)
    r = torch.clamp(v_x * torch.sign(cmd_x) / denom, -1.0, 1.0)

    if upright_gate:
        upright = -asset.data.projected_gravity_b[:, 2]   # 1.0 upright
        r = torch.where(upright > math.cos(math.radians(60.0)),
                        r, torch.zeros_like(r))
    return r


def body_upright_pitch(env,
                       deadband_deg: float = 15.0,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                       ) -> torch.Tensor:
    """tinh: L2 penalty on a BODY's world PITCH tilt, outside a deadband.

    body_upright_roll penalises g_b[1] (lateral) and says outright
    "Pitch is left free" -- so before this, NOTHING in the reward stack
    priced a forward fall. The only forward-pitch signal was the
    torso_contact termination, which is a cliff at the moment of impact
    with no gradient leading up to it.

    The deadband exists because the walking reference carries a real
    forward lean; penalising absolute pitch would fight the trajectory
    the policy is being paid to track. Beyond it the penalty grows
    quadratically, so there is a slope pointing away from the fall well
    before the 50 deg bad_orientation termination fires.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    quat = asset.data.body_quat_w[:, body_id]
    g_w = torch.tensor([0.0, 0.0, -1.0], device=quat.device).expand(
        quat.shape[0], 3)
    g_b = quat_rotate_inverse(quat, g_w)
    # x-component = fore/aft tilt. Positive and negative pitch both cost.
    tilt = torch.abs(g_b[:, 0])
    dead = math.sin(math.radians(deadband_deg))
    return torch.square(torch.clamp(tilt - dead, min=0.0))


def forward_progress(env, command_name: str,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """tinh: signed, capped, LINEAR forward-velocity reward.

    Integrates to net displacement along the commanded direction, so
    in-place rocking averages to ~zero — the exp velocity kernel at
    high weight taught the LPA to rock at the commanded speed with no
    travel (reward up, displacement down; tinh-95oj). Backward motion
    is penalized symmetrically; reward caps at 1 when the command is
    met (no bonus for overspeed).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    v_x = asset.data.root_lin_vel_b[:, 0]
    cmd_x = cmd[:, 0]
    denom = torch.clamp(torch.abs(cmd_x), min=0.1)
    return torch.clamp(v_x * torch.sign(cmd_x) / denom, -1.0, 1.0)


def _arm_torso_rows(asset: Articulation, margin: float):
    """Tensors for the bank's arm/torso sphere rows: body ids and
    link-frame offsets of the two sphere sets [R,·], summed radii [R],
    and the per-row margin min(margin, rest - headroom) [R] -- the
    collision_bank clamp, applied per row."""
    from robot_rl.tasks.manager_based.robot_rl.lpa.lpa_collision_spheres import SPHERES
    from robot_rl.tasks.manager_based.robot_rl.lpa import lpa_arm_torso_pairs as tab
    if len(SPHERES) != tab.N_SPHERES:
        raise ValueError(f"sphere bank copy has {len(SPHERES)} spheres, the row table "
                         f"was generated against {tab.N_SPHERES}: regenerate both")
    names = list(asset.data.body_names)
    ia, oa, it, ot, rsum, marg = [], [], [], [], [], []
    for a, t, rest in tab.ROWS:
        la, oa_, ra, _ = SPHERES[a]
        lt, ot_, rt, _ = SPHERES[t]
        ia.append(names.index(la))          # ValueError if the body is absent: loud
        it.append(names.index(lt))
        oa.append(oa_)
        ot.append(ot_)
        rsum.append(ra + rt)
        marg.append(min(margin, rest - tab.HEADROOM_M))
    dev = asset.data.body_pos_w.device
    f = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
    return (torch.tensor(ia, device=dev), f(oa),
            torch.tensor(it, device=dev), f(ot), f(rsum), f(marg))


def _sphere_centers(asset: Articulation, ids, off) -> torch.Tensor:
    """World centres [E, R, 3] of spheres attached to bodies `ids` at
    link-frame offsets `off`."""
    p = asset.data.body_pos_w[:, ids]                       # [E, R, 3]
    q = asset.data.body_quat_w[:, ids]                      # [E, R, 4]
    return p + quat_apply(q, off.unsqueeze(0).expand(p.shape[0], -1, -1))


def arm_torso_clearance(env, margin: float = 0.02,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                        ) -> torch.Tensor:
    """tinh (am-m7l.14): price the ARM-to-TORSO margin BEFORE contact.

    Every reset on the clad walking line is torso_contact -- forearm
    or hand into the chest, zero falls, on four checkpoints -- and the
    only term that saw it was that termination: a cliff at >1 N with
    no gradient leading up to it. Joint-space tracking weight did not
    move the arms (cladwalkarm, 8x: 73 resets vs 74) and the banked
    trajopt reference could not be solved (am-m7l.11). So this prices
    the geometry itself, on the policy's live state.

    THE ROWS ARE THE BANK'S, WITH THE BANK'S CLAMP. The trajopt's
    collision bank prices only what is separated at rest: each pair's
    margin is min(margin, rest_clearance - 2 mm). lpa_arm_torso_pairs.ROWS
    (generated by //modules/collision:arm_torso_pairs) carries the
    arm/TORSO_ROLL sphere rows of the bank with their rest clearance
    (all 224 are separated at q=0: shoulder-roll rows 6-9 mm, upper-arm
    80 mm, forearm/hand 241 mm), and this term clamps its margin per
    row the way collision_bank does per link pair.

    MEASURED PARITY (2026-09-01, lpa_sphere_parity_probe on the
    cladwalkclr smoke bundle, settled reset pose): these rows on Isaac
    body poses give min 2.8 mm on R_SHOULDER_ROLL/TORSO_ROLL (bank rest
    6.2 mm; the trajopt's cycle worst is 2.9 mm on that pair), median
    393 mm, and the eight tightest rows sit within 2-4 mm of the bank's
    rest values. Body frames and offsets agree. An earlier all-pairs
    draft of this term printed -370 mm min / -236 mm median at its
    first call; that number was the draft's, not the geometry's.

    Per row: d = |c_arm - c_torso| - r_arm - r_torso, penalty
    ((m_row - d)/m_row)^2 for d < m_row else 0, summed over rows -- a
    quadratic ramp, zero when clear, 1 per row at touch, growing with
    intrusion. Body frames are the URDF link frames the offsets assume
    (modules/wbc/lpa_sim/lpa_isaac_fk_parity.py holds that parity).

    The first call prints the minimum row clearance across envs. At a
    standing-ish reset pose it should be small and mostly POSITIVE
    (the bank's binding pair rests at 6.1 mm); hundreds of millimetres
    negative means the offsets are in the wrong frame or the row table
    is stale, and the term is pricing nothing real.

    Returns the SUM over rows (weight it negative). Row tensors are
    cached on the env after the first call.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cache = getattr(env, "_arm_torso_clearance_cache", None)
    if cache is None:
        cache = _arm_torso_rows(asset, margin)
        env._arm_torso_clearance_cache = cache
    ia, oa, it, ot, rsum, marg = cache
    ca = _sphere_centers(asset, ia, oa)                     # [E, R, 3]
    ct = _sphere_centers(asset, it, ot)                     # [E, R, 3]
    d = torch.linalg.norm(ca - ct, dim=-1) - rsum[None, :]  # [E, R]
    if not getattr(env, "_arm_torso_clearance_printed", False):
        env._arm_torso_clearance_printed = True
        dmin = d.amin(dim=1)
        print(f"[arm_torso_clearance] rows={d.shape[1]} "
              f"first-call min clearance: min {dmin.min().item()*1e3:.1f} mm, "
              f"median {dmin.median().item()*1e3:.1f} mm, "
              f"max {dmin.max().item()*1e3:.1f} mm "
              f"(margin {margin*1e3:.0f} mm, row margins "
              f"{marg.min().item()*1e3:.1f}..{marg.max().item()*1e3:.1f} mm)",
              flush=True)
    pen = torch.clamp(marg[None, :] - d, min=0.0) / marg[None, :]
    return torch.sum(pen * pen, dim=1)


# Arm points the mirror term compares, in each ELBOW body frame: the
# elbow origin and the hand sphere centre from the collision bank
# (lpa_collision_spheres L_ELBOW#3 / R_ELBOW#3).
_ARM_POINTS = (("L_ELBOW", (0.0, 0.018, -0.43)),
               ("R_ELBOW", (0.0, -0.015, -0.43)))


def arm_phase_mirror(env, shift_s: float = 0.745, unshifted_cap: float = 0.04,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                     ) -> torch.Tensor:
    """tinh (am-m7l.18): the two arms must be the SAME arm, half a gait
    cycle apart -- measured in space, not in joint angles.

    Asa on the cladwalkclr render (2026-09-01): 'the right arm looks
    good, but the left arm is very different from it. can we have a
    term that penalizes differences from left to right, but only when
    shifted by the counterswing phase?'

    WHY CARTESIAN. The reference IS an exact physical mirror with a
    half-cycle delay: with pinocchio FK in the CORE frame, each hand
    matches the y-mirror of the other hand half a cycle earlier to
    1 mm (284 mm unshifted), each elbow to 2 mm (163 mm)
    (automaton //modules/trajopt/walk_stack:ref_mirror_probe). In
    JOINT space the same relation is not the axis-string table: the
    fit on the reference is roll S=+1, yaw S=-1, elbow S=-1 (exact)
    and pitch S=+1 with a 0.564 rad offset and a 0.11 rad residual,
    because the two sides' joint conventions differ (tinh-as7q.21).
    A joint-space draft with the axis-string signs trained the left
    roll and elbow toward the wrong sign (cladwalksym, killed). Space
    is convention-free, is what the reference satisfies, and is what
    a viewer sees.

    THE TERM. For each side, two points in the base frame -- the elbow
    body origin and the hand sphere centre (bank offsets, in the
    elbow frame). With M = diag(1, -1, 1) the sagittal mirror:
        e_L(t) = p_L(t) - M p_R(t - shift),  e_R(t) = p_R(t) - M p_L(t - shift)
    shifted = mean squared distance over the two points and both
    directions, in m^2.

    THE DEGENERATE SOLUTION, AND WHY THE SECOND HALF EXISTS. "Left
    equals mirrored right one step earlier" is ALSO satisfied by both
    arms swinging together at twice the gait frequency: one full arm
    cycle per step makes a one-step delay a whole period. cladwalksym3
    (2026-09-01) found exactly that -- its training shift check read
    6 cm and Asa saw both arms exactly in sync. So the term is
        err = shifted - min(unshifted, unshifted_cap)
    where unshifted = the same mirror error with no delay. A true
    counterswing has shifted ~0 and unshifted large (the reference:
    0.05-0.06 m^2 on these points, 284 mm mean |dx| on the hands);
    the in-phase solution has both ~0 and forfeits the cap. With the
    cap at 0.04 m^2 the counterswing is worth 0.04 x weight per step
    over the degenerate one, and nothing beyond the reference's own
    asymmetry is rewarded.

    shift_s is the step period: 0.745 s for the clad stomp (library
    cycle 1.489 s, half-periodic), 37 control steps at 50 Hz. The
    one-time print, once a quarter of the envs have a full buffer,
    gives shifted vs unshifted error: for a counterswing the shifted
    one must be the small one. On the reference the ratio is ~1:250.

    Per env a ring buffer holds the last shift+1 point sets; the term
    is zero until the episode is older than the shift, so a reset
    never pairs this episode's arm with the previous one's. Called
    exactly once per step by the reward manager, which is what
    advances the buffer.
    """
    return _phase_mirror(env, asset_cfg, "arm", _ARM_POINTS, shift_s, unshifted_cap)


# Feet: ankle origins. The reference's ankle paths mirror at half cycle
# to 0.3 cm rms and differ by 36.9 cm rms unshifted (ref_mirror_probe,
# 2026-09-01); cladwalkclr/cladwalksym3 measured 18.6-20.2 cm rms at the
# shift -- 'one step longer than the other' (Asa, am-m7l.19).
_FOOT_POINTS = (("L_ANKLE", (0.0, 0.0, 0.0)),
                ("R_ANKLE", (0.0, 0.0, 0.0)))


def foot_phase_mirror(env, shift_s: float = 0.745, unshifted_cap: float = 0.10,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                      ) -> torch.Tensor:
    """tinh (am-m7l.19): each foot's path is the other foot's, mirrored,
    half a cycle later. Same construction as arm_phase_mirror on the
    ankle origins (one point per side); cap 0.10 m^2, below the
    reference's own unshifted 0.136, so alternating legs earn the cap
    and a hop (both feet together) forfeits it."""
    return _phase_mirror(env, asset_cfg, "foot", _FOOT_POINTS, shift_s, unshifted_cap)


def _phase_mirror(env, asset_cfg, key: str, points, shift_s: float, unshifted_cap: float
                  ) -> torch.Tensor:
    """Shared body of arm_phase_mirror / foot_phase_mirror. `points` is
    ((L_body, offset), (R_body, offset)); each side contributes the body
    origin and, if the offset is non-zero, the offset point too."""
    asset: Articulation = env.scene[asset_cfg.name]
    attr = f"_{key}_phase_mirror_state"
    st = getattr(env, attr, None)
    if st is None:
        names = list(asset.data.body_names)
        dev = asset.data.body_pos_w.device
        k = int(round(shift_s / env.step_dt))
        if k < 1:
            raise ValueError(f"shift_s {shift_s} is under one control step {env.step_dt}")
        two = any(any(abs(c) > 0 for c in o) for _b, o in points)
        st = {
            "ids": torch.tensor([names.index(b) for b, _o in points], device=dev),
            "off": torch.tensor([o for _b, o in points], dtype=torch.float32, device=dev),
            "two": two,
            "mirror": torch.tensor([1.0, -1.0, 1.0], device=dev),
            "k": k,
            "buf": torch.zeros(env.num_envs, k + 1, 2, 2 if two else 1, 3, device=dev),
            "ptr": 0,
            # calls since this env's episode began, counted HERE: the
            # buffer is only as old as the calls that filled it, and
            # episode_length_buf can be older than that (a resumed run
            # whose envs already count as settled at the first call read
            # zero slots and printed 0.257 m^2 -- the arm points' |p|^2).
            "age": torch.zeros(env.num_envs, dtype=torch.long, device=dev),
            "last_ep": torch.full((env.num_envs,), -1, dtype=torch.long, device=dev),
            # the one-time print is a CYCLE AVERAGE: a single-step snapshot
            # after a synchronized reset samples one gait phase for every
            # env (measured 0.0037 where the 600-step probe gave 0.031).
            "acc": None, "acc_n": 0, "printed": False,
        }
        setattr(env, attr, st)
    root_p = asset.data.root_pos_w                              # [E, 3]
    root_q = asset.data.root_quat_w                             # [E, 4]
    bp = asset.data.body_pos_w[:, st["ids"]]                    # [E, 2, 3] body origins
    if st["two"]:
        bq = asset.data.body_quat_w[:, st["ids"]]               # [E, 2, 4]
        tip = bp + quat_apply(bq, st["off"].unsqueeze(0).expand(bp.shape[0], -1, -1))
        pts_w = torch.stack([bp, tip], dim=2)                   # [E, side, point, 3]
    else:
        pts_w = bp.unsqueeze(2)
    npt = pts_w.shape[2]
    rel = pts_w - root_p[:, None, None, :]
    rq = root_q[:, None, None, :].expand(-1, 2, npt, -1)
    pts = quat_rotate_inverse(rq.reshape(-1, 4), rel.reshape(-1, 3)).reshape(pts_w.shape)
    k, buf, ptr = st["k"], st["buf"], st["ptr"]
    buf[:, ptr] = pts
    old = (ptr + 1) % (k + 1)                                   # t - k
    pts_d = buf[:, old]
    st["ptr"] = old
    m = st["mirror"]
    e_l = pts[:, 0] - m * pts_d[:, 1]                           # L(t) vs M R(t-k)   [E, npt, 3]
    e_r = pts[:, 1] - m * pts_d[:, 0]
    shifted = 0.5 * (torch.sum(e_l * e_l, dim=-1).mean(dim=1) + torch.sum(e_r * e_r, dim=-1).mean(dim=1))
    e0 = pts[:, 0] - m * pts[:, 1]                              # L(t) vs M R(t): must NOT be small
    unshifted = torch.sum(e0 * e0, dim=-1).mean(dim=1)
    err = shifted - torch.clamp(unshifted, max=unshifted_cap)
    # Valid only once THIS buffer has seen two shifts of the current
    # episode: one shift back is the reset transient, and the buffer's
    # age is counted by calls, not by episode_length_buf (see "age").
    ep = env.episode_length_buf
    continuing = ep == st["last_ep"] + 1
    st["age"] = torch.where(continuing, st["age"] + 1, torch.zeros_like(st["age"]))
    st["last_ep"] = ep.clone()
    valid = st["age"] >= 2 * k
    err = torch.where(valid, err, torch.zeros_like(err))
    if not st["printed"]:
        settled = st["age"] >= 3 * k
        if int(settled.sum()) >= max(8, env.num_envs // 4):
            s = torch.stack([shifted[settled].mean(), unshifted[settled].mean(), err[settled].mean()])
            st["acc"] = s if st["acc"] is None else st["acc"] + s
            st["acc_n"] += 1
            if st["acc_n"] >= 2 * k:                            # two shifts = one full cycle
                a = (st["acc"] / st["acc_n"]).tolist()
                st["printed"] = True
                print(f"[{key}_phase_mirror] shift {shift_s:.3f} s = {k} steps; cycle average over "
                      f"{st['acc_n']} steps of settled envs: shifted mirror error {a[0]:.5f} m^2 "
                      f"(rms {a[0]**0.5*100:.1f} cm) vs unshifted {a[1]:.5f} m^2 (rms {a[1]**0.5*100:.1f} cm); "
                      f"term = shifted - min(unshifted, {unshifted_cap}) = {a[2]:.5f} "
                      f"(counterswing => shifted << unshifted; both small = in phase = degenerate)", flush=True)
    return err
